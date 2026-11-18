"""Tests merged from: test_length_contracts.py, test_length_policy.py, test_length_predictor.py, test_length_predictor_fit.py, test_fit_mlm_length.py, test_variable_length_eval.py, test_evaluation_length.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

from types import SimpleNamespace
import unittest
import torch
from torch.utils.data import Subset
from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.eval.generative import build_masked_judge_prompt
from cdrseqgen.methods.mlm.length import (
    FixedNativeStrategy,
    VariableLengthStrategy,
    LengthHead,
    LengthPredictor,
    prepare_fixed_length_prompt,
    prepare_gap_prompt,
    resolve_length_policy,
    sample_length_conditioned_spans,
)
from cdrseqgen.methods.mlm.length import (
    LengthConstraints,
    native_mask_length,
    resolve_length_policy,
)
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from cdrseqgen.artifacts import LENGTH_CONFIG_FILENAME, LENGTH_HEAD_FILENAME
from cdrseqgen.methods.mlm.length import sample_length_conditioned_spans
from cdrseqgen.methods.mlm.length import resolve_length_policy
from cdrseqgen.methods.mlm.length import SCHEMA_VERSION, LengthHead, LengthPredictor
from torch import nn
from cdrseqgen.methods.mlm.length import fit_length_predictor
from contextlib import contextmanager
from unittest.mock import Mock, patch, sentinel
import fit_mlm_length as command
from fit_mlm_length import resolve_config_path
from benchmark.adapters.iglm import IgLMSampler
from cdrseqgen.methods.mlm.evaluation import run_generative_eval
from cdrseqgen.eval.metrics import calculate
from cdrseqgen.methods.mlm.length import GeneratedSpan
from cdrseqgen.methods.mlm.length import resolve_generation_length_policy


# ====================================================================================================
# from test_length_contracts.py
# MLM prompt budgets, mixed-precision heads and nested judge views.
# ====================================================================================================

class Frameworks:
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return dict(input_ids=torch.tensor([index + 5, 4, 6]),
                    labels=torch.tensor([-100, 5, -100]), target_len=1)

    def framework_parts(self, index):
        return dict(pt1=("AC", "DE", "FG", "HI", "KL", "MN")[index], pt2="PQ",
                    target="A", target_length=1, max_len=64,
                    partner_chain=None, antigen=None)


class BfloatModel:
    config = SimpleNamespace(max_position_embeddings=16)

    def __call__(self, input_ids, **kwargs):
        hidden = torch.ones(*input_ids.shape, 2, dtype=torch.bfloat16)
        return {"hidden_states": (hidden,)}


class LengthContractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())

    def test_fixed_and_reserved_spans_cannot_exceed_the_input_budget(self):
        for build in (lambda: prepare_fixed_length_prompt(self.tokenizer, "AC", "DE", 20, max_len=10),
                      lambda: prepare_gap_prompt(self.tokenizer, "AC", "DE", reserve_length=20, max_len=10)):
            with self.subTest(build=build):
                with self.assertRaisesRegex(ValueError, "requires.*allows"):
                    build()
        self.tokenizer.uses_special_tokens = True
        with self.assertRaisesRegex(ValueError, "requires.*allows"):
            prepare_fixed_length_prompt(self.tokenizer, "", "", 9, max_len=10)

    def test_declared_tokenizer_and_model_capacities_are_enforced_before_forward(self):
        self.tokenizer.max_positions = 8
        with self.assertRaisesRegex(ValueError, "max_positions"):
            prepare_fixed_length_prompt(self.tokenizer, "", "", 9, max_len=32)
        del self.tokenizer.max_positions
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=3))
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            sample_length_conditioned_spans(
                model, self.tokenizer, "cpu", torch.Generator(), pt1="AC", pt2="DE",
                policy=resolve_length_policy(target_length=2), max_len=16,
                decoding="independent", num_samples=1, temperature=0, top_k=0, top_p=1)

    def test_bfloat_features_work_for_head_training_and_length_inference(self):
        head = LengthHead(2, 3)
        features = torch.tensor([[0.1, 0.5]], dtype=torch.bfloat16, requires_grad=True)
        actual = head(features)
        torch.testing.assert_close(actual, head(features.float()), rtol=0, atol=0)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(head.classifier.weight.grad).all())
        predictor = LengthPredictor(head, dict(supported_lengths=[1, 2, 3], reserve_length=3))
        selected, probabilities, _ = predictor.select(
            BfloatModel(), self.tokenizer, "cpu", torch.Generator().manual_seed(5),
            pt1="AC", pt2="DE", policy=resolve_length_policy(), max_len=16,
            num_samples=2, temperature=0)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in probabilities.values()))

    def test_length_fitting_checks_the_reserved_span_against_model_capacity(self):
        from unittest.mock import Mock
        from cdrseqgen.methods.mlm.length import _extract_dataset_features
        model = Mock(config=SimpleNamespace(max_position_embeddings=16))
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            _extract_dataset_features(model, self.tokenizer, Frameworks(), device="cpu",
                                      batch_size=2, reserve_length=20)
        model.assert_not_called()

    def test_both_strategies_map_nested_subsets_to_the_correct_judge_framework(self):
        full = Frameworks()
        nested = Subset(Subset(full, [4, 1, 5]), [2, 0])
        for judge in (full, nested):
            options = dict(model=object(), tokenizer=self.tokenizer, device="cpu",
                           generator=torch.Generator(), eval_dataset=nested, judge_dataset=judge,
                           judge_tokenizer=self.tokenizer, decoding="independent", num_samples=1,
                           temperature=0, top_k=0, top_p=1)
            fixed = FixedNativeStrategy(**options)
            ids, _ = fixed.judge_inputs(0, "A")
            self.assertEqual(ids[0], 10)
            variable = VariableLengthStrategy(length_predictor=object(),
                length_policy=resolve_length_policy(), length_temperature=1, **options)
            self.assertEqual(variable.judge_inputs(0, "AC"),
                             build_masked_judge_prompt(self.tokenizer, full.framework_parts(5), 2))


# ====================================================================================================
# from test_length_policy.py
# ====================================================================================================

class LengthPolicyTest(unittest.TestCase):
    def test_equal_bounds_are_an_explicit_fixed_length(self):
        policy = resolve_length_policy(min_length=12, max_length=12)
        self.assertEqual(policy.mode, "fixed")
        self.assertEqual(policy.length, 12)
        self.assertEqual(policy.source, "explicit_bounds")

    def test_native_length_preserves_current_evaluation_policy(self):
        decision = native_mask_length(9)
        self.assertEqual(decision.length, 9)
        self.assertEqual(decision.source, "native_mask_count")

    def test_native_mask_length_is_a_fixed_policy_with_the_recorded_payload(self):
        # This dict is written into every fixed-native evaluation record.
        policy = native_mask_length(9)
        self.assertEqual(policy.mode, "fixed")
        self.assertFalse(policy.requires_predictor)
        self.assertEqual(policy.as_dict(), {
            "mode": "fixed", "source": "native_mask_count",
            "minimum": 9, "maximum": 9, "selected_length": 9,
        })
        for bad in (0, -3, True, 2.5):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "length must be"):
                native_mask_length(bad)

    def test_native_truth_takes_precedence_when_request_agrees(self):
        policy = resolve_length_policy(
            native_length=11, use_native_length=True, min_length=11, max_length=11
        )
        self.assertEqual(policy.length, 11)
        self.assertEqual(policy.source, "native_truth")

    def test_deprecated_target_length_remains_supported(self):
        policy = resolve_length_policy(target_length=10)
        self.assertEqual(policy.length, 10)
        self.assertEqual(policy.source, "legacy_target_length")

    def test_ranges_are_general_policies_with_inclusive_bounds(self):
        policies = [
            resolve_length_policy(min_length=11, max_length=15),
            resolve_length_policy(min_length=11),
            resolve_length_policy(max_length=15),
        ]
        self.assertTrue(all(policy.mode == "bounded" for policy in policies))
        self.assertTrue(all(policy.requires_predictor for policy in policies))
        self.assertEqual(policies[0].constraints.as_dict(), {"minimum": 11, "maximum": 15})
        self.assertTrue(policies[0].constraints.contains(11))
        self.assertTrue(policies[0].constraints.contains(15))
        self.assertFalse(policies[0].constraints.contains(16))

    def test_no_bounds_is_inferred_and_native_truth_does_not_leak(self):
        policy = resolve_length_policy(native_length=12)
        self.assertEqual(policy.mode, "inferred")
        self.assertTrue(policy.requires_predictor)
        fixed = resolve_length_policy(native_length=12, use_native_length=True)
        self.assertEqual(fixed.length, 12)
        self.assertEqual(fixed.source, "native_truth")

    def test_use_native_length_without_a_native_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires a native length"):
            resolve_length_policy(use_native_length=True)

    def test_conflicting_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_length_policy(target_length=12, min_length=12, max_length=12)
        with self.assertRaisesRegex(ValueError, "conflicts with native"):
            resolve_length_policy(
                native_length=10, use_native_length=True, min_length=12, max_length=12
            )

    def test_invalid_constraints_are_rejected(self):
        for kwargs in (
            {"min_length": 0, "max_length": 0},
            {"min_length": 15, "max_length": 11},
            {"min_length": True, "max_length": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LengthConstraints(
                        minimum=kwargs["min_length"], maximum=kwargs["max_length"]
                    )


# ====================================================================================================
# from test_length_predictor.py
# ====================================================================================================

class HiddenModel:
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False,
                 return_dict=True):
        batch, length = input_ids.shape
        hidden = torch.ones(batch, length, 2, device=input_ids.device)
        return {"logits": torch.zeros(batch, length, 25), "hidden_states": (hidden,)}


class FixedSelectionPredictor:
    def select(self, *_args, **_kwargs):
        return [2, 3, 2], {2: 0.6, 3: 0.4}, type("Prompt", (), {
            "input_ids": [4], "gap_position": 0, "reserve_length": 3,
        })()


class LengthPredictorTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())
        self.config = {
            "schema_version": SCHEMA_VERSION,
            "feature_dim": 2,
            "supported_lengths": [2, 3, 4],
            "reserve_length": 4,
        }
        head = LengthHead(2, 3)
        with torch.no_grad():
            head.normalization.weight.fill_(1)
            head.normalization.bias.zero_()
            head.classifier.weight.zero_()
            head.classifier.bias.copy_(torch.tensor([0.0, 2.0, 5.0]))
        self.predictor = LengthPredictor(head, self.config)

    def test_bounds_mask_classes_before_selection(self):
        policy = resolve_length_policy(min_length=2, max_length=3)
        selected, probabilities, _prompt = self.predictor.select(
            HiddenModel(), self.tokenizer, torch.device("cpu"),
            torch.Generator().manual_seed(1),
            pt1="AC", pt2="WY", policy=policy, num_samples=3, temperature=0,
        )
        self.assertEqual(selected, [3, 3, 3])
        self.assertNotIn(4, probabilities)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)

    def test_impossible_bounds_fail_with_observed_support(self):
        policy = resolve_length_policy(min_length=7)
        with self.assertRaisesRegex(ValueError, "contain no class"):
            self.predictor.select(
                HiddenModel(), self.tokenizer, torch.device("cpu"),
                torch.Generator().manual_seed(1),
                pt1="AC", pt2="WY", policy=policy, num_samples=1,
            )

    def test_artifact_roundtrip(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_file(self.predictor.head.state_dict(), str(path / LENGTH_HEAD_FILENAME))
            (path / LENGTH_CONFIG_FILENAME).write_text(json.dumps(self.config))
            loaded = LengthPredictor.load(path, device=torch.device("cpu"))
            self.assertEqual(loaded.lengths, (2, 3, 4))

    @patch("cdrseqgen.methods.mlm.length.sample_masked_span")
    def test_variable_generator_groups_draws_by_selected_length(self, sample_span):
        def sample_for_prompt(*args, **kwargs):
            length = len(args[5])
            count = kwargs["num_samples"]
            return [self.tokenizer.encode("A" * length) for _ in range(count)]

        sample_span.side_effect = sample_for_prompt
        generated, metadata = sample_length_conditioned_spans(
            model=HiddenModel(), tokenizer=self.tokenizer, device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(2),
            pt1="AC", pt2="WY", policy=resolve_length_policy(),
            length_predictor=FixedSelectionPredictor(), decoding="independent",
            num_samples=3, temperature=1.0, top_k=0, top_p=1.0,
        )
        self.assertEqual([span.length for span in generated], [2, 3, 2])
        self.assertEqual(metadata["selected_length_counts"], {"2": 2, "3": 1})
        self.assertEqual(sample_span.call_count, 2)


# ====================================================================================================
# from test_length_predictor_fit.py
# ====================================================================================================

class TinyFrameworkDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def framework_parts(self, index):
        pt1, length = self.rows[index]
        return {
            "pt1": pt1,
            "target": "A" * length,
            "target_length": length,
            "pt2": "WY",
            "partner_chain": None,
            "antigen": None,
            "max_len": 16,
        }


class TinyHiddenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(25, 4)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                return_dict=True):
        hidden = self.embedding(input_ids)
        return {"logits": hidden, "hidden_states": (hidden,)}


class LengthPredictorFitTests(unittest.TestCase):
    def test_frozen_backbone_fit_writes_loadable_artifacts_with_empty_eval(self):
        tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())
        model = TinyHiddenModel()
        before = model.embedding.weight.detach().clone()
        train = TinyFrameworkDataset([("AC", 2), ("AC", 2), ("DE", 3), ("DE", 3)])
        empty_eval = TinyFrameworkDataset([])

        with tempfile.TemporaryDirectory() as directory:
            predictor = fit_length_predictor(
                model,
                tokenizer,
                train,
                empty_eval,
                output_dir=directory,
                device=torch.device("cpu"),
                epochs=1,
                learning_rate=0.01,
                batch_size=2,
            )
            path = Path(directory)
            self.assertTrue((path / LENGTH_CONFIG_FILENAME).exists())
            self.assertTrue((path / LENGTH_HEAD_FILENAME).exists())
            config = json.loads((path / LENGTH_CONFIG_FILENAME).read_text())

        self.assertEqual(predictor.lengths, (2, 3))
        self.assertTrue(torch.equal(before, model.embedding.weight.detach()))
        self.assertEqual(config["train_length_counts"], {"2": 2, "3": 2})
        self.assertEqual(config["train_mode_length"], 2)
        self.assertIsNone(config["eval_accuracy"])
        self.assertIsNone(config["eval_mode_baseline_accuracy"])
        self.assertIsNone(config["eval_length_mae"])

    def test_eval_diagnostics_include_mode_baseline_and_mae(self):
        tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())
        model = TinyHiddenModel()
        train = TinyFrameworkDataset([("AC", 2), ("DE", 2), ("FG", 3)])
        evaluation = TinyFrameworkDataset([("AC", 2), ("DE", 3), ("FG", 4)])

        with tempfile.TemporaryDirectory() as directory:
            predictor = fit_length_predictor(
                model,
                tokenizer,
                train,
                evaluation,
                output_dir=directory,
                device=torch.device("cpu"),
                epochs=1,
                learning_rate=0.01,
                batch_size=2,
            )
            config = predictor.config

        self.assertEqual(config["eval_examples"], 3)
        self.assertEqual(config["eval_supported_examples"], 2)
        self.assertEqual(config["eval_unseen_lengths"], [4])
        self.assertAlmostEqual(config["eval_mode_baseline_accuracy"], 1 / 3)
        self.assertAlmostEqual(config["eval_mode_baseline_length_mae"], 1.0)
        self.assertIsInstance(config["eval_accuracy"], float)
        self.assertIsInstance(config["eval_supported_accuracy"], float)
        self.assertIsInstance(config["eval_length_mae"], float)


# ====================================================================================================
# from test_fit_mlm_length.py
# ====================================================================================================

class ResolveMLMLengthConfigTests(unittest.TestCase):
    def test_discovers_canonical_run_config(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            config = checkpoint / "run_config.json"
            config.write_text(json.dumps({"model_type": "scratch"}))

            self.assertEqual(resolve_config_path(str(checkpoint)), config.resolve())

    def test_explicit_config_overrides_checkpoint_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "run_config.json").write_text("{}")
            explicit = root / "training.json"
            explicit.write_text("{}")

            self.assertEqual(
                resolve_config_path(str(checkpoint), str(explicit)), explicit.resolve()
            )

    def test_missing_config_fails_with_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "pass --config explicitly"):
                resolve_config_path(directory)


class FitLengthCommandTests(unittest.TestCase):
    @contextmanager
    def harness(self, *flags, method="mlm", saved_method="mlm"):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "run_config.json").write_text(json.dumps({"method": saved_method}))
            config = checkpoint / "original.json"
            config.write_text(json.dumps({"method": method}))
            args = command.parse_args(["--model_dir", directory, "--config", str(config),
                                       "--device", "cpu", *flags])
            parsed = (SimpleNamespace(model_type="scratch"),
                      SimpleNamespace(length_predictor_epochs=7,
                                      length_predictor_learning_rate=0.02,
                                      length_predictor_batch_size=4),
                      SimpleNamespace(), SimpleNamespace(seed=43))
            backbone = Mock()
            with patch.object(command, "configure_logging"), \
                    patch.object(command, "HfArgumentParser"), \
                    patch.object(command, "parse_config_dict", return_value=parsed), \
                    patch.object(command, "resolve", return_value=backbone) as resolve, \
                    patch.object(command, "get_datasets", return_value=(sentinel.train, sentinel.eval,
                                                                         sentinel.tokenizer)) as datasets, \
                    patch.object(command, "fit_length_predictor", return_value=sentinel.head) as fit:
                yield args, backbone, resolve, datasets, fit

    def test_saved_tokenizer_and_config_defaults_reach_fitter(self):
        with self.harness() as (args, backbone, resolve, datasets, fit):
            self.assertIs(command.main(args), sentinel.head)
            self.assertEqual(datasets.call_args.kwargs["checkpoint_path"], args.model_dir)
            backbone.load_pretrained_model.assert_called_once_with(args.model_dir)
            options = fit.call_args.kwargs
            self.assertEqual((options["epochs"], options["learning_rate"], options["batch_size"]),
                             (7, 0.02, 4))
            self.assertEqual(options["seed"], 43)
            self.assertEqual(options["output_dir"], args.model_dir)

    def test_explicit_fit_overrides_are_forwarded(self):
        with self.harness("--epochs", "3", "--learning_rate", "0.001", "--batch_size", "2") as setup:
            args, _, _, _, fit = setup
            command.main(args)
            self.assertEqual((fit.call_args.kwargs["epochs"], fit.call_args.kwargs["learning_rate"],
                              fit.call_args.kwargs["batch_size"]), (3, 0.001, 2))

    def test_invalid_overrides_fail_before_data_or_weights_load(self):
        for flag, value in (("--epochs", "0"), ("--epochs", "-1"), ("--batch_size", "0"),
                            ("--learning_rate", "0"), ("--learning_rate", "nan"),
                            ("--learning_rate", "inf")):
            with self.subTest(flag=flag, value=value), self.harness(flag, value) as setup:
                args, backbone, resolve, datasets, fit = setup
                with self.assertRaisesRegex(ValueError, flag[2:]):
                    command.main(args)
                resolve.assert_not_called()
                datasets.assert_not_called()
                backbone.load_pretrained_model.assert_not_called()
                fit.assert_not_called()

    def test_both_explicit_and_saved_configs_must_be_mlm(self):
        for supplied, saved in (("ar", "mlm"), ("mlm", "s2s")):
            with self.subTest(supplied=supplied, saved=saved), \
                    self.harness(method=supplied, saved_method=saved) as setup:
                args, _, resolve, datasets, fit = setup
                with self.assertRaisesRegex(ValueError, "method"):
                    command.main(args)
                resolve.assert_not_called()
                datasets.assert_not_called()
                fit.assert_not_called()


class RootEvaluationConfigTests(unittest.TestCase):
    def test_nonobject_config_fails_before_checkpoint_routing(self):
        import evaluate
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text("[]")
            args = evaluate.parse_args(["--config", str(config), "--model_dir", directory])
            with patch.object(evaluate, "read_run_config") as read, \
                    self.assertRaisesRegex(ValueError, "JSON object"):
                evaluate.main(args)
            read.assert_not_called()


# ====================================================================================================
# from test_variable_length_eval.py
# ====================================================================================================

class FrameworkDataset:
    def __init__(self):
        self.parts = {
            "pt1": "AC",
            "target": "AAAA",
            "target_length": 4,
            "pt2": "WY",
            "partner_chain": None,
            "antigen": None,
            "max_len": 32,
        }

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return {"target_len": 4}

    def framework_parts(self, _index):
        return dict(self.parts)


class EvalModel:
    def eval(self):
        return self


class VariableLengthEvalTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())

    def test_unequal_lengths_use_edit_metrics_and_length_matched_aar(self):
        metrics = calculate(["AAAA", "AAA"], native="AAAA")
        self.assertEqual(metrics["length_match_rate"], 0.5)
        self.assertEqual(metrics["aar_mean"], 1.0)
        self.assertEqual(metrics["aar_length_matched_mean"], 1.0)
        self.assertAlmostEqual(metrics["length_absolute_error_mean"], 0.5)
        self.assertGreater(metrics["edit_similarity_mean"], 0.8)

    def test_iglm_natural_prepare_does_not_consult_native_length_limit(self):
        sampler = IgLMSampler.__new__(IgLMSampler)
        sampler.length_policy = "natural"
        sampler.max_length = 5
        sampler.max_new_tokens = None
        sampler._starting_tokens = lambda _record: torch.zeros((1, 4), dtype=torch.long)
        sampler.prepare({"example_id": "x", "native_length": 100})
        self.assertIsNone(sampler.target_total_length)

    @patch("cdrseqgen.methods.mlm.length.sample_length_conditioned_spans")
    def test_held_out_inferred_mode_keeps_truth_scoring_only(self, generate):
        ids4 = self.tokenizer.encode("AAAA")
        ids3 = self.tokenizer.encode("AAA")
        generate.return_value = (
            [
                GeneratedSpan(ids4, 4, [4] * 4, list(range(4)), {}),
                GeneratedSpan(ids3, 3, [4] * 3, list(range(3)), {}),
            ],
            {"mode": "inferred", "source": "length_predictor", "selected_lengths": [4, 3]},
        )

        payload = run_generative_eval(
            EvalModel(),
            self.tokenizer,
            FrameworkDataset(),
            device=torch.device("cpu"),
            length_mode="inferred",
            length_predictor=object(),
            num_samples=2,
            num_examples=-1,
            skip_perplexity=True,
        )

        record = payload["records"][0]
        self.assertEqual(record["generated_lengths"], [4, 3])
        self.assertEqual(record["metrics"]["length_match_rate"], 0.5)
        self.assertEqual(record["metrics"]["aar_mean"], 1.0)
        call = generate.call_args.kwargs
        self.assertEqual(call["policy"].mode, "inferred")
        self.assertNotIn("native", call)


# ====================================================================================================
# from test_evaluation_length.py
# ====================================================================================================

class EvaluationLengthModeTests(unittest.TestCase):
    def test_modes_resolve_without_silent_bound_ignoring(self):
        self.assertIsNone(resolve_generation_length_policy("fixed_native"))
        self.assertEqual(resolve_generation_length_policy("inferred").mode, "inferred")
        bounded = resolve_generation_length_policy("bounded", min_length=11, max_length=15)
        self.assertEqual(bounded.mode, "bounded")
        self.assertEqual((bounded.constraints.minimum, bounded.constraints.maximum), (11, 15))

    def test_contradictory_modes_fail(self):
        with self.assertRaisesRegex(ValueError, "inferred mode"):
            resolve_generation_length_policy("inferred", min_length=11)
        with self.assertRaisesRegex(ValueError, "fixed_native mode"):
            resolve_generation_length_policy("fixed_native", max_length=15)
        with self.assertRaisesRegex(ValueError, "bounded mode"):
            resolve_generation_length_policy("bounded")
        with self.assertRaisesRegex(ValueError, "equal bounds"):
            resolve_generation_length_policy("bounded", min_length=12, max_length=12)


if __name__ == "__main__":
    unittest.main()
