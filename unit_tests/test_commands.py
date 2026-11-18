"""Tests merged from: test_command_dispatch.py, test_inference_entrypoints.py, test_inference_validation.py, test_evaluate_contract.py, test_generation_callers.py, test_eos_command_setup.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel
import evaluate
import generate
import train
from cdrseqgen.methods.common import generation as shared_generation
from cdrseqgen.methods.common import evaluation as shared_evaluation
from unittest.mock import Mock, patch
import torch
from cdrseqgen.datasets.dataset_mlm import (
    AntibodyTokenizer,
    MaskedInfillingDataset,
    TokenizerAdapter,
)
from cdrseqgen.methods.common import arguments as shared_arguments
from cdrseqgen.methods.common.eos_decoding import _partition_attempts
import hashlib
import os
from unittest.mock import patch
from benchmark.adapters.cdrseqgen import resolve_decoding as benchmark_decoding
from cdrseqgen.methods.mlm.eval_cli import prepare_evaluation
from cdrseqgen.methods.mlm.evaluation import resolve_decoding
from cdrseqgen.methods.mlm.gen_cli import prepare_generation
from evaluate import compatibility_problems, default_output_path, get_evaluation_source_info
from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.methods.mlm import gen_cli as mlm_gen_cli
from cdrseqgen.methods.mlm.evaluation import run_generative_eval
import importlib
from cdrseqgen.datasets.dataset_ar import CAUSAL_DECODE_STEPS
from cdrseqgen.datasets.dataset_s2s import S2S_DECODE_STEPS
from cdrseqgen.methods.ar.models.iglm import resolve_conditioning


# ====================================================================================================
# from test_command_dispatch.py
# Single-entrypoint routing and shared lifecycle contracts with loading mocked.
# ====================================================================================================

def method_with_modules(**modules):
    return SimpleNamespace(name="mlm", module=Mock(side_effect=modules.__getitem__))


class CommandLifecycleTests(unittest.TestCase):
    def test_training_dispatch_preserves_arguments_for_each_method(self):
        for name in train.METHODS:
            with self.subTest(method=name):
                argv = ["train.py", "--method", name, "--seed", "43"]
                observed = []
                command = SimpleNamespace(main=Mock(side_effect=lambda: observed.append(list(train.sys.argv))))
                method = method_with_modules(train_cli=command)
                with patch("sys.argv", argv), patch("train.resolve", return_value=method) as resolve:
                    train.main()
                resolve.assert_called_once_with(name)
                method.module.assert_called_once_with("train_cli")
                command.main.assert_called_once_with()
                self.assertEqual(observed, [argv])

    def test_generation_forwards_case_order_and_method_state(self):
        cases = [{"name": "first"}, {"name": "second"}]
        config = {"source": "run_config", "model_type": "scratch", "max_len": 64}
        args = SimpleNamespace(model_dir="/tmp/checkpoint", max_len=None,
                               seed=43, force=False, output="none")
        command = Mock()
        command.prepare_generation.return_value = sentinel.policies
        command.generation_state.return_value = sentinel.state
        model, tokenizer = Mock(), Mock()
        backbone = Mock()
        backbone.build_tokenizer.return_value = tokenizer
        backbone.load_pretrained_model.return_value = model
        backbones = SimpleNamespace(resolve=Mock(return_value=backbone))
        method = method_with_modules(gen_cli=command, backbones=backbones)

        with patch.object(shared_generation, "load_test_cases", return_value=cases), \
                patch.object(shared_generation, "check_compatibility", return_value=([], [])), \
                patch.object(shared_generation, "resolve", return_value=method), \
                patch.object(shared_generation.torch.cuda, "is_available", return_value=False), \
                patch("builtins.print"):
            shared_generation.run_generation_pipeline(
                "mlm", args, config,
                prepare_generation=command.prepare_generation,
                generation_state=command.generation_state,
                generate_case=command.generate_case,
            )

        self.assertEqual(args.max_len, 64)
        command.prepare_generation.assert_called_once_with(args, cases, config)
        self.assertIs(command.generation_state.call_args.args[1], sentinel.policies)
        self.assertEqual(len(command.generate_case.call_args_list), 2)
        for invocation, case in zip(command.generate_case.call_args_list, cases):
            self.assertIs(invocation.args[0], model)
            self.assertIs(invocation.args[1], tokenizer)
            self.assertIs(invocation.args[3], args)
            self.assertIs(invocation.args[5], case)
            self.assertIs(invocation.args[6], sentinel.state)
            self.assertIs(invocation.args[7], generate._print_case)

    def test_invalid_generation_options_stop_before_weights_are_loaded(self):
        command = Mock()
        command.prepare_generation.side_effect = ValueError("unsupported length policy")
        backbones = Mock()
        method = method_with_modules(gen_cli=command, backbones=backbones)
        args = SimpleNamespace(model_dir="/tmp/checkpoint", max_len=64, force=False)
        with patch.object(shared_generation, "load_test_cases", return_value=[{}]), \
                patch.object(shared_generation, "check_compatibility", return_value=([], [])), \
                patch.object(shared_generation, "resolve", return_value=method), \
                patch("builtins.print"):
            with self.assertRaisesRegex(SystemExit, "unsupported length policy"):
                shared_generation.run_generation_pipeline(
                    "mlm", args, {"source": "run_config", "model_type": "scratch"},
                    prepare_generation=command.prepare_generation,
                    generation_state=command.generation_state,
                    generate_case=command.generate_case,
                )
        method.module.assert_not_called()
        backbones.resolve.assert_not_called()

    def test_evaluation_forwards_method_options_and_keeps_contexts_separate(self):
        command = Mock()
        command.evaluation_options.return_value = {
            "length_mode": "inferred", "length_predictor": sentinel.length_head,
        }
        model, tokenizer = Mock(), Mock()
        dataset = [sentinel.example]
        backbone = Mock()
        backbone.load_pretrained_model.return_value = model
        backbones = SimpleNamespace(resolve=Mock(return_value=backbone))
        training = SimpleNamespace(get_datasets=Mock(return_value=([], dataset, tokenizer)))
        arguments = SimpleNamespace(ModelArguments=object, DataArguments=object,
                                    GenerationArguments=object)
        method = method_with_modules(eval_cli=command, backbones=backbones,
                                     training=training, arguments=arguments)
        model_args = SimpleNamespace(model_type="scratch", anchor_trim=3)
        data_args = SimpleNamespace(data_path="/tmp/corpus.json", target_region="h3",
                                    eval_split="validation", include_partner_chain=True,
                                    include_antigen=False, context_dropout=0.5)
        parsed = (model_args, data_args, SimpleNamespace(), SimpleNamespace())
        run_config = {"method": "mlm", "target_region": "h3"}

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"method": "mlm"}))
            argv = ["evaluate.py", "--config", str(config), "--model_dir", directory,
                    "--output", "none", "--judge", "", "--partner_context", "without",
                    "--judge_partner_context", "with", "--skip_perplexity"]
            with patch("sys.argv", argv), \
                    patch.object(shared_evaluation, "resolve", return_value=method), \
                    patch("cdrseqgen.methods.common.evaluation.HfArgumentParser"), \
                    patch("cdrseqgen.methods.common.evaluation.parse_config_dict", return_value=parsed), \
                    patch("cdrseqgen.methods.common.evaluation.compatibility_problems", return_value=[]), \
                    patch("cdrseqgen.methods.common.evaluation.get_evaluation_source_info", return_value={}), \
                    patch("cdrseqgen.methods.common.evaluation.evaluation_context_view", return_value=dataset) as context_view, \
                    patch("cdrseqgen.methods.common.evaluation.torch.cuda.is_available", return_value=False), \
                    patch("builtins.print"):
                args = evaluate.parse_args()
                shared_evaluation.run_evaluation_pipeline(
                    "mlm", args, {"method": "mlm"}, run_config,
                    prepare_evaluation=command.prepare_evaluation,
                    evaluation_options=command.evaluation_options,
                    evaluate=command.evaluate,
                )

        context_view.assert_called_once_with(dataset, include_partner_chain=False)
        command.evaluate.assert_called_once()
        invocation = command.evaluate.call_args
        self.assertEqual(invocation.args, (model, tokenizer, dataset))
        self.assertEqual(invocation.kwargs["length_mode"], "inferred")
        self.assertIs(invocation.kwargs["length_predictor"], sentinel.length_head)
        self.assertEqual(invocation.kwargs["anchor_trim"], 3)
        self.assertFalse(invocation.kwargs["meta"]["include_partner_chain"])
        self.assertTrue(invocation.kwargs["meta"]["judge_include_partner_chain"])
        self.assertEqual(invocation.kwargs["meta"]["eval_context_dropout"], 0.0)


class SingleEntrypointRoutingTests(unittest.TestCase):
    def test_generation_calls_one_method_main_for_each_registered_method(self):
        for name in train.METHODS:
            with self.subTest(method=name):
                args = SimpleNamespace(model_dir="/tmp/checkpoint")
                config = {"method": name}
                command = SimpleNamespace(main=Mock(return_value=sentinel.results))
                method = method_with_modules(gen_cli=command)
                with patch("generate.read_run_config", return_value=config), \
                        patch("generate.checkpoint_method", return_value=method), \
                        patch("builtins.print"):
                    self.assertIs(generate.main(args), sentinel.results)
                method.module.assert_called_once_with("gen_cli")
                command.main.assert_called_once_with(args, config)

    def test_evaluation_calls_one_method_main_for_each_registered_method(self):
        for name in train.METHODS:
            with self.subTest(method=name), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "config.json"
                supplied = {"method": name}
                config_path.write_text(json.dumps(supplied))
                args = SimpleNamespace(config=str(config_path), model_dir=directory)
                run_config = {"method": name, "target_region": "h3"}
                command = SimpleNamespace(main=Mock(return_value=sentinel.payload))
                method = method_with_modules(eval_cli=command)
                method.name = name
                with patch("evaluate.parse_args", return_value=args), \
                        patch("evaluate.read_run_config", return_value=run_config), \
                        patch("evaluate.resolve", return_value=method), \
                        patch("evaluate.checkpoint_method") as check:
                    self.assertIs(evaluate.main(), sentinel.payload)
                method.module.assert_called_once_with("eval_cli")
                command.main.assert_called_once_with(args, supplied, run_config)
                check.assert_called_once_with(run_config, name)

    def test_evaluation_rejects_architecture_mismatch_before_method_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({"method": "mlm"}))
            args = SimpleNamespace(config=str(config_path), model_dir=directory)
            with patch("evaluate.parse_args", return_value=args), \
                    patch("evaluate.read_run_config", return_value={"method": "ar"}), \
                    patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "different method"):
                    evaluate.main()


# ====================================================================================================
# from test_inference_entrypoints.py
# Exercise root -> MLM command -> workflow -> real synthetic candidate scoring.
# ====================================================================================================

class InferenceEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())
        self.model = Mock()
        residue = self.tokenizer.encode("A")[0]

        def forward(*, input_ids):
            logits = torch.zeros(*input_ids.shape, 32)
            logits[..., residue] = 10
            return {"logits": logits}

        self.model.side_effect = forward
        self.backbone = Mock()
        self.backbone.build_tokenizer.return_value = self.tokenizer
        self.backbone.load_pretrained_model.return_value = self.model

    def test_generation_runs_fixed_case_and_writes_compatibility_without_length_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"method": "mlm", "model_type": "scratch", "target_region": "h3",
                      "max_len": 32, "include_partner_chain": False, "include_antigen": False,
                      "source": "cli"}
            (root / "run_config.json").write_text(json.dumps(config))
            output = root / "generated.json"
            args = SimpleNamespace(
                model_dir=directory, model_type=None, max_len=None, target_loop=None,
                test_file=None, pt1="AC", pt2="DE", target_truth="WY",
                target_length=None, min_length=2, max_length=2, use_native_length=False,
                partner_chain=None, antigen=None, max_new_tokens=None,
                decoding=None, num_samples=2, temperature=0.0,
                top_k=0, top_p=1.0, seed=42, skip_perplexity=True,
                show_samples=0, force=False, output=str(output),
            )
            with patch("cdrseqgen.methods.mlm.backbones.resolve", return_value=self.backbone), \
                    patch("cdrseqgen.methods.mlm.length.LengthPredictor.load") as load_head, \
                    patch("torch.cuda.is_available", return_value=False), \
                    patch("builtins.print"):
                results = generate.main(args)
            payload = json.loads(output.read_text())

        load_head.assert_not_called()
        self.assertEqual(args.max_len, 32)
        self.assertEqual(args.decoding, "independent")
        self.assertEqual(self.model.call_count, 1)
        self.assertEqual(payload["cases"], results)
        self.assertEqual(payload["cases"][0]["samples"], [
            {"seq": "AA", "length": 2, "length_match": True, "recovery": 0.0},
        ])
        self.assertEqual(payload["cases"][0]["length_policy"]["source"], "explicit_bounds")
        self.assertEqual(payload["model"]["compatibility"], {
            "problems": [], "warnings": [], "forced": False,
        })
        self.assertEqual(payload["model"]["expects"]["source"], "run_config")

    def test_evaluation_scores_fixed_native_and_refuses_repeated_output_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain = dict(fwr1="AC", cdr1="D", fwr2="E", cdr2="F", fwr3="G",
                         cdr3="AW", fwr4="WY", full_aa="ACDEFGAWWY")
            data_path = root / "synthetic.json"
            data_path.write_text(json.dumps([
                {"heavy": chain, "light": chain, "split": "validation"},
            ]))
            dataset = MaskedInfillingDataset(str(data_path), self.tokenizer, max_len=32)
            config = {
                "method": "mlm", "model_type": "scratch", "target_region": "h3",
                "data_path": str(data_path), "eval_split": "validation",
                "max_len": 32, "masking_policy": "full_span",
                "include_partner_chain": False, "include_antigen": False,
                "output_dir": directory, "report_to": "none", "use_cpu": True,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            (root / "run_config.json").write_text(json.dumps(config))
            output = root / "evaluated.json"
            argv = ["evaluate.py", "--config", str(config_path), "--model_dir", directory,
                    "--output", str(output), "--num_examples", "-1", "--num_samples", "2",
                    "--temperature", "0", "--skip_perplexity"]
            with patch("sys.argv", argv), \
                    patch("cdrseqgen.methods.mlm.backbones.resolve", return_value=self.backbone), \
                    patch("cdrseqgen.methods.mlm.training.get_datasets", return_value=([], dataset, self.tokenizer)) as get_data, \
                    patch("cdrseqgen.methods.mlm.length.LengthPredictor.load") as load_head, \
                    patch("torch.cuda.is_available", return_value=False), \
                    patch("builtins.print"):
                payload = evaluate.main()
                with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
                    evaluate.main()
            saved = json.loads(output.read_text())

        load_head.assert_not_called()
        get_data.assert_called_once()
        self.backbone.load_pretrained_model.assert_called_once()
        self.assertEqual(payload["records"], saved["records"])
        self.assertEqual(saved["records"][0]["samples"], ["AA"])
        self.assertEqual(saved["records"][0]["metrics"]["aar_mean"], 0.5)
        self.assertEqual(saved["length_policy"]["source"], "native_mask_count")
        self.assertEqual(saved["compatibility"], {"problems": [], "forced": False})

    def test_ar_s2s_generation_entries_reject_fixed_bounds_before_loading(self):
        for method in ["ar", "s2s"]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                config = {"method": method, "model_type": "scratch", "target_region": "h3",
                          "max_len": 32, "include_partner_chain": False, "include_antigen": False}
                (Path(directory) / "run_config.json").write_text(json.dumps(config))
                args = SimpleNamespace(
                    model_dir=directory, model_type=None, max_len=None, target_loop=None,
                    test_file=None, pt1="AC", pt2="DE", target_truth=None,
                    target_length=None, min_length=2, max_length=2, use_native_length=False,
                    partner_chain=None, antigen=None, max_new_tokens=None,
                    decoding=None, force=False,
                )
                with patch(f"cdrseqgen.methods.{method}.backbones.resolve") as backbone, \
                        patch("builtins.print"):
                    with self.assertRaisesRegex(SystemExit, "EOS"):
                        generate.main(args)
                backbone.assert_not_called()

    def test_ar_s2s_evaluation_entries_reject_native_length_before_loading(self):
        for method in ["ar", "s2s"]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = {"method": method}
                path = root / "config.json"
                path.write_text(json.dumps(config))
                (root / "run_config.json").write_text(json.dumps(config))
                argv = ["evaluate.py", "--config", str(path), "--model_dir", directory,
                        "--length_mode", "fixed_native", "--output", "none"]
                with patch("sys.argv", argv), \
                        patch(f"cdrseqgen.methods.{method}.backbones.resolve") as backbone:
                    with self.assertRaisesRegex(ValueError, "EOS"):
                        evaluate.main()
                backbone.assert_not_called()


# ====================================================================================================
# from test_inference_validation.py
# Reject invalid inference options and contradictory EOS records before use.
# ====================================================================================================

class SamplingValidationTests(unittest.TestCase):
    def test_valid_boundaries_work_for_command_and_training_names(self):
        values = dict(num_samples=1, temperature=0.0, top_k=0, top_p=1.0,
                      length_temperature=0.0, num_examples=-1)
        for prefix in ("", "gen_"):
            with self.subTest(prefix=prefix):
                args = SimpleNamespace(**{prefix + key: value for key, value in values.items()})
                shared_arguments.validate_sampling_args(args, prefix=prefix)
        # The command parser's absent max_len means use the checkpoint's cap.
        shared_arguments.validate_sampling_args(SimpleNamespace(max_len=None, show_samples=0))
        shared_arguments.validate_sampling_args(SimpleNamespace())

    def test_invalid_numeric_values_name_the_rejected_field(self):
        invalid = {
            "num_samples": (0, -1, True, 1.5, None),
            "temperature": (-0.1, float("nan"), float("inf"), None),
            "length_temperature": (-0.1, float("nan"), float("inf"), None),
            "top_k": (-1, True, 1.5, None),
            "top_p": (0, -0.1, 1.1, float("nan"), float("inf"), None),
            "num_examples": (-2, True, 1.5, None),
        }
        for prefix in ("", "gen_"):
            for field, values in invalid.items():
                for value in values:
                    key = prefix + field
                    with self.subTest(field=key, value=value):
                        with self.assertRaisesRegex(ValueError, field):
                            shared_arguments.validate_sampling_args(
                                SimpleNamespace(**{key: value}), prefix=prefix)

    def test_invalid_command_caps_and_display_limits_are_rejected(self):
        for field, values in (("max_len", (0, -1, True, 2.5)),
                              ("show_samples", (-1, True, 2.5, None))):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        shared_arguments.validate_sampling_args(SimpleNamespace(**{field: value}))

    def test_generation_preflight_rejects_bad_parser_values_before_loading(self):
        settings = (("--temperature", "nan"), ("--top_p", "inf"),
                    ("--num_samples", "0"), ("--top_k", "-1"),
                    ("--length_temperature", "nan"), ("--max_len", "0"),
                    ("--show_samples", "-1"))
        for flag, value in settings:
            with self.subTest(flag=flag, value=value):
                args = generate.parse_args(["--model_dir", "/tmp/checkpoint", "--output", "none",
                                            flag, value])
                backbone = Mock()
                method = SimpleNamespace(name="mlm", module=Mock(return_value=SimpleNamespace(
                    resolve=Mock(return_value=backbone))))
                with patch.object(shared_generation, "resolve", return_value=method), \
                        patch.object(shared_generation, "load_test_cases", return_value=[{}]), \
                        patch.object(shared_generation, "check_compatibility", return_value=([], [])), \
                        patch("builtins.print"):
                    with self.assertRaises((ValueError, SystemExit)) as caught:
                        shared_generation.run_generation_pipeline(
                            "mlm", args, {"source": "run_config", "model_type": "scratch", "max_len": 64},
                            prepare_generation=Mock(), generation_state=Mock(), generate_case=Mock())
                self.assertIn(flag[2:], str(caught.exception))
                backbone.build_tokenizer.assert_not_called()
                backbone.load_pretrained_model.assert_not_called()

    def test_evaluation_preflight_rejects_bad_parser_values_before_resources(self):
        settings = (("--temperature", "inf"), ("--top_p", "nan"),
                    ("--num_examples", "-2"), ("--num_samples", "0"),
                    ("--length_temperature", "nan"))
        for flag, value in settings:
            with self.subTest(flag=flag, value=value):
                args = evaluate.parse_args(["--model_dir", "/tmp/checkpoint", "--config", "/tmp/config.json",
                                            "--output", "none", flag, value])
                method = SimpleNamespace(name="mlm", module=Mock(side_effect=AssertionError(
                    "method resources must not be requested before numeric preflight")))
                with patch.object(shared_evaluation, "resolve", return_value=method):
                    with self.assertRaises((ValueError, SystemExit)) as caught:
                        shared_evaluation.run_evaluation_pipeline(
                            "mlm", args, {"method": "mlm"}, None,
                            prepare_evaluation=Mock(), evaluation_options=Mock(), evaluate=Mock())
                self.assertIn(flag[2:], str(caught.exception))
                method.module.assert_not_called()


class EosDrawContractTests(unittest.TestCase):
    def test_legacy_and_native_records_keep_disjoint_nonnegative_counts(self):
        results = [dict(loop="AC", complete=True), dict(loop="A", complete=False),
                   dict(loop="", complete=True),
                   dict(loop="", complete=False, valid=False, terminated=True),
                   dict(loop="AC", complete=True, valid=True, terminated=True)]
        kept, counts = _partition_attempts(results)
        self.assertEqual([record["loop"] for record in kept], ["AC", "AC"])
        self.assertEqual(counts, dict(attempts=5, complete_nonempty=2, complete_empty=1,
                                     incomplete=1, invalid=1))
        self.assertEqual(sum(counts[key] for key in ("complete_nonempty", "complete_empty", "incomplete", "invalid")), 5)
        self.assertTrue(all(value >= 0 for value in counts.values()))

    def test_contradictory_or_malformed_sampler_records_raise_clear_errors(self):
        invalid = [dict(loop="AC", complete=True, valid=False, terminated=True),
                   dict(loop="AC", complete=False, valid=True),
                   dict(loop="AC", complete=True, terminated=False),
                   dict(loop="AC", complete="yes"), dict(loop="AC", complete=True, valid=1),
                   dict(loop=["A", "C"], complete=True), dict(loop="AC"), dict(complete=True)]
        for record in invalid:
            with self.subTest(record=record):
                with self.assertRaises(ValueError) as caught:
                    _partition_attempts([record])
                self.assertTrue(str(caught.exception))


# ====================================================================================================
# from test_evaluate_contract.py
# ====================================================================================================

class EvaluationCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(model_type="scratch")
        self.data = SimpleNamespace(
            target_region="h3",
            include_partner_chain=False,
            include_antigen=False,
            data_path="data/processed/processed_oas_tuning.json",
            max_len=128,
            masking_policy="full_span",
        )
        self.run = {
            "target_region": "h3",
            "model_type": "scratch",
            "include_partner_chain": False,
            "include_antigen": False,
            "data_path": self.data.data_path,
            "max_len": 128,
            "masking_policy": "full_span",
        }

    def test_matching_input_contract_has_no_problems(self):
        self.assertEqual(compatibility_problems(self.run, self.model, self.data), [])

    def test_every_input_shaping_mismatch_is_reported(self):
        mismatched = dict(self.run)
        mismatched.update({
            "target_region": "h1",
            "model_type": "Exscientia/IgBert",
            "include_partner_chain": True,
            "include_antigen": True,
            "data_path": "different.json",
            "max_len": 512,
            "masking_policy": "mixed",
        })
        problems = compatibility_problems(mismatched, self.model, self.data)
        self.assertEqual(len(problems), 7)
        self.assertTrue(any("target CDR" in item for item in problems))
        self.assertTrue(any("backbone/tokenizer" in item for item in problems))

    def test_absent_legacy_fields_are_unknown_not_mismatches(self):
        self.assertEqual(
            compatibility_problems({"target_region": "h3"}, self.model, self.data),
            [],
        )


class DecoderContractTests(unittest.TestCase):
    def test_full_span_ignores_an_old_unsafe_recorded_default(self):
        chosen = resolve_decoding(
            None,
            run_config={"masking_policy": "full_span", "gen_decoding": "iterative"},
        )
        self.assertEqual(chosen, "independent")

    def test_missing_policy_is_treated_as_legacy_full_span(self):
        self.assertEqual(
            resolve_decoding(None, run_config={"gen_decoding": "iterative"}),
            "independent",
        )

    def test_mixed_policy_uses_its_recorded_decoder_or_independent(self):
        self.assertEqual(
            resolve_decoding(
                None,
                run_config={"masking_policy": "mixed", "gen_decoding": "independent"},
            ),
            "independent",
        )
        self.assertEqual(
            resolve_decoding(None, run_config={"masking_policy": "mixed"}),
            "independent",
        )

    def test_explicit_off_distribution_request_remains_possible(self):
        self.assertEqual(
            resolve_decoding(
                "iterative",
                run_config={"masking_policy": "full_span", "gen_decoding": "independent"},
            ),
            "iterative",
        )

    def test_mlm_commands_and_benchmark_share_decoder_precedence(self):
        cases = [
            ({}, None, "independent"),
            ({"masking_policy": "full_span", "gen_decoding": "iterative"}, None, "independent"),
            ({"gen_decoding": "iterative"}, None, "independent"),
            ({"masking_policy": None, "gen_decoding": "iterative"}, None, "independent"),
            ({"masking_policy": "mixed"}, None, "independent"),
            ({"masking_policy": "mixed", "gen_decoding": None}, None, "independent"),
            ({"masking_policy": "mixed", "gen_decoding": ""}, None, "independent"),
            ({"masking_policy": "mixed", "gen_decoding": "invalid"}, None, "independent"),
            ({"masking_policy": "mixed", "gen_decoding": "independent"}, None, "independent"),
            ({"masking_policy": "mixed", "gen_decoding": "iterative"}, None, "iterative"),
            ({"masking_policy": " Mixed ", "gen_decoding": "iterative"}, None, "iterative"),
            ({"masking_policy": "mixed", "gen_decoding": "iterative"}, "independent", "independent"),
            ({"masking_policy": "mixed", "gen_decoding": "independent"}, "iterative", "iterative"),
            ({"masking_policy": "full_span"}, "iterative", "iterative"),
        ]
        for config, requested, expected in cases:
            with self.subTest(config=config, requested=requested):
                generation_args = SimpleNamespace(
                    decoding=requested, max_new_tokens=None, target_length=None,
                    min_length=None, max_length=None, use_native_length=False,
                )
                evaluation_args = SimpleNamespace(
                    decoding=requested, length_mode="fixed_native", max_new_tokens=None,
                    min_length=None, max_length=None,
                )
                with patch("builtins.print"):
                    prepare_generation(generation_args, [{"target_length": 2}], config)
                    prepare_evaluation(evaluation_args, config)
                self.assertEqual(generation_args.decoding, expected)
                self.assertEqual(evaluation_args.decoding, expected)
                self.assertEqual(benchmark_decoding(config, requested), expected)


class EvaluationArtifactTests(unittest.TestCase):
    def test_default_output_is_parameter_addressed(self):
        args = SimpleNamespace(
            model_dir="/tmp/checkpoint",
            length_mode="inferred",
            min_length=None,
            max_length=None,
            length_temperature=1.0,
            decoding="independent",
            num_examples=25,
            stratify=True,
            num_samples=20,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            seed=42,
            skip_perplexity=False,
            judge="Exscientia/IgBert",
            judge_path=None,
        )
        path = default_output_path(args)
        self.assertIn(os.path.join("checkpoint", "evaluations"), path)
        self.assertIn("inferred_independent_n25x20", path)
        self.assertIn("IgBert", path)

    @patch("cdrseqgen.methods.common.evaluation.git_state", return_value={
        "repository_commit": "abc", "repository_dirty": False,
    })
    def test_provenance_hashes_config_checkpoint_and_metadata(self, _git):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "experiment.json"
            config.write_text("{}")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "run_config.json").write_text("{}")
            weights = checkpoint / "model.safetensors"
            weights.write_bytes(b"weights")
            data = root / "processed.json"
            data.write_text("[]")
            metadata = root / "processed.metadata.json"
            metadata.write_text(json.dumps({"processed_file_sha256": "declared"}))

            provenance = get_evaluation_source_info(config, checkpoint, data)

        _git.assert_called_once_with(Path(__file__).resolve().parents[1])
        self.assertEqual(
            provenance["checkpoint_weights"][0]["sha256"],
            hashlib.sha256(b"weights").hexdigest(),
        )
        self.assertEqual(provenance["dataset"]["declared_processed_file_sha256"], "declared")
        self.assertEqual(provenance["repository_commit"], "abc")


# ====================================================================================================
# from test_generation_callers.py
# ====================================================================================================

class FakeModel:
    def eval(self):
        return self


class SharedGenerationCallerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())

    @patch("cdrseqgen.methods.mlm.length.sample_masked_span")
    def test_generate_uses_shared_decoder_without_changing_fixed_length(self, shared_decode):
        shared_decode.return_value = [
            self.tokenizer.encode("WY"),
            self.tokenizer.encode("WW"),
        ]
        args = SimpleNamespace(
            partner_chain=None,
            antigen=None,
            max_len=20,
            decoding="independent",
            num_samples=2,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            skip_perplexity=True,
            show_samples=0,
            seed=42,
        )

        record = mlm_gen_cli.generate_case(
            FakeModel(),
            self.tokenizer,
            torch.device("cpu"),
            args,
            torch.Generator().manual_seed(42),
            {"name": "fixed", "pt1": "AC", "pt2": "DE", "target_truth": "WY", "min_length": 2, "max_length": 2},
            None,
            generate._print_case,
        )

        self.assertEqual(record["target_length"], 2)
        self.assertEqual(record["length_policy"]["source"], "explicit_bounds")
        self.assertEqual([sample["seq"] for sample in record["samples"]], ["WY", "WW"])
        self.assertEqual(shared_decode.call_args.kwargs["decoding"], "independent")
        self.assertEqual(len(shared_decode.call_args.args[5]), 2)

    @patch("cdrseqgen.methods.mlm.length.sample_masked_span")
    def test_held_out_eval_uses_shared_decoder_and_native_mask_count(self, shared_decode):
        native_ids = self.tokenizer.encode("WY")
        shared_decode.return_value = [native_ids, self.tokenizer.encode("WW")]
        dataset = [{
            "input_ids": torch.tensor([self.tokenizer.mask_id] * 2),
            "labels": torch.tensor(native_ids),
            "target_len": 2,
        }]

        payload = run_generative_eval(
            FakeModel(),
            self.tokenizer,
            dataset,
            device=torch.device("cpu"),
            decoding="iterative",
            num_samples=2,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            num_examples=-1,
            seed=42,
            skip_perplexity=True,
        )

        self.assertEqual(payload["records"][0]["target_len"], 2)
        self.assertEqual(
            payload["records"][0]["length_policy"]["source"],
            "native_mask_count",
        )
        self.assertEqual(payload["length_policy"]["mode"], "fixed")

        self.assertEqual(payload["records"][0]["samples"], ["WY", "WW"])
        self.assertEqual(shared_decode.call_args.kwargs["decoding"], "iterative")
        self.assertEqual(len(shared_decode.call_args.args[5]), 2)


    @patch("cdrseqgen.methods.mlm.length.sample_masked_span")
    def test_held_out_eval_uses_the_requested_anchor_trim(self, shared_decode):
        native_ids = self.tokenizer.encode("ACDEFG")
        shared_decode.return_value = [self.tokenizer.encode("AYDEFG")]
        dataset = [{
            "input_ids": torch.tensor([self.tokenizer.mask_id] * 6),
            "labels": torch.tensor(native_ids),
            "target_len": 6,
        }]

        payload = run_generative_eval(
            FakeModel(),
            self.tokenizer,
            dataset,
            device=torch.device("cpu"),
            decoding="independent",
            num_samples=1,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            num_examples=-1,
            seed=42,
            skip_perplexity=True,
            anchor_trim=1,
        )

        # With one residue trimmed at each end, positions 1..4 are core and only position 1
        # differs: 3/4. The old hard-coded trim of two would instead report 2/2.
        self.assertEqual(payload["anchor_trim_each_end"], 1)
        self.assertAlmostEqual(
            payload["records"][0]["metrics"]["core_recovery_mean"],
            0.75,
        )


# ====================================================================================================
# from test_eos_command_setup.py
# EOS command contracts shared by AR/S2S, without models or inference.
# ====================================================================================================

METHODS = (("ar", "causal", CAUSAL_DECODE_STEPS),
           ("s2s", "seq2seq", S2S_DECODE_STEPS))

def arguments(**overrides):
    values = dict(decoding=None, length_mode=None, min_length=None, max_length=None,
                  target_length=None, use_native_length=None, length_temperature=1.0,
                  max_new_tokens=None, target_loop=None)
    return SimpleNamespace(**(values | overrides))


def module(method, kind):
    return importlib.import_module(f"cdrseqgen.methods.{method}.{kind}")


def prepare(method, kind, args, config, cases=None):
    if kind == "generation":
        return module(method, kind).prepare_generation(args, [{}] if cases is None else cases, config)
    return module(method, kind).prepare_evaluation(args, config)


class EosCommandSetupTests(unittest.TestCase):
    def test_modes_and_decode_cap_precedence_through_both_commands(self):
        settings = ((7, {"gen_max_new_tokens": 9}, 7),
                    (None, {"gen_max_new_tokens": 9}, 9), (None, {}, None))
        for method, decoder, default in METHODS:
            for kind in ("generation", "evaluation"):
                for requested, config, expected in settings:
                    with self.subTest(method=method, kind=kind, requested=requested, config=config):
                        args = arguments(max_new_tokens=requested)
                        self.assertIsNone(prepare(method, kind, args, config))
                        self.assertEqual(args.max_new_tokens, default if expected is None else expected)
                        self.assertEqual(args.decoding, decoder)
                        if kind == "evaluation":
                            self.assertEqual(args.length_mode, "inferred")
                            self.assertEqual(module(method, kind).evaluation_options(args, "cpu"),
                                             {"max_new_tokens": args.max_new_tokens})

    def test_invalid_explicit_saved_and_null_caps_still_fail(self):
        for method, _, default in METHODS:
            for kind in ("generation", "evaluation"):
                for value in (0, -1, True, 1.5, default + 1, None):
                    configs = [(None, {"gen_max_new_tokens": value})]
                    if value is not None:
                        configs.append((value, {"gen_max_new_tokens": 8}))
                    for requested, config in configs:
                        with self.subTest(method=method, kind=kind, value=value, requested=requested):
                            args = arguments(max_new_tokens=requested)
                            with self.assertRaisesRegex(ValueError, "max_new_tokens"):
                                prepare(method, kind, args, config)

    def test_generation_accepts_missing_decode_cap_attribute(self):
        for method, _, default in METHODS:
            args = arguments()
            del args.max_new_tokens
            prepare(method, "generation", args, {})
            self.assertEqual(args.max_new_tokens, default)

    def test_case_length_controls_reject_zero_and_report_the_exact_key(self):
        for method, _, _ in METHODS:
            for key in ("min_length", "max_length", "target_length", "use_native_length"):
                for value in (0, 8, True):
                    with self.subTest(method=method, key=key, value=value):
                        with self.assertRaises(ValueError) as caught:
                            prepare(method, "generation", arguments(), {}, [{key: value}])
                        self.assertEqual(str(caught.exception),
                                         f"{method.upper()} chooses length via EOS; {key} is unsupported")

    def test_case_none_or_false_overrides_cli_target_length_controls(self):
        for method, _, _ in METHODS:
            for key in ("target_length", "use_native_length"):
                for value in (None, False):
                    with self.subTest(method=method, key=key, value=value):
                        args = arguments(**{key: 8})
                        prepare(method, "generation", args, {}, [{key: value}])
                        with self.assertRaisesRegex(ValueError, key):
                            prepare(method, "generation", arguments(**{key: 8}), {}, [{}])

    def test_evaluation_rejects_native_length_and_incompatible_decoder(self):
        for method, _, _ in METHODS:
            for overrides in ({"length_mode": "fixed_native"}, {"decoding": "independent"},
                              {"min_length": 8}, {"length_temperature": 2.0}):
                with self.subTest(method=method, overrides=overrides):
                    with self.assertRaises(ValueError):
                        prepare(method, "evaluation", arguments(**overrides), {})

    def test_ar_target_region_still_controls_native_iglm_chain(self):
        for requested, expected, chain in ((None, "h3", "[HEAVY]"), ("l3", "l3", "[LIGHT]")):
            args = arguments(target_loop=requested)
            prepare("ar", "generation", args, {"target_region": "h3"})
            self.assertEqual(args.target_region, expected)
            self.assertEqual(resolve_conditioning({"target_region": args.target_region})["chain_token"], chain)

    def test_self_scorer_keeps_legacy_import_and_shared_command_ownership(self):
        for method, _, _ in METHODS:
            sampling = module(method, "sampling")
            self.assertIs(module(method, "evaluation").perplexities, sampling.perplexities)
            self.assertIs(module(method, "generation").perplexities, sampling.perplexities)
            self.assertEqual(sampling.perplexities.__module__, sampling.__name__)


if __name__ == "__main__":
    unittest.main()
