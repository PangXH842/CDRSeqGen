"""Tests merged from: test_mlm_generation.py, test_mlm_recovery.py, test_mlm_training_workflow.py, test_masking_policy.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import unittest
from unittest.mock import patch
from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.methods.mlm.length import prepare_fixed_length_prompt
from cdrseqgen.methods.mlm.sampling import sample_masked_span
from unittest.mock import call, patch
import numpy as np
from cdrseqgen.eval.metrics import core_anchor_indices
from cdrseqgen.methods.mlm import recovery
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel
from cdrseqgen.methods.common import evaluation as shared_evaluation
from cdrseqgen.methods.common import training as shared_training
from cdrseqgen.methods.mlm import train_cli, training
import json
import tempfile
from pathlib import Path
import torch
from cdrseqgen.datasets.dataset_mlm import (
    MaskedInfillingDataset,
    AntibodyTokenizer,
    TokenizerAdapter,
    mlm_collate_fn,
    normalize_masking_policy,
)
from cdrseqgen.methods.mlm.training import get_datasets


# ====================================================================================================
# from test_mlm_generation.py
# ====================================================================================================

class FixedLengthMLMGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())

    def test_prompt_preserves_exact_mask_count_and_framework_order(self):
        prompt = prepare_fixed_length_prompt(
            self.tokenizer, "AC", "WY", 3, max_len=20
        )

        expected = (
            self.tokenizer.encode("AC")
            + [self.tokenizer.mask_id] * 3
            + self.tokenizer.encode("WY")
        )
        self.assertEqual(prompt.input_ids, expected)
        self.assertEqual(prompt.mask_positions, [2, 3, 4])
        self.assertEqual(prompt.target_length, 3)
        self.assertEqual(prompt.metadata()["input_tokens"], 7)

    def test_prompt_reports_training_compatible_context_truncation(self):
        prompt = prepare_fixed_length_prompt(
            self.tokenizer, "ACDE", "FGHI", 3, max_len=7
        )

        self.assertEqual(len(prompt.input_ids), 7)
        self.assertEqual(prompt.pt1_tokens, 4)
        self.assertEqual(prompt.pt2_tokens, 0)
        self.assertEqual(prompt.pt1_truncated_tokens, 0)
        self.assertEqual(prompt.pt2_truncated_tokens, 4)

    @patch("cdrseqgen.methods.mlm.sampling.sample_iterative")
    @patch("cdrseqgen.methods.mlm.sampling.sample_independent")
    def test_decoder_dispatch_preserves_sampling_arguments(self, independent, iterative):
        independent.return_value = [[1, 2]]
        result = sample_masked_span(
            model="model",
            tokenizer=self.tokenizer,
            device="cpu",
            generator="generator",
            input_ids=[4, 4],
            mask_positions=[0, 1],
            decoding="independent",
            num_samples=5,
            temperature=0.8,
            top_k=7,
            top_p=0.9,
        )

        self.assertEqual(result, [[1, 2]])
        independent.assert_called_once_with(
            model="model",
            input_ids=[4, 4],
            mask_positions=[0, 1],
            banned_ids=self.tokenizer.special_ids,
            device="cpu",
            num_samples=5,
            temperature=0.8,
            top_k=7,
            top_p=0.9,
            generator="generator",
        )
        iterative.assert_not_called()

        iterative.return_value = [[2, 1]]
        result = sample_masked_span(
            model="model",
            tokenizer=self.tokenizer,
            device="cpu",
            generator="generator",
            input_ids=[4, 4],
            mask_positions=[0, 1],
            decoding="iterative",
            num_samples=5,
            temperature=0.8,
            top_k=7,
            top_p=0.9,
        )

        self.assertEqual(result, [[2, 1]])
        iterative.assert_called_once()


# ====================================================================================================
# from test_mlm_recovery.py
# Trainer recovery uses shared region boundaries and retains token pooling.
# ====================================================================================================

class MLMRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(recovery, "_warned_low_core_coverage", False))

    def test_shared_loop_offsets_map_to_input_columns_and_preserve_short_rows(self):
        selected = np.zeros((3, 10), dtype=bool)
        selected[0, 2:8] = True   # loop length 6, after two context positions
        selected[1, 1:5] = True   # loop length 4: all anchor at trim 2
        with patch.object(recovery, "core_anchor_indices", wraps=core_anchor_indices) as split:
            core, anchor = recovery._core_anchor_masks(selected, trim=2)
        self.assertEqual(split.call_args_list, [call(6, 2), call(4, 2), call(0, 2)])
        self.assertEqual(np.flatnonzero(core[0]).tolist(), [4, 5])
        self.assertEqual(np.flatnonzero(anchor[0]).tolist(), [2, 3, 6, 7])
        self.assertFalse(core[1:].any())
        self.assertEqual(np.flatnonzero(anchor[1]).tolist(), [1, 2, 3, 4])
        self.assertFalse(anchor[2].any())
        np.testing.assert_array_equal(core | anchor, selected)
        self.assertFalse((core & anchor).any())

    def test_zero_trim_assigns_every_supervised_position_to_core(self):
        selected = np.array([[False, True, True, True, False]])
        core, anchor = recovery._core_anchor_masks(selected, trim=0)
        np.testing.assert_array_equal(core, selected)
        self.assertFalse(anchor.any())

    def test_accuracy_pools_tokens_while_coverage_counts_rows(self):
        labels = np.full((2, 10), -100)
        labels[0, 2:8] = [1, 2, 3, 4, 5, 6]
        labels[1, 1:4] = [1, 2, 3]
        predicted = np.full((2, 10), 7)
        predicted[0, 2:8] = labels[0, 2:8]
        logits = np.eye(8)[predicted]
        # Row 0 is entirely correct; row 1 is entirely wrong and has no core.
        # Token pooling gives 6/9 rather than the per-row mean of 1/2.
        scores = recovery.compute_metrics(((logits, np.zeros((2, 1))), labels), trim=2)
        self.assertEqual(scores, {
            "accuracy": 6/9, "core_accuracy": 1.,
            "anchor_accuracy": 4/7, "core_coverage": 1/2,
        })

    def test_no_core_is_distinguished_from_no_correct_tokens_by_coverage(self):
        labels = np.array([[0, 1, 2, 3]])
        logits = np.eye(4)[labels]
        with patch.object(recovery.logger, "warning") as warning:
            scores = recovery.compute_metrics((logits, labels), trim=2)
        self.assertEqual(scores, {
            "accuracy": 1., "core_accuracy": 0.,
            "anchor_accuracy": 1., "core_coverage": 0.,
        })
        warning.assert_called_once()

    def test_entirely_unsupervised_input_keeps_existing_return_keys(self):
        labels = np.full((1, 4), -100)
        scores = recovery.compute_metrics((np.zeros((1, 4, 5)), labels), trim=2)
        self.assertEqual(scores, dict(accuracy=0., core_accuracy=0., anchor_accuracy=0.))

    def test_low_coverage_warns_once_and_describes_population_without_dismissing_score(self):
        with patch.object(recovery.logger, "warning") as warning:
            recovery._warn_if_low_core_coverage(0.5, 2, 4, trim=2)
            warning.assert_not_called()
            recovery._warn_if_low_core_coverage(0.25, 1, 4, trim=2)
            recovery._warn_if_low_core_coverage(0.25, 1, 4, trim=2)
        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn("1/4", message)
        self.assertIn("25.0%", message)
        self.assertIn("non-empty core", message)
        self.assertIn("whole-span accuracy", message)
        self.assertNotIn("noise", message)


# ====================================================================================================
# from test_mlm_training_workflow.py
# MLM command/workflow contracts; model, training and generation are mocked.
# ====================================================================================================

class MLMCommandTests(unittest.TestCase):
    def test_command_configures_logging_parses_and_delegates(self):
        arguments = (sentinel.model_args, sentinel.data_args,
                     sentinel.gen_args, sentinel.training_args)
        events = Mock()
        with patch.object(train_cli, "configure_logging") as logging, \
                patch.object(train_cli, "setup_args", return_value=arguments) as parse, \
                patch.object(train_cli, "run_training_pipeline") as run:
            events.attach_mock(logging, "logging")
            events.attach_mock(parse, "parse")
            events.attach_mock(run, "run")
            self.assertIsNone(train_cli.main())
            run.assert_called_once_with(
                *arguments, prepare_training=training.prepare_training,
                finish_training=training.finish_training)
        self.assertEqual([call[0] for call in events.mock_calls], ["logging", "parse", "run"])


class MLMTrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.model_args = SimpleNamespace(
            model_type="scratch", hidden_size=128, num_layers=4,
            num_attention_heads=8, intermediate_size=512, loss_type="weighted",
            core_loss_weight=3., anchor_loss_weight=1., anchor_trim=2,
        )
        self.data_args = SimpleNamespace(
            max_len=384, train_length_predictor=True, length_predictor_epochs=20,
            length_predictor_learning_rate=0.01, length_predictor_batch_size=16,
        )
        self.gen_args = SimpleNamespace(
            run_generative_eval=True, gen_decoding="iterative", gen_length_mode="bounded",
            gen_min_length=6, gen_max_length=24, gen_length_temperature=0.7,
        )
        self.training_args = SimpleNamespace(
            seed=42, output_dir="/tmp/mock-mlm-workflow", device="cpu",
            do_train=True, do_eval=True, label_smoothing_factor=0.1,
        )
        self.arguments = (self.model_args, self.data_args, self.gen_args, self.training_args)
        self.tokenizer = SimpleNamespace(vocab_size=25, pad_id=7)
        self.train_dataset, self.eval_dataset = [sentinel.train], [sentinel.eval]
        self.trainer = SimpleNamespace(model=sentinel.initial_model)
        self.backbone = SimpleNamespace(load_training_model=Mock(return_value=sentinel.initial_model))
        self.events = Mock()

        self.seed = self.mock_stage("set_seed")
        self.resolve = self.mock_stage("resolve", return_value=self.backbone)
        self.datasets = self.mock_stage(
            "get_datasets", return_value=(self.train_dataset, self.eval_dataset, self.tokenizer))
        self.latest = self.mock_stage("update_latest_link")
        self.events.attach_mock(self.backbone.load_training_model, "load_model")
        self.callbacks = self.mock_stage("build_callbacks", return_value=[sentinel.callback])
        self.make_trainer = self.mock_stage("MLMLossTrainer", side_effect=self.make_trainer_state)
        self.lifecycle = self.enterContext(patch.object(
            shared_training, "train_and_save", side_effect=self.select_checkpoint))
        self.events.attach_mock(self.lifecycle, "train_and_save")
        self.fit = self.enterContext(patch(
            "cdrseqgen.methods.mlm.length.fit_length_predictor",
            return_value=sentinel.length_predictor))
        self.events.attach_mock(self.fit, "fit_length_predictor")
        self.generate = self.enterContext(patch.object(
            shared_evaluation, "post_training_generation"))
        self.events.attach_mock(self.generate, "post_training_generation")

    def mock_stage(self, name, **kwargs):
        mocked = self.enterContext(patch.object(training, name, **kwargs))
        self.events.attach_mock(mocked, name)
        return mocked

    def make_trainer_state(self, **kwargs):
        self.trainer.train_dataset = kwargs["train_dataset"]
        self.trainer.eval_dataset = kwargs["eval_dataset"]
        self.trainer.loss_config = kwargs["loss_config"]
        return self.trainer

    def run_workflow(self):
        return shared_training.run_training_pipeline(
            *self.arguments, prepare_training=training.prepare_training,
            finish_training=training.finish_training)

    def select_checkpoint(self, trainer, args, *, tokenizer):
        if args.do_train:
            trainer.model = sentinel.selected_model

    def test_full_workflow_preserves_order_and_uses_selected_model_for_length_head(self):
        self.assertIs(self.run_workflow(), self.trainer)
        self.assertEqual([call[0] for call in self.events.mock_calls], [
            "set_seed", "resolve", "get_datasets", "update_latest_link", "load_model",
            "build_callbacks", "MLMLossTrainer", "train_and_save",
            "fit_length_predictor", "post_training_generation",
        ])
        self.seed.assert_called_once_with(42)
        self.resolve.assert_called_once_with("scratch")
        self.datasets.assert_called_once_with(
            self.model_args, self.data_args, self.training_args, self.backbone)
        self.latest.assert_called_once_with(self.training_args.output_dir)
        self.backbone.load_training_model.assert_called_once_with(
            "scratch", vocab_size=25, hidden_size=128, num_layers=4,
            num_attention_heads=8, intermediate_size=512, max_len=384)

        options = self.make_trainer.call_args.kwargs
        self.assertIs(options["model"], sentinel.initial_model)
        self.assertIs(options["args"], self.training_args)
        self.assertIs(options["train_dataset"], self.train_dataset)
        self.assertIs(options["eval_dataset"], self.eval_dataset)
        self.assertEqual(options["loss_config"].loss_type, "weighted_cross_entropy")
        self.assertEqual(options["loss_config"].label_smoothing, 0.1)
        self.assertIs(options["data_collator"].func, training.mlm_collate_fn)
        self.assertEqual(options["data_collator"].keywords, {"pad_token_id": 7})
        self.assertIs(options["compute_metrics"].func, training.compute_metrics)
        self.assertEqual(options["compute_metrics"].keywords, {"trim": 2})
        self.assertEqual(options["callbacks"], [sentinel.callback])
        self.lifecycle.assert_called_once_with(
            self.trainer, self.training_args, tokenizer=self.tokenizer)
        self.fit.assert_called_once_with(
            sentinel.selected_model, self.tokenizer, self.train_dataset, self.eval_dataset,
            output_dir=self.training_args.output_dir, device="cpu", epochs=20,
            learning_rate=0.01, batch_size=16, seed=42)
        self.generate.assert_called_once_with(
            "mlm", self.trainer, self.tokenizer, self.eval_dataset,
            self.data_args, self.gen_args, self.training_args,
            anchor_trim=2, method_options={
                "decoding": "iterative", "length_mode": "bounded",
                "length_predictor": sentinel.length_predictor,
                "min_length": 6, "max_length": 24, "length_temperature": 0.7,
            })

    def test_evaluation_only_skips_length_fitting_and_keeps_generation_available(self):
        self.training_args.do_train = False
        self.run_workflow()
        self.lifecycle.assert_called_once()
        self.fit.assert_not_called()
        self.assertIs(self.trainer.model, sentinel.initial_model)
        self.assertIsNone(self.generate.call_args.kwargs["method_options"]["length_predictor"])

    def test_extensions_can_both_be_disabled(self):
        self.data_args.train_length_predictor = False
        self.gen_args.run_generative_eval = False
        self.run_workflow()
        self.lifecycle.assert_called_once()
        self.fit.assert_not_called()
        self.generate.assert_not_called()

    def test_length_fitting_does_not_depend_on_generation_being_enabled(self):
        self.gen_args.run_generative_eval = False
        self.run_workflow()
        self.fit.assert_called_once()
        self.generate.assert_not_called()

    def test_dataset_failure_leaves_latest_and_model_untouched(self):
        self.datasets.side_effect = ValueError("invalid evaluation split")
        with self.assertRaisesRegex(ValueError, "invalid evaluation split"):
            self.run_workflow()
        self.latest.assert_not_called()
        self.backbone.load_training_model.assert_not_called()
        self.make_trainer.assert_not_called()
        self.lifecycle.assert_not_called()
        self.fit.assert_not_called()
        self.generate.assert_not_called()

    def test_training_failure_stops_before_post_training_extensions(self):
        self.lifecycle.side_effect = RuntimeError("training failed")
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.run_workflow()
        self.fit.assert_not_called()
        self.generate.assert_not_called()


# ====================================================================================================
# from test_masking_policy.py
# ====================================================================================================

def _chain(cdr1="CDEF"):
    regions = {
        "fwr1": "AA",
        "cdr1": cdr1,
        "fwr2": "GG",
        "cdr2": "HI",
        "fwr3": "KK",
        "cdr3": "LMN",
        "fwr4": "PP",
    }
    regions["full_aa"] = "".join(regions.values())
    return regions


def _records():
    return [
        {"heavy": _chain(), "light": _chain("QRST"), "split": split}
        for split in ("train", "validation")
    ]


class _Backbone:
    def build_tokenizer(self, _model_type, checkpoint_path=None):
        return TokenizerAdapter.from_custom(AntibodyTokenizer())


class MaskingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "processed.json"
        self.path.write_text(json.dumps(_records()))
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())

    def tearDown(self):
        self.temp.cleanup()

    def _dataset(self, **kwargs):
        return MaskedInfillingDataset(
            str(self.path), self.tokenizer, target_region="h1", **kwargs
        )

    def test_full_span_remains_the_default(self):
        item = self._dataset()[0]
        masked = item["labels"] != -100
        self.assertEqual(item["masked_count"], item["target_len"])
        self.assertTrue(torch.all(item["input_ids"][masked] == self.tokenizer.mask_id))

    def test_mixed_policy_reveals_and_unsupervises_native_residues(self):
        torch.manual_seed(7)
        item = self._dataset(
            masking_policy="mixed", mixed_full_span_probability=0.0
        )[0]
        supervised = item["labels"] != -100
        span_start = len(self.tokenizer.encode("AA"))
        span = item["input_ids"][span_start:span_start + item["target_len"]]
        target = torch.tensor(self.tokenizer.encode("CDEF"))
        span_labels = item["labels"][span_start:span_start + item["target_len"]]
        visible = span != self.tokenizer.mask_id

        self.assertGreater(item["masked_count"], 0)
        self.assertLess(item["masked_count"], item["target_len"])
        self.assertEqual(int(supervised.sum()), item["masked_count"])
        self.assertEqual(int((span == self.tokenizer.mask_id).sum()), item["masked_count"])
        self.assertTrue(torch.equal(span[visible], target[visible]))
        self.assertTrue(torch.all(span_labels[visible] == -100))
        offsets = item["target_offsets"][span_start:span_start + item["target_len"]]
        self.assertTrue(torch.equal(offsets, torch.arange(item["target_len"])))

    def test_collate_preserves_padded_loss_metadata(self):
        first = self._dataset()[0]
        second = self._dataset()[1]
        batch = mlm_collate_fn([first, second], pad_token_id=self.tokenizer.pad_id)
        self.assertEqual(batch["target_offsets"].shape, batch["labels"].shape)
        self.assertEqual(batch["target_lengths"].tolist(), [4, 4])

    def test_pipeline_uses_distinct_shared_data_views_and_full_span_eval(self):
        data_args = SimpleNamespace(
            data_path=str(self.path),
            max_len=64,
            test_size=0.5,
            eval_split="validation",
            target_region="h1",
            include_partner_chain=False,
            include_antigen=False,
            context_dropout=0.0,
            masking_policy="mixed",
            mixed_full_span_probability=0.0,
        )
        model_args = SimpleNamespace(model_type="scratch")
        training_args = SimpleNamespace(seed=42)

        train, evaluation, _ = get_datasets(
            model_args, data_args, training_args, _Backbone()
        )

        self.assertIsNot(train.dataset, evaluation.dataset)
        self.assertIs(train.dataset.data, evaluation.dataset.data)
        self.assertEqual(train.dataset.masking_policy, "mixed")
        self.assertEqual(evaluation.dataset.masking_policy, "full_span")
        self.assertEqual(evaluation[0]["masked_count"], evaluation[0]["target_len"])

    def test_policy_alias_and_invalid_probability_are_validated(self):
        self.assertEqual(normalize_masking_policy("fullspan"), "full_span")
        with self.assertRaisesRegex(ValueError, "masking_policy"):
            normalize_masking_policy("single_token")
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self._dataset(masking_policy="mixed", mixed_full_span_probability=1.1)


if __name__ == "__main__":
    unittest.main()
