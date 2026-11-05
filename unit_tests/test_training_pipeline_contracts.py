"""Training-pipeline contracts that the rest of the suite reached only through subprocess runs.

A line trace of the suite showed these shared pieces were never executed in-process: the early-stopping
branch of build_callbacks, AR/S2S prepare_training, S2S compute_metrics (the checkpoint-selection
metric), initialize_run with its run naming and provenance, and the draw path where every attempt fails.
Real tiny CPU models and corpora throughout; nothing is mocked except where a stub is the point.
"""
import contextlib
import json
import logging
import os
import random
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import EarlyStoppingCallback, GPT2Config, GPT2LMHeadModel, HfArgumentParser, TrainingArguments

from cdrseqgen.datasets.dataset_ar import span_layout_metadata
from cdrseqgen.methods.common import run_setup
from cdrseqgen.methods.common.arguments import initialize_run, parse_config_dict
from cdrseqgen.methods.common.callbacks import LogMetricsCallback, build_callbacks
from cdrseqgen.methods.common.eos_decoding import EosDecodingStrategy
from cdrseqgen.methods.common.training import run_training_pipeline
from cdrseqgen.methods.ar import training as ar_training
from cdrseqgen.methods.ar.models import iglm
from cdrseqgen.methods.registry import resolve
from cdrseqgen.methods.s2s import training as s2s_training

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def write_corpus(path, train=16, validation=8, seed=0):
    rng = random.Random(seed)
    records = []
    for index in range(train + validation):
        loop = "".join(rng.choice(RESIDUES) for _ in range(rng.randint(4, 9)))
        chain = dict(fwr1="AAAA", cdr1="CC", fwr2="DDDD", cdr2="EE", fwr3="FFFF", cdr3=loop, fwr4="HHHH")
        chain["full_aa"] = "".join(chain.values())
        records.append({"heavy": chain, "light": chain, "species": "human",
                        "split": "train" if index < train else "validation"})
    Path(path).write_text(json.dumps(records))


def parsed(config):
    module = resolve(config["method"]).module("arguments")
    parser = HfArgumentParser((module.ModelArguments, module.DataArguments, module.GenerationArguments,
                               TrainingArguments))
    return parse_config_dict(parser, config)


def tiny_config(method, data, output, **extra):
    config = dict(method=method, model_type="scratch", hidden_size=32, num_layers=2, num_attention_heads=2,
                  data_path=str(data), target_region="h3", eval_split="validation", max_len=192,
                  include_partner_chain=False, output_dir=str(output), do_train=True, do_eval=True,
                  per_device_train_batch_size=4, per_device_eval_batch_size=4, num_train_epochs=2,
                  learning_rate=1e-3, eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
                  save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="core_accuracy",
                  greater_is_better=True, early_stopping_patience=2, report_to="none", seed=42, data_seed=42,
                  use_cpu=True, dataloader_num_workers=0, run_generative_eval=False)
    config.update(extra)
    return config


class CallbackTests(unittest.TestCase):
    def callbacks(self, patience, load_best, threshold=0.0):
        model_args = SimpleNamespace(early_stopping_patience=patience, early_stopping_threshold=threshold)
        training_args = SimpleNamespace(load_best_model_at_end=load_best, metric_for_best_model="core_accuracy")
        return build_callbacks(model_args, training_args)

    def test_metric_logging_is_always_present_and_early_stopping_only_when_requested(self):
        for patience, load_best in ((None, True), (0, True), (3, False)):
            with self.subTest(patience=patience, load_best=load_best):
                found = self.callbacks(patience, load_best)
                self.assertEqual([type(c) for c in found], [LogMetricsCallback])
        found = self.callbacks(3, True, threshold=0.001)
        self.assertEqual([type(c) for c in found], [LogMetricsCallback, EarlyStoppingCallback])
        self.assertEqual((found[1].early_stopping_patience, found[1].early_stopping_threshold), (3, 0.001))

    def test_trainer_log_events_reach_the_logger_with_and_without_an_epoch(self):
        callback = LogMetricsCallback()
        with self.assertLogs("cdrseqgen.methods.common.callbacks", level="INFO") as captured:
            callback.on_log(None, None, None, logs={"epoch": 1.0, "loss": 0.123456})
            callback.on_log(None, None, None, logs={"train_runtime": 3.0})
            callback.on_log(None, None, None, logs=None)
        self.assertEqual(len(captured.records), 2)
        self.assertIn("epoch   1.00", captured.output[0])
        self.assertIn("loss=0.1235", captured.output[0])
        self.assertIn("train_runtime=3", captured.output[1])


class RunSetupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(contextlib.chdir(self.root))
        self.enterContext(patch.dict(os.environ, {run_setup.RESULTS_ROOT_ENV: str(self.root / "batch")}))
        root_logger = logging.getLogger()
        original_handlers = set(root_logger.handlers)

        def close_run_logs():
            for handler in list(root_logger.handlers):
                if isinstance(handler, logging.FileHandler) and handler not in original_handlers:
                    root_logger.removeHandler(handler)
                    handler.close()

        self.addCleanup(close_run_logs)
        for name in (run_setup.RUN_SUFFIX_ENV, run_setup.SKIP_LATEST_ENV):
            os.environ.pop(name, None)
        write_corpus(self.root / "processed_oas_multi.json")

    def init(self, method="ar", marker="ar", **config):
        arguments = parsed(tiny_config(method, self.root / "processed_oas_multi.json", "unset", **config))
        return initialize_run(*arguments, marker=marker)

    def test_run_directory_is_named_from_resolved_arguments(self):
        cases = [({}, "h3_OAS4_ar_scratch"),
                 ({"include_partner_chain": True}, "h3_OAS4_LC_ar_scratch"),
                 ({"target_region": "l3", "include_partner_chain": True}, "l3_OAS4_HC_ar_scratch")]
        for index, (config, name) in enumerate(cases):
            with self.subTest(name=name):
                training_args = self.init(**config)
                self.assertEqual(Path(training_args.output_dir), self.root / "batch" / name)
                self.assertEqual(training_args.run_name, name)

    def test_suffix_names_the_trial_and_a_clash_never_overwrites(self):
        with patch.dict(os.environ, {run_setup.RUN_SUFFIX_ENV: "lr3e-5"}):
            first = self.init()
        second = self.init()
        with self.assertLogs("cdrseqgen.methods.common.run_setup", level="WARNING"):
            third = self.init()
        self.assertEqual(Path(first.output_dir).name, "h3_OAS4_ar_scratch_lr3e-5")
        self.assertEqual(Path(second.output_dir).name, "h3_OAS4_ar_scratch")
        self.assertEqual(Path(third.output_dir).name, "h3_OAS4_ar_scratch_2")

    def test_provenance_and_log_are_written_into_the_run(self):
        training_args = self.init()
        run_dir = Path(training_args.output_dir)
        saved = json.loads((run_dir / "run_config.json").read_text())
        self.assertEqual((saved["model_type"], saved["target_region"], saved["seed"]), ("scratch", "h3", 42))
        provenance = saved["provenance"]
        self.assertEqual(provenance["model_source"], "scratch")
        self.assertEqual(Path(provenance["dataset"]["path"]), (self.root / "processed_oas_multi.json").resolve())
        self.assertTrue(provenance["dataset"]["size_bytes"] > 0)
        self.assertEqual(set(provenance["runtime"]["packages"]),
                         {"torch", "transformers", "accelerate", "safetensors", "pandas", "scikit-learn"})
        self.assertTrue((run_dir / "train.log").is_file())

    def test_results_root_prefers_the_environment_and_otherwise_makes_a_timestamped_batch(self):
        self.assertEqual(run_setup.results_root(), str(self.root / "batch"))
        os.environ.pop(run_setup.RESULTS_ROOT_ENV)
        chosen = Path(run_setup.results_root())
        self.assertEqual(chosen.parent, Path("./results/runs"))
        self.assertRegex(chosen.name, r"^\d{8}_\d{6}$")

    def test_latest_follows_the_newest_batch_unless_told_not_to_and_never_replaces_a_directory(self):
        runs = Path("results/runs")
        run_setup.update_latest_link(str(runs / "b1" / "run"))
        self.assertEqual(os.readlink(runs / "latest"), "b1")
        run_setup.update_latest_link(str(runs / "b2" / "run"))
        self.assertEqual(os.readlink(runs / "latest"), "b2")
        with patch.dict(os.environ, {run_setup.SKIP_LATEST_ENV: "1"}):
            run_setup.update_latest_link(str(runs / "b3" / "run"))
        self.assertEqual(os.readlink(runs / "latest"), "b2")
        run_setup.update_latest_link(str(runs / "directly_in_runs"))        # no batch folder: never self-link
        self.assertEqual(os.readlink(runs / "latest"), "b2")
        (runs / "latest").unlink()
        (runs / "latest").mkdir()
        with self.assertLogs("cdrseqgen.methods.common.run_setup", level="WARNING"):
            run_setup.update_latest_link(str(runs / "b4" / "run"))
        self.assertTrue((runs / "latest").is_dir() and not (runs / "latest").is_symlink())

    def test_dataset_tags_keep_incomparable_corpora_apart(self):
        for path, tag in (("data/processed_oas_multi.json", "OAS4"), ("data/processed_oas_data.json", "OAS"),
                          ("data/processed_sabdab2_data.json", "SD2")):
            self.assertEqual(run_setup.dataset_tag(path), tag)
        with self.assertLogs("cdrseqgen.methods.common.run_setup", level="WARNING") as captured:
            self.assertEqual(run_setup.dataset_tag("data/mystery.json"), run_setup.UNKNOWN_DATASET_TAG)
        self.assertIn("mystery.json", captured.output[0])


class MetricAlignmentTests(unittest.TestCase):
    """AR shifts by one and S2S does not; both must report the same recovery for the same loop."""
    LOOP = [5, 6, 7, 8, 9, 10]            # six residues, so two anchors at each end and a core of two
    EOS = 2

    def ar(self, wrong=None, shift=0):
        labels = [-100, -100] + self.LOOP + [self.EOS]
        guesses = list(self.LOOP) + [self.EOS]
        if wrong is not None:
            guesses[wrong] = 24
        predictions = [0] + [0] + guesses            # predictions[t] guesses labels[t + 1]; two prompt slots
        predictions = predictions[1 + shift:] + [0] * shift
        return ar_training.compute_metrics((np.array([predictions]), np.array([labels])), trim=2)

    def s2s(self, wrong=None, shift=0):
        labels = self.LOOP + [self.EOS]
        guesses = list(labels)
        if wrong is not None:
            guesses[wrong] = 24
        guesses = guesses[shift:] + [0] * shift
        return s2s_training.compute_metrics((np.array([guesses]), np.array([labels])), trim=2)

    def test_both_alignments_report_the_same_recovery_for_the_same_errors(self):
        self.assertEqual(self.ar(), self.s2s())
        self.assertEqual(self.s2s()["accuracy"], 1.0)
        for wrong, core, anchor in ((3, 0.5, 1.0), (0, 1.0, 0.75), (1, 1.0, 0.75)):
            with self.subTest(wrong=wrong):
                result = self.s2s(wrong=wrong)
                self.assertEqual(result, self.ar(wrong=wrong))
                self.assertAlmostEqual(result["core_accuracy"], core)
                self.assertAlmostEqual(result["anchor_accuracy"], anchor)
                self.assertAlmostEqual(result["accuracy"], 5 / 6)
                self.assertEqual(result["core_coverage"], 1.0)

    def test_the_masked_method_reports_the_same_numbers_for_the_same_errors(self):
        """Sweep tables and checkpoint selection compare core_accuracy across the three methods."""
        from cdrseqgen.methods.mlm.recovery import compute_metrics as masked_metrics
        labels = np.array([[-100, -100] + self.LOOP + [-100]])            # framework, hidden loop, framework
        for wrong in (None, 3, 0):
            with self.subTest(wrong=wrong):
                guesses = np.array([[0, 0] + self.LOOP + [0]])
                if wrong is not None:
                    guesses[0, 2 + wrong] = 24
                logits = np.zeros((1, guesses.shape[1], 25))
                logits[0, np.arange(guesses.shape[1]), guesses[0]] = 1.0
                self.assertEqual(masked_metrics((logits, labels), trim=2), self.s2s(wrong=wrong))

    def test_an_off_by_one_alignment_would_be_visible(self):
        self.assertLess(self.s2s(shift=1)["accuracy"], 1.0)
        self.assertLess(self.ar(shift=1)["accuracy"], 1.0)

    def test_logits_and_decoded_ids_give_the_same_numbers(self):
        labels = np.array([self.LOOP + [self.EOS]])
        ids = np.array([[5, 6, 7, 24, 9, 10, 2]])
        logits = np.zeros((1, 7, 25))
        logits[0, np.arange(7), ids[0]] = 1.0
        self.assertEqual(s2s_training.compute_metrics((logits, labels), trim=2),
                         s2s_training.compute_metrics((ids, labels), trim=2))

    def test_unequal_loops_pool_tokens_and_count_eos_only_rows_in_core_coverage(self):
        # EOS-only rows count towards coverage; entirely unsupervised rows do not.
        labels = np.full((5, 11), -100)
        guesses = np.zeros_like(labels)
        for row, loop in enumerate((list(range(5, 13)), list(range(5, 10)), [5, 6], [])):
            labels[row, 1:len(loop) + 2] = loop + [self.EOS]
            guesses[row, 1:len(loop) + 1] = loop
        guesses[0, [1, 4]] = 24       # one wrong anchor and one wrong core residue
        guesses[1, 1:6] = 24          # every residue of the shorter loop is wrong
        # All EOS guesses stay wrong: terminator accuracy must not dilute residue recovery.
        ar_labels = np.pad(labels, ((0, 0), (1, 0)), constant_values=-100)
        ar_guesses = np.pad(guesses, ((0, 0), (0, 1)), constant_values=0)

        for trim, expected in (
            (2, {"accuracy": 8 / 15, "core_accuracy": 3 / 5,
                 "anchor_accuracy": 5 / 10, "core_coverage": 2 / 4}),
            (0, {"accuracy": 8 / 15, "core_accuracy": 8 / 15,
                 "anchor_accuracy": 0.0, "core_coverage": 3 / 4}),
        ):
            with self.subTest(trim=trim):
                result = s2s_training.compute_metrics((guesses, labels), trim=trim)
                self.assertEqual(result,
                                 ar_training.compute_metrics((ar_guesses, ar_labels), trim=trim))
                for key, value in expected.items():
                    self.assertAlmostEqual(result[key], value, msg=key)


class StubStrategy(EosDecodingStrategy):
    """Samples scripted draws, so the accounting is checked without a model."""

    def __init__(self, draws, **kwargs):
        super().__init__(model=None, device="cpu", generator=None, judge_dataset=None, judge_tokenizer=None,
                         num_samples=len(draws), temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=97, **kwargs)
        self.draws = draws

    def context(self, framework):
        return [1, 2, 3]

    def sample(self, context):
        return self.draws

    def score(self, spans):
        return [1.0] * len(spans)


def draw(loop="", complete=True, valid=None, terminated=None):
    valid = complete if valid is None else valid
    return {"loop": loop, "length": len(loop), "complete": complete, "valid": valid,
            "terminated": complete if terminated is None else terminated, "raw_token_ids": [3, 2]}


class FailedFrameworkTests(unittest.TestCase):
    class Corpus:
        def framework_parts(self, index):
            return {"target": "ACDE", "target_length": 4}

    def strategy(self, draws, protocol=False):
        tokenizer = SimpleNamespace(encode=lambda s: [RESIDUES.index(c) + 5 for c in s])
        if protocol:
            tokenizer.causal_sampling_metadata = {"id": "stub"}
        return StubStrategy(draws, tokenizer=tokenizer, eval_dataset=self.Corpus())

    def test_a_framework_whose_every_draw_fails_returns_nothing_but_is_still_counted(self):
        strategy = self.strategy([draw(complete=False), draw(complete=False, terminated=True, valid=False),
                                  draw("", complete=True)], protocol=True)
        self.assertIsNone(strategy.draw(7))
        self.assertEqual((strategy.attempts, strategy.complete_nonempty, strategy.complete_empty,
                          strategy.incomplete, strategy.invalid), (3, 0, 1, 1, 1))
        self.assertEqual([entry["dataset_index"] for entry in strategy.generation_attempts], [7])
        self.assertEqual(len(strategy.generation_attempts[0]["draws"]), 3)

    def test_a_mixed_framework_keeps_its_valid_draws_and_reports_the_failures(self):
        draws = [draw("ACDE"), draw("AC"), draw("ACDE"), draw(""),
                 draw("ACDE", complete=False),
                 draw("A", complete=False, terminated=True, valid=False)]
        strategy = self.strategy(draws, protocol=True)
        drawn = strategy.draw(0)
        self.assertEqual(drawn.seqs, ["ACDE", "AC", "ACDE"])
        self.assertEqual({k: drawn.record_extra[k] for k in ("attempts", "incomplete", "empty", "invalid",
                                                             "length_matches")},
                         {"attempts": 6, "incomplete": 1, "empty": 1, "invalid": 1, "length_matches": 2})
        # an unterminated draw is penalised by at least one residue, a length match is not
        self.assertEqual(drawn.record_extra["length_absolute_error_sum_capped"], 0 + 2 + 0 + 4 + 1 + 3)
        self.assertEqual((strategy.attempts, strategy.complete_nonempty, strategy.complete_empty,
                          strategy.incomplete, strategy.invalid, strategy.length_matches,
                          strategy.length_absolute_error_capped), (6, 3, 1, 1, 1, 2, 10))
        self.assertEqual(strategy.generation_attempts,
                         [{"dataset_index": 0, "conditioning": None, "draws": draws}])

    def test_default_layouts_keep_their_records_free_of_native_protocol_fields(self):
        strategy = self.strategy([draw("ACDE"), draw(complete=False)])
        drawn = strategy.draw(0)
        self.assertNotIn("invalid", drawn.record_extra)
        self.assertEqual(strategy.generation_attempts, [])
        self.assertEqual(strategy.invalid, 0)


class WorkflowTests(unittest.TestCase):
    """AR and S2S run prepare -> train/evaluate/save -> finish through the shared pipeline, in process."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(contextlib.chdir(self.root))
        self.enterContext(patch.dict(os.environ, {run_setup.SKIP_LATEST_ENV: "1"}))
        write_corpus(self.root / "corpus.json")
        self.output = self.root / "results/runs/batch/run"
        self.output.mkdir(parents=True)
        (self.output / "run_config.json").write_text("{}")

    def run_pipeline(self, method, training, **extra):
        arguments = parsed(tiny_config(method, self.root / "corpus.json", self.output, **extra))
        model_args, data_args, gen_args, training_args = arguments
        captured = {}

        def prepare(*args):
            captured["trainer"], tokenizer = training.prepare_training(*args)
            return captured["trainer"], tokenizer

        run_training_pipeline(*arguments, prepare_training=prepare,
                              finish_training=partial(training.finish_training, anchor_trim=model_args.anchor_trim))
        return captured["trainer"], arguments

    def check_artifacts(self, trainer):
        for name in ("model.safetensors", "config.json", "train_results.json", "eval_results.json",
                     "epoch_metrics.csv"):
            self.assertTrue((self.output / name).is_file(), name)
        results = json.loads((self.output / "eval_results.json").read_text())
        for key in ("eval_loss", "eval_accuracy", "eval_core_accuracy", "eval_anchor_accuracy"):
            self.assertTrue(np.isfinite(results[key]), key)
        self.assertIsNotNone(trainer.state.best_metric)
        stoppers = [c for c in trainer.callback_handler.callbacks if isinstance(c, EarlyStoppingCallback)]
        self.assertEqual([c.early_stopping_patience for c in stoppers], [2])
        self.assertTrue(any(isinstance(c, LogMetricsCallback) for c in trainer.callback_handler.callbacks))

    def test_scratch_ar_runs_the_whole_pipeline_and_records_its_layout(self):
        trainer, _ = self.run_pipeline("ar", ar_training)
        self.check_artifacts(trainer)
        saved = json.loads((self.output / "run_config.json").read_text())
        self.assertEqual(saved["span_layout"]["id"], "cdrseqgen_causal")
        self.assertNotIn("causal_sampling", saved)
        self.assertTrue((self.output / "span_layout.json").is_file())

    def test_scratch_s2s_runs_the_whole_pipeline(self):
        trainer, _ = self.run_pipeline("s2s", s2s_training)
        self.check_artifacts(trainer)
        self.assertFalse((self.output / "span_layout.json").exists())

    def test_iglm_backbone_trains_through_the_same_pipeline_and_records_its_protocol(self):
        source = self.root / "tiny-iglm"
        source.mkdir()
        (source / "vocab.txt").write_text("\n".join(iglm.NATIVE_VOCAB) + "\n")
        config = GPT2Config(vocab_size=33, n_positions=512, n_ctx=512, n_embd=16, n_layer=1, n_head=4)
        config.cdrseqgen_backbone = "iglm"
        GPT2LMHeadModel(config).save_pretrained(source)
        config.span_layout = span_layout_metadata(iglm.load_tokenizer(str(source)))   # a converted source keeps its layout
        config.save_pretrained(source)
        (source / "span_layout.json").write_text(json.dumps(config.span_layout))
        trainer, arguments = self.run_pipeline("ar", ar_training, model_type=f"iglm:{source}")
        self.check_artifacts(trainer)
        saved = json.loads((self.output / "run_config.json").read_text())
        self.assertEqual(saved["span_layout"]["id"], "iglm_infill")
        self.assertEqual(saved["causal_sampling"]["id"], "iglm_infill")
        self.assertTrue((self.output / "tokenizer_config.json").is_file())
        layout = json.loads((self.output / "span_layout.json").read_text())
        self.assertEqual(layout, saved["span_layout"])

    def test_a_dataset_failure_leaves_the_latest_link_untouched_and_success_updates_it(self):
        write_corpus(self.root / "corpus.json")
        for method, training in (("ar", ar_training), ("s2s", s2s_training)):
            with self.subTest(method=method), patch.dict(os.environ, {run_setup.SKIP_LATEST_ENV: ""}):
                arguments = parsed(tiny_config(method, self.root / "corpus.json", self.output,
                                               eval_split="no_such_split"))
                with self.assertRaises(ValueError):
                    training.prepare_training(arguments[0], arguments[1], arguments[3])
                self.assertFalse(os.path.lexists("results/runs/latest"))
                arguments = parsed(tiny_config(method, self.root / "corpus.json", self.output))
                training.prepare_training(arguments[0], arguments[1], arguments[3])
                self.assertEqual(os.readlink("results/runs/latest"), "batch")
                os.remove("results/runs/latest")


if __name__ == "__main__":
    unittest.main()
