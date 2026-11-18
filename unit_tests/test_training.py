"""Tests merged from: test_training_pipeline_contracts.py, test_sampling_contracts.py, test_method_integration.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
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
from transformers import (
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    HfArgumentParser,
    TrainingArguments,
)
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
from cdrseqgen.datasets.dataset_ar import causal_adapter
from cdrseqgen.datasets.dataset_s2s import s2s_adapter
from cdrseqgen.methods.ar.sampling import perplexities as ar_perplexities
from cdrseqgen.methods.ar.sampling import sample_loops as ar_sample
from cdrseqgen.methods.common.sampling import warp_logits
from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood, sample_masked_span
from cdrseqgen.methods.s2s.sampling import perplexities as s2s_perplexities
from cdrseqgen.methods.s2s.sampling import sample_loops as s2s_sample
from unittest.mock import Mock, patch
from transformers import HfArgumentParser, TrainingArguments
from cdrseqgen.datasets.dataset_ar import CAUSAL_DECODE_STEPS, LOOP_LENGTH_RESERVE
from cdrseqgen.methods.ar.arguments import (
    DataArguments as ARDataArguments,
    GenerationArguments as ARGenerationArguments,
    ModelArguments as ARModelArguments,
    validate_args as validate_ar_args,
)
from cdrseqgen.methods.common.arguments import (
    BaseDataArguments,
    BaseGenerationArguments,
    BaseModelArguments,
    parse_config_and_cli,
    parse_config_dict,
    validate_common_args,
)
from cdrseqgen.methods.common.training import train_and_save
from cdrseqgen.methods.mlm.arguments import GenerationArguments as MLMGenerationArguments
from train import resolve_method


# ====================================================================================================
# from test_training_pipeline_contracts.py
# Training-pipeline contracts that the rest of the suite reached only through subprocess runs.
#
# A line trace of the suite showed these shared pieces were never executed in-process: the early-stopping
# branch of build_callbacks, AR/S2S prepare_training, S2S compute_metrics (the checkpoint-selection
# metric), initialize_run with its run naming and provenance, and the draw path where every attempt fails.
# Real tiny CPU models and corpora throughout; nothing is mocked except where a stub is the point.
# ====================================================================================================

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


# ====================================================================================================
# from test_sampling_contracts.py
# Sampler/output contracts exercised with deterministic CPU models, without weights.
# ====================================================================================================

def output(logits, form):
    if form == "dict":
        return {"logits": logits}
    if form == "object":
        return SimpleNamespace(logits=logits)
    return (logits, "unused cached state")


class MaskedModel(torch.nn.Module):
    def __init__(self, form):
        super().__init__()
        self.form = form

    def forward(self, input_ids):
        batch, width = input_ids.shape
        logits = torch.arange(9, dtype=torch.float32).expand(batch, width, 9).clone()
        return output(logits, self.form)


class EosModel(torch.nn.Module):
    """Emit A, C, EOS for decoding; also supply deterministic self-scoring logits."""

    def __init__(self, form, tokenizer, prompt_length):
        super().__init__()
        self.form = form
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.script = tokenizer.encode("AC") + [tokenizer.eos_id]

    def encode(self, input_ids, attention_mask=None):
        return torch.zeros(*input_ids.shape, 4)

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        ids = input_ids if decoder_input_ids is None else decoder_input_ids
        step = ids.shape[1] - (self.prompt_length if decoder_input_ids is None else 1)
        token = self.script[min(max(step, 0), len(self.script) - 1)]
        logits = torch.zeros(*ids.shape, self.tokenizer.vocab_size)
        logits[:, -1, token] = 20.0
        return output(logits, self.form)


class StaggeredModel(EosModel):
    """Rows stop after zero, one and two residues, keeping completed rows in the batch."""

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        ids = input_ids if decoder_input_ids is None else decoder_input_ids
        step = ids.shape[1] - (self.prompt_length if decoder_input_ids is None else 1)
        logits = torch.zeros(*ids.shape, self.tokenizer.vocab_size)
        for row in range(ids.shape[0]):
            token = self.tokenizer.eos_id if step >= row else self.script[0]
            logits[row, -1, token] = 20.0
        return {"logits": logits}


class ExtremeModel(EosModel):
    """Make candidate residues extremely unlikely or produce non-finite loss."""

    def __init__(self, tokenizer, peak):
        super().__init__("dict", tokenizer, 4)
        self.peak = peak

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        ids = input_ids if decoder_input_ids is None else decoder_input_ids
        logits = torch.zeros(*ids.shape, self.tokenizer.vocab_size)
        logits[..., self.tokenizer.encode("Y")[0]] = self.peak
        return {"logits": logits}


class OutputFormatTests(unittest.TestCase):
    def test_both_mlm_decoders_preserve_draws_and_rng_with_tuple_outputs(self):
        tokenizer = SimpleNamespace(special_ids={0, 1}, mask_id=1)
        for decoder in ("independent", "iterative"):
            expected = None
            for form in ("dict", "object", "tuple"):
                with self.subTest(decoder=decoder, form=form):
                    generator = torch.Generator().manual_seed(73)
                    samples = sample_masked_span(
                        MaskedModel(form), tokenizer, "cpu", generator,
                        [2, 1, 1, 3], [1, 2], decoding=decoder, num_samples=4,
                        temperature=0.8, top_k=3, top_p=0.9)
                    actual = (samples, generator.get_state())
                    if expected is None:
                        expected = actual
                    self.assertEqual(samples, expected[0])
                    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_eos_decoders_preserve_completion_and_rng_with_tuple_outputs(self):
        for tokenizer, sampler in ((causal_adapter(), ar_sample), (s2s_adapter(), s2s_sample)):
            for temperature in (0.0, 1.0):
                expected = None
                for form in ("dict", "object", "tuple"):
                    with self.subTest(sampler=sampler.__module__, temperature=temperature, form=form):
                        generator = torch.Generator().manual_seed(73)
                        draws = sampler(
                            EosModel(form, tokenizer, 4), tokenizer, [5, 4, 6, 3],
                            device="cpu", num_samples=3, temperature=temperature,
                            max_new_tokens=5, generator=generator)
                        actual = (draws, generator.get_state())
                        if expected is None:
                            expected = actual
                        self.assertEqual(draws, expected[0])
                        self.assertEqual(len(draws), 1 if temperature == 0 else 3)
                        self.assertTrue(all(d["complete"] and d["loop"] == "AC" for d in draws))
                        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_self_scores_are_identical_for_all_supported_output_formats(self):
        for tokenizer, scorer in ((causal_adapter(), ar_perplexities), (s2s_adapter(), s2s_perplexities)):
            expected = None
            for form in ("dict", "object", "tuple"):
                with self.subTest(scorer=scorer.__module__, form=form):
                    spans = [([5, 4, 6, 3], tokenizer.encode(loop)) for loop in ("AC", "DE")]
                    scores = scorer(EosModel(form, tokenizer, 4), tokenizer, "cpu", spans)
                    if expected is None:
                        expected = scores
                    self.assertEqual(scores, expected)
                    self.assertTrue(all(score is not None for score in scores))
        expected = None
        for form in ("dict", "object", "tuple"):
            with self.subTest(scorer="MLM", form=form):
                scores = pseudo_log_likelihood(MaskedModel(form), [2, 1, 1, 3], [1, 2],
                                               [5, 6], {0, 1}, 1, "cpu")
                if expected is None:
                    expected = scores
                self.assertEqual(scores, expected)


class BoundaryTests(unittest.TestCase):
    def test_empty_mlm_spans_skip_the_model_and_preserve_rng(self):
        def forbidden_model(**kwargs):
            self.fail("an empty span must not invoke the sequence model")

        tokenizer = SimpleNamespace(special_ids={0, 1}, mask_id=1)
        for decoder in ("independent", "iterative"):
            for temperature in (0.0, 1.0):
                with self.subTest(decoder=decoder, temperature=temperature):
                    generator = torch.Generator().manual_seed(73)
                    before = generator.get_state().clone()
                    samples = sample_masked_span(
                        forbidden_model, tokenizer, "cpu", generator, [2, 3], [],
                        decoding=decoder, num_samples=4, temperature=temperature,
                        top_k=0, top_p=1.0)
                    self.assertEqual(samples, [[]] * (1 if temperature == 0 else 4))
                    torch.testing.assert_close(generator.get_state(), before, rtol=0, atol=0)

    def test_pll_rejects_missing_or_extra_candidate_tokens_before_scoring(self):
        def forbidden_model(**kwargs):
            self.fail("a malformed candidate must fail before model execution")

        for positions, tokens in (([1, 2], [5]), ([1], [5, 6]), ([], [5])):
            with self.subTest(positions=positions, tokens=tokens), self.assertRaisesRegex(
                    ValueError, "token_ids must contain one token per mask position"):
                pseudo_log_likelihood(forbidden_model, [2, 1, 1, 3], positions,
                                      tokens, {0, 1}, 1, "cpu")

    def test_eos_rows_stop_independently_without_changing_rng_consumption(self):
        for tokenizer, sampler in ((causal_adapter(), ar_sample), (s2s_adapter(), s2s_sample)):
            with self.subTest(sampler=sampler.__module__):
                generator = torch.Generator().manual_seed(73)
                draws = sampler(
                    StaggeredModel("dict", tokenizer, 4), tokenizer, [5, 4, 6, 3],
                    device="cpu", num_samples=3, temperature=1.0, top_k=1,
                    max_new_tokens=5, generator=generator)
                self.assertEqual([d["loop"] for d in draws], ["", "A", "AA"])
                self.assertTrue(all(d["complete"] for d in draws))
                expected_rng = torch.Generator().manual_seed(73)
                for step in range(3):
                    probabilities = torch.zeros(3, tokenizer.vocab_size)
                    for row in range(3):
                        token = tokenizer.eos_id if step >= row else tokenizer.encode("A")[0]
                        probabilities[row, token] = 1.0
                    torch.multinomial(probabilities, 1, generator=expected_rng)
                torch.testing.assert_close(generator.get_state(), expected_rng.get_state(),
                                           rtol=0, atol=0)

    def test_eos_self_scorers_handle_overflow_and_nonfinite_losses(self):
        for tokenizer, scorer in ((causal_adapter(), ar_perplexities), (s2s_adapter(), s2s_perplexities)):
            for peak in (1000.0, float("nan"), float("inf")):
                with self.subTest(scorer=scorer.__module__, peak=peak):
                    scores = scorer(ExtremeModel(tokenizer, peak), tokenizer, "cpu",
                                    [([5, 4, 6, 3], tokenizer.encode("AC"))])
                    self.assertEqual(scores, [None])

    def test_nucleus_keeps_the_crossing_token_and_top_k_keeps_cutoff_ties(self):
        nucleus = warp_logits(torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05])), top_p=0.6)
        self.assertEqual(torch.isfinite(nucleus).nonzero().flatten().tolist(), [0, 1])
        tied = warp_logits(torch.tensor([4.0, 3.0, 3.0, 1.0]), top_k=2)
        self.assertEqual(torch.isfinite(tied).nonzero().flatten().tolist(), [0, 1, 2])


# ====================================================================================================
# from test_method_integration.py
# Cross-method contracts introduced by the dispatching training stack.
# ====================================================================================================

class DispatcherTests(unittest.TestCase):
    def test_cli_method_override_remains_for_the_method_parser(self):
        argv = ["train.py", "configs/mlm/heavy/h3_scratch.json", "--method", "ar"]
        method, routed = resolve_method(argv)
        self.assertEqual(method, "ar")
        self.assertEqual(routed, argv)

    def test_equals_form_and_last_override_win(self):
        method, _ = resolve_method(
            ["train.py", "--method", "mlm", "--method=AR"])
        self.assertEqual(method, "ar")

    def test_config_method_is_normalized(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump({"method": " AR "}, handle)
            handle.flush()
            method, _ = resolve_method(["train.py", handle.name])
        self.assertEqual(method, "ar")

    def test_override_is_also_recorded_by_the_method_parser(self):
        parser = HfArgumentParser((
            ARModelArguments, ARDataArguments, ARGenerationArguments, TrainingArguments))
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump({"method": "mlm", "output_dir": "/tmp/unused"}, handle)
            handle.flush()
            argv = ["train.py", handle.name, "--method", "ar"]
            with patch("sys.argv", argv):
                model, _, _, _ = parse_config_and_cli(parser)

        # The old dispatcher removed --method before parsing: it routed to AR but persisted
        # method=mlm in run_config.json. The override must reach both decisions.
        self.assertEqual(model.method, "ar")


class MethodRegistryTests(unittest.TestCase):
    """A method is only usable if every component the dispatchers reach for actually exists."""

    COMMAND_PROTOCOLS = {
        "train_cli": ("main",),
        "eval_cli": ("main", "prepare_evaluation", "evaluation_options", "evaluate"),
        "gen_cli": ("main", "prepare_generation", "generation_state", "generate_case"),
    }

    def test_every_registered_method_resolves_its_components(self):
        from cdrseqgen.methods.registry import METHODS, resolve
        for name, method in METHODS.items():
            with self.subTest(method=name):
                for component in ("arguments", "backbones", "training", "evaluation",
                                  *self.COMMAND_PROTOCOLS):
                    self.assertTrue(hasattr(method.module(component), "__name__"))
                self.assertTrue(callable(method.module("eval_cli").evaluate))
                self.assertTrue(callable(method.module("train_cli").main))
                self.assertTrue(callable(method.module("training").get_datasets))
                self.assertIs(resolve(name), method)

    def test_every_method_exposes_the_identical_command_protocols(self):
        """All three root commands must find the functions they call in every method."""
        from cdrseqgen.methods.registry import METHODS
        for name, method in METHODS.items():
            for component, functions in self.COMMAND_PROTOCOLS.items():
                command = method.module(component)
                for function in functions:
                    with self.subTest(method=name, command=component, function=function):
                        self.assertTrue(callable(getattr(command, function, None)))

    def test_post_training_evaluation_forwards_to_the_command_interface(self):
        """Preserve arguments and return values through the shared evaluation route."""
        from cdrseqgen.methods.registry import METHODS
        args = (object(), object(), object())
        options = {"length_mode": "inferred", "anchor_trim": 3, "seed": 43}
        for name, method in METHODS.items():
            with self.subTest(method=name):
                command = method.module("eval_cli")
                with patch.object(command, "evaluate", return_value={"records": []}) as evaluate:
                    self.assertEqual(method.evaluate(*args, **options), {"records": []})
                evaluate.assert_called_once_with(*args, **options)

    def test_a_checkpoint_cannot_be_reinterpreted_by_another_method(self):
        from cdrseqgen.methods.registry import checkpoint_method
        self.assertEqual(checkpoint_method({"method": "s2s"}).name, "s2s")
        with self.assertRaisesRegex(ValueError, "different method"):
            checkpoint_method({"method": "s2s"}, "ar")

    def test_decoding_choices_cover_every_method(self):
        """generate.py and evaluate.py hard-coded this list, so a method added later was rejected
        by argparse before its own cli could validate the request. Deriving it from the registry
        means an enumeration that cannot fall behind."""
        from cdrseqgen.methods.registry import METHODS, decoding_choices
        choices = decoding_choices()
        self.assertEqual(len(choices), len(set(choices)), "duplicate decoder name across methods")
        for name, method in METHODS.items():
            with self.subTest(method=name):
                self.assertTrue(method.decoders, f"{name} declares no decoder for --decoding")
                for decoder in method.decoders:
                    self.assertIn(decoder, choices)

    def test_the_dispatcher_routes_the_seq2seq_method(self):
        method, _ = resolve_method(["train.py", "--method", "s2s"])
        self.assertEqual(method, "s2s")


class CommonArgumentTests(unittest.TestCase):
    def _args(self):
        return (
            BaseModelArguments(),
            BaseDataArguments(),
            BaseGenerationArguments(),
        )

    def test_both_methods_expose_the_same_generation_seed(self):
        self.assertEqual(ARGenerationArguments().gen_seed, 42)
        self.assertEqual(MLMGenerationArguments().gen_seed, 42)

    def test_shared_generation_ranges_are_rejected_before_run_creation(self):
        invalid = [
            ("gen_num_examples", -2),
            ("gen_num_samples", 0),
            ("gen_temperature", -0.1),
            ("gen_top_k", -1),
            ("gen_top_p", 0.0),
            ("gen_top_p", 1.1),
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                model, data, generation = self._args()
                setattr(generation, field, value)
                with self.assertRaises(ValueError):
                    validate_common_args(
                        model, data, generation, expected_method="mlm")

    def test_entrypoint_method_mismatch_is_rejected(self):
        model, data, generation = self._args()
        with self.assertRaisesRegex(ValueError, "entrypoint received"):
            validate_common_args(model, data, generation, expected_method="ar")

    def test_json_null_does_not_break_an_unrelated_cli_override(self):
        from dataclasses import dataclass

        @dataclass
        class Args:
            optional_count: int = None
            count: int = 1

        parser = HfArgumentParser(Args)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump({"optional_count": None, "count": 2}, handle)
            handle.flush()
            argv = ["prog", handle.name, "--count", "3"]
            with patch("sys.argv", argv):
                (parsed,) = parse_config_and_cli(parser)
        self.assertIsNone(parsed.optional_count)
        self.assertEqual(parsed.count, 3)

    def test_legacy_warmup_ratio_preserves_schedule_and_explicit_steps_take_precedence(self):
        parser = HfArgumentParser(TrainingArguments)
        base = dict(output_dir='/tmp/unused', use_cpu=True, report_to='none', warmup_ratio=.06)
        (training,) = parse_config_dict(parser, {**base, 'warmup_steps': 0})
        self.assertEqual(training.get_warmup_steps(1000), 60)
        (training,) = parse_config_dict(parser, {**base, 'warmup_steps': 12})
        self.assertEqual(training.get_warmup_steps(1000), 12)
        (training,) = parse_config_dict(parser, {**base, 'warmup_ratio': None})
        self.assertEqual(training.get_warmup_steps(1000), 0)

    def test_warmup_conversion_survives_an_unrelated_cli_override(self):
        parser = HfArgumentParser(TrainingArguments)
        with tempfile.NamedTemporaryFile('w', suffix='.json') as handle:
            json.dump(dict(output_dir='/tmp/unused', use_cpu=True, report_to='none',
                           warmup_ratio=.06, warmup_steps=0), handle)
            handle.flush()
            for tail in ([], ['--learning_rate', '0.0001']):
                with self.subTest(tail=tail), patch('sys.argv', ['train.py', handle.name, *tail]):
                    (training,) = parse_config_and_cli(parser)
                self.assertEqual(training.get_warmup_steps(1000), 60)

    def test_warmup_ratio_remains_native_for_a_parser_supporting_both_fields(self):
        from dataclasses import dataclass
        @dataclass
        class LegacyArgs:
            warmup_steps: int = 0
            warmup_ratio: float = 0
        (parsed,) = parse_config_dict(HfArgumentParser(LegacyArgs), dict(warmup_ratio=.06))
        self.assertEqual(parsed.warmup_ratio, .06)
        self.assertEqual(parsed.warmup_steps, 0)


class SharedTrainingLifecycleTests(unittest.TestCase):
    @patch("cdrseqgen.methods.common.training.plot_training_curves")
    @patch("cdrseqgen.methods.common.training.save_training_csv")
    def test_train_and_eval_metrics_and_tokenizer_are_persisted(self, save_csv, plot_curves):
        trainer = Mock()
        trainer.train.return_value = SimpleNamespace(metrics={"train_loss": 1.0})
        trainer.evaluate.return_value = {"eval_loss": 0.5}
        tokenizer = Mock()
        args = SimpleNamespace(do_train=True, do_eval=True, output_dir="/tmp/example")

        train_and_save(trainer, args, tokenizer=tokenizer)

        trainer.save_model.assert_called_once_with(args.output_dir)
        trainer.save_metrics.assert_any_call("train", {"train_loss": 1.0})
        trainer.save_metrics.assert_any_call("eval", {"eval_loss": 0.5})
        tokenizer.save_pretrained.assert_called_once_with(args.output_dir)
        save_csv.assert_called_once_with(trainer, args.output_dir)
        plot_curves.assert_called_once_with(args.output_dir)


class Seq2SeqConfigTests(unittest.TestCase):
    def test_decode_cap_covers_the_decoder_budget(self):
        from cdrseqgen.datasets.dataset_s2s import MAX_LOOP_RESIDUES, S2S_DECODE_STEPS
        from cdrseqgen.methods.s2s.arguments import (DataArguments, GenerationArguments,
                                                     ModelArguments, validate_args)
        generation = GenerationArguments()
        self.assertEqual(generation.gen_max_new_tokens, MAX_LOOP_RESIDUES + 1)
        self.assertEqual(generation.gen_max_new_tokens, S2S_DECODE_STEPS)
        generation.gen_max_new_tokens += 1
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            validate_args(ModelArguments(), DataArguments(), generation)

    def test_stack_depths_must_be_positive_integers_or_unset(self):
        from cdrseqgen.methods.s2s.arguments import (DataArguments, GenerationArguments,
                                                     ModelArguments, validate_args)
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value):
                model = ModelArguments(num_encoder_layers=value)
                with self.assertRaisesRegex(ValueError, "num_encoder_layers"):
                    validate_args(model, DataArguments(), GenerationArguments())
        validate_args(ModelArguments(num_encoder_layers=2, num_decoder_layers=2),
                      DataArguments(), GenerationArguments())

    def test_the_shipped_configs_pin_both_stack_depths(self):
        """The comparison configs must not inherit num_layers on both stacks: that is 1.86M
        parameters against the causal arm's 800k, and the batch would confound architecture with
        2.3x capacity while every label still read 'scratch'."""
        import json
        from pathlib import Path
        for name in ("h3_s2s_scratch.json", "h3_s2s_scratch_lc.json"):
            with self.subTest(config=name):
                config = json.loads((Path("configs/s2s") / name).read_text())
                self.assertEqual(config["method"], "s2s")
                self.assertIn("num_encoder_layers", config)
                self.assertIn("num_decoder_layers", config)


class CausalConfigTests(unittest.TestCase):
    def test_decode_cap_covers_the_reserve_but_cannot_overrun_the_prompt_budget(self):
        generation = ARGenerationArguments()
        self.assertEqual(generation.gen_max_new_tokens, LOOP_LENGTH_RESERVE + 1)
        self.assertEqual(generation.gen_max_new_tokens, CAUSAL_DECODE_STEPS)
        generation.gen_max_new_tokens += 1
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            validate_ar_args(ARModelArguments(), ARDataArguments(), generation)

    def test_a_second_oas_corpus_gets_its_own_dataset_tag(self):
        """Two donor sets have two test splits, so they must not share a run-name tag."""
        from cdrseqgen.methods.common.run_setup import dataset_tag
        self.assertEqual(dataset_tag("data/processed/processed_oas_tuning.json"), "OAS")
        self.assertEqual(dataset_tag("data/processed/processed_oas_multi.json"), "OAS4")
        self.assertEqual(dataset_tag("data/processed/processed_sabdab2_data.json"), "SD2")


if __name__ == "__main__":
    unittest.main()
