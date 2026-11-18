"""Tests merged from: test_artifacts.py, test_run_config_loading.py, test_run_recording.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from cdrseqgen.artifacts import atomic_write_json, file_identity
from cdrseqgen.methods.common.run_setup import write_run_config
from types import SimpleNamespace
from cdrseqgen.artifacts import RUN_CONFIG_FILENAME, find_run_config, read_run_config
from generate import resolve_config
import csv
import logging
import os
import sys
from cdrseqgen.methods.common import logging as run_logging
from cdrseqgen.methods.common import run_setup
from cdrseqgen.methods.common.callbacks import LogMetricsCallback, build_callbacks
from cdrseqgen.visualization.training_results import save_training_csv


# ====================================================================================================
# from test_artifacts.py
# ====================================================================================================

class AtomicArtifactTests(unittest.TestCase):
    def test_atomic_writer_creates_parents_and_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "artifact.json"
            atomic_write_json(output, {"value": 3})
            self.assertEqual(json.loads(output.read_text()), {"value": 3})
            self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_serialization_failure_preserves_the_previous_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.json"
            output.write_text('{"old": true}\n')
            with self.assertRaises(TypeError):
                atomic_write_json(output, {"not_json": {1, 2, 3}})
            self.assertEqual(json.loads(output.read_text()), {"old": True})
            self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_file_identity_records_size_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"abc")
            identity = file_identity(path)
        self.assertEqual(identity["size_bytes"], 3)
        self.assertEqual(identity["sha256"], hashlib.sha256(b"abc").hexdigest())


@dataclass
class ModelArgs:
    model_type: str = "scratch"


@dataclass
class DataArgs:
    data_path: str
    target_region: str = "h3"


@dataclass
class GenerationArgs:
    gen_decoding: str = "independent"


class TrainingArgs:
    def to_dict(self):
        return {"seed": 42}


class RunConfigProvenanceTests(unittest.TestCase):
    @patch("cdrseqgen.methods.common.run_setup._package_versions", return_value={"torch": "test"})
    @patch("cdrseqgen.methods.common.run_setup.git_state", return_value={
        "repository_commit": "abc", "repository_dirty": False,
    })
    def test_run_config_carries_dataset_runtime_and_repository_identity(self, _git, _versions):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "processed.json"
            data.write_text("[]")
            metadata = root / "processed.metadata.json"
            metadata.write_text(json.dumps({"processed_file_sha256": "declared"}))

            payload = write_run_config(
                str(root),
                ModelArgs(),
                DataArgs(str(data)),
                GenerationArgs(),
                TrainingArgs(),
            )
            written = json.loads((root / "run_config.json").read_text())

        self.assertEqual(written, payload)
        self.assertEqual(payload["provenance"]["dataset"]["declared_processed_file_sha256"], "declared")
        self.assertEqual(payload["provenance"]["runtime"]["packages"]["torch"], "test")
        self.assertEqual(payload["provenance"]["repository_commit"], "abc")
        _git.assert_called_once_with(Path(__file__).resolve().parents[1])


# ====================================================================================================
# from test_run_config_loading.py
# ====================================================================================================

class RunConfigLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_current_filename_wins_over_the_legacy_one(self):
        (self.dir / "run_meta.json").write_text(json.dumps({"target_region": "l2"}))
        (self.dir / RUN_CONFIG_FILENAME).write_text(json.dumps({"target_region": "h3"}))

        config, path = read_run_config(self.dir), find_run_config(self.dir)

        self.assertEqual(config["target_region"], "h3")
        self.assertEqual(path.name, RUN_CONFIG_FILENAME)

    def test_legacy_filename_still_loads(self):
        (self.dir / "run_meta.json").write_text(json.dumps({"target_region": "h3"}))
        self.assertEqual(read_run_config(self.dir)["target_region"], "h3")

    def test_absent_config_is_distinguishable_from_a_corrupt_one(self):
        self.assertIsNone(find_run_config(self.dir))
        self.assertIsNone(read_run_config(self.dir))

        (self.dir / RUN_CONFIG_FILENAME).write_text("{not json")
        self.assertIsNotNone(find_run_config(self.dir))
        with self.assertRaises(json.JSONDecodeError):
            read_run_config(self.dir)

    def test_an_empty_config_is_a_config_not_an_absence(self):
        (self.dir / RUN_CONFIG_FILENAME).write_text("{}")
        self.assertEqual(read_run_config(self.dir), {})


class GenerateConfigSourceTests(unittest.TestCase):
    """resolve_config marks CLI-built configs source='cli', and check_compatibility skips both
    context checks for those — so a corrupt run config must not degrade into that path."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.args = SimpleNamespace(
            model_dir=str(self.dir), model_type=None, target_loop=None,
            partner_chain=None, antigen=None, max_len=128,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_a_saved_config_is_trusted_provenance(self):
        (self.dir / RUN_CONFIG_FILENAME).write_text(json.dumps({
            "target_region": "h3", "model_type": "scratch", "include_partner_chain": True,
        }))
        configs = resolve_config(self.args)
        self.assertEqual(configs["source"], "run_config")
        self.assertTrue(configs["include_partner_chain"])

    def test_missing_config_falls_back_to_cli(self):
        configs = resolve_config(self.args)
        self.assertEqual(configs["source"], "cli")

    def test_preloaded_snapshot_is_reused_and_override_warning_is_retained(self):
        saved = {"model_type": "scratch", "include_partner_chain": True}
        self.args.model_type = "other-backbone"
        with patch("cdrseqgen.methods.common.generation.read_run_config") as reload, \
                self.assertLogs("cdrseqgen.methods.common.generation", level="WARNING") as logs:
            configs = resolve_config(self.args, run_config=saved)

        reload.assert_not_called()
        self.assertEqual(saved["model_type"], "scratch")
        self.assertEqual(configs["model_type"], "other-backbone")
        self.assertEqual(configs["source"], "run_config")
        self.assertTrue(configs["include_partner_chain"])
        self.assertIn("Model type mismatch", logs.output[0])

    def test_corrupt_config_raises_rather_than_silently_disabling_context_checks(self):
        (self.dir / RUN_CONFIG_FILENAME).write_text("{truncated")
        with self.assertRaises(json.JSONDecodeError):
            resolve_config(self.args)


# ====================================================================================================
# from test_run_recording.py
# Shared logging and run-recording contracts using temporary files only.
# ====================================================================================================

class RunLoggingTests(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.handlers = list(self.root.handlers)
        self.level = self.root.level
        self.exception_hook = sys.excepthook
        self.root.setLevel(logging.INFO)
        self.addCleanup(self.restore_logging)

    def restore_logging(self):
        for handler in list(self.root.handlers):
            if handler not in self.handlers:
                self.root.removeHandler(handler)
                handler.close()
        self.root.setLevel(self.level)
        sys.excepthook = self.exception_hook

    def test_metric_callback_reaches_one_file_handler_without_duplicate_records(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(run_logging.attach_file_logging(directory))
            self.assertEqual(run_logging.attach_file_logging(directory), str(log_path))
            handlers = [h for h in self.root.handlers if isinstance(h, logging.FileHandler)
                        and h.baseFilename == str(log_path.resolve())]
            self.assertEqual(len(handlers), 1)

            callbacks = build_callbacks(
                SimpleNamespace(early_stopping_patience=0),
                SimpleNamespace(load_best_model_at_end=False),
            )
            self.assertEqual(len(callbacks), 1)
            self.assertIsInstance(callbacks[0], LogMetricsCallback)
            callbacks[0].on_log(None, None, None, logs={
                "epoch": 2, "loss": 0.375, "eval_core_accuracy": 0.25,
            })
            text = log_path.read_text()
            self.assertEqual(text.count("loss=0.375"), 1)
            self.assertIn("eval_core_accuracy=0.25", text)
            self.assertIn("epoch   2.00", text)

    def test_uncaught_exception_is_logged_and_forwarded_to_original_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(run_logging.attach_file_logging(directory))
            error = ValueError("fixture failure")
            with patch.object(sys, "__excepthook__") as original:
                sys.excepthook(ValueError, error, None)
                original.assert_called_once_with(ValueError, error, None)
            text = log_path.read_text()
            self.assertIn("Run failed with an uncaught exception", text)
            self.assertIn("ValueError: fixture failure", text)


class RunOutputTests(unittest.TestCase):
    def test_run_name_collisions_select_a_new_path_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trial").mkdir()
            (root / "trial_2").mkdir()
            with self.assertLogs(run_setup.__name__, level="WARNING"):
                chosen = run_setup.unique_run_dir(str(root), "trial")
            self.assertEqual(Path(chosen), root / "trial_3")
            self.assertFalse(Path(chosen).exists())

    def test_latest_tracks_batch_and_preserves_a_real_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            batch = runs / "batch"
            run = batch / "trial"
            run.mkdir(parents=True)
            with patch.object(run_setup, "DEFAULT_RESULTS_ROOT", str(runs)), \
                    patch.dict(os.environ, {"CDRSEQGEN_SKIP_LATEST": "0"}):
                run_setup.update_latest_link(str(run))
                latest = runs / "latest"
                self.assertTrue(latest.is_symlink())
                self.assertEqual(latest.resolve(), batch)
                latest.unlink()
                latest.mkdir()
                marker = latest / "preserve.txt"
                marker.write_text("existing results")
                with self.assertLogs(run_setup.__name__, level="WARNING"):
                    run_setup.update_latest_link(str(run))
                self.assertFalse(latest.is_symlink())
                self.assertEqual(marker.read_text(), "existing results")

    def test_epoch_csv_preserves_trajectory_before_best_checkpoint_reevaluation(self):
        trainer = SimpleNamespace(state=SimpleNamespace(log_history=[
            {"epoch": 1., "loss": 9.},
            {"epoch": 1., "eval_loss": 2., "eval_accuracy": 0.3},
            {"epoch": 2., "eval_loss": 3., "eval_accuracy": 0.2},
            {"epoch": 2., "eval_loss": 2., "eval_accuracy": 0.3},
        ]))
        with tempfile.TemporaryDirectory() as directory:
            save_training_csv(trainer, directory)
            with (Path(directory) / "epoch_metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[1]["eval_loss"]), 3.)
        self.assertEqual(float(rows[1]["eval_accuracy"]), 0.2)
        self.assertIn("eval_core_coverage", rows[1])


if __name__ == "__main__":
    unittest.main()
