"""Shared logging and run-recording contracts using temporary files only."""
import csv
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cdrseqgen.methods.common import logging as run_logging
from cdrseqgen.methods.common import run_setup
from cdrseqgen.methods.common.callbacks import LogMetricsCallback, build_callbacks
from cdrseqgen.visualization.training_results import save_training_csv


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
