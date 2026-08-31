"""Tests for the sweep ranking and the auto-reporting steps the shell scripts call.

Two guarantees matter here beyond arithmetic:

  * the ranking direction must follow the metric. Perplexity is lower-is-better and recovery is
    higher-is-better, and a sweep that silently ranks perplexity backwards would nominate the worst
    setting as the winner while looking entirely normal.
  * reporting must never fail the thing it reports on. ``scripts/run.sh`` and ``scripts/sweep.sh``
    call these after the GPU work is finished, so a batch that cannot be parsed has to degrade to a
    warning rather than an exception.
"""
import csv
import json
import os
import tempfile
import unittest

from cdrseqgen.visualization.batch_tables import write_batch_tables
from cdrseqgen.visualization.sweep_tables import (
    RANKERS,
    load_trials,
    rank_settings,
    summarize_sweep,
)

COLUMNS = ["lr", "warmup", "weight_decay", "epochs", "hidden", "layers",
           "loss", "core_w", "anchor_w", "seed", "scheduler", "status", "run_dir"]


def _build_sweep(root, trials):
    """Write a runs.tsv plus one run directory per trial. `trials` is (loss, seed, core, judge)."""
    rows = []
    for loss, seed, core, judge in trials:
        run_dir = os.path.join(root, f"run_{loss}_{seed}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "epoch_metrics.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "eval_loss", "eval_accuracy", "eval_core_accuracy"])
            writer.writerow([1, 1.0, core + 0.1, core - 0.01])
            writer.writerow([2, 0.9, core + 0.1, core])
        if judge is not None:
            with open(os.path.join(run_dir, "generation_eval.json"), "w") as handle:
                json.dump({"summary": {"judge_perplexity_mean": judge}}, handle)
        rows.append({
            "lr": "1e-3", "warmup": "0.06", "weight_decay": "0.01", "epochs": "30",
            "hidden": "__keep__", "layers": "__keep__", "loss": loss,
            "core_w": "1.0", "anchor_w": "1.0", "seed": str(seed), "scheduler": "linear",
            "status": "completed", "run_dir": run_dir,
        })
    map_path = os.path.join(root, "runs.tsv")
    with open(map_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return map_path


class RankingDirectionTests(unittest.TestCase):
    TRIALS = [("cross_entropy", 42, 0.32, 7.0), ("cross_entropy", 43, 0.33, 7.2),
              ("weighted_cross_entropy", 42, 0.30, 4.0), ("weighted_cross_entropy", 43, 0.31, 4.2)]

    def test_perplexity_ranks_lowest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path = _build_sweep(tmp, self.TRIALS)
            trials = load_trials(map_path, "judge_ppl")
            ranked = rank_settings(trials, RANKERS["judge_ppl"][2])
            self.assertEqual(ranked[0]["loss"], "weighted_cross_entropy")
            self.assertAlmostEqual(ranked[0]["mean_score"], 4.1)

    def test_recovery_ranks_highest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path = _build_sweep(tmp, self.TRIALS)
            trials = load_trials(map_path, "core")
            ranked = rank_settings(trials, RANKERS["core"][2])
            self.assertEqual(ranked[0]["loss"], "cross_entropy")
            self.assertAlmostEqual(ranked[0]["mean_score"], 0.325)

    def test_seeds_collapse_into_one_row_carrying_their_spread(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path = _build_sweep(tmp, self.TRIALS)
            ranked = rank_settings(load_trials(map_path, "core"), True)
            self.assertEqual(len(ranked), 2)
            self.assertTrue(all(entry["n_seeds"] == 2 for entry in ranked))
            self.assertGreater(ranked[0]["sd_score"], 0)


class TrialSelectionTests(unittest.TestCase):
    def test_only_completed_trials_are_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path = _build_sweep(tmp, [("cross_entropy", 42, 0.32, 7.0)])
            with open(map_path) as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            rows[0]["status"] = "failed"
            with open(map_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(load_trials(map_path, "core"), [])

    def test_a_trial_without_the_generative_metric_is_unranked_not_zero(self):
        # Scoring a missing judge value as 0.0 would make it the winner on a lower-is-better metric.
        with tempfile.TemporaryDirectory() as tmp:
            map_path = _build_sweep(tmp, [("cross_entropy", 42, 0.32, None)])
            trials = load_trials(map_path, "judge_ppl")
            self.assertEqual(len(trials), 1)
            self.assertIsNone(trials[0]["score"])


class ReportingNeverFailsTheRunTests(unittest.TestCase):
    """scripts/run.sh and scripts/sweep.sh call these after the GPU work has already succeeded."""

    def test_summarize_returns_none_when_there_is_no_runs_tsv(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(summarize_sweep(tmp))

    def test_summarize_returns_none_when_no_trial_carries_the_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            _build_sweep(tmp, [("cross_entropy", 42, 0.32, None)])
            self.assertIsNone(summarize_sweep(tmp, "judge_ppl"))

    def test_summarize_writes_a_summary_next_to_runs_tsv(self):
        with tempfile.TemporaryDirectory() as tmp:
            _build_sweep(tmp, [("cross_entropy", 42, 0.32, 7.0)])
            self.assertIsNotNone(summarize_sweep(tmp, "core", quiet=True))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "summary.tsv")))

    def test_batch_tables_returns_none_on_an_unparseable_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "not_a_run_name"))
            self.assertIsNone(write_batch_tables(tmp))


if __name__ == "__main__":
    unittest.main()
