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
    def test_retry_is_not_a_second_seed_and_full_setting_identity_separates_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path = _build_sweep(tmp, [("cross_entropy", 42, 0.32, 7.0)])
            trial = load_trials(map_path, "core")[0]
            retry = dict(trial, score=0.34)
            ranked = rank_settings([trial, retry], True)
            self.assertEqual(ranked[0]["n_seeds"], 1)
            self.assertEqual(ranked[0]["mean_score"], 0.34)
            ranked = rank_settings([dict(trial, setting_id="ar"),
                                    dict(trial, setting_id="s2s")], True)
            self.assertEqual(len(ranked), 2)

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


class CheckpointAndValidityTests(unittest.TestCase):
    def test_epoch_scores_follow_the_checkpoint_selection_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path = _build_sweep(tmp, [("cross_entropy", 42, 0.32, None)])
            run_dir = os.path.join(tmp, "run_cross_entropy_42")
            with open(os.path.join(run_dir, "epoch_metrics.csv"), "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["epoch", "eval_loss", "eval_accuracy", "eval_core_accuracy"])
                writer.writerow([1, 0.9, 0.50, 0.90])
                writer.writerow([2, 0.8, 0.80, 0.40])
            with open(os.path.join(run_dir, "run_config.json"), "w") as handle:
                json.dump({
                    "metric_for_best_model": "accuracy",
                    "greater_is_better": True,
                }, handle)

            core_trial = load_trials(map_path, "core")[0]
            whole_trial = load_trials(map_path, "whole")[0]
            self.assertEqual(core_trial["selection_metric"], "eval_accuracy")
            self.assertEqual(core_trial["best_epoch"], 2.0)
            self.assertEqual(core_trial["score"], 0.40)
            self.assertEqual(whole_trial["score"], 0.80)
            self.assertEqual(
                rank_settings([core_trial], higher_is_better=True)[0]["selection_metric"],
                "eval_accuracy",
            )

    def test_length_rankers_include_invalid_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path = _build_sweep(tmp, [("cross_entropy", 42, 0.32, None)])
            run_dir = os.path.join(tmp, "run_cross_entropy_42")
            payload = {
                # Conditional metrics look perfect because only one valid draw survived.
                "summary": {"length_match_rate": 1.0, "length_absolute_error_mean": 0.0},
                "generation_validity": {
                    "attempts": 20,
                    "complete_nonempty": 1,
                    "incomplete": 19,
                    "valid_generation_rate": 0.05,
                    "fixed_length_acceptance_rate": 0.05,
                    "length_absolute_error_mean_capped": 9.5,
                },
            }
            with open(os.path.join(run_dir, "generation_eval.json"), "w") as handle:
                json.dump(payload, handle)

            self.assertEqual(load_trials(map_path, "length_match")[0]["score"], 0.05)
            self.assertEqual(load_trials(map_path, "length_mae")[0]["score"], 9.5)
            self.assertEqual(load_trials(map_path, "valid_rate")[0]["score"], 0.05)

    def test_legacy_conditional_mae_is_unranked_when_decodes_were_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path = _build_sweep(tmp, [("cross_entropy", 42, 0.32, None)])
            run_dir = os.path.join(tmp, "run_cross_entropy_42")
            with open(os.path.join(run_dir, "generation_eval.json"), "w") as handle:
                json.dump({
                    "summary": {"length_absolute_error_mean": 0.0},
                    "generation_validity": {
                        "attempts": 20, "valid_generation_rate": 0.05,
                    },
                }, handle)
            self.assertIsNone(load_trials(map_path, "length_mae")[0]["score"])


class ReportingNeverFailsTheRunTests(unittest.TestCase):
    """scripts/run.sh and scripts/sweep.sh call these after the GPU work has already succeeded."""

    def test_summarize_returns_none_when_there_is_no_runs_tsv(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(summarize_sweep(tmp))

    def test_saved_summary_names_ranking_separately_from_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            _build_sweep(tmp, [("cross_entropy", 42, 0.32, 7.0)])
            summarize_sweep(tmp, "judge_ppl", quiet=True)
            with open(os.path.join(tmp, "summary.tsv")) as handle:
                row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(row["rank_metric"], "judge_ppl")
            self.assertEqual(row["selection_metric"], "eval_core_accuracy")
            self.assertEqual(row["higher_is_better"], "False")

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
