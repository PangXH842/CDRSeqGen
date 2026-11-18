"""Tests merged from: test_confidence_plot.py, test_plotting.py, test_sweep_tables.py, test_batch_tables.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import unittest
from cdrseqgen.visualization.confidence_plot import (
    baseline_recovery,
    rho_distribution,
    selection_curve,
)
import os
import csv
import json
from pathlib import Path
import tempfile
import matplotlib
from cdrseqgen.visualization.batch_tables import write_batch_tables
from cdrseqgen.visualization.sweep_tables import (
    RANKERS,
    _selected_epoch,
    load_trials,
    rank_settings,
    summarize_sweep,
)
from cdrseqgen.visualization.batch_tables import (
    RUN_RE,
    _fmt,
    _common_recovery_policy,
    build_table,
    dataset_alias,
    markdown_table,
    recovery_summary,
    unseen_fraction,
)


# ====================================================================================================
# from test_confidence_plot.py
# ====================================================================================================

def record(scores, recoveries, key="self_perplexity"):
    return {"candidates": [{"recovery": r, key: s} for s, r in zip(scores, recoveries)]}


class SelectionCurveTests(unittest.TestCase):
    def test_low_perplexity_is_selected_first(self):
        """Perplexity is lower-is-better; sorting the wrong way would invert every conclusion."""
        r = record([1.0, 5.0, 9.0], [0.9, 0.5, 0.1])
        curve = selection_curve([r], "self_perplexity", 3)
        self.assertAlmostEqual(curve[0], 0.9)                 # best candidate first
        self.assertAlmostEqual(curve[2], (0.9 + 0.5 + 0.1) / 3)  # k = N -> the mean

    def test_curve_converges_to_the_random_baseline_at_k_equals_n(self):
        r = record([3.0, 1.0, 2.0], [0.2, 0.8, 0.4])
        self.assertAlmostEqual(selection_curve([r], "self_perplexity", 3)[-1],
                               baseline_recovery([r]))

    def test_frameworks_shorter_than_k_contribute_all_they_have(self):
        """Dropping them instead would change the population as k grows, so the curve would
        measure which frameworks survive rather than how well the score ranks."""
        short, long = record([1.0, 2.0], [1.0, 0.0]), record([1.0, 2.0, 3.0], [0.6, 0.3, 0.0])
        curve = selection_curve([short, long], "self_perplexity", 3)
        self.assertAlmostEqual(curve[2], (0.5 + 0.3) / 2)     # short still averaged over its 2

    def test_candidates_without_a_defined_recovery_are_excluded(self):
        r = {"candidates": [{"recovery": None, "self_perplexity": 0.1},
                            {"recovery": 0.4, "self_perplexity": 5.0},
                            {"recovery": 0.6, "self_perplexity": 6.0}]}
        # the wrong-length candidate has the best perplexity but must not be picked
        self.assertAlmostEqual(selection_curve([r], "self_perplexity", 1)[0], 0.4)

    def test_missing_scores_do_not_shift_the_ranking(self):
        r = {"candidates": [{"recovery": 0.9, "judge_perplexity": 1.0},
                            {"recovery": 0.1, "judge_perplexity": None},
                            {"recovery": 0.5, "judge_perplexity": 2.0}]}
        self.assertAlmostEqual(selection_curve([r], "judge_perplexity", 1)[0], 0.9)


class RhoDistributionTests(unittest.TestCase):
    def test_one_rho_per_framework_oriented_for_perplexity(self):
        good = record([1.0, 2.0, 3.0], [0.9, 0.5, 0.1])   # confident == accurate
        bad = record([1.0, 2.0, 3.0], [0.1, 0.5, 0.9])    # confident == wrong
        self.assertAlmostEqual(rho_distribution([good], "self_perplexity")[0], 1.0)
        self.assertAlmostEqual(rho_distribution([bad], "self_perplexity")[0], -1.0)

    def test_frameworks_with_too_few_candidates_are_skipped(self):
        self.assertEqual(rho_distribution([record([1.0, 2.0], [0.5, 0.4])], "self_perplexity"), [])


if __name__ == "__main__":
    unittest.main()


class AutoFigureTests(unittest.TestCase):
    """evaluate.py / post-training generation write the figure beside the JSON automatically."""

    def _payload(self, with_scores=True):
        cands = [{"recovery": 0.9, "self_perplexity": 1.0 if with_scores else None,
                  "judge_perplexity": None},
                 {"recovery": 0.5, "self_perplexity": 5.0 if with_scores else None,
                  "judge_perplexity": None},
                 {"recovery": 0.1, "self_perplexity": 9.0 if with_scores else None,
                  "judge_perplexity": None}]
        return {"records": [{"candidates": cands}]}

    def test_figure_is_written_next_to_the_json(self):
        import tempfile, os
        from cdrseqgen.methods.mlm.evaluation import write_confidence_figure
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "generation_eval.json")
            path = write_confidence_figure(self._payload(), out)
            self.assertEqual(path, os.path.join(tmp, "generation_eval_confidence.png"))
            self.assertGreater(os.path.getsize(path), 0)

    def test_nothing_to_plot_is_skipped_not_failed(self):
        """--skip_perplexity with no judge leaves no scores; that's a config choice, not an error."""
        import tempfile, os
        from cdrseqgen.methods.mlm.evaluation import write_confidence_figure
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "generation_eval.json")
            self.assertIsNone(write_confidence_figure(self._payload(with_scores=False), out))
            self.assertEqual(os.listdir(tmp), [])

    def test_no_output_path_writes_nothing(self):
        from cdrseqgen.methods.mlm.evaluation import write_confidence_figure
        self.assertIsNone(write_confidence_figure(self._payload(), None))


# ====================================================================================================
# from test_plotting.py
# Contract tests for the shared plotting layer.
#
# These do not check that a chart looks right — that needs eyes. They check the guarantees the
# analysis figures rely on when they compose panels: that a primitive reports whether it drew
# anything (so a caller can substitute an empty panel instead of shipping blank axes), that
# mismatched inputs fail loudly at construction rather than producing a silently wrong chart, and
# that an unmeasured value is skipped rather than drawn as a zero.
#
# The last one is the science-critical case: on a recovery axis a 0.0 bar and an absent bar look
# identical and mean opposite things.
# ====================================================================================================

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from cdrseqgen.visualization import plot_style
from cdrseqgen.visualization.bargraph_plotter import BarSeries, draw_grouped_bars
from cdrseqgen.visualization.batch_plots import _context_pair
from cdrseqgen.visualization.linegraph_plotter import LineSeries, draw_lines
from cdrseqgen.visualization.scatterplot_plotter import ScatterSeries, draw_scatter
from cdrseqgen.visualization.sweep_plots import SWEEP_SUBDIRS, load_trials as load_sweep_trials

class PlotStyleTests(unittest.TestCase):
    def test_colors_and_markers_wrap_instead_of_raising(self):
        # A sweep can carry more settings than the palette has entries; wrapping keeps the chart
        # drawable, and the marker cycle keeps wrapped series distinguishable.
        self.assertEqual(plot_style.color(0), plot_style.color(len(plot_style.PALETTE)))
        self.assertEqual(plot_style.marker(0), plot_style.marker(len(plot_style.MARKERS)))

    def test_percent_axis_applies_only_to_the_named_axis(self):
        _, ax = plt.subplots()
        plot_style.style_axes(ax, percent_axis="y")
        self.assertIn("%", ax.yaxis.get_major_formatter().format_data(0.5))
        self.assertNotIn("%", ax.xaxis.get_major_formatter().format_data(0.5))
        plt.close("all")

    def test_empty_panel_states_the_reason_in_the_panel(self):
        _, ax = plt.subplots()
        plot_style.empty_panel(ax, "No generative eval found")
        self.assertEqual([t.get_text() for t in ax.texts], ["No generative eval found"])
        # Ticks are cleared so a blank panel does not read as a real axis with no data on it.
        self.assertEqual(list(ax.get_xticks()), [])
        plt.close("all")

    def test_figures_live_in_a_subdirectory_of_the_batch(self):
        # Records stay at the batch root (runs.tsv, summary.tsv, batch_tables.md) and images go
        # below, so a round's folder listing stays readable as the chart count grows.
        self.assertEqual(plot_style.figures_dir(os.path.join("results", "runs", "b1")),
                         os.path.join("results", "runs", "b1", "figures"))

    def test_save_figure_creates_a_missing_directory(self):
        figure, _ = plt.subplots()
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "nested", "deeper")
            path = plot_style.save_figure(figure, target, "chart")
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(path.endswith("chart.png"))


class LineGraphTests(unittest.TestCase):
    def test_mismatched_x_and_y_fail_at_construction(self):
        with self.assertRaises(ValueError):
            LineSeries([1, 2, 3], [0.1, 0.2], "broken")

    def test_draw_reports_whether_anything_was_drawn(self):
        _, ax = plt.subplots()
        self.assertFalse(draw_lines(ax, []))
        self.assertFalse(draw_lines(ax, [LineSeries([], [], "empty")]))
        self.assertTrue(draw_lines(ax, [LineSeries([1, 2], [0.3, 0.4], "real")]))
        plt.close("all")

    def test_a_band_is_drawn_beneath_its_line(self):
        _, ax = plt.subplots()
        draw_lines(ax, [LineSeries([1, 2], [0.3, 0.4], "banded", band=([0.2, 0.3], [0.4, 0.5]))])
        self.assertEqual(len(ax.collections), 1)  # the fill_between
        self.assertEqual(len(ax.lines), 1)
        plt.close("all")


def _bar_container(ax):
    """The bars themselves. Passing yerr also appends an ErrorbarContainer, and it sorts first, so
    indexing ax.containers[0] finds different objects depending on whether error bars were drawn."""
    return next(c for c in ax.containers if isinstance(c, BarContainer))


class BarGraphTests(unittest.TestCase):
    def test_value_and_error_lengths_must_agree(self):
        with self.assertRaises(ValueError):
            BarSeries("broken", [0.1, 0.2], errors=[0.01])

    def test_series_must_cover_every_category(self):
        _, ax = plt.subplots()
        with self.assertRaises(ValueError):
            draw_grouped_bars(ax, ["core", "anchor"], [BarSeries("short", [0.1])])
        plt.close("all")

    def test_an_unmeasured_value_is_skipped_rather_than_drawn_as_zero(self):
        _, ax = plt.subplots()
        drew = draw_grouped_bars(ax, ["core", "anchor"], [BarSeries("partial", [0.25, None])])
        self.assertTrue(drew)
        heights = [patch.get_height() for patch in ax.patches]
        self.assertEqual(heights, [0.25])
        plt.close("all")

    def test_draw_reports_false_when_no_series_has_a_finite_value(self):
        _, ax = plt.subplots()
        self.assertFalse(draw_grouped_bars(ax, ["core"], []))
        self.assertFalse(draw_grouped_bars(ax, ["core"], [BarSeries("none", [None])]))
        plt.close("all")

    def test_a_single_replicate_gets_no_error_bar(self):
        # The spread of one run is undefined, not zero; a zero-length cap would claim a precision
        # the run cannot support. Callers pass 0.0 for "no spread measured", so 0.0 must draw nothing.
        _, ax = plt.subplots()
        draw_grouped_bars(ax, ["core"], [BarSeries("one seed", [0.25], errors=[0.0])])
        self.assertIsNone(_bar_container(ax).errorbar)
        plt.close("all")

    def test_a_measured_spread_does_get_an_error_bar(self):
        _, ax = plt.subplots()
        draw_grouped_bars(ax, ["core"], [BarSeries("three seeds", [0.25], errors=[0.004])])
        self.assertIsNotNone(_bar_container(ax).errorbar)
        plt.close("all")


class ScatterTests(unittest.TestCase):
    def test_mismatched_axes_fail_at_construction(self):
        with self.assertRaises(ValueError):
            ScatterSeries("broken", [0.5, 0.6], [7.0])

    def test_non_finite_points_are_dropped_not_plotted_at_zero(self):
        _, ax = plt.subplots()
        drew = draw_scatter(ax, [ScatterSeries("mixed", [0.5, float("nan")], [7.0, 8.0])])
        self.assertTrue(drew)
        self.assertEqual(len(ax.collections[0].get_offsets()), 1)
        plt.close("all")

    def test_draw_reports_false_when_nothing_is_plottable(self):
        _, ax = plt.subplots()
        self.assertFalse(draw_scatter(ax, [ScatterSeries("empty", [], [])]))
        plt.close("all")


class SweepSubdirectoryTests(unittest.TestCase):
    def test_planned_learning_rates_are_separate_settings_and_retries_are_not_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, trials = [], []
            for lr, seed in (("1e-4", 42), ("1e-4", 43), ("1e-3", 42)):
                run = root / f"run_{lr}_{seed}"
                run.mkdir()
                config = dict(loss_type="cross_entropy", learning_rate=float(lr), seed=seed)
                (run / "run_config.json").write_text(json.dumps(config))
                (run / "epoch_metrics.csv").write_text("epoch,eval_core_accuracy\n1,0.3\n")
                (run / "generation_eval.json").write_text(json.dumps({"summary": {}}))
                rows.append(dict(run_dir=str(run), status="completed", setting_id=lr, seed=seed))
                trials.append(dict(setting_id=lr, config=config, grid=dict(lr=lr, seed=str(seed))))
            rows.append(dict(rows[0]))
            with (root / "runs.tsv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            (root / "plan.json").write_text(json.dumps({"trials": trials}))
            groups = load_sweep_trials(str(root))
            self.assertEqual({k: len(v) for k, v in groups.items()}, {"lr=1e-4": 2, "lr=1e-3": 1})

    def test_a_sweeps_own_output_dirs_are_not_mistaken_for_trials(self):
        # load_trials falls back to globbing subdirectories when runs.tsv is missing — the
        # interrupted-sweep case — so figures/ and logs/ must be excluded by name or every one
        # would be reported as a trial with no run config.
        self.assertIn(plot_style.FIGURES_DIRNAME, SWEEP_SUBDIRS)
        self.assertIn("logs", SWEEP_SUBDIRS)


class BatchContextPairingTests(unittest.TestCase):
    """The context tag is per-loop, not per-batch, and pairing on a single batch-wide tag silently
    drops half the loops from the ablation — heavy targets carry _LC, light targets carry _HC, and
    SAbDab2 has no context-free run at all."""

    RUNS = {
        ("h3", "OAS", "", "scratch"): "a",
        ("h3", "OAS", "_LC", "scratch"): "b",
        ("l3", "OAS", "", "scratch"): "c",
        ("l3", "OAS", "_HC", "scratch"): "d",
        ("h3", "SD2", "_LC", "scratch"): "e",
        ("h3", "SD2", "_LC_AG", "scratch"): "f",
        ("h1", "SD2", "_LC", "scratch"): "g",
    }

    def test_a_heavy_loop_pairs_against_its_light_chain_context(self):
        self.assertEqual(_context_pair(self.RUNS, "h3", "OAS"), ("", "_LC"))

    def test_a_light_loop_pairs_against_its_heavy_chain_context(self):
        self.assertEqual(_context_pair(self.RUNS, "l3", "OAS"), ("", "_HC"))

    def test_sabdab_pairs_partner_chain_against_partner_plus_antigen(self):
        # No context-free SAbDab2 run exists, so the baseline is itself a context regime.
        self.assertEqual(_context_pair(self.RUNS, "h3", "SD2"), ("_LC", "_LC_AG"))

    def test_a_loop_with_one_regime_has_no_pair(self):
        self.assertIsNone(_context_pair(self.RUNS, "h1", "SD2"))


# ====================================================================================================
# from test_sweep_tables.py
# Tests for the sweep ranking and the auto-reporting steps the shell scripts call.
#
# Two guarantees matter here beyond arithmetic:
#
#   * the ranking direction must follow the metric. Perplexity is lower-is-better and recovery is
#     higher-is-better, and a sweep that silently ranks perplexity backwards would nominate the worst
#     setting as the winner while looking entirely normal.
#   * reporting must never fail the thing it reports on. ``scripts/run.sh`` and ``scripts/sweep.sh``
#     call these after the GPU work is finished, so a batch that cannot be parsed has to degrade to a
#     warning rather than an exception.
# ====================================================================================================

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


class SelectedEpochConfigTests(unittest.TestCase):
    """The run config names the selection metric; a missing or unreadable one must fall back to
    core accuracy instead of failing the report that is built after the GPU work has finished."""

    ROWS = [{"epoch": 1, "eval_core_accuracy": 0.2, "eval_loss": 1.0},
            {"epoch": 2, "eval_core_accuracy": 0.5, "eval_loss": 2.0},
            {"epoch": 3, "eval_core_accuracy": 0.4, "eval_loss": 0.5}]

    def select(self, config_text=None):
        with tempfile.TemporaryDirectory() as tmp:
            if config_text is not None:
                with open(os.path.join(tmp, "run_config.json"), "w") as handle:
                    handle.write(config_text)
            return _selected_epoch(tmp, self.ROWS)

    def test_without_a_run_config_the_best_core_accuracy_epoch_is_selected(self):
        row, key = self.select()
        self.assertEqual((row["epoch"], key), (2, "eval_core_accuracy"))

    def test_a_corrupt_run_config_falls_back_the_same_way(self):
        row, key = self.select("{not json")
        self.assertEqual((row["epoch"], key), (2, "eval_core_accuracy"))

    def test_the_run_config_names_the_selection_metric_and_direction(self):
        row, key = self.select(json.dumps({"metric_for_best_model": "loss", "greater_is_better": False}))
        self.assertEqual((row["epoch"], key), (3, "eval_loss"))


# ====================================================================================================
# from test_batch_tables.py
# ====================================================================================================

FIELDS = (
    "epoch",
    "eval_accuracy",
    "eval_core_accuracy",
    "eval_anchor_accuracy",
    "eval_core_coverage",
)

def write_metrics(run_dir, rows, *, include_coverage=True):
    run_dir.mkdir(parents=True)
    fields = FIELDS if include_coverage else FIELDS[:-1]
    with (run_dir / "epoch_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


class CoverageAwareReportingTests(unittest.TestCase):
    def test_adequate_coverage_selects_core_best_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_metrics(run, [
                {"epoch": 1, "eval_accuracy": 0.90, "eval_core_accuracy": 0.20,
                 "eval_anchor_accuracy": 0.95, "eval_core_coverage": 0.80},
                {"epoch": 2, "eval_accuracy": 0.80, "eval_core_accuracy": 0.40,
                 "eval_anchor_accuracy": 0.90, "eval_core_coverage": 0.80},
                # Overfitting tail: the last row must NOT be the reported one (README: "read
                # best-epoch, not last-epoch"). Without it, idxmax and "take the last row" agree.
                {"epoch": 3, "eval_accuracy": 0.75, "eval_core_accuracy": 0.31,
                 "eval_anchor_accuracy": 0.88, "eval_core_coverage": 0.80},
            ])
            summary = recovery_summary(str(run))

        self.assertEqual(summary["primary_metric"], "core")
        self.assertEqual(summary["selection_metric"], "eval_core_accuracy")
        self.assertEqual(summary["epoch"], 2)
        self.assertEqual(summary["primary_value"], 0.40)
        self.assertEqual(summary["whole_accuracy"], 0.80)
        self.assertEqual(summary["core_coverage"], 0.80)

    def test_low_coverage_selects_whole_and_marks_core_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_metrics(run, [
                {"epoch": 1, "eval_accuracy": 0.70, "eval_core_accuracy": 0.99,
                 "eval_anchor_accuracy": 0.70, "eval_core_coverage": 0.10},
                {"epoch": 2, "eval_accuracy": 0.80, "eval_core_accuracy": 0.10,
                 "eval_anchor_accuracy": 0.80, "eval_core_coverage": 0.10},
                {"epoch": 3, "eval_accuracy": 0.72, "eval_core_accuracy": 0.05,
                 "eval_anchor_accuracy": 0.75, "eval_core_coverage": 0.10},
            ])
            summary = recovery_summary(str(run))

        self.assertEqual(summary["primary_metric"], "whole")
        self.assertEqual(summary["selection_metric"], "eval_accuracy")
        self.assertEqual(summary["epoch"], 2)
        self.assertEqual(summary["primary_value"], 0.80)
        self.assertIsNone(summary["core_accuracy"])
        self.assertIn("below", summary["reason"])

    def test_missing_coverage_falls_back_to_explicit_whole_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "legacy"
            write_metrics(run, [
                {"epoch": 1, "eval_accuracy": 0.60, "eval_core_accuracy": 0.90,
                 "eval_anchor_accuracy": 0.60, "eval_core_coverage": 1.0},
                {"epoch": 2, "eval_accuracy": 0.70, "eval_core_accuracy": 0.10,
                 "eval_anchor_accuracy": 0.70, "eval_core_coverage": 1.0},
                {"epoch": 3, "eval_accuracy": 0.65, "eval_core_accuracy": 0.08,
                 "eval_anchor_accuracy": 0.66, "eval_core_coverage": 1.0},
            ], include_coverage=False)
            summary = recovery_summary(str(run))

        self.assertEqual(summary["primary_metric"], "whole")
        self.assertEqual(summary["epoch"], 2)
        self.assertIsNone(summary["core_coverage"])
        self.assertEqual(summary["reason"], "core coverage unavailable")

    def test_loop_name_does_not_determine_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "l2_OAS_scratch"
            write_metrics(run, [
                {"epoch": 1, "eval_accuracy": 0.30, "eval_core_accuracy": 0.40,
                 "eval_anchor_accuracy": 0.20, "eval_core_coverage": 0.80},
            ])
            table = build_table("loops", {("l2", "OAS", "", "scratch"): str(run)})

        self.assertIn("| L2 | core | 80.0% | 0.4000", table)
        self.assertNotIn("L2 whole-span", table)

    def test_mixed_policies_are_detected_instead_of_subtracted(self):
        core = {"primary_metric": "core", "core_coverage": 1.0}
        whole = {"primary_metric": "whole", "core_coverage": 0.1}
        self.assertEqual(_common_recovery_policy((core, whole)), ("mixed", 0.1))

    def test_markdown_renderer_rejects_wrong_row_width(self):
        with self.assertRaisesRegex(ValueError, "cells"):
            markdown_table("Example", ("a", "b"), (("only one",),))


class DatasetTagPlumbingTests(unittest.TestCase):
    """A new corpus tag must survive parsing AND the table lookups, or a batch renders empty."""

    def test_run_names_parse_for_every_declared_dataset_tag(self):
        # RUN_RE used to spell out OAS|SD2|DS. Adding OAS4 for the four-donor corpus then made
        # every OAS4 run unparseable: discover_runs returned nothing and a finished batch produced
        # empty tables and dashboards with no error anywhere.
        from cdrseqgen.methods.common.run_setup import DATASET_TAGS, UNKNOWN_DATASET_TAG
        for tag in {t for _, t in DATASET_TAGS} | {UNKNOWN_DATASET_TAG}:
            with self.subTest(tag=tag):
                match = RUN_RE.match(f"h3_{tag}_LC_scratch")
                self.assertIsNotNone(match, f"h3_{tag}_LC_scratch does not parse")
                self.assertEqual(match.group("ds"), tag)

    def test_unseen_fraction_measures_the_split_it_names(self):
        """It used to count every non-train entry, merging validation AND test on a 3-way corpus."""
        entries = [
            {"heavy": {"cdr3": "AAA"}, "split": "train"},
            {"heavy": {"cdr3": "AAA"}, "split": "validation"},
            {"heavy": {"cdr3": "BBB"}, "split": "validation"},
            {"heavy": {"cdr3": "CCC"}, "split": "test"},
            {"heavy": {"cdr3": "DDD"}, "split": "test"},
        ]
        _, _, n, unseen = unseen_fraction(entries, "cdr3", split="validation")
        self.assertEqual((n, unseen), (2, 1))       # not (4, 3)
        _, _, n, unseen = unseen_fraction(entries, "cdr3", split="test")
        self.assertEqual((n, unseen), (2, 2))

    def test_unseen_fraction_falls_back_when_there_is_no_such_split(self):
        entries = [{"heavy": {"cdr3": "AAA"}, "split": "train"},
                   {"heavy": {"cdr3": "BBB"}, "split": "test"}]
        _, _, n, unseen = unseen_fraction(entries, "cdr3", split="validation")
        self.assertEqual((n, unseen), (1, 1))

    def test_a_missing_metric_renders_as_a_dash_not_nan(self):
        """`aggregate` returns None for an uncollected metric; the table must not crash or print
        'nan'. Both were live: causal H3 has no per-position entropy, since that needs
        length-matched candidates."""
        self.assertEqual(_fmt(None), "—")
        self.assertEqual(_fmt(0.5), "0.500")

    def test_a_second_oas_corpus_resolves_onto_the_canonical_oas_rows(self):
        # Tables are written against "OAS"; a batch on OAS4 must still fill them.
        self.assertEqual(dataset_alias({("h3", "OAS4", "", "scratch"): "d"}), {"OAS": "OAS4"})
        self.assertEqual(dataset_alias({("h3", "OAS", "", "scratch"): "d"}), {})
        # Ambiguous or absent: leave the canonical tag alone rather than guessing.
        self.assertEqual(dataset_alias({("h3", "OAS", "", "s"): "d",
                                        ("h3", "OAS4", "", "s"): "e"}), {})
        self.assertEqual(dataset_alias({("h3", "SD2", "_LC", "scratch"): "d"}), {})


class MixedLossDetectionTests(unittest.TestCase):
    """A core-weighted run optimises the number the recovery tables report, so a batch that mixes
    objectives is not one comparison — it must say so rather than render silently."""

    def _batch(self, tmp, losses):
        import json, os
        runs = {}
        for i, loss in enumerate(losses):
            run = Path(tmp) / f"h3_OAS_scratch_{i}"
            run.mkdir(parents=True)
            if loss is not None:
                (run / "run_config.json").write_text(json.dumps({"loss_type": loss}))
            runs[("h3", "OAS", "", f"m{i}")] = str(run)
        return runs

    def test_single_objective_is_not_flagged(self):
        import tempfile
        from cdrseqgen.visualization.batch_tables import loss_warning
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(loss_warning(self._batch(tmp, ["cross_entropy", "cross_entropy"])))

    def test_mixed_objectives_are_flagged(self):
        import tempfile
        from cdrseqgen.visualization.batch_tables import loss_warning
        with tempfile.TemporaryDirectory() as tmp:
            warning = loss_warning(self._batch(tmp, ["cross_entropy", "weighted_cross_entropy"]))
            self.assertIsNotNone(warning)
            self.assertIn("weighted_cross_entropy", warning)

    def test_runs_predating_the_field_count_as_cross_entropy(self):
        """Old runs have no loss_type; they trained with plain cross-entropy, so a batch of them
        plus a new cross-entropy run is still one comparison."""
        import tempfile
        from cdrseqgen.visualization.batch_tables import loss_warning
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(loss_warning(self._batch(tmp, [None, "cross_entropy"])))


class MethodComparisonTableTests(unittest.TestCase):
    def test_plm_encoder_families_retain_the_s2s_method(self):
        from cdrseqgen.visualization.batch_tables import model_family, discover_runs, FAMILY_LABELS
        self.assertEqual(model_family('s2s_esm2_t12_35M_UR50D'), 's2s_esm2')
        self.assertEqual(model_family('s2s_IgBert'), 's2s_igbert')
        self.assertEqual(model_family('s2s_prot_bert'), 's2s_protbert')
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in ('esm2_t12_35M_UR50D', 's2s_esm2_t12_35M_UR50D',
                           's2s_IgBert', 's2s_prot_bert'):
                (Path(tmp) / ('h3_OAS4_' + suffix)).mkdir()
            runs = discover_runs(tmp)
        self.assertEqual(len(runs), 4)
        self.assertEqual({k[-1] for k in runs},
                         {'esm2', 's2s_esm2', 's2s_igbert', 's2s_protbert'})
        self.assertTrue(all(k[-1] in FAMILY_LABELS for k in runs))

    def test_every_registered_method_has_a_display_family(self):
        """A method whose run names parse to a family absent from FAMILY_LABELS is silently
        dropped from every table and dashboard — the run completes, the artifacts are written, and
        the results simply never appear. Adding a method must add its family."""
        from cdrseqgen.methods.registry import METHODS
        from cdrseqgen.visualization.batch_tables import FAMILY_LABELS
        for name in METHODS:
            # `mlm` is the historical default and its runs carry no method marker in the name.
            family = "scratch" if name == "mlm" else f"{name}_scratch"
            self.assertIn(family, FAMILY_LABELS,
                          f"method {name!r} produces runs named *_{family} with no table family")

    def test_family_to_method_mapping_comes_from_the_registry(self):
        """Dashboards branch on this to decide whether teacher-forced bars are comparable. A
        prefix guess (`startswith("ar_")`) filed every method added after `ar` under "masked",
        which inverts the warning: `s2s` predicts position i from the true prefix exactly as `ar`
        does, so it is the MASKED arm that is the odd one out."""
        from cdrseqgen.methods.registry import METHODS
        from cdrseqgen.visualization.batch_tables import FAMILY_LABELS, method_of_family
        self.assertEqual(method_of_family("scratch"), "mlm")
        self.assertEqual(method_of_family("esm2"), "mlm")
        self.assertEqual(method_of_family("ar_scratch"), "ar")
        self.assertEqual(method_of_family("s2s_scratch"), "s2s")
        for family in FAMILY_LABELS:
            self.assertIn(method_of_family(family), METHODS)

    def test_compare_suite_table_contains_masked_and_causal_runs(self):
        row = {
            "epoch": 1,
            "eval_accuracy": 0.5,
            "eval_core_accuracy": 0.4,
            "eval_anchor_accuracy": 0.8,
            "eval_core_coverage": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            masked = Path(tmp) / "h3_OAS_scratch"
            causal = Path(tmp) / "h3_OAS_ar_scratch"
            write_metrics(masked, [row])
            write_metrics(causal, [{**row, "eval_core_accuracy": 0.45}])
            table = build_table("methods", {
                ("h3", "OAS", "", "scratch"): str(masked),
                ("h3", "OAS", "", "ar_scratch"): str(causal),
            })

        self.assertIn("scratch (masked)", table)
        self.assertIn("scratch (causal)", table)
        self.assertIn("0.4000", table)
        self.assertIn("0.4500", table)
        self.assertIn("+0.050", table)

    def test_the_encoder_decoder_arm_reaches_the_comparison_table(self):
        row = {
            "epoch": 1,
            "eval_accuracy": 0.5,
            "eval_core_accuracy": 0.4,
            "eval_anchor_accuracy": 0.8,
            "eval_core_coverage": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            masked = Path(tmp) / "h3_OAS_scratch"
            seq2seq = Path(tmp) / "h3_OAS_s2s_scratch"
            write_metrics(masked, [row])
            write_metrics(seq2seq, [{**row, "eval_core_accuracy": 0.47}])
            table = build_table("methods", {
                ("h3", "OAS", "", "scratch"): str(masked),
                ("h3", "OAS", "", "s2s_scratch"): str(seq2seq),
            })

        self.assertIn("scratch (enc-dec)", table)
        self.assertIn("0.4700", table)


class GenerationPolicyReportingTests(unittest.TestCase):
    def test_length_policy_and_recovery_population_come_from_each_artifact(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            runs = {}
            # The same model family can use native or inferred length: never infer policy from
            # the method name. An entirely unmatched framework must not inflate the denominator.
            for family, source, mode, expected in (
                ('scratch', 'length_predictor', 'inferred', 'inferred'),
                ('ar_scratch', 'eos_token', 'inferred', 'EOS'),
                ('esm2', 'native_mask_count', 'fixed', 'native length'),
            ):
                run = Path(directory) / family
                run.mkdir()
                payload = {'length_policy': {'mode': mode, 'source': source},
                           'records': [{'metrics': {'aar_mean': 0.5}}, {'metrics': {'aar_mean': None}}]}
                (run / 'generation_eval.json').write_text(json.dumps(payload))
                runs[('h3', 'OAS', '', family)] = str(run)
                table = build_table('design', runs)
                self.assertIn(f'| {expected} | 1/2 | 0.500 |', table)
            self.assertNotIn('a masked run is given the length', table)
            self.assertIn('populations may differ', table)

    def test_legacy_artifact_without_policy_is_reported_as_unknown(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'generation_eval.json').write_text(json.dumps({
                'records': [{'metrics': {'aar_mean': 0.5}}]}))
            table = build_table('design', {('h3', 'OAS', '', 'scratch'): directory})
        self.assertIn('| unknown | 1/1 | 0.500 |', table)


class TrialIdentityReportingTests(unittest.TestCase):
    def make_batch(self, directory):
        runs = []
        for seed in (42, 43):
            for dropout in (0, .5):
                run = Path(directory) / f'h3_OAS4_LC_s2s_esm2_t12_35M_UR50D_cd{dropout}_s{seed}'
                write_metrics(run, [dict(epoch=1, eval_accuracy=.5, eval_core_accuracy=.4,
                    eval_anchor_accuracy=.8, eval_core_coverage=1)])
                (run/'run_config.json').write_text(json.dumps(dict(method='s2s',
                    model_type='facebook/esm2_t12_35M_UR50D', seed=seed,
                    context_dropout=dropout, include_partner_chain=True)))
                for filename, partner in (('generation_eval.json', True),
                                          ('generation_eval_without_partner.json', False)):
                    (run/filename).write_text(json.dumps(dict(include_partner_chain=partner,
                        judge_include_partner_chain=True,
                        generation_validity=dict(valid_generation_rate=1, length_absolute_error_mean_capped=3),
                        summary=dict(edit_similarity_mean=.39, core_recovery_mean=.2,
                            length_match_rate=.08, judge_perplexity_mean=3.2, judge_coverage=1),
                        records=[dict(metrics=dict(core_recovery_mean=.2))])))
                runs.append(run)
        return runs

    def test_repeated_esm2_trials_are_all_discovered_and_reported(self):
        from cdrseqgen.visualization.batch_tables import discover_runs, has_repeated_families, render_tables
        with tempfile.TemporaryDirectory() as directory:
            expected = self.make_batch(directory)
            runs = discover_runs(directory)
            self.assertEqual(len(runs), 4)
            self.assertEqual(set(runs.values()), {str(p) for p in expected})
            self.assertTrue(has_repeated_families(runs))
            table = render_tables(directory, runs, wanted=[])
            self.assertIn('4 runs', table)
            for run in expected:
                self.assertEqual(table.count(f'| {run.name} |'), 3)
            self.assertIn('Judge partner', table)
            self.assertIn('Core frameworks', table)
            self.assertNotIn('nan', table)

    def test_control_report_distinguishes_training_generator_and_judge_context(self):
        from cdrseqgen.visualization.batch_tables import context_ablation_table
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_batch(directory)[0]
            (run/'generation_eval_without_partner.json').unlink()
            config = json.loads((run/'run_config.json').read_text())
            config.update(include_partner_chain=False, gen_judge_partner_context='with')
            (run/'run_config.json').write_text(json.dumps(config))
            payload = json.loads((run/'generation_eval.json').read_text())
            payload['include_partner_chain'] = False
            (run/'generation_eval.json').write_text(json.dumps(payload))
            table = context_ablation_table({('h3', 'OAS4', '', 's2s_esm2'): str(run)})
            self.assertIn('| without | with |', table)
            self.assertIn('| 1/1 |', table)

    def test_trial_dashboard_keeps_all_seed_and_dropout_categories(self):
        from unittest.mock import patch
        from cdrseqgen.visualization.batch_plots import plot_batch
        from cdrseqgen.visualization.bargraph_plotter import draw_grouped_bars
        with tempfile.TemporaryDirectory() as directory:
            self.make_batch(directory)
            with patch('cdrseqgen.visualization.batch_plots.draw_grouped_bars', wraps=draw_grouped_bars) as draw:
                paths = plot_batch(directory, save_dir=str(Path(directory)/'figures'))
            self.assertEqual(len(paths), 1)
            self.assertTrue(Path(paths[0]).is_file())
            self.assertEqual(draw.call_count, 2)
            self.assertTrue(all(len(call.args[1]) == 4 for call in draw.call_args_list))


if __name__ == "__main__":
    unittest.main()
