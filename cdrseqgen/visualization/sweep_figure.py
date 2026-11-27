"""Compare a sweep's settings across seeds in a four-panel dashboard PNG.

Aggregates replicate seeds per setting and delegates chart drawing to ``plot_line``, ``plot_bar``,
and ``plot_scatter`` over the shared conventions in ``plot_style``. Reads saved artifacts only
(``epoch_metrics.csv``, ``generation_eval.json``, ``run_config.json``); no GPU or checkpoint load.

Usage:
    python -m cdrseqgen.visualization.sweep_figure --sweep results/runs/20260825_165553_sweep_loss
    python -m cdrseqgen.visualization.sweep_figure --sweep <dir> --bins 9 --title "H3 objectives"

Panels (2x2 -> ``<sweep>/figures/<filename>.png``):
  * A. Validation core recovery per epoch -- seed mean with min–max replicate band.
  * B. Free-running ``core@L`` and ``anchor@L`` -- grouped bars (seed mean ± SD) split at the
       anchor boundary.
  * C. Recovery across normalized loop position -- macro-averaged within each example first, then
       across examples and seeds so longer loops carry no extra weight.
  * D. Judge perplexity vs. pairwise Hamming diversity -- one scatter marker per trial so seed
       spread stays visible.

Functions:
  * objective_label     -- format a run config's loss objective as a readable label.
  * canonical_objective -- normalize equivalent weighted-loss ratios to detect duplicates.
  * load_trials         -- load completed sweep trials from ``runs.tsv``/``plan.json`` or subdirs.
  * report_duplicates   -- log settings that share a canonical objective class.
  * plot_sweep          -- build and save the 2x2 sweep dashboard PNG.
  * main                -- CLI entry point.

Notes:
  * Works on interrupted sweeps: when ``runs.tsv`` is absent, falls back to scanning trial
    subdirectories while skipping sweep output folders in ``SWEEP_SUBDIRS``.
  * Imports ``plot_style`` before ``matplotlib.pyplot`` so the headless ``Agg`` backend is set.
"""
import argparse
import json
import os
import statistics
from collections import defaultdict

import pandas as pd

from cdrseqgen.artifacts import find_run_config
from cdrseqgen.eval.metrics import position_recovery
from cdrseqgen.visualization import plot_style
import matplotlib.pyplot as plt
from cdrseqgen.visualization.plot_bar import BarSeries, draw_grouped_bars
from cdrseqgen.visualization.plot_line import LineSeries, draw_lines
from cdrseqgen.visualization.plot_scatter import ScatterSeries, draw_scatter
from cdrseqgen import artifacts


# --- Trial Loading & Objective Grouping ----------------------------------------------------------

# A sweep's own output directories, which sit beside its trial folders and are not trials.
SWEEP_SUBDIRS = frozenset({plot_style.FIGURES_DIRNAME, "logs", ".configs"})


def objective_label(config):
    """Human-readable name for a run's training objective, from its saved run_config.json."""
    loss = str(config.get("loss_type", "cross_entropy"))
    if loss == "cross_entropy":
        return "cross-entropy"
    core = float(config.get("core_loss_weight", 1.0))
    anchor = float(config.get("anchor_loss_weight", 1.0))
    return f"weighted {core:g}/{anchor:g}"


def canonical_objective(config):
    """The equivalence class of an objective, so duplicate settings can be reported as duplicates.

    The weighted loss is a weighted MEAN over supervised tokens, so only the core-to-anchor ratio
    survives normalisation: `3.0/0.0` trains exactly the model `1.0/0.0` trains, and equal weights
    reduce to plain cross-entropy. Two settings sharing a class are the same experiment run twice --
    useful as a determinism control, misleading if read as two objectives.
    """
    loss = str(config.get("loss_type", "cross_entropy"))
    if loss == "cross_entropy":
        return "cross_entropy"
    core = float(config.get("core_loss_weight", 1.0))
    anchor = float(config.get("anchor_loss_weight", 1.0))
    if anchor == 0:
        return "core_only"
    if core == 0:
        return "anchor_only"
    return "cross_entropy" if core == anchor else f"ratio:{core / anchor:g}"


def load_trials(sweep_dir):
    """Every completed run under a sweep directory, grouped by objective label.

    Uses `runs.tsv` when the sweep wrote one, and otherwise falls back to any subdirectory holding
    a run config -- an interrupted sweep never reaches its summary step, so its finished trials must
    stay readable without it.
    """
    map_path = os.path.join(sweep_dir, "runs.tsv")
    settings_by_run = {}
    plan_path = os.path.join(sweep_dir, "plan.json")
    if os.path.exists(map_path):
        # Journal identifiers are opaque strings, even when they resemble numeric values.
        rows = pd.read_csv(map_path, sep="\t", dtype=str)
        if os.path.exists(plan_path) and "setting_id" in rows:
            with open(plan_path) as handle:
                plan = json.load(handle)
            # Objective-only grouping pools different LRs/depths as if they were seeds.
            varying = [key for key in plan["trials"][0]["grid"] if key != "seed" and
                       len({t["grid"][key] for t in plan["trials"]}) > 1]
            labels = {t["setting_id"]: (
                ", ".join(f"{k}={t['grid'][k]}" for k in varying) or objective_label(t["config"]))
                for t in plan["trials"]}
            rows = rows.drop_duplicates(["setting_id", "seed"], keep="last")
            settings_by_run = {r["run_dir"]: (r["setting_id"], labels[r["setting_id"]])
                               for _, r in rows.iterrows()}
        run_dirs = [r for r in rows.loc[rows["status"] == "completed", "run_dir"] if isinstance(r, str)]
    else:
        # A sweep's own subdirectories are not trials; without this the fallback would report every
        # one of them as a run missing its config, which is noise on exactly the interrupted sweep
        # this path exists to rescue.
        run_dirs = sorted(
            os.path.join(sweep_dir, name) for name in os.listdir(sweep_dir)
            if os.path.isdir(os.path.join(sweep_dir, name)) and name not in SWEEP_SUBDIRS
        )

    groups = defaultdict(list)
    for run_dir in run_dirs:
        meta_path = find_run_config(run_dir)
        if not meta_path:
            print(f"⚠️ No run config in {run_dir} — skipped")
            continue
        with open(meta_path) as handle:
            config = json.load(handle)
        gen_path = os.path.join(run_dir, artifacts.GENERATION_EVAL_FILENAME)
        generation = None
        if os.path.exists(gen_path):
            with open(gen_path) as handle:
                generation = json.load(handle)
        else:
            print(f"⚠️ No generation_eval.json in {run_dir} — panels B–D will skip this trial")
        csv_path = os.path.join(run_dir, artifacts.EPOCH_METRICS_FILENAME)
        canonical, label = settings_by_run.get(
            run_dir, (canonical_objective(config), objective_label(config)))
        groups[label].append({
            "run_dir": run_dir,
            "seed": config.get("seed"),
            "canonical": canonical,
            "curve": pd.read_csv(csv_path) if os.path.exists(csv_path) else None,
            "generation": generation,
        })
    return groups


def report_duplicates(groups):
    """Print any labels that name the same objective; they are replicates, not separate settings."""
    by_class = defaultdict(set)
    for label, trials in groups.items():
        for trial in trials:
            by_class[trial["canonical"]].add(label)
    for canonical, labels in by_class.items():
        if len(labels) > 1:
            print(f"ℹ️ {' and '.join(sorted(labels))} are the same objective ({canonical}). "
                  "Their gap measures run-to-run nondeterminism, not an effect of the setting.")


# --- Per-Setting Aggregation Helpers -------------------------------------------------------------

def _seed_averaged_curve(trials, column, label):
    """One setting's epoch curve as a LineSeries: mean over seeds, band spanning them.

    Seeds stop at different epochs under early stopping. The band is therefore built only where
    more than one seed is still present, and epochs past that point carry the mean alone — a
    single-seed tail drawn as a zero-width band would read as agreement between seeds.
    """
    per_epoch = defaultdict(list)
    for trial in trials:
        curve = trial["curve"]
        if curve is None or column not in curve.columns:
            continue
        for epoch, value in zip(curve["epoch"], curve[column]):
            per_epoch[float(epoch)].append(float(value))
    if not per_epoch:
        return None
    epochs = sorted(per_epoch)
    means = [statistics.mean(per_epoch[e]) for e in epochs]
    lower = [min(per_epoch[e]) if len(per_epoch[e]) > 1 else statistics.mean(per_epoch[e])
             for e in epochs]
    upper = [max(per_epoch[e]) if len(per_epoch[e]) > 1 else statistics.mean(per_epoch[e])
             for e in epochs]
    return LineSeries(epochs, means, label, band=(lower, upper))


def _summary_values(trials, key):
    """One value per trial from its generation_eval summary, skipping trials that lack it."""
    values = []
    for trial in trials:
        generation = trial["generation"]
        if not generation:
            continue
        value = generation.get("summary", {}).get(key)
        if isinstance(value, (int, float)) and value == value:
            values.append(float(value))
    return values


# --- Sweep Dashboard Panels (2x2 -> <filename>.png: Panels A–D) ----------------------------------

# Panel A: validation core recovery trajectory per epoch (seed mean + min–max band)
def _curve_panel(ax, groups):
    lines = [_seed_averaged_curve(trials, "eval_core_accuracy", label)
             for label, trials in groups.items()]
    if not draw_lines(ax, [line for line in lines if line], markers=False):
        return plot_style.empty_panel(ax, "No epoch_metrics.csv found")
    plot_style.style_axes(ax, title="A. Validation core recovery (teacher-forced)",
                          xlabel="Epoch", ylabel="eval_core_accuracy", legend=True)


# Panel B: free-running core@L vs. anchor@L grouped bars (seed mean ± SD)
def _decomposition_panel(ax, groups):
    metrics = [("core_recovery_mean", "core@L"), ("anchor_recovery_mean", "anchor@L")]
    series = []
    for label, trials in groups.items():
        values = [_summary_values(trials, key) for key, _ in metrics]
        if not any(values):
            continue
        series.append(BarSeries(
            label,
            [statistics.mean(v) if v else None for v in values],
            [statistics.stdev(v) if len(v) > 1 else 0.0 for v in values],
        ))
    if not draw_grouped_bars(ax, [pretty for _, pretty in metrics], series):
        return plot_style.empty_panel(ax, "No generative eval found")
    plot_style.style_axes(ax, title="B. Free-running recovery, split at the anchor boundary",
                          ylabel="Recovery (length-matched candidates)",
                          percent_axis="y", legend=True, grid="y")


# Panel C: free-running recovery across normalized loop position (N -> C terminus)
def _position_curve(trials, bins, label):
    """Recovery against normalized loop position, macro-averaged over examples then over seeds."""
    per_seed = []
    for trial in trials:
        generation = trial["generation"]
        if not generation:
            continue
        across_examples = [[] for _ in range(bins)]
        for record in generation.get("records", []):
            native = record.get("native")
            samples = record.get("samples") or []
            if not native or not samples:
                continue
            within_example = [[] for _ in range(bins)]
            for point in position_recovery(samples, native):
                if point["recovery"] is None:
                    continue
                index = min(bins - 1, max(0, int(point["relative_position"] * (bins - 1) + 0.5)))
                within_example[index].append(point["recovery"])
            # Average inside the example first, so a long loop contributing several residues to one
            # bin does not outweigh a short loop contributing one.
            for index, values in enumerate(within_example):
                if values:
                    across_examples[index].append(statistics.mean(values))
        curve = [statistics.mean(v) if v else None for v in across_examples]
        if any(value is not None for value in curve):
            per_seed.append(curve)
    if not per_seed:
        return None
    x_values, y_values = [], []
    for index in range(bins):
        present = [curve[index] for curve in per_seed if curve[index] is not None]
        if present:
            x_values.append(index / (bins - 1))
            y_values.append(statistics.mean(present))
    return LineSeries(x_values, y_values, label) if x_values else None


def _position_panel(ax, groups, bins):
    lines = [_position_curve(trials, bins, label) for label, trials in groups.items()]
    if not draw_lines(ax, [line for line in lines if line]):
        return plot_style.empty_panel(ax, "No stored generations")
    plot_style.style_axes(ax, title="C. Recovery across normalized loop position",
                          xlabel="Relative position (N-terminus → C-terminus)",
                          ylabel="Recovery", percent_axis="y", legend=True)


# Panel D: judge perplexity vs. pairwise Hamming diversity scatter (one point per trial)
def _tradeoff_panel(ax, groups):
    series = []
    for label, trials in groups.items():
        xs = _summary_values(trials, "diversity_hamming")
        ys = _summary_values(trials, "judge_perplexity_mean")
        if xs and len(xs) == len(ys):
            series.append(ScatterSeries(label, xs, ys))
    if not draw_scatter(ax, series):
        return plot_style.empty_panel(ax, "No judged generations")
    plot_style.style_axes(ax, title="D. Judged plausibility against diversity",
                          xlabel="Diversity (pairwise Hamming)",
                          ylabel="Judge perplexity  ↓ better", percent_axis="x", legend=True)


# --- Dashboard Composer & CLI --------------------------------------------------------------------

def plot_sweep(sweep_dir, save_dir=None, bins=11, title=None, filename="objective_comparison"):
    groups = load_trials(sweep_dir)
    if not groups:
        print(f"🚫 No readable runs under {sweep_dir}")
        return None
    report_duplicates(groups)

    # Sorted so a setting keeps the same colour between two runs of this script; every panel then
    # reads the same series in the same order.
    groups = dict(sorted(groups.items()))
    for label, trials in groups.items():
        seeds = ", ".join(str(t["seed"]) for t in trials)
        print(f"   {label:<22} {len(trials)} trial(s), seeds [{seeds}]")

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    _curve_panel(axes[0][0], groups)
    _decomposition_panel(axes[0][1], groups)
    _position_panel(axes[1][0], groups, bins)
    _tradeoff_panel(axes[1][1], groups)

    figure.suptitle(title or f"Sweep comparison — {os.path.basename(sweep_dir.rstrip('/'))}",
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    # Under the sweep, so its evidence is one directory rather than two; runs.tsv and summary.tsv
    # stay at the sweep root beside it, images here.
    save_dir = save_dir or plot_style.figures_dir(sweep_dir)
    return plot_style.save_figure(figure, save_dir, filename)


def main():
    parser = argparse.ArgumentParser(
        description="Compare a sweep's settings across seeds on one four-panel chart.")
    parser.add_argument("--sweep", required=True, help="Sweep directory containing the run folders.")
    parser.add_argument("--save_dir", default=None,
                        help="Output directory. Defaults to <sweep>/figures/.")
    parser.add_argument("--bins", type=int, default=11,
                        help="Normalized-position bins for panel C (at least 2).")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument("--filename", default="objective_comparison",
                        help="Output filename without extension.")
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    plot_sweep(args.sweep, save_dir=args.save_dir, bins=args.bins,
               title=args.title, filename=args.filename)


if __name__ == "__main__":
    main()
