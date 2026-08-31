"""The visual counterpart of `batch_tables`: a training batch -> one dashboard PNG per dataset.

`batch_tables` renders the (loop x dataset x context x model) matrix as Markdown for the README.
This renders the same runs, read through the same coverage-aware policy, as charts — the shape of
the matrix is easier to see than to read, particularly the H3 gap and the anchor inflation.

Usage:
    python -m cdrseqgen.visualization.batch_plots --batch results/runs/20260818_192606
    python -m cdrseqgen.visualization.batch_plots --batch <dir> --dataset OAS

Panels (per dataset):
  * A. best-epoch recovery per loop, scratch against ESM-2 -- Findings §1/§2
  * B. whole-span against core and anchor  -- the anchor inflation of Findings §3, which is why the
       headline whole-span number overstates what the model designs
  * C. the context effect per loop         -- Findings §4, drawn as a signed delta so "no effect"
       is a bar at zero rather than two bars a reader has to subtract by eye
  * D. recovery against judged perplexity  -- Findings §7, where the two disagree

**One figure per dataset, never a shared axis.** OAS and SAbDab2 are evaluated on different test
sets, so their accuracies are not comparable; putting them on one axis is the mistake the `_OAS` /
`_SD2` run tags exist to prevent. This module enforces that by splitting on the dataset tag rather
than by asking the caller to remember.

Reads only what a finished run writes, through `batch_tables.discover_runs` / `recovery_summary`,
so the tables and the charts cannot disagree about which epoch or which metric a run reports.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from cdrseqgen.visualization import plot_style
from cdrseqgen.visualization.bargraph_plotter import BarSeries, draw_grouped_bars
from cdrseqgen.visualization.batch_tables import _epoch_metrics, discover_runs, recovery_summary
from cdrseqgen.visualization.scatterplot_plotter import ScatterSeries, draw_scatter

GENERATION_EVAL_FILENAME = "generation_eval.json"

LOOP_ORDER = ("h1", "h2", "h3", "l1", "l2", "l3")
FAMILIES = ("scratch", "esm2")
FAMILY_LABELS = {"scratch": "scratch", "esm2": "ESM-2"}


def _loops_present(runs, dataset):
    return [loop for loop in LOOP_ORDER
            if any(key[0] == loop and key[1] == dataset for key in runs)]


def _contexts_present(runs, dataset):
    """Context tags seen for a dataset, bare context first so a delta has a baseline to subtract."""
    tags = {key[2] for key in runs if key[1] == dataset}
    return sorted(tags, key=len)


def _summary(runs, key, cache):
    if key not in cache:
        run_dir = runs.get(key)
        cache[key] = recovery_summary(run_dir) if run_dir else None
    return cache[key]


def _generation_summary(run_dir):
    path = os.path.join(run_dir, GENERATION_EVAL_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle).get("summary", {})


def _recovery_panel(ax, runs, dataset, cache, context):
    loops = _loops_present(runs, dataset)
    series = []
    for family in FAMILIES:
        values = [
            (lambda s: s["primary_value"] if s else None)(
                _summary(runs, (loop, dataset, context, family), cache))
            for loop in loops
        ]
        series.append(BarSeries(FAMILY_LABELS[family], values))
    if not draw_grouped_bars(ax, [loop.upper() for loop in loops], series):
        return plot_style.empty_panel(ax, "No recovery metrics")
    plot_style.style_axes(ax, title="A. Best-epoch recovery per loop",
                          ylabel="Recovery (coverage-aware)", percent_axis="y",
                          legend=True, grid="y")


def _breakdown_panel(ax, runs, dataset, cache, context):
    """Whole-span against core and anchor, for one model family.

    Only scratch is drawn: showing both families doubles the bars without changing the point, which
    is the gap between the three metrics rather than the gap between models.
    """
    loops = _loops_present(runs, dataset)
    metrics = [("whole_accuracy", "whole-span"), ("core_accuracy", "core"), ("anchor", "anchor")]
    series = []
    for key, label in metrics:
        values = []
        for loop in loops:
            summary = _summary(runs, (loop, dataset, context, "scratch"), cache)
            if not summary:
                values.append(None)
            elif key == "anchor":
                # Not recorded in the summary; recover it from the identity that whole-span is the
                # token-weighted blend of core and anchor only when coverage is full, so read it
                # from the CSV instead of inferring it.
                values.append(_anchor_accuracy(runs.get((loop, dataset, context, "scratch"))))
            else:
                values.append(summary.get(key))
        series.append(BarSeries(label, values))
    if not draw_grouped_bars(ax, [loop.upper() for loop in loops], series, value_labels=False):
        return plot_style.empty_panel(ax, "No recovery breakdown")
    plot_style.style_axes(ax, title="B. Anchor inflation (scratch)", ylabel="Recovery",
                          percent_axis="y", legend=True, grid="y")


def _anchor_accuracy(run_dir):
    """Anchor recovery at the run's best core epoch, or None when the column is absent."""
    if not run_dir:
        return None
    frame = _epoch_metrics(run_dir)
    if frame is None or "eval_anchor_accuracy" not in frame.columns:
        return None
    column = "eval_core_accuracy" if "eval_core_accuracy" in frame.columns else "eval_accuracy"
    best = frame.loc[frame[column].idxmax()]
    return float(best["eval_anchor_accuracy"])


def _context_pair(runs, loop, dataset):
    """(baseline, augmented) context tags for one loop, or None when the batch has no pair.

    The tag is not global. A heavy target's partner chain is the light chain (`_LC`) and a light
    target's is the heavy chain (`_HC`), so a single batch-wide tag matches only half the loops —
    and on SAbDab2 no context-free run exists at all, where the available pair is instead partner
    chain against partner chain + antigen. Taking the two shortest tags for *this* loop covers all
    three cases without naming any of them.
    """
    tags = sorted({key[2] for key in runs if key[0] == loop and key[1] == dataset},
                  key=lambda tag: (len(tag), tag))
    return (tags[0], tags[1]) if len(tags) >= 2 else None


def _context_panel(ax, runs, dataset, cache):
    """Signed recovery change from adding context, per loop and family."""
    loops = [loop for loop in _loops_present(runs, dataset) if _context_pair(runs, loop, dataset)]
    if not loops:
        return plot_style.empty_panel(ax, "No paired context runs in this batch")

    series = []
    for family in FAMILIES:
        values = []
        for loop in loops:
            base, augmented = _context_pair(runs, loop, dataset)
            bare = _summary(runs, (loop, dataset, base, family), cache)
            with_context = _summary(runs, (loop, dataset, augmented, family), cache)
            values.append(with_context["primary_value"] - bare["primary_value"]
                          if bare and with_context else None)
        series.append(BarSeries(FAMILY_LABELS[family], values))

    # Name the added context per loop rather than in the title, since it differs across loops.
    categories = []
    for loop in loops:
        base, augmented = _context_pair(runs, loop, dataset)
        added = augmented[len(base):].strip("_") or "ctx"
        categories.append(f"{loop.upper()}\n+{added}")

    if not draw_grouped_bars(ax, categories, series, value_format="{:+.3f}"):
        return plot_style.empty_panel(ax, "No paired context runs")
    ax.axhline(0, color="#333", linewidth=1)
    # The seed noise floor on a per-loop delta (README Findings §4). A bar inside this band is not
    # distinguishable from noise, which is the whole reading of that finding.
    ax.axhspan(-0.006, 0.006, color="#999", alpha=0.18, zorder=0)
    plot_style.style_axes(ax, title="C. Effect of adding context", ylabel="Δ recovery",
                          legend=True, grid="y")


def _design_panel(ax, runs, dataset, context):
    """Recovery against judged perplexity: the two need not agree (Findings §7)."""
    series = []
    for family in FAMILIES:
        xs, ys, labels = [], [], []
        for loop in LOOP_ORDER:
            run_dir = runs.get((loop, dataset, context, family))
            if not run_dir:
                continue
            summary = _generation_summary(run_dir)
            if not summary:
                continue
            recovery = summary.get("aar_length_matched_mean", summary.get("aar_mean"))
            judge = summary.get("judge_perplexity_mean")
            if recovery is None or judge is None:
                continue
            xs.append(recovery)
            ys.append(judge)
            labels.append(loop.upper())
        if xs:
            series.append(ScatterSeries(FAMILY_LABELS[family], xs, ys))
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    if not draw_scatter(ax, series):
        return plot_style.empty_panel(ax, "No generative eval in this batch")
    plot_style.style_axes(ax, title="D. Recovery against judged plausibility",
                          xlabel="Free-running recovery", ylabel="Judge perplexity  ↓ better",
                          percent_axis="x", legend=True)


def plot_batch_dataset(runs, dataset, batch_dir, save_dir=None):
    """One dashboard for one dataset. Returns the written path, or None when nothing was plotted."""
    contexts = _contexts_present(runs, dataset)
    if not contexts:
        return None
    cache = {}
    primary = contexts[0]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    _recovery_panel(axes[0][0], runs, dataset, cache, primary)
    _breakdown_panel(axes[0][1], runs, dataset, cache, primary)
    _context_panel(axes[1][0], runs, dataset, cache)
    _design_panel(axes[1][1], runs, dataset, primary)

    batch_name = os.path.basename(batch_dir.rstrip("/"))
    figure.suptitle(f"{dataset} batch — {batch_name}", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    # Under the batch it describes, so a round is self-contained; batch_tables.md stays at the
    # batch root beside it, images here.
    save_dir = save_dir or plot_style.figures_dir(batch_dir)
    return plot_style.save_figure(figure, save_dir, f"batch_{dataset.lower()}")


def plot_batch(batch_dir, save_dir=None, dataset=None):
    """One dashboard per dataset in the batch — never one chart spanning two test sets."""
    runs = discover_runs(batch_dir)
    if not runs:
        print(f"🚫 No parseable runs under {batch_dir}")
        return []
    datasets = sorted({key[1] for key in runs}) if dataset is None else [dataset]
    written = []
    for name in datasets:
        if not any(key[1] == name for key in runs):
            print(f"⚠️ No {name} runs in this batch — skipped")
            continue
        count = sum(1 for key in runs if key[1] == name)
        print(f"   {name}: {count} runs")
        path = plot_batch_dataset(runs, name, batch_dir, save_dir=save_dir)
        if path:
            written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Chart a training batch, one dashboard per dataset.")
    parser.add_argument("--batch", required=True, help="Batch directory holding the run folders.")
    parser.add_argument("--dataset", default=None,
                        help="Only this dataset tag (OAS, SD2). Default: every one present.")
    parser.add_argument("--save_dir", default=None,
                        help="Output directory. Defaults to <batch>/figures/.")
    args = parser.parse_args()
    plot_batch(args.batch, save_dir=args.save_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
