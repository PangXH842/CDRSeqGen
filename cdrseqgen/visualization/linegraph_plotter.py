"""Line charts, and the training-curve reader that is their commonest caller.

  * LineSeries        -- one line: x, y, a label, and an optional band around it
  * draw_lines        -- render lines onto an axes somebody else owns (the composable entry point)
  * plot_epoch_curves -- read epoch_metrics.csv files and draw them as a standalone PNG
  * slugify           -- filename-safe version of a title

`plot_epoch_curves` covers both comparison modes, because every line is the same triple
(run_dir, column, legend_label):
  * one metric across many runs   -- [(run_a, metric, "A"), (run_b, metric, "B"), ...]
  * many metrics from one run     -- [(run, "eval_accuracy", "whole"), (run, "eval_core_accuracy",
                                      "core"), ...]  <- the whole/core/anchor breakdown

Called three ways: train.py auto-plots every run's curves at the end of training; this file is a CLI
for ad-hoc cross-run charts (python -m cdrseqgen.visualization.linegraph_plotter --help); and `draw_lines`
is a panel inside the multi-panel analysis figures (sweep_plots).

Note:
  * A missing CSV or column is reported and skipped rather than aborting the chart, so one bad
    path in a multi-run comparison doesn't lose the other lines.
"""
from __future__ import annotations

import os
import re
import argparse
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt

from cdrseqgen.visualization import plot_style


@dataclass
class LineSeries:
    """One line. `band` is an optional (lower, upper) pair parallel to `y`, drawn as a shaded range.

    The band carries the spread of whatever `y` averages — seeds, usually — so a reader can see
    whether two means are separated by more than their replicates.
    """

    x: list
    y: list
    label: str
    band: tuple | None = None
    color: str | None = None
    linestyle: str = "-"

    def __post_init__(self):
        if len(self.x) != len(self.y):
            raise ValueError(
                f"LineSeries '{self.label}': {len(self.x)} x-values against {len(self.y)} y-values"
            )


def draw_lines(ax, series, *, markers=True, color_offset=0):
    """Draw `series` onto `ax`. Returns True if anything was drawn.

    A band is drawn beneath its line at low alpha; where a band would be zero-width it is the
    caller's job to omit it, since a collapsed band reads as agreement rather than as one replicate.
    """
    drew = False
    for index, entry in enumerate(series):
        if not len(entry.x):
            continue
        drew = True
        colour = entry.color or plot_style.color(index + color_offset)
        if entry.band is not None:
            lower, upper = entry.band
            ax.fill_between(entry.x, lower, upper, color=colour, alpha=0.18, linewidth=0)
        ax.plot(
            entry.x, entry.y,
            color=colour,
            marker=plot_style.marker(index + color_offset) if markers else None,
            markersize=4, linewidth=2, linestyle=entry.linestyle, label=entry.label,
        )
    return drew


def slugify(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def read_epoch_series(series):
    """Turn (run_dir, column, label) triples into LineSeries, skipping what cannot be read."""
    csv_cache = {}  # a run_dir reused across lines (the breakdown case) is read once
    lines = []
    for run_dir, column, label in series:
        csv_path = os.path.join(run_dir, "epoch_metrics.csv")
        if csv_path not in csv_cache:
            csv_cache[csv_path] = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
        frame = csv_cache[csv_path]

        if frame is None:
            print(f"❌ Could not find epoch_metrics.csv in {run_dir}")
        elif column not in frame.columns:
            print(f"⚠️ Metric '{column}' not found in {csv_path}")
        else:
            lines.append(LineSeries(list(frame["epoch"]), list(frame[column]), label))
    return lines


def plot_epoch_curves(series, title="Comparison", ylabel="Value", save_dir=".", filename="plot"):
    """Plot any number of epoch-vs-value lines (from epoch_metrics.csv files) on one chart"""
    figure, ax = plt.subplots(figsize=(10, 6))
    if not draw_lines(ax, read_epoch_series(series)):
        print("🚫 No valid data found to plot. Check your folder paths and metric names!")
        plt.close(figure)
        return

    plot_style.style_axes(ax, title=title, xlabel="Epoch", ylabel=ylabel, legend=True,
                          legend_fontsize=10)
    # A single-chart PNG is physically smaller than a multi-panel figure, so it needs the higher
    # density to stay legible when dropped into a document at full width.
    plot_style.save_figure(figure, save_dir, filename, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamically compare training runs / metrics on one chart.")
    # Each --run is one line: LABEL FOLDER [METRIC]. METRIC is optional and defaults to --metric,
    # so same-metric comparisons stay terse (metric once) while a per-run METRIC unlocks breakdowns —
    # point several --run at the SAME folder with different metrics to plot whole/core/anchor together.
    parser.add_argument("--run", action="append", nargs="+", metavar="LABEL FOLDER [METRIC]", required=True,
                        help="A line to plot: LABEL FOLDER [METRIC]. Repeat per line. "
                             "Same metric across runs: --run scratch ./A --run esm2 ./B (with --metric). "
                             "Breakdown of one run: --run whole ./R eval_accuracy --run core ./R eval_core_accuracy ...")
    parser.add_argument("--metric", type=str, default="eval_core_accuracy",
                        help="Default column when a --run omits its METRIC: eval_loss, eval_accuracy, "
                             "eval_core_accuracy (default), eval_anchor_accuracy, ...")
    parser.add_argument("--ylabel", type=str, default=None, help="Y-axis label. Defaults to the metric name, or 'Value' when metrics are mixed.")
    parser.add_argument("--title", type=str, default=None, help="Optional custom title for the plot")
    parser.add_argument("--filename", type=str, default=None, help="Optional custom filename for the saved plot (without extension)")
    parser.add_argument("--save_dir", type=str, default="results/figures/training", help="Directory to save the plot (created if missing). Defaults to results/pngs.")
    args = parser.parse_args()

    # One (folder, metric, label) line per --run; a run's METRIC falls back to the global --metric
    series = []
    for entry in args.run:
        if len(entry) == 2:
            label, folder, metric = entry[0], entry[1], args.metric
        elif len(entry) == 3:
            label, folder, metric = entry
        else:
            parser.error(f"--run takes 'LABEL FOLDER [METRIC]' (2 or 3 values), got {len(entry)}: {entry}")
        series.append((folder, metric, label))

    # A single shared metric drives the default title / ylabel / filename; mixed metrics (a breakdown)
    # have no one metric to name, so fall back to generic labels.
    metrics_used = {metric for _, metric, _ in series}
    uniform_metric = next(iter(metrics_used)) if len(metrics_used) == 1 else None

    if args.ylabel:
        ylabel = args.ylabel
    elif uniform_metric:
        ylabel = uniform_metric.replace("_", " ").title()
    else:
        ylabel = "Value"

    if args.title:
        plot_title = args.title
    elif uniform_metric:
        plot_title = f"{uniform_metric.replace('_', ' ').title()} Comparison"
    else:
        plot_title = "Metric breakdown"

    # Keep filename in sync with a custom title unless --filename overrides it explicitly
    if args.filename:
        plot_filename = args.filename
    elif args.title:
        plot_filename = slugify(args.title)
    elif uniform_metric:
        plot_filename = f"{uniform_metric}_curve"
    else:
        plot_filename = "breakdown"

    plot_epoch_curves(
        series,
        title=plot_title,
        ylabel=ylabel,
        save_dir=args.save_dir,
        filename=plot_filename,
    )
