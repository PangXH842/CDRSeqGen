"""Conventions every CDRSeqGen chart shares: palette, axes furniture, empty panels, saving.

The plotting layer has three levels, and this is the bottom one:

  * plot_style          -- what all charts must agree on (this file). Draws no data.
  * chart types         -- linegraph_plotter, bargraph_plotter, scatterplot_plotter. Each exposes a
                           ``draw_*(ax, ...)`` that renders onto an axes somebody else owns.
  * analysis figures    -- sweep_plots, confidence_plot, record_results, benchmark/plot_comparison.
                           These choose panels and compose chart types; they own the figure.

The split exists so a panel can be reused. Before it, the four plotting sites had grown four
palettes, four grid alphas and three DPIs, and every one of them had reimplemented "there is
nothing to plot here" differently — so two charts of the same quantity did not look like the same
quantity.

The palette is the benchmark comparison's, because those figures back a reported result and their
colours were the ones already chosen deliberately.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Muted and distinguishable in greyscale print. Extended past the benchmark's five so a sweep with
# several settings does not wrap onto a colour it has already used.
PALETTE = ("#2878B5", "#E07A24", "#3A923A", "#8A60A8", "#C44E52",
           "#7F7F7F", "#17A2B8", "#B5852A")

MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")

# Multi-panel figures are physically larger, so they reach the same pixel density at a lower DPI.
# A single-chart PNG passes 300 explicitly.
DEFAULT_DPI = 200

GRID_ALPHA = 0.4
GRID_STYLE = "--"

# Charts for a batch or sweep go in a subdirectory of it, so the folder's own records stay legible:
# runs.tsv, summary.tsv and batch_tables.md at the root, images under figures/. Same split as
# results/ itself, one level down.
FIGURES_DIRNAME = "figures"


def figures_dir(batch_dir):
    """Where a batch's or sweep's charts belong: <batch>/figures/."""
    return os.path.join(batch_dir, FIGURES_DIRNAME)


def color(index):
    """Series colour by position, wrapping at the end of the palette."""
    return PALETTE[index % len(PALETTE)]


def marker(index):
    """Series marker by position; pairs with `color` so a series is identifiable without colour."""
    return MARKERS[index % len(MARKERS)]


def style_axes(ax, *, title=None, xlabel=None, ylabel=None, percent_axis=None,
               legend=False, legend_fontsize=8, grid=True):
    """Apply the shared axes furniture. Every panel in the project goes through here.

    `percent_axis` takes "x", "y" or "both" and formats a 0–1 fraction as a percentage, which is
    what every recovery/diversity axis wants; leaving it None keeps raw units for the axes that are
    not fractions (loss, perplexity, epoch counts).

    `grid` takes True for both axes, or "x"/"y" for one — bar charts want horizontal rules only,
    since a vertical rule through a category has nothing to mark.
    """
    if title:
        ax.set_title(title, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if percent_axis in ("x", "both"):
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    if percent_axis in ("y", "both"):
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if grid:
        ax.grid(True, axis=grid if grid in ("x", "y") else "both",
                linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    if legend:
        ax.legend(fontsize=legend_fontsize)
    return ax


def empty_panel(ax, message):
    """Say why a panel is blank, in the panel, instead of leaving empty axes or aborting the figure.

    A multi-panel figure is usually assembled from several runs, and one missing input should cost
    that panel rather than the whole chart — the same rule `linegraph_plotter` follows for a missing
    CSV column.
    """
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="#555")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def save_figure(figure, save_dir, filename, *, dpi=DEFAULT_DPI, close=True):
    """Write a figure to `save_dir/filename.png`, creating the directory, and report the path."""
    os.makedirs(save_dir, exist_ok=True)
    output = os.path.join(save_dir, f"{filename}.png")
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(figure)
    print(f"✅ Plot successfully saved as: {output}")
    return output
