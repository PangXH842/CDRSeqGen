"""Shared visual conventions for CDRSeqGen charts: palette, axes furniture, empty panels, saving.

Architecture:
  * Style layer      -- ``plot_style`` (this module): shared visual defaults; draws no data.
  * Chart primitives -- ``plot_bar``, ``plot_line``, ``plot_scatter``: composable ``draw_*`` helpers.
  * Analysis figures -- ``batch_figure``, ``sweep_figure``, ``confidence_figure``,
                        ``training_figure``, and ``benchmark/analysis/plot_comparison``.

Functions:
  * figures_dir -- standard ``<batch>/figures/`` output path for a batch or sweep.
  * color       -- palette colour by series index (wrapping cleanly).
  * marker      -- marker shape by series index, paired with ``color`` for print readability.
  * style_axes  -- apply wrapped bold titles, axis labels, percent formatting, grid, and legend.
  * empty_panel -- render a clean explanatory placeholder when a panel's input is missing.
  * save_figure -- write PNG/PDF outputs with consistent DPI and tight bounding boxes.

Notes:
  * Configures matplotlib's non-interactive ``Agg`` backend and TrueType font embedding (type 42)
    on import so every chart works cleanly on headless servers and exports editable vector PDFs.
  * The palette matches the benchmark comparison figures so reported series look consistent across
    training, sweep, and benchmark charts.
"""
from __future__ import annotations

import textwrap
import os

# --- Backend, Font & Shared Style Constants ------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.setdefault("pdf.fonttype", 42)
plt.rcParams.setdefault("ps.fonttype", 42)

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


# --- Output Paths, Palette & Markers -------------------------------------------------------------

def figures_dir(batch_dir):
    """Where a batch's or sweep's charts belong: <batch>/figures/."""
    return os.path.join(batch_dir, FIGURES_DIRNAME)


def color(index):
    """Series colour by position, wrapping at the end of the palette."""
    return PALETTE[index % len(PALETTE)]


def marker(index):
    """Series marker by position; pairs with `color` so a series is identifiable without colour."""
    return MARKERS[index % len(MARKERS)]


# --- Axes Styling, Empty Panels & Saving ---------------------------------------------------------

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
        # Matplotlib neither wraps nor clips a title, so an over-long one runs straight into the
        # NEXT panel's — which is what an appended caveat ("... ⚠ masked vs sequential NOT
        # comparable") did to every two-column dashboard. Break the caveat onto its own line, then
        # wrap whatever is still long. A short title passes through untouched.
        if "⚠" in title:
            head, _, caveat = title.partition("⚠")
            title = f"{head.rstrip()}\n⚠{caveat}"
        title = "\n".join(textwrap.fill(line, width=58) for line in title.split("\n"))
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
    that panel rather than the whole chart — the same rule `plot_line` follows for a missing
    CSV column.
    """
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="#555")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def save_figure(figure, save_dir, filename, *, dpi=DEFAULT_DPI, close=True, formats=("png",)):
    """Write a figure to `save_dir/filename.<ext>` (default PNG), creating the directory.

    Returns the first written path (preserving the PNG path for callers expecting a single string).
    """
    os.makedirs(save_dir, exist_ok=True)
    written = []
    for ext in formats:
        ext = ext.lstrip(".")
        output = os.path.join(save_dir, f"{filename}.{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext != "pdf":
            kwargs["dpi"] = dpi
        figure.savefig(output, **kwargs)
        written.append(output)
    if close:
        plt.close(figure)
    print(f"✅ Plot successfully saved as: {', '.join(written)}")
    return written[0]
