"""Grouped bar charts for comparing multiple series across shared categories.

Types & functions:
  * BarSeries         -- one series' bar heights across categories, with optional error bars.
  * draw_grouped_bars -- render grouped bars onto an existing axes (composable entry point).
  * plot_grouped_bars -- create a standalone figure, render grouped bars, and save as PNG.

Notes:
  * Missing (``None`` or non-finite) values are skipped rather than plotted as zero, so unmeasured
    categories are never confused with zero recovery.
  * Series with no finite values reserve no horizontal slot while remaining series keep their
    original palette index for stable cross-chart colouring.
  * Zero or missing errors omit error bars rather than drawing misleading zero-length caps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from cdrseqgen.visualization import plot_style
import matplotlib.pyplot as plt


# --- Data Series ---------------------------------------------------------------------------------

@dataclass
class BarSeries:
    """One series across the chart's categories. `errors` is the half-length of the error bar."""

    label: str
    values: list
    errors: list | None = None
    color: str | None = None

    def __post_init__(self):
        if self.errors is not None and len(self.errors) != len(self.values):
            raise ValueError(
                f"BarSeries '{self.label}': {len(self.errors)} errors for {len(self.values)} values"
            )


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


# --- Composable Axes Renderer (used by multi-panel figures) --------------------------------------

def draw_grouped_bars(ax, categories, series, *, value_labels=True, value_format="{:.2f}",
                      color_offset=0, bar_span=0.8):
    """Draw `series` as grouped bars over `categories`. Returns True if anything was drawn.

    Error bars are omitted where the error is zero or missing: a single-seed setting has no measured
    spread, and a zero-length bar would claim a precision one run cannot support.
    """
    series = list(series)
    if not series:
        return False
    for entry in series:
        if len(entry.values) != len(categories):
            raise ValueError(
                f"BarSeries '{entry.label}' has {len(entry.values)} values for "
                f"{len(categories)} categories"
            )

    # A slot is reserved only for a series that actually draws something. Callers pass one series
    # per REGISTERED family, so a batch that holds three of the four left the absent family's slot
    # empty -- a bar-width hole inside the group that reads as a missing bar rather than an absent
    # family. Colour still comes from the ORIGINAL index, so a family keeps its colour whatever else
    # the batch contains; filtering that too would recolour every method as batches gained arms.
    drawable = [(index, entry) for index, entry in enumerate(series)
                if any(_finite(value) for value in entry.values)]
    if not drawable:
        return False

    width = bar_span / len(drawable)
    drew = False
    for slot, (index, entry) in enumerate(drawable):
        offset = (slot - (len(drawable) - 1) / 2) * width
        positions, heights, errors = [], [], []
        for category_index, value in enumerate(entry.values):
            if not _finite(value):
                continue
            positions.append(category_index + offset)
            heights.append(float(value))
            error = entry.errors[category_index] if entry.errors else None
            errors.append(float(error) if _finite(error) and error else 0.0)
        if not positions:
            continue
        drew = True
        bars = ax.bar(
            positions, heights, width=width,
            color=entry.color or plot_style.color(index + color_offset),
            label=entry.label,
            yerr=errors if any(errors) else None,
            capsize=3 if any(errors) else 0,
        )
        if value_labels:
            ax.bar_label(bars, labels=[value_format.format(h) for h in heights],
                         padding=2, fontsize=8)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    return drew


# --- Standalone Figure Wrapper (save_dir/filename.png) -------------------------------------------

def plot_grouped_bars(categories, series, *, title=None, ylabel=None, percent_axis=None,
                      save_dir=".", filename="bars", figsize=(9, 6), dpi=300):
    """Standalone grouped-bar chart: one figure, one axes, saved to `save_dir/filename.png`."""
    figure, ax = plt.subplots(figsize=figsize)
    if not draw_grouped_bars(ax, categories, series):
        plot_style.empty_panel(ax, "No values to plot")
        plot_style.save_figure(figure, save_dir, filename, dpi=dpi)
        return None
    plot_style.style_axes(ax, title=title, ylabel=ylabel, percent_axis=percent_axis,
                          legend=True, grid="y")
    return plot_style.save_figure(figure, save_dir, filename, dpi=dpi)
