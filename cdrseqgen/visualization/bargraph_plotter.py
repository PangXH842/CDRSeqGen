"""Grouped bar charts: several series compared across the same categories.

  * BarSeries        -- one series' heights across every category, with optional error bars
  * draw_grouped_bars -- render onto an axes somebody else owns (the composable entry point)
  * plot_grouped_bars -- create a figure, draw, save (standalone use)

The shape this exists for is "N settings x M metrics": objectives against core@L/anchor@L, models
against a recovery decomposition, runs against length buckets. Each of those was previously bar code
inlined into whichever figure needed it.

A missing value is passed as None and is skipped rather than drawn as a zero, because on a recovery
axis a zero bar and an unmeasured bar look identical and mean opposite things.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from cdrseqgen.visualization import plot_style


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
