"""Scatter charts: where runs sit against each other on two axes at once.

  * ScatterSeries  -- one series' points, optionally annotated with its label
  * draw_scatter   -- render onto an axes somebody else owns (the composable entry point)
  * plot_scatter   -- create a figure, draw, save (standalone use)

Used for the operating-point charts — recovery against diversity, judged plausibility against
diversity — where the question is a trade-off rather than a trend, and a bar or a line would imply
an ordering the x-axis does not have.

Passing several points in one series (one per seed) rather than their mean is deliberate: the
scatter of replicates is the reader's only cue for whether a gap between two settings is real.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt

from cdrseqgen.visualization import plot_style


@dataclass
class ScatterSeries:
    """One series. `x` and `y` are parallel; each pair is one trial, not one aggregate."""

    label: str
    x: list
    y: list
    color: str | None = None

    def __post_init__(self):
        if len(self.x) != len(self.y):
            raise ValueError(
                f"ScatterSeries '{self.label}': {len(self.x)} x-values against {len(self.y)} y-values"
            )


def _pairs(entry):
    return [
        (float(x), float(y)) for x, y in zip(entry.x, entry.y)
        if isinstance(x, (int, float)) and isinstance(y, (int, float))
        and math.isfinite(x) and math.isfinite(y)
    ]


def draw_scatter(ax, series, *, size=70, annotate=False, color_offset=0):
    """Draw `series` as points. Returns True if anything was drawn.

    `annotate` labels each series next to its first point, for charts with few enough points that a
    legend would be the longer way to read them.
    """
    drew = False
    for index, entry in enumerate(series):
        points = _pairs(entry)
        if not points:
            continue
        drew = True
        colour = entry.color or plot_style.color(index + color_offset)
        xs, ys = zip(*points)
        ax.scatter(xs, ys, s=size, color=colour, edgecolor="white", linewidth=0.8,
                   zorder=3, label=entry.label)
        if annotate:
            ax.annotate(entry.label, points[0], xytext=(7, 6), textcoords="offset points",
                        fontsize=9, color=colour)
    return drew


def plot_scatter(series, *, title=None, xlabel=None, ylabel=None, percent_axis=None,
                 annotate=False, save_dir=".", filename="scatter", figsize=(8, 6), dpi=300):
    """Standalone scatter chart: one figure, one axes, saved to `save_dir/filename.png`."""
    figure, ax = plt.subplots(figsize=figsize)
    if not draw_scatter(ax, series, annotate=annotate):
        plot_style.empty_panel(ax, "No points to plot")
        plot_style.save_figure(figure, save_dir, filename, dpi=dpi)
        return None
    plot_style.style_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel,
                          percent_axis=percent_axis, legend=not annotate)
    return plot_style.save_figure(figure, save_dir, filename, dpi=dpi)
