"""Contract tests for the shared plotting layer.

These do not check that a chart looks right — that needs eyes. They check the guarantees the
analysis figures rely on when they compose panels: that a primitive reports whether it drew
anything (so a caller can substitute an empty panel instead of shipping blank axes), that
mismatched inputs fail loudly at construction rather than producing a silently wrong chart, and
that an unmeasured value is skipped rather than drawn as a zero.

The last one is the science-critical case: on a recovery axis a 0.0 bar and an absent bar look
identical and mean opposite things.
"""
import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer

from cdrseqgen.visualization import plot_style
from cdrseqgen.visualization.bargraph_plotter import BarSeries, draw_grouped_bars
from cdrseqgen.visualization.batch_plots import _context_pair
from cdrseqgen.visualization.linegraph_plotter import LineSeries, draw_lines
from cdrseqgen.visualization.scatterplot_plotter import ScatterSeries, draw_scatter
from cdrseqgen.visualization.sweep_plots import SWEEP_SUBDIRS


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


if __name__ == "__main__":
    unittest.main()
