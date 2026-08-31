"""Turning finished run artifacts into something a human reads — charts and tables both.

Nothing here loads a model or touches a GPU. Every module reads what a completed run already wrote
(``epoch_metrics.csv``, ``generation_eval.json``, ``run_config.json``) plus the processed dataset
JSONs, so any figure can be regenerated long after the run, and a chart can never disagree with the
run it describes.

Three levels, bottom up:

  * ``plot_style``       -- conventions every chart shares: palette, axes furniture, empty panels,
                            saving. Draws no data.
  * chart types          -- ``linegraph_plotter``, ``bargraph_plotter``, ``scatterplot_plotter``.
                            Each exposes a ``draw_*(ax, ...)`` that renders onto an axes owned by
                            somebody else, and a ``plot_*`` that makes a standalone figure.
  * analysis figures     -- ``sweep_plots`` (settings x seeds), ``batch_plots`` (the loop x dataset
                            x context x model matrix), ``confidence_plot`` (per-candidate scores),
                            ``batch_tables`` (the same matrix as Markdown for the README).
                            These choose panels, do their own aggregation, and own the figure.

The split is the usual rule in this project — a capability is added in one place. A new chart type
is one module beside the others and every figure can use it; a new figure composes existing chart
types without touching them.

``batch_tables`` emits Markdown rather than a PNG and still belongs here: it is the same concern
(present a batch's results), reads the same artifacts through the same coverage-aware policy, and
``batch_plots`` builds on its run discovery so the tables and the charts cannot drift apart.
"""
