"""Render saved CDRSeqGen run artifacts into charts and tabular reports.

Nothing here loads a model or touches a GPU. Modules read saved run artifacts
(``epoch_metrics.csv``, ``generation_eval.json``, ``run_config.json``) and processed dataset JSONs.
``training_figure`` also exports ``epoch_metrics.csv`` from Trainer log history so training curves
can be regenerated offline at any time.

Architecture (bottom up):
  * Style layer      -- ``plot_style``: shared palette, markers, axes furniture, empty panels,
                        and figure saving. Draws no data.
  * Chart primitives -- ``plot_bar``, ``plot_line``, ``plot_scatter``: composable
                        ``draw_*(ax, ...)`` renderers plus standalone ``plot_*`` wrappers.
  * Analysis figures -- ``training_figure`` (per-run training curves), ``batch_figure`` (per-dataset
                        batch dashboards), ``sweep_figure`` (multi-seed sweep dashboard),
                        ``confidence_figure`` (top-k selection and Spearman rho diagnostics).
  * Tabular reports  -- ``batch_table`` (Markdown tables for a batch) and ``sweep_table``
                        (ranked ``summary.tsv`` and text report for a sweep).

Notes:
  * New chart primitives belong beside ``plot_bar``/``plot_line``/``plot_scatter`` so every figure
    can reuse them; new analysis figures compose existing primitives without modifying them.
  * ``batch_table`` and ``batch_figure`` share run discovery and coverage-aware recovery selection
    so tabular and visual reports stay consistent.
"""
