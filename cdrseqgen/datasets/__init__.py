"""Data layer — all data *code* lives here:

  * dataset_common.py — runtime, method-neutral: tokenizers, TokenizerAdapter, framework/target
                        split, context injection. Runs every batch.
  * dataset_{mlm,ar,s2s}.py — one view per method on that shared base.
  * preprocess/       — offline: raw downloads -> processed JSON, plus the fetch scripts. Runs once,
                        from the repo root: ``python -m cdrseqgen.datasets.preprocess.oas_paired``.

The data *files* deliberately stay OUTSIDE the package, in the gitignored top-level ``data/``
(~6.5GB raw+processed). A package should hold importable code, not a blob that IDE indexers, linters
and ``grep -r`` have to crawl — and the configs' ``data_path`` plus scripts/run.sh already point
there, so it stays put.

The rule, in one line: **cdrseqgen/datasets/ is the code that produces and consumes the data;
data/ is the data.**
"""
