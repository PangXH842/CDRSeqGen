"""Data layer — all data *code* lives here:

  * dataset.py        — runtime: processed JSON -> tensors (tokenizers, TokenizerAdapter,
                        AntibodyInfillingDataset, collator). Runs every batch.
  * preprocess/       — offline: raw downloads -> processed JSON, plus the fetch scripts. Runs once,
                        from the repo root: ``python -m cdrseqgen.data.preprocess.oas_paired``.

The data *files* deliberately stay OUTSIDE the package, in the gitignored top-level ``dataset/data/``
(~4.4GB raw+processed). A package should hold importable code, not a blob that IDE indexers, linters
and ``grep -r`` have to crawl — and the configs' ``data_path`` plus scripts/run.sh already point
there, so it stays put.

The rule, in one line: **cdrseqgen/data/ is the code that produces and consumes the data;
dataset/data/ is the data.**
"""
