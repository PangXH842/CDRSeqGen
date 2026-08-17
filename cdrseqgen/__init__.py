"""CDRSeqGen — modular antibody CDR-generation library.

The package separates complete generation methods from shared infrastructure:
  * methods/ — method-owned models, objectives, generation, and native evaluation.
  * data/    — processed-data loading, tokenization, and batching.
  * eval/    — architecture-neutral sequence metrics.
  * train/   — shared run logging and artifact reporting.
  * utils/   — result tables and plots.

The root entrypoints are ``train.py``, ``generate.py``, ``evaluate.py``, and
``fit_mlm_length.py``; they import their implementations from this package.
"""
