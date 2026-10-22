"""CDRSeqGen — modular antibody CDR-generation library.

The package separates complete generation methods from shared infrastructure:
  * methods/ — method-owned models, objectives, generation and evaluation;
               common/ supplies shared workflow, logging and run recording.
  * datasets/ — processed-data loading, tokenization and batching.
  * eval/ — architecture-neutral sequence metrics and generated-output reporting.
  * visualization/ — result tables and plots.

The root entrypoints are ``train.py``, ``generate.py``, ``evaluate.py``, and
``fit_mlm_length.py``; they import their implementations from this package.
"""
