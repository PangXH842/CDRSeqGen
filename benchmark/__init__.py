"""Reproducible cross-model benchmark harness for CDRSeqGen.

Organized into three layers:
  core       _common.py, manifest.py, run.py, scorer.py, blind_check.py
  adapters/  model-specific samplers (cdrseqgen, iglm, igcraft, retrieval) and registry.py
  analysis/  post-hoc statistics, clustering/empirical baseline, followup/extend/variability,
             leakage audit, and matched comparison plotting
"""
