"""Evaluation — both halves of "how good is this model?".

  * recovery.py   -- recovery-side: did we reproduce the native loop (compute_metrics for the Trainer,
                     whole-span vs anchor-excluded core, core coverage)
  * metrics.py    -- design-side: diversity, novelty, unique fraction, per-position entropy
  * sampling.py   -- decoders (independent / iterative Mask-Predict) + pseudo-perplexity scoring
  * generation.py -- the generative-eval core shared by train.py and gen_eval.py

The split matters: a model can score 0.99 recovery while emitting one sequence with total certainty
(measured on H1), so the two halves can rank models differently. See README Findings.
"""
