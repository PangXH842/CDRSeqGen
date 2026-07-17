"""Evaluation: decoders + sampling (sampling.py), generative sequence metrics (metrics.py),
and training-curve plotting (linegraph_plotter.py).

Note: the HF-Trainer token-recovery metrics (compute_metrics, anchor/core split) currently live in
the root train.py so eval_recovery.py can share them; they'd move here if that sharing grows.
"""
