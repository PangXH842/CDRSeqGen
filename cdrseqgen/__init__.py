"""CDRSeqGen — modular antibody CDR-infilling library.

Source is grouped by concern so the sequence→structure (Phase 2) expansion has a home:
  * data/   — tokenizers, the infilling Dataset, collator.
  * models/ — the backbone registry and the from-scratch transformer.
  * eval/   — decoders/sampling, generative metrics, and plotting.

The entrypoint scripts (train.py, predict.py, eval_recovery.py) live at the repo root and are run from
there (e.g. ``python train.py configs/h3_scratch.json``); they import this package as ``cdrseqgen.*``.
"""
