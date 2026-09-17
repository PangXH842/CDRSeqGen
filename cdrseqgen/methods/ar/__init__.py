"""Autoregressive span infilling — the causal sibling of ``methods/mlm``.

Why it exists: three Phase-1 findings converge on the masked-LM *objective class* rather than
anything tunable inside it — iterative and independent decoding are within noise because the learned
marginals are near-independent (Step 4), the model's own confidence barely tracks its correctness
(§12), and re-weighting the loss across loop positions moves nothing (§13). The one axis where IgLM
— autoregressive — beats CDRSeqGen decisively is judged plausibility (§11).

  * models/transformer.py -- from-scratch causal decoder, model_type "scratch"
  * backbones.py          -- causal backbone registry, resolve(model_type) -> CausalBackbone

The data view lives in :mod:`cdrseqgen.datasets.dataset_ar`, beside the masked one, because
``cdrseqgen/datasets/`` holds all data code.
"""
