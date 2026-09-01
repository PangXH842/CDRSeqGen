"""Machinery every generation method needs, so no two methods can drift apart on it.

  * arguments.py  -- base ModelArguments / DataArguments / GenerationArguments, config+CLI parsing,
                     and run-directory setup. A method subclasses the bases and adds only its own.
  * sampling.py   -- logit shaping (temperature/top-k/top-p) and model-output normalisation
  * callbacks.py  -- Trainer callbacks: metric logging always, early stopping on request

The membership test: **things a method needs to PRODUCE output**. Turning records into model inputs
belongs in ``data/dataset_common.py``; measuring outputs belongs in ``cdrseqgen/eval/``.

Everything here started life duplicated and was consolidated after a second method existed to check
it against — which is how ``build_callbacks`` was found to have silently lost ``LogMetricsCallback``
on the causal side, reintroducing a `train.log` regression fixed in July.
"""
from cdrseqgen.methods.common.callbacks import build_callbacks  # noqa: F401
from cdrseqgen.methods.common.sampling import (  # noqa: F401
    NEG_INF, ban_tokens, logits_of, warp_logits)
