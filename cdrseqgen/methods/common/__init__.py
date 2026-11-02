"""Machinery every generation method needs, so no two methods can drift apart on it.

  * arguments.py  -- base ModelArguments / DataArguments / GenerationArguments, config+CLI parsing,
                     and run-directory setup. A method subclasses the bases and adds only its own.
  * sampling.py   -- logit shaping (temperature/top-k/top-p) and model-output normalisation
  * callbacks.py  -- Trainer callbacks: metric logging always, early stopping on request
  * logging.py    -- console/file logging and uncaught-exception recording
  * run_setup.py  -- run directories, latest link, dataset tags and config/provenance
  * record_results.py -- epoch CSV export and training curves
  * losses.py     -- aligned token cross-entropy; each method owns alignment and weighting
  * backbones.py  -- model descriptors, per-method registry mechanics and loading helpers
  * transformer.py -- custom-model validation, rotary self-attention, FFNs and initialization
  * eos_decoding.py -- AR and S2S only: recovery metrics, dataset split, evaluation strategy and
                     generation record, which they share down to the sampler and label alignment
  * training.py   -- external-loss Trainer workflow, lifecycle and finished-run artifacts
  * generation.py -- standalone generation lifecycle: checks, loading, cases and output
  * evaluation.py -- standalone evaluation loading/provenance and post-training context regimes

The membership test: **things a method needs to PRODUCE output**. Turning records into model inputs
belongs in ``datasets/dataset_common.py``; measuring outputs belongs in ``cdrseqgen/eval/``.

Everything here started life duplicated and was consolidated after a second method existed to check
it against — which is how ``build_callbacks`` was found to have silently lost ``LogMetricsCallback``
on the causal side, reintroducing a `train.log` regression fixed in July.
"""
from cdrseqgen.methods.common.callbacks import build_callbacks  # noqa: F401
from cdrseqgen.methods.common.sampling import (  # noqa: F401
    NEG_INF, ban_tokens, logits_of, warp_logits)
