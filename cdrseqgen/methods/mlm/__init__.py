"""Masked-LM training and CDR generation method.

This package owns the MLM-compatible backbones, training objective, masked
sampling, fixed/variable-length generation, and MLM-native evaluation. Future
autoregressive or encoder-decoder methods can live beside it without inheriting
mask-token or pseudo-log-likelihood assumptions.
"""

from .generation import (
    FixedLengthPrompt,
    GeneratedSpan,
    prepare_fixed_length_prompt,
    sample_length_conditioned_spans,
    sample_masked_span,
)
from .length import (
    LengthConstraints,
    LengthDecision,
    LengthPolicy,
    UnsupportedLengthPolicy,
    fixed_length,
    fixed_policy,
    native_mask_length,
    resolve_fixed_length,
    resolve_generation_length_policy,
    resolve_length_policy,
)
from .length_predictor import LengthPredictor, fit_length_predictor

__all__ = [
    "FixedLengthPrompt",
    "GeneratedSpan",
    "LengthConstraints",
    "LengthDecision",
    "LengthPolicy",
    "LengthPredictor",
    "UnsupportedLengthPolicy",
    "fit_length_predictor",
    "fixed_length",
    "fixed_policy",
    "native_mask_length",
    "prepare_fixed_length_prompt",
    "resolve_fixed_length",
    "resolve_generation_length_policy",
    "resolve_length_policy",
    "sample_length_conditioned_spans",
    "sample_masked_span",
]
