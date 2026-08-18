"""Masked-LM training and CDR generation method.

This package owns the MLM-compatible backbones, training objective, masked
sampling, fixed/variable-length generation, and MLM-native evaluation. Future
autoregressive or encoder-decoder methods can live beside it without inheriting
mask-token or pseudo-log-likelihood assumptions.

Import from the submodules directly (``from cdrseqgen.methods.mlm.length import
resolve_length_policy``). This file deliberately re-exports nothing: the previous
package-level surface duplicated every public name, was imported by no caller, and
forced the whole method — including torch and the length predictor — to load for
any single import.
"""
