"""MLM reconstruction metrics used during Trainer validation.

Caller:
  mlm.training.prepare_training() -> Trainer(compute_metrics=...) -> compute_metrics()

Functions:
  * compute_metrics -- argmax recovery over whole-loop, core and anchor tokens;
                       also return the fraction of rows contributing core tokens
  * _core_anchor_masks -- adapt eval.metrics.core_anchor_indices to Trainer masks
  * _warn_if_low_core_coverage -- log low core coverage once per process

Notes:
  * Validation uses full-span masking: all native loop residues are hidden,
    while the mask count supplies native length. These scores measure masked
    reconstruction; generated candidates are evaluated separately.
  * Anchors are defined positionally by trimming each end, rather than inferred
    from genes or structure. Accuracy alone can hide differences in the core.
    The boundary comes from eval.metrics.core_anchor_indices(), also used in
    generated-sequence evaluation and benchmark scoring.
  * ANCHOR_TRIM re-exports the shared default from cdrseqgen.eval. Training
    explicitly passes the configured loss_config.anchor_trim to compute_metrics.
  * Trainer logging and shared recording code persist the returned metrics.
    This module warns about low coverage; artifact reporting chooses its own
    fallback, and checkpoint selection follows the configured Trainer metric.
"""
import logging

import numpy as np
from transformers import EvalPrediction

from cdrseqgen.eval import MIN_CORE_COVERAGE, ANCHOR_TRIM as _ANCHOR_TRIM
from cdrseqgen.eval.metrics import core_anchor_indices

logger = logging.getLogger(__name__)

# Shared default, re-exported for existing MLM evaluation callers.
ANCHOR_TRIM = _ANCHOR_TRIM

_warned_low_core_coverage = False


def _core_anchor_masks(masked_positions: np.ndarray, trim: int = ANCHOR_TRIM):
    """Map the shared core/anchor split onto supervised input positions.

    Example: one six-residue loop, trim=2
      position:   0 1 2 3 4 5 6 7 8 9
      supervised: . . 1 1 1 1 1 1 . .
      region:     . . A A C C A A . .   (A=anchor, C=core)
      shared core offsets: [2, 3] -> input columns [4, 5]

    Notes:
      * Input is a boolean [N, S] array; framework/padding positions are False.
      * A row with <= 2*trim supervised tokens contributes only anchors.
      * Full-span validation supplies one contiguous target block per row.
        This helper trims the first/last True positions; it cannot recover
        original loop offsets from a partially supervised mixed-mask view.
      * trim is expected to be a non-negative integer; inputs are not validated.
    """
    core = np.zeros_like(masked_positions)
    anchor = np.zeros_like(masked_positions)
    for r in range(masked_positions.shape[0]):
        cols = np.nonzero(masked_positions[r])[0]
        core_idxs, anchor_idxs = core_anchor_indices(cols.size, trim)
        core[r, cols[core_idxs]] = True
        anchor[r, cols[anchor_idxs]] = True
    return core, anchor


def _warn_if_low_core_coverage(
    coverage: float, rows_with_core: int, n_rows: int, trim: int = ANCHOR_TRIM
) -> None:
    """Log a warning once per process when coverage < MIN_CORE_COVERAGE.

    Example: one of four rows contributes core tokens
      coverage = 1/4 = 0.25 < 0.5 (the shared threshold) -> warn
      later low-coverage calls in the same process -> no repeated warning

    Coverage at the threshold does not warn. This function does not alter
    metrics, select a reporting fallback or change checkpoint selection.
    Low coverage describes a restricted scoring population, not proof that
    its core accuracy is invalid.
    """
    global _warned_low_core_coverage

    if coverage >= MIN_CORE_COVERAGE or _warned_low_core_coverage:
        return
    _warned_low_core_coverage = True

    logger.warning(
        f"⚠️ core_accuracy covers only {rows_with_core}/{n_rows} eval rows "
        f"({coverage:.1%}; anchor_trim={trim}). It describes only rows with a non-empty core. "
        "Interpret it alongside core_coverage and whole-span accuracy."
    )


def compute_metrics(eval_pred: EvalPrediction, *, trim: int = ANCHOR_TRIM) -> dict:
    """Return token-pooled argmax recovery and the fraction of rows with a core.

    Example: one fully masked loop, trim=2
      native:  A C D E F G
      argmax:  A Q D Q F G
      region:  A A C C A A
      correct: 1 0 1 0 1 1
      accuracy:        4/6
      core_accuracy:   1/2
      anchor_accuracy: 3/4
      core_coverage:   1/1

    Inputs:
      predictions: [N, S, V] logits, or a tuple whose first item is logits
      labels:      [N, S] native token IDs; -100 excludes framework/padding

    Notes:
      * Recovery pools correct/total tokens across rows; it is not a mean
        of per-loop recovery scores. core_coverage counts rows instead.
      * Core/anchor splitting assumes the full-span evaluation view.
      * An empty core/anchor denominator returns 0.0. Read core_coverage
        alongside core_accuracy to distinguish no core from no correct tokens.
      * If every label is -100, the existing early return contains only the
        three accuracy keys, all 0.0; core_coverage is omitted.
    """
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        logits = predictions[0]
    else:
        logits = predictions

    preds = np.argmax(logits, axis=-1)

    # Exclude framework and padding labels.
    masked_positions = (labels != -100)

    total_masked = masked_positions.sum()
    if total_masked == 0:
        return {"accuracy": 0.0, "core_accuracy": 0.0, "anchor_accuracy": 0.0}

    correct = (preds == labels) & masked_positions

    core_mask, anchor_mask = _core_anchor_masks(masked_positions, trim=trim)
    core_total = core_mask.sum()
    anchor_total = anchor_mask.sum()

    # Coverage counts contributing rows, while accuracy pools their tokens.
    rows_with_core = int((core_mask.sum(axis=-1) > 0).sum())
    n_rows = int(masked_positions.shape[0])
    core_coverage = (rows_with_core / n_rows) if n_rows else 0.0
    _warn_if_low_core_coverage(core_coverage, rows_with_core, n_rows, trim=trim)

    return {
        "accuracy": float(correct.sum() / total_masked),
        "core_accuracy": float((correct & core_mask).sum() / core_total) if core_total else 0.0,
        "anchor_accuracy": float((correct & anchor_mask).sum() / anchor_total) if anchor_total else 0.0,
        "core_coverage": float(core_coverage),
    }
