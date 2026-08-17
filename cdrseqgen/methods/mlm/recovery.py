"""MLM Trainer adapter for recovery on supervised mask positions.

The counterpart to metrics.py: this is the *recovery* half of the evaluation (did we reproduce the
native loop), metrics.py is the *design* half (diversity, novelty, entropy). Passed to Trainer as
compute_metrics, so it runs every eval_strategy step and lands in epoch_metrics.csv.

  * compute_metrics -- whole-span, anchor-excluded core, and anchor-only recovery + core coverage

Notes:
  * ANCHOR_TRIM residues are trimmed from EACH end of the loop to separate the V/J-gene-determined
    "anchor" residues (recovered ~80-100% by any model) from the hypervariable "core" that is
    actually being designed. Whole-span accuracy is dominated by the anchors: on H3 it reads ~0.46
    while the core reads ~0.32. Override the trim with env ANCHOR_TRIM.
  * A loop of <= 2*ANCHOR_TRIM residues has no core at all, so core_accuracy can be computed over a
    tiny unrepresentative slice while looking like an ordinary score. Every run therefore records
    `core_coverage`; coverage below MIN_CORE_COVERAGE warns and makes artifact reporting fall back
    explicitly to whole-span. OAS light-chain CDR2 is one observed case, not a coded exception.
"""
import os
import logging

import numpy as np
from transformers import EvalPrediction

from cdrseqgen.eval import MIN_CORE_COVERAGE

logger = logging.getLogger(__name__)

# Residues trimmed from EACH end of the target loop to separate the V/J-gene-determined "anchor"
# residues (predicted ~80-100% regardless of model) from the hypervariable "core" that is actually
# being designed. The raw whole-span `accuracy` is dominated by the anchors and can mask a real
# effect on the core (see README Known Issues). Override with env ANCHOR_TRIM if needed.
ANCHOR_TRIM = int(os.environ.get("ANCHOR_TRIM", "2"))

_warned_low_core_coverage = False


def _core_anchor_masks(masked_positions: np.ndarray, trim: int = ANCHOR_TRIM):
    """Split each row's contiguous target span into core (middle) and anchor (trimmed ends) masks.

    The target loop is a single contiguous block of non-(-100) positions per row (see
    AntibodyInfillingDataset: [cls]+pt1+mask+pt2+[end]). Rows whose loop is too short to have a
    core after trimming `trim` from each end contribute only anchor tokens.
    """
    core = np.zeros_like(masked_positions)
    for r in range(masked_positions.shape[0]):
        cols = np.nonzero(masked_positions[r])[0]
        if cols.size > 2 * trim:
            core[r, cols[trim:cols.size - trim]] = True
    anchor = masked_positions & ~core
    return core, anchor


def _warn_if_low_core_coverage(coverage: float, rows_with_core: int, n_rows: int) -> None:
    """Warn once per process when core_accuracy is backed by too few eval rows to be meaningful."""
    global _warned_low_core_coverage
    if coverage >= MIN_CORE_COVERAGE or _warned_low_core_coverage:
        return
    _warned_low_core_coverage = True
    logger.warning(
        f"⚠️ core_accuracy covers only {rows_with_core}/{n_rows} eval rows ({coverage:.1%}) — this "
        f"target's loops are mostly <= {2 * ANCHOR_TRIM} residues, so most have no anchor-excluded "
        f"core and the score is noise. Report WHOLE-SPAN accuracy for this loop (see README, "
        f"'How to read the numbers'). Known case: light-chain CDR2 (L2)."
    )


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Sequence recovery on the masked loop: whole-span, plus anchor-excluded core and anchor-only."""
    predictions, labels = eval_pred

    # If the model outputs a tuple (logits, hidden_states), extract the logits
    if isinstance(predictions, tuple):
        logits = predictions[0]
    else:
        logits = predictions

    # Take argmax over the vocab dimension (axis=-1) to get predicted token IDs
    preds = np.argmax(logits, axis=-1)

    # Create a boolean mask where labels are NOT -100 (i.e., the target loop only)
    masked_positions = (labels != -100)

    # Check if there are any masked tokens to evaluate (to avoid division by zero)
    total_masked = masked_positions.sum()
    if total_masked == 0:
        return {"accuracy": 0.0, "core_accuracy": 0.0, "anchor_accuracy": 0.0}

    correct = (preds == labels) & masked_positions

    # Split the span into hypervariable core vs. V/J-determined anchor ends.
    core_mask, anchor_mask = _core_anchor_masks(masked_positions)
    core_total = core_mask.sum()
    anchor_total = anchor_mask.sum()

    # Fraction of eval rows that contribute ANY core token. A loop of <= 2*ANCHOR_TRIM residues has
    # no core at all, so core_accuracy can be computed over a tiny unrepresentative slice while
    # looking like an ordinary score — light-chain CDR2 is 3aa in 98.8% of OAS, leaving ~30 of 2111
    # rows, which is why L2 core reads 1.0000 for scratch and 0.5957 for scratch+HC (both noise;
    # whole-span says 0.9921 vs 0.9590). And when NO row has a core, the ratio below falls back to
    # 0.0 — indistinguishable from "got everything wrong". Recording coverage makes both visible.
    rows_with_core = int((core_mask.sum(axis=-1) > 0).sum())
    n_rows = int(masked_positions.shape[0])
    core_coverage = (rows_with_core / n_rows) if n_rows else 0.0
    _warn_if_low_core_coverage(core_coverage, rows_with_core, n_rows)

    return {
        "accuracy": float(correct.sum() / total_masked),
        "core_accuracy": float((correct & core_mask).sum() / core_total) if core_total else 0.0,
        "anchor_accuracy": float((correct & anchor_mask).sum() / anchor_total) if anchor_total else 0.0,
        "core_coverage": float(core_coverage),
    }
