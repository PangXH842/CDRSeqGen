"""Token-loss arithmetic shared by method-specific objectives.

Functions:
  * token_cross_entropy -- flatten aligned logits/labels and apply PyTorch CE

Notes:
  * The caller aligns predictions with labels: MLM uses matching positions;
    a causal objective would need to shift them before calling this helper.
  * Labels equal to IGNORE_INDEX contribute no loss. Core/anchor selection
    and other method-specific weighting belong to the calling method.
"""
import torch.nn.functional as F


IGNORE_INDEX = -100


def token_cross_entropy(logits, labels, *, label_smoothing, reduction):
    """Apply CE to aligned token predictions, ignoring labels equal to -100.

    Shapes:
      logits: [B, S, V] -> [B*S, V]
      labels: [B, S]    -> [B*S]
      result: scalar for "mean"/"sum", or [B*S] for "none"

    "mean" divides by the number of supervised tokens, across this batch.
    Label smoothing and reduction are passed directly to PyTorch.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
