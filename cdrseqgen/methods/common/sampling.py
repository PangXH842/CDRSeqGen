"""Logit shaping and model-output normalisation, shared by every sampler.

These are pure tensor operations with no method content. They live here for a comparability reason:
an AR-versus-MLM table reports both models "at top_p=0.9", and that sentence is only true if one
implementation produced both. Two copies that drifted — in whether top-p keeps the boundary token,
say — would silently compare different sampling regimes while every label still read 0.9.
"""
import torch

NEG_INF = float("-inf")


def ban_tokens(logits, banned_ids):
    """Set banned token logits to -inf (softmax probability = 0)."""
    if banned_ids:
        logits[..., list(banned_ids)] = NEG_INF
    return logits


def warp_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Apply temperature, then top-k, then top-p to softmax logits

    Cases:
      * temperature = 0: greedy, should not use this function
      * top_k = 0: disabled, all logits kept, ~ top_k = inf
      * top_p = 1.0: all low probability tail tokens kept

    Note:
      * top_p keeps at least 1 highest-probability token
    """
    if temperature and temperature > 0:
        logits = logits / temperature

    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, NEG_INF)      # set all remaining logits to -inf

    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Drop tokens once the cumulative mass exceeds top_p, but always keep the first (shift right)
        remove_sorted = cumulative > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, NEG_INF)            # set tail logits (>top_p) to -inf

    return logits


def logits_of(outputs):
    """Extract logits from whichever shape a model returns.

    The custom models return a plain dict; Hugging Face returns a ``ModelOutput``. Both happen to
    support ``["logits"]``, which is why bypassing this normaliser looks harmless — but a backbone
    returning a tuple would break only at sampling time, on the pretrained path, long after the
    forward pass that produced it looked fine.
    """
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    raise ValueError(f"Cannot extract logits from {type(outputs)}")
