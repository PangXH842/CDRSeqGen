"""Share logit filtering, model-output access and perplexity conversion.

Entry paths:
  mlm/ar/s2s.sampling -> logits_of -> ban_tokens -> warp_logits -> token selection
  ar/s2s.sampling.perplexities -> perplexity_from_nll

Functions:
  * ban_tokens          -- exclude token IDs by modifying logits in place
  * warp_logits         -- apply temperature, top-k and nucleus filtering
  * logits_of           -- read logits from dict, attribute or tuple model outputs
  * perplexity_from_nll -- convert a scalar mean NLL to a finite score or None

Notes:
  * Default method samplers share these operations; method code owns prompts,
    decoding order, stopping and RNG use. Native sampler hooks retain their protocol.
  * Greedy selection happens in the caller and bypasses stochastic filtering.
  * Filtering assumes floating logits with vocabulary on the final dimension.
"""
import math

import torch

NEG_INF = float("-inf")


def ban_tokens(logits, banned_ids):
    """Set banned vocabulary entries to -inf in place; return the same tensor.

    Example: logits=[1, 2, 3], banned_ids={1} -> [1, -inf, 3].
    All leading dimensions are preserved. Callers copy logits first when the
    unfiltered values are still needed. An empty banned_ids leaves them unchanged.
    """
    if banned_ids:
        logits[..., list(banned_ids)] = NEG_INF
    return logits


def warp_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Return filtered logits, applying temperature -> top-k -> top-p.

    Example: sorted probabilities [0.50, 0.30, 0.20], top_p=0.7
      cumulative: [0.50, 0.80, 1.00] -> keep the first two tokens

    Temperature divides logits when positive. top_k=0 disables its filter;
    ties at the kth logit remain eligible. top_p=0 or 1 disables nucleus filtering;
    otherwise the first token crossing its threshold is retained.
    The input is not modified. Callers select greedy tokens separately when
    temperature=0; this helper does not choose tokens or consume RNG state.
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
    """Return logits from a dict, a .logits attribute or a nonempty tuple.

    Custom models use {"logits": tensor}; Hugging Face models normally expose
    .logits, or return (logits, ...) with return_dict=False and no labels supplied.
    Tuple output must put logits first. Unsupported types and empty tuples raise
    ValueError; dictionaries must carry a "logits" key. Returned logits are
    neither copied nor reshaped.
    """
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise ValueError(f"Cannot extract logits from {type(outputs)}")


def perplexity_from_nll(nll):
    """Return exp(mean NLL), or None when a finite score cannot be represented.

    Accept a scalar tensor or number. Example: NLL=log(2) -> perplexity=2.
    Non-finite NLL and exponential overflow produce None, so one unscorable
    candidate does not interrupt the evaluation. Ordinary values use math.exp.
    """
    value = float(nll)
    if not math.isfinite(value):
        return None
    try:
        return math.exp(value)
    except OverflowError:
        return None
