"""Decoding strategies + sequence scoring for masked-LM CDR infilling.

  * warp_logits           -- apply temperature / top-k / top-p to logits
  * sample_independent    -- sample each masked residue individually
  * sample_iterative      -- sample and update from most confident masked residue
  * pseudo_log_likelihood -- calculate pll and pseudo-perplexity of filled loop

Notes:
  * Comment letter keys: S = sequence length, L = masked length, V = vocab size, N = samples length
  * Every decoder bans non-amino-acid ids so only real residues are emitted
  * Both sampling methods can handle temperature=0 (greedy) cases
"""
import torch

NEG_INF = float("-inf")


def _logits_of(outputs):
    """Function to extract logits from each kind of model output"""
    if isinstance(outputs, dict):
        return outputs["logits"]
    elif hasattr(outputs, "logits"):
        return outputs.logits
    else:
        raise ValueError(f"Cannot extract logits from {type(outputs)}")


def _ban(logits, banned_ids):
    """Set banned token logits to -inf (softmax probability = 0)"""
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


@torch.no_grad()
def sample_independent(model, input_ids, mask_positions, banned_ids, *, device,
                     num_samples=1, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """Sample each masked position independently in one forward pass"""
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = _logits_of(model(input_ids=x))[0]                      # [S, V]
    mask_logits = _ban(logits[mask_positions].clone(), banned_ids)  # [L, V]

    if temperature == 0:                                            # greedy: argmax, one sequence
        return [torch.argmax(mask_logits, dim=-1).tolist()]

    mask_logits = warp_logits(mask_logits, temperature, top_k, top_p)
    probs = torch.softmax(mask_logits, dim=-1)                      # [L, V]
    draws = torch.multinomial(probs, num_samples, replacement=True, generator=generator)  # [L, N]
    return draws.t().tolist()                                       # N sequences, each length L


@torch.no_grad()
def sample_iterative(model, input_ids, mask_positions, banned_ids, mask_id, *, device,
                     num_samples=1, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """Confidence-ordered iterative sampling, update after every next most-confident
    masked position is filled
    """
    L = len(mask_positions)
    greedy = temperature == 0
    N = 1 if greedy else num_samples                                # argmax is deterministic -> one
    if L == 0:
        return [[] for _ in range(N)]

    mp = torch.tensor(mask_positions, dtype=torch.long, device=device)       # [L]
    cur = torch.tensor([input_ids], dtype=torch.long, device=device).repeat(N, 1)  # [N, S]
    cur[:, mp] = mask_id                                                     # start all-masked
    filled = torch.zeros(N, L, dtype=torch.bool, device=device)
    rows = torch.arange(N, device=device)

    for _ in range(L):
        logits = _logits_of(model(input_ids=cur))                           # [N, S, V]
        pos_logits = _ban(logits[:, mp, :].clone(), banned_ids)             # [N, L, V]
        if not greedy:
            pos_logits = warp_logits(pos_logits, temperature, top_k, top_p)
        probs = torch.softmax(pos_logits, dim=-1)                           # [N, L, V]

        # Easy-first: commit each sample's most-confident still-unfilled position
        confidence = probs.max(dim=-1).values.masked_fill(filled, NEG_INF)  # [N, L]
        chosen = confidence.argmax(dim=-1)                                  # [N]
        if greedy:
            token = probs[rows, chosen].argmax(dim=-1)                      # [N] deterministic
        else:
            token = torch.multinomial(probs[rows, chosen], 1, generator=generator).squeeze(1)  # [N]

        cur[rows, mp[chosen]] = token
        filled[rows, chosen] = True

    return cur[:, mp].tolist()                                              # N sequences length L


@torch.no_grad()
def pseudo_log_likelihood(model, input_ids, mask_positions, token_ids, banned_ids, mask_id, device):
    """Pseudo-log-likelihood of a filled loop; returns (pll, pseudo_perplexity)"""
    L = len(mask_positions)
    if L == 0:
        return 0.0, 1.0

    base = list(input_ids)
    for pos, tok in zip(mask_positions, token_ids):
        base[pos] = tok                                       # fill the candidate residues in
    batch = torch.tensor([base], dtype=torch.long, device=device).repeat(L, 1)  # [L, S]
    mp = torch.tensor(mask_positions, dtype=torch.long, device=device)
    rows = torch.arange(L, device=device)
    batch[rows, mp] = mask_id                                 # variant i re-masks position i only

    logits = _logits_of(model(input_ids=batch))              # [L, S, V]
    at_masked = _ban(logits[rows, mp, :].clone(), banned_ids)  # [L, V] logit rows at each masked pos
    log_probs = torch.log_softmax(at_masked, dim=-1)         # amino-acid-restricted log-probs
    tok = torch.tensor(token_ids, dtype=torch.long, device=device)
    per_pos = log_probs[rows, tok]                           # [L] log p(true residue | rest)

    pll = float(per_pos.sum())
    pseudo_ppl = float(torch.exp(-per_pos.mean()))
    return pll, pseudo_ppl
