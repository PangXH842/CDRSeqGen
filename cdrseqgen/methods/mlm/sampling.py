"""Fill MLM masks and score completed candidate loops.

Functions:
  * sample_masked_span    -- select the independent or iterative decoder for a masked prompt
  * sample_independent    -- fill all positions from one forward pass over a masked prompt
  * sample_iterative      -- fill one position per draw per pass, in confidence order
  * pseudo_log_likelihood -- re-mask each candidate position and return (PLL, pseudo-ppl)
  * _logits_of            -- extract logits from a dict or model-output object

Callers:
  * mlm.generation and fixed-native MLM evaluation call sample_masked_span.
  * MLM evaluation strategies use pseudo_log_likelihood for generator self-scoring.
  * eval.generative uses the same scorer for the shared masked-LM judge.

Notes:
  * ban_tokens and warp_logits are shared helpers imported from methods.common.
  * Both decoders exclude banned_ids and return one candidate at temperature=0.
  * Tensor comments use S=prompt length, L=loop length, V=vocabulary size, N=draws.
"""

from __future__ import annotations

from typing import Any

import torch

from cdrseqgen.methods.common import NEG_INF, ban_tokens, warp_logits


def sample_masked_span(
    model,
    tokenizer,
    device,
    generator,
    input_ids: list[int],
    mask_positions: list[int],
    *,
    decoding: str,
    num_samples: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[list[int]]:
    """Return candidate token-ID lists for an already assembled masked prompt.

    Dispatch:
      decoding="independent" -> sample_independent
      otherwise              -> sample_iterative

    The caller validates the decoder name. Both decoders exclude tokenizer.special_ids
    and return one greedy candidate when temperature=0, regardless of num_samples.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "input_ids": input_ids,
        "mask_positions": mask_positions,
        "banned_ids": tokenizer.special_ids,
        "device": device,
        "num_samples": num_samples,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "generator": generator,
    }
    if decoding == "independent":
        return sample_independent(**kwargs)
    return sample_iterative(**kwargs, mask_id=tokenizer.mask_id)


def _logits_of(outputs):
    """Return dict['logits'] or outputs.logits; raise ValueError for other outputs."""
    if isinstance(outputs, dict):
        return outputs["logits"]
    elif hasattr(outputs, "logits"):
        return outputs.logits
    else:
        raise ValueError(f"Cannot extract logits from {type(outputs)}")


_ban = ban_tokens   # kept as a local alias so existing call sites are untouched


def sample_independent(model, input_ids, mask_positions, banned_ids, *, device,
                     num_samples=1, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """Return candidate token-ID lists using one model forward pass.

    Example: two masks, two requested draws
      model sees: [mask] [mask] -> position-wise distributions
      draw 1:        A      C
      draw 2:        D      C

    Predictions at different positions use the same supplied prompt; sampled
    residues are not fed back into the model. temperature=0 returns one argmax
    sequence and bypasses top-k/top-p filtering.
    """
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
    """Return candidates by filling one most-confident unfilled position per pass.

    Example: one draw with three masks (position order is chosen by the model)
      pass 1: [mask] [mask] [mask] -> [mask]   C    [mask]
      pass 2: [mask]   C    [mask] ->   A      C    [mask]
      pass 3:   A      C    [mask] ->   A      C      D

    Notes:
      * Each draw can choose a different position order. Confidence is the maximum
        residue probability at a position after banning/filtering logits.
      * Filled residues stay fixed and become context for later forward passes.
      * All supplied mask_positions are reset to mask_id before generation.
      * temperature=0 returns one greedy sequence and bypasses top-k/top-p filtering.
      * An empty span returns one empty list per effective draw.
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
    """Return (sum of conditional log-probabilities, masked-LM pseudo-perplexity).

    Example: candidate="ACD"
      model input 1: [mask]   C      D      -> log p(A | C, D, context)
      model input 2:   A    [mask]   D      -> log p(C | A, D, context)
      model input 3:   A      C    [mask]   -> log p(D | A, C, context)
      PLL:        sum of these log-probabilities
      pseudo-ppl: exp(-PLL / 3); p=0.5 at every position gives 2.0

    Notes:
      * token_ids must correspond one-to-one with mask_positions.
      * All single-mask variants are scored together in one batched forward pass.
      * Probabilities are normalized after excluding banned_ids; standard callers
        use this to restrict scoring to amino-acid tokens.
      * An empty span returns (0.0, 1.0).
    """
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
