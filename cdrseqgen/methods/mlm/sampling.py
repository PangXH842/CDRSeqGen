"""Decode masked CDR spans and calculate masked-LM self/judge scores.

Entry paths:
  mlm.length -> sample_masked_span -> independent or iterative decoder
  mlm.generation / length evaluation strategies -> pseudo_log_likelihood
  eval.generative.score_with_judge -> pseudo_log_likelihood

Functions:
  * sample_masked_span    -- select the independent or iterative decoder for a masked prompt
  * sample_independent    -- fill all positions from one forward pass over a masked prompt
  * sample_iterative      -- fill one position per draw per pass, in confidence order
  * pseudo_log_likelihood -- re-mask each candidate position and return (PLL, pseudo-ppl)

Notes:
  * Callers assemble prompts, choose lengths and put models in evaluation mode.
    These helpers disable autograd and leave model mode unchanged.
  * Both decoders exclude banned tokens and return one candidate at temperature=0.
    Sampling retains duplicates; iterative fills residues back into its prompt.
  * methods.common supplies logit extraction, token banning and sampling filters.
    Pseudo-likelihood re-masks completed candidates for self or judge scoring.
"""

from __future__ import annotations

from typing import Any

import torch

from cdrseqgen.methods.common import NEG_INF, ban_tokens, logits_of, warp_logits


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
      decoding="iterative"   -> sample_iterative
      anything else          -> ValueError, so a mislabelled run cannot silently use another decoder

    Both decoders exclude tokenizer.special_ids and return one greedy candidate
    when temperature=0, regardless of num_samples.
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
    if decoding == "iterative":
        return sample_iterative(**kwargs, mask_id=tokenizer.mask_id)
    raise ValueError(f"unknown decoding {decoding!r}; expected 'independent' or 'iterative'")


@torch.no_grad()
def sample_independent(model, input_ids, mask_positions, banned_ids, *, device,
                     num_samples=1, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """Return candidate token-ID lists using one model forward pass.

    Example: two masks, two requested draws
      model sees: [mask] [mask] -> position-wise distributions
      draw 1:        A      C
      draw 2:        D      C

    Predictions at different positions use the same supplied prompt; sampled
    residues are not fed back into the model. temperature=0 returns one argmax
    sequence and bypasses top-k/top-p filtering. An empty span returns one empty
    list per effective draw without a model forward or RNG consumption.
    """
    if len(mask_positions) == 0:
        num_draws = 1 if temperature == 0 else num_samples
        return [[] for _ in range(num_draws)]

    prompt = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = logits_of(model(input_ids=prompt))[0]                    # [sequence, vocab]
    mask_logits = ban_tokens(logits[mask_positions].clone(), banned_ids)  # [span, vocab]

    if temperature == 0:                                            # greedy: argmax, one sequence
        return [torch.argmax(mask_logits, dim=-1).tolist()]

    mask_logits = warp_logits(mask_logits, temperature, top_k, top_p)
    probs = torch.softmax(mask_logits, dim=-1)                       # [span, vocab]
    draws = torch.multinomial(probs, num_samples, replacement=True, generator=generator)
    return draws.t().tolist()                                       # [draws, span]


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
    span_length = len(mask_positions)
    greedy = temperature == 0
    num_draws = 1 if greedy else num_samples
    if span_length == 0:
        return [[] for _ in range(num_draws)]

    positions = torch.tensor(mask_positions, dtype=torch.long, device=device)
    candidates = torch.tensor([input_ids], dtype=torch.long, device=device).repeat(num_draws, 1)
    candidates[:, positions] = mask_id                              # [draws, sequence]
    filled = torch.zeros(num_draws, span_length, dtype=torch.bool, device=device)
    rows = torch.arange(num_draws, device=device)

    for _ in range(span_length):
        logits = logits_of(model(input_ids=candidates))             # [draws, sequence, vocab]
        pos_logits = ban_tokens(logits[:, positions, :].clone(), banned_ids)
        if not greedy:
            pos_logits = warp_logits(pos_logits, temperature, top_k, top_p)
        probs = torch.softmax(pos_logits, dim=-1)                    # [draws, span, vocab]

        # Commit each draw's most-confident still-unfilled position.
        confidence = probs.max(dim=-1).values.masked_fill(filled, NEG_INF)
        chosen = confidence.argmax(dim=-1)                         # [draws]
        if greedy:
            token = probs[rows, chosen].argmax(dim=-1)
        else:
            token = torch.multinomial(probs[rows, chosen], 1, generator=generator).squeeze(1)

        candidates[rows, positions[chosen]] = token
        filled[rows, chosen] = True

    return candidates[:, positions].tolist()                        # [draws, span]


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
      * token_ids must correspond one-to-one with mask_positions; unequal
        lengths raise ValueError before any model forward.
      * All single-mask variants are scored together in one batched forward pass.
      * Probabilities are normalized after excluding banned_ids; standard callers
        use this to restrict scoring to amino-acid tokens.
      * An empty span returns (0.0, 1.0).
    """
    span_length = len(mask_positions)
    if len(token_ids) != span_length:
        raise ValueError("token_ids must contain one token per mask position")
    if span_length == 0:
        return 0.0, 1.0

    base = list(input_ids)
    for position, token_id in zip(mask_positions, token_ids):
        base[position] = token_id
    batch = torch.tensor([base], dtype=torch.long, device=device).repeat(span_length, 1)
    positions = torch.tensor(mask_positions, dtype=torch.long, device=device)
    rows = torch.arange(span_length, device=device)
    batch[rows, positions] = mask_id                             # variant i re-masks position i

    logits = logits_of(model(input_ids=batch))                  # [span, sequence, vocab]
    at_masked = ban_tokens(logits[rows, positions, :].clone(), banned_ids)
    log_probs = torch.log_softmax(at_masked, dim=-1)             # [span, vocab]
    candidate_tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
    token_log_probs = log_probs[rows, candidate_tokens]         # log p(residue | rest)

    pll = float(token_log_probs.sum())
    pseudo_ppl = float(torch.exp(-token_log_probs.mean()))
    return pll, pseudo_ppl
