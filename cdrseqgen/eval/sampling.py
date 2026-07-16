"""Decoding strategies + sequence scoring for masked-LM CDR infilling.

predict.py used to be argmax-only: one forward pass, torch.argmax per masked position. That is
(a) deterministic — you get exactly one sequence, so diversity/novelty are undefined — and
(b) *independent* — each loop residue is chosen from its own marginal with every OTHER loop
position still masked, so the strong intra-loop dependencies (acute for CDR-H3) are ignored.
Recovery-vs-native then rewards copying the germline-ish consensus, which is the opposite of what
a design objective wants (see README Findings/Issues). This module is the sampler that unblocks it.

  * warp_logits         -- temperature / top-k / top-p (nucleus) logit warping.
  * decode_greedy       -- the original argmax behaviour (kept, verbatim semantics, for repro).
  * sample_independent  -- per-position sampling from the warped marginals. ONE forward pass for
                           any number of samples: fast, but the L positions are independent.
  * sample_iterative    -- confidence-ordered iterative refill ("Mask-Predict", Ghazvininejad
                           et al. 2019). Fills one position at a time, easiest (most-confident)
                           first, re-running the model so each new residue is conditioned on the
                           already-committed ones. Captures intra-loop dependencies; costs L
                           forward passes, batched across all samples so it stays cheap.
  * pseudo_log_likelihood / pseudo_perplexity -- Salazar et al. (2020) pseudo-log-likelihood over
                           the loop: mask each residue in turn and read its log-prob given the
                           rest. The masked-LM stand-in for perplexity — a *naturalness* score
                           that, unlike recovery, does NOT reward matching the native sequence.

All model-facing code lives here; pure sequence statistics (diversity, novelty) live in metrics.py.
Every decoder bans the tokenizer's non-amino-acid ids (special_ids) so only real residues are
emitted, and scoring restricts the softmax to that same amino-acid support so the log-probs are
consistent with what the sampler could actually draw and comparable across ESM/BERT/scratch vocabs.
"""
import torch

NEG_INF = float("-inf")


def _logits_of(outputs):
    """Backbone-agnostic logit access (scratch returns a dict; HF models return a ModelOutput)."""
    return outputs["logits"] if isinstance(outputs, dict) else outputs.logits


def _ban(logits, banned_ids):
    """In-place: set every non-amino-acid (special) token id to -inf on the last (vocab) dim."""
    if banned_ids:
        logits[..., list(banned_ids)] = NEG_INF
    return logits


def warp_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Apply temperature, then top-k, then top-p (nucleus) filtering on the last dim.

    Operates on any leading batch shape ([V], [L,V], [N,L,V]). top_k/top_p always keep at least
    one token, so a row can never become all -inf here (banned ids are handled separately).
    """
    if temperature and temperature != 1.0:
        logits = logits / temperature

    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, NEG_INF)

    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Drop tokens once the cumulative mass exceeds top_p, but always keep the first (shift right).
        remove_sorted = cumulative > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, NEG_INF)

    return logits


@torch.no_grad()
def decode_greedy(model, input_ids, mask_positions, banned_ids, device):
    """Argmax per masked position — the original predict.py behaviour. Returns one sequence."""
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = _logits_of(model(input_ids=x))[0]                      # [S, V]
    mask_logits = _ban(logits[mask_positions].clone(), banned_ids)  # [L, V]
    return [torch.argmax(mask_logits, dim=-1).tolist()]


@torch.no_grad()
def sample_independent(model, input_ids, mask_positions, banned_ids, num_samples,
                       temperature, top_k, top_p, device, generator=None):
    """Sample each masked position independently from its warped marginal.

    ONE forward pass regardless of num_samples: the loop positions never see each other's draws,
    so this is the fast baseline that deliberately ignores intra-loop dependencies.
    """
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = _logits_of(model(input_ids=x))[0]                      # [S, V]
    mask_logits = _ban(logits[mask_positions].clone(), banned_ids)  # [L, V]
    mask_logits = warp_logits(mask_logits, temperature, top_k, top_p)
    probs = torch.softmax(mask_logits, dim=-1)                      # [L, V]
    draws = torch.multinomial(probs, num_samples, replacement=True, generator=generator)  # [L, N]
    return draws.t().tolist()                                       # N sequences, each length L


@torch.no_grad()
def sample_iterative(model, input_ids, mask_positions, banned_ids, num_samples,
                     temperature, top_k, top_p, device, mask_id, generator=None):
    """Confidence-ordered iterative refill (Mask-Predict), batched across all samples.

    Each step: run the model on the current (partially filled) batch, and for every sample commit
    the single still-masked position the model is most confident about, sampling its residue from
    the warped distribution. After L steps every position is filled, and — crucially — each residue
    was drawn conditioned on the ones committed before it, so the joint over the loop is respected.
    Cost is L forward passes total (the N samples ride in the batch dimension).
    """
    L = len(mask_positions)
    if L == 0:
        return [[] for _ in range(num_samples)]

    N = num_samples
    mp = torch.tensor(mask_positions, dtype=torch.long, device=device)       # [L]
    cur = torch.tensor([input_ids], dtype=torch.long, device=device).repeat(N, 1)  # [N, S]
    cur[:, mp] = mask_id                                                     # start all-masked
    filled = torch.zeros(N, L, dtype=torch.bool, device=device)
    rows = torch.arange(N, device=device)

    for _ in range(L):
        logits = _logits_of(model(input_ids=cur))                           # [N, S, V]
        pos_logits = _ban(logits[:, mp, :].clone(), banned_ids)             # [N, L, V]
        pos_logits = warp_logits(pos_logits, temperature, top_k, top_p)
        probs = torch.softmax(pos_logits, dim=-1)                           # [N, L, V]

        # Pick each sample's most-confident still-unfilled position (easy-first order).
        confidence = probs.max(dim=-1).values.masked_fill(filled, NEG_INF)  # [N, L]
        chosen = confidence.argmax(dim=-1)                                  # [N]
        token = torch.multinomial(probs[rows, chosen], 1, generator=generator).squeeze(1)  # [N]

        cur[rows, mp[chosen]] = token
        filled[rows, chosen] = True

    return cur[:, mp].tolist()                                              # N sequences length L


@torch.no_grad()
def pseudo_log_likelihood(model, input_ids, mask_positions, token_ids, banned_ids, mask_id, device):
    """Pseudo-log-likelihood of a filled loop (Salazar et al. 2020); returns (pll, pseudo_perplexity).

    The masked-LM analogue of a language-model likelihood: place the candidate residues into the
    context, then for each loop position re-mask ONLY that position and read the model's log-prob
    of the true residue given all the others. Summing gives the PLL; pseudo_perplexity =
    exp(-PLL / L). Unlike recovery this is native-free — it scores whether a designed loop looks
    *plausible to the model*, which is the quality axis recovery cannot provide. One forward pass
    per candidate (the L single-mask variants ride in the batch dimension).
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
