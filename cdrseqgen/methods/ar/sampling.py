"""Causal decoding: generate a loop from a span-infilling prompt until <eos>.

  * sample_loops -- batched autoregressive decode, returns one loop string per requested candidate

This is the whole sampler. The masked side needs two decoders, a length policy layer, a gap-sentinel
classifier and mask expansion (``mlm/sampling.py``, ``mlm/generation.py`` and ``mlm/length.py``); a
causal model needs none of it, because **the model chooses its own length by emitting <eos>**.
Findings §10's length head — which scores below the "always predict 15" baseline — exists only to
work around the masked model's inability to do that.

Three decisions worth stating:

  * **temperature = 0 means greedy and returns ONE candidate**, matching the masked sampler's
    convention, so `--temperature 0` behaves identically across methods.
  * **Special tokens are banned except <eos>.** The model could otherwise emit <mask>, <sep> or
    <pad> mid-loop, which is not a residue and would be scored as one. Banning them at sampling
    time is the causal analogue of the masked sampler's `banned_ids`.
  * **Prompts are LEFT-padded for batching.** With right padding the newest token sits at a
    different index in every row, so a single `logits[:, -1]` read would be wrong for all but the
    longest prompt. This is why `causal_collate_fn` (right-padded, for teacher forcing) must not be
    reused here — its own docstring says so.

A draw is *invalid* if the model never emitted <eos> within ``max_new_tokens``; that is reported
rather than silently truncated, because a truncated loop is not a length the model chose.
"""
import torch

from cdrseqgen.datasets.dataset_ar import (CAUSAL_DECODE_STEPS, layout_for,
                                       validate_causal_decode_steps, validate_layout)
from cdrseqgen.methods.common import ban_tokens, logits_of, warp_logits


def _left_pad(prompts, pad_id, device):
    """[B, L] left-padded ids + attention mask, so every row's last token is at index -1."""
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        ids[row, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long)
        mask[row, width - len(prompt):] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def sample_loops(model, tokenizer, prompt_ids, *, device, num_samples=1, temperature=1.0,
                 top_k=0, top_p=1.0, max_new_tokens=CAUSAL_DECODE_STEPS, generator=None):
    """Decode `num_samples` loops from one prompt.

    Returns a list of dicts: ``{"loop": str, "length": int, "complete": bool}``. ``complete`` is
    False when <eos> never arrived inside ``max_new_tokens`` — such a draw has no length the model
    chose, so a caller should treat it as a failure rather than a short loop.
    """
    validate_causal_decode_steps(max_new_tokens)
    # Both argument checks up front, before the model is touched: a layout the sampler cannot read
    # back would otherwise be discovered only as empty loops in the results.
    layout = validate_layout(layout_for(tokenizer), tokenizer)
    greedy = temperature == 0
    draws = 1 if greedy else num_samples
    model.eval()

    ids, mask = _left_pad([list(prompt_ids)] * draws, tokenizer.pad_id, device)
    # <eos> must stay reachable; everything else non-residue is banned so a loop is all residues.
    terminator = layout.terminator_id(tokenizer)
    banned = {i for i in tokenizer.special_ids if i != terminator}

    generated = [[] for _ in range(draws)]
    finished = torch.zeros(draws, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits = logits_of(model(input_ids=ids, attention_mask=mask))[:, -1, :].float()
        logits = ban_tokens(logits, banned)
        if greedy:
            nxt = logits.argmax(-1)
        else:
            probs = torch.softmax(warp_logits(logits, temperature, top_k, top_p), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)

        nxt = torch.where(finished, torch.full_like(nxt, tokenizer.pad_id), nxt)
        for row in range(draws):
            if not finished[row] and nxt[row].item() != terminator:
                generated[row].append(int(nxt[row].item()))
        finished |= nxt.eq(terminator)
        if bool(finished.all()):
            break

        ids = torch.cat([ids, nxt[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones_like(nxt)[:, None]], dim=1)

    return [{"loop": tokenizer.decode(tokens), "length": len(tokens),
             "complete": bool(finished[row])}
            for row, tokens in enumerate(generated)]
