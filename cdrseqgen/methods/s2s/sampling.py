"""Seq2seq decoding: generate a loop from an encoded framework until <eos>.

  * sample_loops -- batched decode against one encoder pass, one loop string per candidate

Contract-identical to ``methods/ar/sampling.py`` — same return shape, same ``complete`` semantics,
same shared ``warp_logits``/``ban_tokens`` so "top_p=0.9" means one thing across all three methods —
with two mechanical differences that follow from having two streams:

  * **The encoder runs ONCE per framework and every candidate reuses it.** Twenty candidates cost
    one framework encode plus twenty short decodes, where the causal sampler re-reads the whole
    prompt on every step of every draw. This is not just a speed note: it is also why the candidates
    are guaranteed to condition on byte-identical framework states, so any variation between them is
    sampling noise rather than a re-tokenised prompt.

  * **No left-padding, and no padding at all.** The causal sampler must left-pad because prompts
    differ in length and ``logits[:, -1]`` would otherwise read a different position in each row.
    Here every decoder row starts at ``<bos>`` and grows in lockstep, so the last position is the
    last position for all of them. ``s2s_collate_fn``'s warning about right padding therefore does
    not apply to this path — there is nothing to pad.

The two shared decisions carry over unchanged:

  * **temperature = 0 means greedy and returns ONE candidate**, matching both siblings.
  * **Special tokens are banned except <eos>**, so a loop is all residues; the model could otherwise
    emit ``<mask>`` or ``<pad>`` mid-loop and have it scored as a residue.

A draw is *invalid* if the model never emitted <eos> within ``max_new_tokens``; that is reported
rather than silently truncated, because a truncated loop is not a length the model chose.
"""
import torch

from cdrseqgen.datasets.dataset_s2s import S2S_DECODE_STEPS, validate_s2s_decode_steps
from cdrseqgen.methods.common import ban_tokens, logits_of, warp_logits


@torch.no_grad()
def sample_loops(model, tokenizer, encoder_ids, *, device, num_samples=1, temperature=1.0,
                 top_k=0, top_p=1.0, max_new_tokens=S2S_DECODE_STEPS, generator=None):
    """Decode `num_samples` loops from one encoder input.

    Returns a list of dicts: ``{"loop": str, "length": int, "complete": bool}``. ``complete`` is
    False when <eos> never arrived inside ``max_new_tokens`` — such a draw has no length the model
    chose, so a caller should treat it as a failure rather than a short loop.
    """
    validate_s2s_decode_steps(max_new_tokens)
    greedy = temperature == 0
    draws = 1 if greedy else num_samples
    model.eval()

    source = torch.tensor([list(encoder_ids)], dtype=torch.long, device=device)
    encoder_mask = torch.ones_like(source)
    # One encode, shared by every candidate for this framework.
    encoded = model.encode(source, attention_mask=encoder_mask).expand(draws, -1, -1)
    encoder_mask = encoder_mask.expand(draws, -1)

    banned = {i for i in tokenizer.special_ids if i != tokenizer.eos_id}
    decoder_ids = torch.full((draws, 1), tokenizer.bos_id, dtype=torch.long, device=device)

    generated = [[] for _ in range(draws)]
    finished = torch.zeros(draws, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        outputs = model(encoder_hidden_states=encoded, attention_mask=encoder_mask,
                        decoder_input_ids=decoder_ids)
        logits = logits_of(outputs)[:, -1, :].float()
        logits = ban_tokens(logits, banned)
        if greedy:
            nxt = logits.argmax(-1)
        else:
            probs = torch.softmax(warp_logits(logits, temperature, top_k, top_p), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)

        nxt = torch.where(finished, torch.full_like(nxt, tokenizer.pad_id), nxt)
        for row in range(draws):
            if not finished[row] and nxt[row].item() != tokenizer.eos_id:
                generated[row].append(int(nxt[row].item()))
        finished |= nxt.eq(tokenizer.eos_id)
        if bool(finished.all()):
            break

        decoder_ids = torch.cat([decoder_ids, nxt[:, None]], dim=1)

    return [{"loop": tokenizer.decode(tokens), "length": len(tokens),
             "complete": bool(finished[row])}
            for row, tokens in enumerate(generated)]
