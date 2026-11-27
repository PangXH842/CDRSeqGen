"""Sample S2S CDR loops and calculate their generator self-perplexities.

Entry paths:
  generation.generate_case / evaluation.Seq2SeqStrategy -> sample_loops / perplexities

Functions:
  * sample_loops -- encode one framework, then decode a batch of candidate attempts
  * perplexities -- score ordered encoder-input/loop pairs under the generating model

Notes:
  * EOS selects length. The decoder bans other special tokens, retains duplicates
    and reports unfinished draws without refilling or forcing EOS.
  * temperature=0 returns one greedy attempt; other temperatures request num_samples.
  * Sampling encodes the framework once. Decoder rows start with BOS and grow
    together, reusing the same encoder states throughout generation.
  * Self-scoring reuses the most recent encoder result across consecutive spans.
    Independent judge scoring lives in eval/.
  * Functions disable gradients; sampling sets eval mode and scoring preserves it.
"""
import torch

from cdrseqgen.datasets.dataset_s2s import (
    S2S_DECODE_STEPS, build_decoder_pair, validate_s2s_decode_steps,
)
from cdrseqgen.methods.common import ban_tokens, logits_of, perplexity_from_nll, warp_logits


@torch.no_grad()
def sample_loops(model, tokenizer, encoder_ids, *, device, num_samples=1, temperature=1.0,
                 top_k=0, top_p=1.0, max_new_tokens=S2S_DECODE_STEPS, generator=None):
    """Return ordered decoding attempts for one already assembled encoder input.

    Steps:
      validate cap -> encode framework once -> prepare BOS decoder batch
                   -> predict/sample one token per step -> attempt records

    Example: max_new_tokens=3
      A C EOS -> {"loop": "AC", "length": 2, "complete": True}
      A C D   -> {"loop": "ACD", "length": 3, "complete": False}

    The cap includes the terminator, which is excluded from returned loops.
    Early EOS can produce a complete empty loop; a missing EOS is incomplete.
    Greedy decoding returns one attempt and bypasses top-k/top-p. Stochastic
    draws retain their original order and duplicates. generator controls RNG.
    """
    # ------ 1. Validate the contract and encode the framework once. ------
    validate_s2s_decode_steps(max_new_tokens)
    greedy = temperature == 0
    draws = 1 if greedy else num_samples
    model.eval()

    source = torch.tensor([list(encoder_ids)], dtype=torch.long, device=device)
    encoder_mask = torch.ones_like(source)
    encoded = model.encode(source, attention_mask=encoder_mask).expand(draws, -1, -1)
    encoder_mask = encoder_mask.expand(draws, -1)

    # ------ 2. Prepare decoder rows and allowed tokens. ------
    banned = {i for i in tokenizer.special_ids if i != tokenizer.eos_id}
    decoder_ids = torch.full((draws, 1), tokenizer.bos_id, dtype=torch.long, device=device)

    # --- [REVISED 2026-11-27] Pre-allocated GPU token buffer (single sync at end) ---
    tokens_buf = torch.full((draws, max_new_tokens), tokenizer.pad_id, dtype=torch.long, device=device)
    lengths = torch.zeros(draws, dtype=torch.long, device=device)
    finished = torch.zeros(draws, dtype=torch.bool, device=device)

    # ------ 3. Advance all attempts; completed rows remain padded. ------
    for step in range(max_new_tokens):
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
        active_content = (~finished) & nxt.ne(tokenizer.eos_id)
        tokens_buf[:, step] = torch.where(active_content, nxt, tokenizer.pad_id)
        lengths += active_content.long()
        finished |= nxt.eq(tokenizer.eos_id)
        if bool(finished.all()):
            break

        decoder_ids = torch.cat([decoder_ids, nxt[:, None]], dim=1)

    # ------ 4. Keep every attempt, including complete-empty and incomplete ones. ------
    buf_cpu = tokens_buf.tolist()
    lengths_cpu = lengths.tolist()
    finished_cpu = finished.tolist()
    return [
        {"loop": tokenizer.decode(row[:length]), "length": length, "complete": complete}
        for row, length, complete in zip(buf_cpu, lengths_cpu, finished_cpu)
    ]
    # --- [/REVISED 2026-11-27] ---


# --- [REVISED 2026-11-27] Batched S2S self-perplexity scoring per consecutive encoder context ---
@torch.inference_mode()
def perplexities(model, tokenizer, device, spans, *, batch_size=64):
    """Return one generator perplexity per ordered (encoder IDs, loop IDs) pair.

    Steps:
      group consecutive identical encoder contexts -> encode once
      -> right-padded batched decoder passes -> aligned answer-token loss -> exp(mean NLL)

    The decoder input/label pair includes the terminator and is already aligned;
    no further shift is applied. Consecutive candidates sharing the same encoder
    input reuse one encoding and decode together in batches.
    """
    if not spans:
        return []
    scores = []
    idx = 0
    total = len(spans)
    while idx < total:
        source_key = tuple(spans[idx][0])
        group = []
        while idx < total and tuple(spans[idx][0]) == source_key:
            group.append(build_decoder_pair(tokenizer, spans[idx][1]))
            idx += 1
        source = torch.tensor([source_key], dtype=torch.long, device=device)
        encoded_one = model.encode(source)
        for offset in range(0, len(group), batch_size):
            chunk = group[offset : offset + batch_size]
            bsz = len(chunk)
            width = max(len(dec_in) for dec_in, _ in chunk)
            decoder = torch.full((bsz, width), tokenizer.pad_id, dtype=torch.long, device=device)
            gold = torch.full((bsz, width), -100, dtype=torch.long, device=device)
            for row, (dec_in, labels) in enumerate(chunk):
                n = len(dec_in)
                decoder[row, :n] = torch.tensor(dec_in, dtype=torch.long, device=device)
                gold[row, :n] = torch.tensor(labels, dtype=torch.long, device=device)
            encoded = encoded_one.expand(bsz, -1, -1)
            logits = logits_of(model(encoder_hidden_states=encoded, decoder_input_ids=decoder)).float()
            tok_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                gold.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(bsz, -1)
            valid = gold.ne(-100)
            counts = valid.sum(dim=1)
            row_nll = (tok_loss * valid).sum(dim=1) / counts.clamp(min=1)
            for row_idx in range(bsz):
                scores.append(perplexity_from_nll(row_nll[row_idx]) if counts[row_idx].item() > 0 else None)
    return scores
# --- [/REVISED 2026-11-27] ---
