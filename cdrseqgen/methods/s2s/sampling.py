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

    generated = [[] for _ in range(draws)]
    finished = torch.zeros(draws, dtype=torch.bool, device=device)

    # ------ 3. Advance all attempts; completed rows remain padded. ------
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
        # Transfer step bookkeeping in batches instead of synchronizing each GPU scalar.
        for tokens, token, was_finished in zip(generated, nxt.tolist(), finished.tolist()):
            if not was_finished and token != tokenizer.eos_id:
                tokens.append(token)
        finished |= nxt.eq(tokenizer.eos_id)
        if bool(finished.all()):
            break

        decoder_ids = torch.cat([decoder_ids, nxt[:, None]], dim=1)

    # ------ 4. Keep every attempt, including complete-empty and incomplete ones. ------
    return [{"loop": tokenizer.decode(tokens), "length": len(tokens), "complete": complete}
            for tokens, complete in zip(generated, finished.tolist())]


@torch.no_grad()
def perplexities(model, tokenizer, device, spans):
    """Return one generator perplexity per ordered (encoder IDs, loop IDs) pair.

    Steps:
      build_decoder_pair -> encode/reuse framework -> one decode per candidate
                         -> aligned answer-token loss -> exp(mean NLL)

    The decoder input/label pair includes the terminator and is already aligned;
    no further shift is applied. Consecutive candidates with identical encoder
    input reuse its encoding. Non-finite or overflowing perplexity yields None.
    An empty spans list returns []. Generation and evaluation share this scorer.
    """
    # Cache only the most recently encoded framework.
    scores = []
    cached_source = None
    encoded = None
    for encoder_ids, loop_ids in spans:
        decoder_input, labels = build_decoder_pair(tokenizer, loop_ids)
        source_key = tuple(encoder_ids)
        if source_key != cached_source:
            source = torch.tensor([source_key], dtype=torch.long, device=device)
            encoded = model.encode(source)
            cached_source = source_key
        decoder = torch.tensor([decoder_input], dtype=torch.long, device=device)
        gold = torch.tensor([labels], dtype=torch.long, device=device)
        logits = logits_of(model(encoder_hidden_states=encoded, decoder_input_ids=decoder)).float()
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), gold.reshape(-1),
            ignore_index=-100, reduction="mean")
        scores.append(perplexity_from_nll(nll))
    return scores
