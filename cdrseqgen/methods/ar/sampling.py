"""Sample AR CDR loops and calculate their generator self-perplexities.

Entry paths:
  generation.generate_case / evaluation.CausalStrategy -> sample_loops / perplexities

Functions:
  * _left_pad    -- align prompt ends and build the matching attention mask
  * sample_loops -- decode attempts from one prompt, using a native hook when present
  * perplexities -- score ordered prompt/answer pairs under the generating model

Notes:
  * EOS selects length. The default decoder bans other special tokens, retains
    duplicates and reports unfinished draws without refilling or forcing EOS.
  * temperature=0 returns one greedy attempt; other temperatures request num_samples.
  * The default decoder batches copies of one prompt. Left padding keeps the last
    input token at index -1; the model rereads the growing prompt at each step.
  * tokenizer.causal_sampler owns native decoding, including IgLM's token rules
    and richer validity/raw-token records. Its results pass through unchanged.
  * Self-perplexity uses the generator; independent judge scoring lives in eval/.
    Functions disable gradients; sampling sets eval mode and scoring preserves it.
"""
import torch

from cdrseqgen.datasets.dataset_ar import (
    CAUSAL_DECODE_STEPS, layout_for, validate_causal_decode_steps, validate_layout,
)
from cdrseqgen.methods.common import ban_tokens, logits_of, perplexity_from_nll, warp_logits


def _left_pad(prompts, pad_id, device):
    """Return left-padded IDs and an attention mask on device.

    Example: prompts=[[A, C], [D]], pad_id=P
      input IDs: [[A, C], [P, D]]
      mask:      [[1, 1], [0, 1]]

    Both tensors have shape [number of prompts, longest prompt length].
    Callers supply at least one nonempty prompt.
    """
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
    """Return ordered decoding attempts for one already assembled AR prompt.

    Steps:
      validate cap/layout -> native hook, or prepare default prompt batch
                          -> predict/sample one token per step -> attempt records

    Example: max_new_tokens=3
      A C EOS -> {"loop": "AC", "length": 2, "complete": True}
      A C D   -> {"loop": "ACD", "length": 3, "complete": False}

    The cap includes the terminator. Default records exclude EOS from the loop;
    early EOS can produce a complete empty loop. A missing EOS is incomplete.
    Native hooks may also distinguish terminated-invalid draws. Greedy decoding
    returns one attempt and bypasses top-k/top-p; stochastic draws retain their
    original order and duplicates. generator controls sampling RNG consumption.
    """
    # ------ 1. Validate the contract before native or default decoding. ------
    validate_causal_decode_steps(max_new_tokens)
    layout = validate_layout(layout_for(tokenizer), tokenizer)
    native_sampler = getattr(tokenizer, "causal_sampler", None)
    if native_sampler is not None:
        return native_sampler(model, tokenizer, prompt_ids, device=device, num_samples=num_samples,
                              temperature=temperature, top_k=top_k, top_p=top_p,
                              max_new_tokens=max_new_tokens, generator=generator)
    # ------ 2. Prepare the default batched prompt and allowed tokens. ------
    greedy = temperature == 0
    draws = 1 if greedy else num_samples
    model.eval()

    ids, mask = _left_pad([list(prompt_ids)] * draws, tokenizer.pad_id, device)
    terminator = layout.terminator_id(tokenizer)
    banned = {i for i in tokenizer.special_ids if i != terminator}

    generated = [[] for _ in range(draws)]
    finished = torch.zeros(draws, dtype=torch.bool, device=device)

    # ------ 3. Advance all attempts; completed rows remain padded. ------
    for _ in range(max_new_tokens):
        logits = logits_of(model(input_ids=ids, attention_mask=mask))[:, -1, :].float()
        logits = ban_tokens(logits, banned)
        if greedy:
            nxt = logits.argmax(-1)
        else:
            probs = torch.softmax(warp_logits(logits, temperature, top_k, top_p), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)

        nxt = torch.where(finished, torch.full_like(nxt, tokenizer.pad_id), nxt)
        # Transfer step bookkeeping in batches instead of synchronizing each GPU scalar.
        for tokens, token, was_finished in zip(generated, nxt.tolist(), finished.tolist()):
            if not was_finished and token != terminator:
                tokens.append(token)
        finished |= nxt.eq(terminator)
        if bool(finished.all()):
            break

        ids = torch.cat([ids, nxt[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones_like(nxt)[:, None]], dim=1)

    # ------ 4. Keep every attempt, including complete-empty and incomplete ones. ------
    return [{"loop": tokenizer.decode(tokens), "length": len(tokens), "complete": complete}
            for tokens, complete in zip(generated, finished.tolist())]


@torch.no_grad()
def perplexities(model, tokenizer, device, spans):
    """Return one generator perplexity per ordered (prompt, loop-token IDs) pair.

    Steps:
      layout.build_answer -> prompt + answer -> one forward pass per candidate
                          -> shifted answer-token loss -> exp(mean NLL)

    Prompt labels are ignored; the answer includes its terminator. Logits/labels
    are shifted once for next-token prediction. Non-finite or overflowing
    perplexity yields None. An empty spans list returns []. Generation and
    evaluation share this scorer.
    """
    scores = []
    for prompt, answer_ids in spans:
        answer = layout_for(tokenizer).build_answer(tokenizer, answer_ids)
        ids = list(prompt) + answer
        labels = [-100] * len(prompt) + answer
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        logits = logits_of(model(input_ids=tensor)).float()
        gold = torch.tensor([labels], dtype=torch.long, device=device)
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), gold[:, 1:].reshape(-1),
            ignore_index=-100, reduction="mean")
        scores.append(perplexity_from_nll(nll))
    return scores
