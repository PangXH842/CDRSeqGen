"""Causal generative evaluation: sample loops from held-out frameworks and score them.

  * prepare_evaluation  -- reject length controls (EOS chooses length) and fix the decode cap
  * evaluation_options  -- the method options the shared command forwards to the evaluator
  * CausalStrategy      -- the three things the shared loop cannot do for a causal model
  * run_generative_eval -- select examples, run the loop, write generation_eval.json
  * perplexities        -- true perplexity of each candidate under the model that produced it

The loop itself lives in :mod:`cdrseqgen.eval.generative` and is shared with the masked method, so
an AR-versus-MLM table is produced by one implementation. The strategy's bookkeeping (validity
accounting, judge prompts, the payload) is shared with the encoder-decoder method in
:mod:`cdrseqgen.methods.common.eos_decoding`; this module supplies only what is causal:

  * ``context(framework)`` -- the prompt training built for this framework
  * ``sample(prompt)``     -- decode to <eos> instead of expanding a mask span
  * ``score(spans)``       -- a TRUE perplexity, not a pseudo one. A masked LM must re-score each
    position with the others visible; a causal model factorises properly, so this is just the NLL of
    the answer under one forward pass.

Candidates report that length came from <eos>, which is the whole point: Findings §10's
gap-sentinel head has no counterpart here. The judge stays a masked LM, so its prompt is masked —
that is the judge's format, not ours.

**These numbers are free-running, not teacher-forced.** ``training.compute_metrics`` reports
argmax over the true prefix and is for checkpoint selection; this samples. The README's "Teacher-
forced accuracy is not generation accuracy" gap applies to the causal side too.
"""
import math

import torch

from cdrseqgen.datasets.dataset_ar import (
    CAUSAL_DECODE_STEPS, build_causal_prompt, layout_for, validate_causal_decode_steps,
)
from cdrseqgen.methods.ar.arguments import validate_inference_args
from cdrseqgen.methods.ar.sampling import sample_loops
from cdrseqgen.methods.common import logits_of
from cdrseqgen.methods.common.eos_decoding import EosDecodingStrategy, run_eos_generative_eval


def prepare_evaluation(args, config):
    validate_inference_args(args)
    if args.length_mode not in (None, 'inferred'):
        raise ValueError('AR evaluation chooses length via EOS; use --length_mode inferred')
    args.length_mode = 'inferred'
    if args.max_new_tokens is None:
        args.max_new_tokens = config.get('gen_max_new_tokens', CAUSAL_DECODE_STEPS)
    validate_causal_decode_steps(args.max_new_tokens)


def evaluation_options(args, device):
    return {'max_new_tokens': args.max_new_tokens}


class CausalStrategy(EosDecodingStrategy):
    """Decode-to-<eos> candidates for the shared evaluation loop."""

    def __init__(self, *, max_new_tokens=CAUSAL_DECODE_STEPS, **options):
        super().__init__(max_new_tokens=max_new_tokens, **options)

    def context(self, framework):
        # No reserve override: the prompt must match the one training built for this framework.
        prompt, _ = build_causal_prompt(self.tokenizer, framework)
        return prompt

    def sample(self, prompt):
        return sample_loops(self.model, self.tokenizer, prompt, **self.sampling_options())

    def score(self, spans):
        return perplexities(self.model, self.tokenizer, self.device, spans)


def run_generative_eval(model, tokenizer, eval_dataset, *, max_new_tokens=CAUSAL_DECODE_STEPS,
                        **options):
    """Sample from held-out frameworks and write generation_eval.json, causal side.

    ``options`` are those of :func:`common.eos_decoding.run_eos_generative_eval`: device, judge
    resources, sampling settings, output path, title and metadata.
    """
    return run_eos_generative_eval(CausalStrategy, "ar", "causal", model, tokenizer, eval_dataset,
                                   max_new_tokens=max_new_tokens, **options)


@torch.no_grad()
def perplexities(model, tokenizer, device, spans):
    """Score prompt/answer pairs under the causal model, including the answer terminator."""
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
        scores.append(float(math.exp(nll.item())) if torch.isfinite(nll) else None)
    return scores
