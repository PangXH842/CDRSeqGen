"""Seq2seq generative evaluation: sample loops from held-out frameworks and score them.

  * prepare_evaluation  -- reject length controls (EOS chooses length) and fix the decode cap
  * evaluation_options  -- the method options the shared command forwards to the evaluator
  * Seq2SeqStrategy     -- the three things the shared loop cannot do for an encoder-decoder
  * run_generative_eval -- select examples, run the loop, write generation_eval.json
  * perplexities        -- true perplexity of each candidate under the model that produced it

The loop itself lives in :mod:`cdrseqgen.eval.generative` and is shared with both siblings, so an
s2s-versus-ar-versus-mlm table is produced by one implementation. The strategy's bookkeeping
(validity accounting, judge prompts, the payload) is shared with the causal method in
:mod:`cdrseqgen.methods.common.eos_decoding`; this module supplies only what is encoder-decoder:

  * ``context(framework)``  -- the encoder input training built for this framework
  * ``sample(encoder_ids)`` -- encode the framework once, decode to <eos>
  * ``score(spans)``        -- a TRUE perplexity, not a pseudo one. Like the causal model and unlike
    the masked one, the decoder factorises properly, so this is the NLL of the loop under one pass.

Candidates report that length came from <eos>, as on the causal side. The judge stays a masked LM,
so its prompt is masked — that is the judge's format.

**These numbers are free-running, not teacher-forced.** ``training.compute_metrics`` reports
argmax over the true prefix and is for checkpoint selection; this samples.
"""
import math

import torch

from cdrseqgen.datasets.dataset_s2s import (
    S2S_DECODE_STEPS, build_decoder_pair, build_encoder_input, validate_s2s_decode_steps,
)
from cdrseqgen.methods.common import logits_of
from cdrseqgen.methods.common.eos_decoding import EosDecodingStrategy, run_eos_generative_eval
from cdrseqgen.methods.s2s.arguments import validate_inference_args
from cdrseqgen.methods.s2s.sampling import sample_loops


def prepare_evaluation(args, config):
    validate_inference_args(args)
    if args.length_mode not in (None, 'inferred'):
        raise ValueError('S2S evaluation chooses length via EOS; use --length_mode inferred')
    args.length_mode = 'inferred'
    if args.max_new_tokens is None:
        args.max_new_tokens = config.get('gen_max_new_tokens', S2S_DECODE_STEPS)
    validate_s2s_decode_steps(args.max_new_tokens)


def evaluation_options(args, device):
    return {'max_new_tokens': args.max_new_tokens}


class Seq2SeqStrategy(EosDecodingStrategy):
    """Decode-to-<eos> candidates for the shared evaluation loop."""

    def __init__(self, *, max_new_tokens=S2S_DECODE_STEPS, **options):
        super().__init__(max_new_tokens=max_new_tokens, **options)

    def context(self, framework):
        # No override of any kind: the encoder input must match the one training built for this
        # framework, which is why build_encoder_input takes no generation-time parameter.
        encoder_ids, _ = build_encoder_input(self.tokenizer, framework)
        return encoder_ids

    def sample(self, encoder_ids):
        return sample_loops(self.model, self.tokenizer, encoder_ids, **self.sampling_options())

    def score(self, spans):
        return perplexities(self.model, self.tokenizer, self.device, spans)


def run_generative_eval(model, tokenizer, eval_dataset, *, max_new_tokens=S2S_DECODE_STEPS,
                        **options):
    """Sample from held-out frameworks and write generation_eval.json, encoder-decoder side.

    ``options`` are those of :func:`common.eos_decoding.run_eos_generative_eval`: device, judge
    resources, sampling settings, output path, title and metadata.
    """
    return run_eos_generative_eval(Seq2SeqStrategy, "s2s", "seq2seq", model, tokenizer,
                                   eval_dataset, max_new_tokens=max_new_tokens, **options)


@torch.no_grad()
def perplexities(model, tokenizer, device, spans):
    """Score (encoder_ids, loop_ids) pairs under the model, including the answer terminator.

    No shift here either: ``build_decoder_pair`` offsets the pair, so the logit at t answers the
    label at t. This mirrors the loss in ``models/transformer.py`` and differs from
    ``methods/ar/evaluation.perplexities``, which must shift because its labels are aligned instead.
    """
    # Candidates from one framework arrive consecutively. Retain just its encoder output,
    # avoiding another full framework encode per candidate without growing a dataset-wide cache.
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
        scores.append(float(math.exp(nll.item())) if torch.isfinite(nll) else None)
    return scores
