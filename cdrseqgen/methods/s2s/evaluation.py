"""Evaluate generated S2S CDR candidates against a selected reference dataset.

Entry paths:
  evaluate.py -> eval_cli.main -> common.evaluation.run_evaluation_pipeline
              -> run_generative_eval
  training.finish_training -> common.evaluation.post_training_generation
                           -> eval_cli.evaluate (alias of run_generative_eval)

Classes/Functions:
  * prepare_evaluation  -- validate EOS options and resolve the decode cap
  * evaluation_options  -- return method options for the shared command pipeline
  * Seq2SeqStrategy     -- supply encoder inputs, sampling and generator self-scoring
  * run_generative_eval -- select frameworks, score candidates and return a payload
  * perplexities        -- score candidate answers under their generating model

Notes:
  * common.evaluation loads standalone resources and controls context regimes;
    this module receives a loaded model, tokenizer, dataset and optional judge.
  * common.eos_decoding owns EOS attempt accounting and the strategy lifecycle.
    eval.generative / eval.metrics own shared scoring, aggregation and reports.
  * EOS selects length. Native references support selection and scoring;
    candidate generation uses the encoder input and allowed context.
  * Generated-loop results are separate from Trainer's teacher-forced recovery.
    Self-perplexity uses the generator; an independent judge has its own inputs.
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
    """Validate S2S evaluation, resolve settings in args and return None.

    Steps:
      inference checks -> set seq2seq/inferred modes -> resolve/check decode cap

    Only inferred length is supported. max_new_tokens comes from args, then saved
    gen_max_new_tokens, then the S2S default. The shared standalone pipeline calls
    this before loading model weights; sampling happens later.
    """
    validate_inference_args(args)
    if args.length_mode not in (None, 'inferred'):
        raise ValueError('S2S evaluation chooses length via EOS; use --length_mode inferred')
    args.length_mode = 'inferred'
    if args.max_new_tokens is None:
        args.max_new_tokens = config.get('gen_max_new_tokens', S2S_DECODE_STEPS)
    validate_s2s_decode_steps(args.max_new_tokens)


def evaluation_options(args, device):
    """Return the resolved decode cap as run_generative_eval keyword arguments.

    S2S has no separate length head to load; device is unused. The shared
    standalone pipeline calls this after loading the sequence model.
    """
    return {'max_new_tokens': args.max_new_tokens}


class Seq2SeqStrategy(EosDecodingStrategy):
    """Supply S2S-specific operations to the shared EOS evaluation strategy.

    context -> build_encoder_input, using the training-format framework encoding
    sample  -> s2s.sampling.sample_loops, encoding context and decoding to EOS
    score   -> perplexities for ordered encoder-input/loop pairs

    The base class handles framework lookup, draw filtering, attempt counters
    and judge inputs. The shared scoring loop calls those inherited operations.
    """

    def __init__(self, *, max_new_tokens=S2S_DECODE_STEPS, **options):
        super().__init__(max_new_tokens=max_new_tokens, **options)

    def context(self, framework):
        """Build the training-format encoder input for the supplied framework."""
        encoder_ids, _ = build_encoder_input(self.tokenizer, framework)
        return encoder_ids

    def sample(self, encoder_ids):
        """Return sampler attempts using this run's decoding settings and RNG."""
        return sample_loops(self.model, self.tokenizer, encoder_ids, **self.sampling_options())

    def score(self, spans):
        """Return generator perplexities for ordered (encoder IDs, loop IDs) pairs."""
        return perplexities(self.model, self.tokenizer, self.device, spans)


def run_generative_eval(model, tokenizer, eval_dataset, *, max_new_tokens=S2S_DECODE_STEPS,
                        **options):
    """Evaluate S2S candidates through the shared EOS workflow; return its payload.

    run_eos_generative_eval() coordinates:
      select frameworks -> Seq2SeqStrategy -> run_eval_loop
                        -> attempt accounting -> report/payload -> optional files

    options forwards device, sampling/selection settings, optional judge resources,
    output_path, title and metadata. Resources are already loaded. output_path=None
    skips file output; the returned payload still contains records and summaries.
    This evaluates the supplied dataset; it does not choose validation versus test.
    """
    return run_eos_generative_eval(Seq2SeqStrategy, "s2s", "seq2seq", model, tokenizer,
                                   eval_dataset, max_new_tokens=max_new_tokens, **options)


@torch.no_grad()
def perplexities(model, tokenizer, device, spans):
    """Return one generator perplexity per ordered (encoder IDs, loop IDs) pair.

    build_decoder_pair() aligns decoder inputs and labels, including the answer
    terminator; no further shift is applied. Each score is exp(mean answer-token
    NLL). Consecutive candidates with the same encoder input reuse its encoding;
    the decoder runs for each candidate. Non-finite NLL yields None.
    Generation case reporting also uses this function.
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
        scores.append(float(math.exp(nll.item())) if torch.isfinite(nll) else None)
    return scores
