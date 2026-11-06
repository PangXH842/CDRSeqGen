"""Evaluate generated AR CDR candidates against a selected reference dataset.

Entry paths:
  evaluate.py -> eval_cli.main -> common.evaluation.run_evaluation_pipeline
              -> run_generative_eval
  training.finish_training -> common.evaluation.post_training_generation
                           -> eval_cli.evaluate (alias of run_generative_eval)

Classes/Functions:
  * prepare_evaluation  -- validate EOS options and resolve the decode cap
  * evaluation_options  -- return method options for the shared command pipeline
  * CausalStrategy      -- supply AR prompts, sampling and generator self-scoring
  * run_generative_eval -- select frameworks, score candidates and return a payload
  * perplexities        -- score candidate answers under their generating model

Notes:
  * common.evaluation loads standalone resources and controls context regimes;
    this module receives a loaded model, tokenizer, dataset and optional judge.
  * common.eos_decoding owns EOS attempt accounting and the strategy lifecycle.
    eval.generative / eval.metrics own shared scoring, aggregation and reports.
  * EOS selects length. Native references support selection and scoring;
    candidate generation uses the framework prompt and allowed context.
  * Generated-loop results are separate from Trainer's teacher-forced recovery.
    Self-perplexity uses the generator; an independent judge has its own inputs.
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
    """Validate AR evaluation, resolve settings in args and return None.

    Steps:
      inference checks -> set causal/inferred modes -> resolve/check decode cap

    Only inferred length is supported. max_new_tokens comes from args, then saved
    gen_max_new_tokens, then the AR default. The shared standalone pipeline calls
    this before loading model weights; sampling happens later.
    """
    validate_inference_args(args)
    if args.length_mode not in (None, 'inferred'):
        raise ValueError('AR evaluation chooses length via EOS; use --length_mode inferred')
    args.length_mode = 'inferred'
    if args.max_new_tokens is None:
        args.max_new_tokens = config.get('gen_max_new_tokens', CAUSAL_DECODE_STEPS)
    validate_causal_decode_steps(args.max_new_tokens)


def evaluation_options(args, device):
    """Return the resolved decode cap as run_generative_eval keyword arguments.

    AR has no separate length head to load; device is unused. The shared
    standalone pipeline calls this after loading the sequence model.
    """
    return {'max_new_tokens': args.max_new_tokens}


class CausalStrategy(EosDecodingStrategy):
    """Supply AR-specific operations to the shared EOS evaluation strategy.

    context -> build_causal_prompt, retaining prompt/conditioning metadata
    sample  -> ar.sampling.sample_loops, including native backbone sampler hooks
    score   -> perplexities for ordered prompt/answer pairs

    The base class handles framework lookup, draw filtering, attempt counters
    and judge inputs. The shared scoring loop calls those inherited operations.
    """

    def __init__(self, *, max_new_tokens=CAUSAL_DECODE_STEPS, **options):
        super().__init__(max_new_tokens=max_new_tokens, **options)

    def context(self, framework):
        """Build the training-format prompt and retain its layout metadata."""
        prompt, self.prompt_metadata = build_causal_prompt(self.tokenizer, framework)
        return prompt

    def sample(self, prompt):
        """Return sampler attempts using this run's decoding settings and RNG."""
        return sample_loops(self.model, self.tokenizer, prompt, **self.sampling_options())

    def score(self, spans):
        """Return generator perplexities for ordered (prompt, loop-token IDs) pairs."""
        return perplexities(self.model, self.tokenizer, self.device, spans)


def run_generative_eval(model, tokenizer, eval_dataset, *, max_new_tokens=CAUSAL_DECODE_STEPS,
                        **options):
    """Evaluate AR candidates through the shared EOS workflow; return its payload.

    run_eos_generative_eval() coordinates:
      select frameworks -> CausalStrategy -> run_eval_loop
                        -> attempt accounting -> report/payload -> optional files

    options forwards device, sampling/selection settings, optional judge resources,
    output_path, title and metadata. Resources are already loaded. output_path=None
    skips file output; the returned payload still contains records and summaries.
    This evaluates the supplied dataset; it does not choose validation versus test.
    """
    return run_eos_generative_eval(CausalStrategy, "ar", "causal", model, tokenizer, eval_dataset,
                                   max_new_tokens=max_new_tokens, **options)


@torch.no_grad()
def perplexities(model, tokenizer, device, spans):
    """Return one generator perplexity per ordered (prompt, loop-token IDs) pair.

    The layout builds each answer, including its terminator. Prompt labels are
    ignored; logits/labels are shifted once for next-token prediction. Each score
    is exp(mean answer-token NLL), using one forward pass per candidate.
    Non-finite NLL yields None. Generation case reporting also uses this function.
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
        scores.append(float(math.exp(nll.item())) if torch.isfinite(nll) else None)
    return scores
