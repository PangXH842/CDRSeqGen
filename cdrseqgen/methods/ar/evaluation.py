"""Causal generative evaluation: sample loops from held-out frameworks and score them.

  * CausalStrategy            -- the four things the shared loop cannot do for a causal model
  * run_causal_generative_eval -- select examples, run the loop, write generation_eval.json

The loop itself lives in :mod:`cdrseqgen.eval.generative` and is shared with the masked method, so
an AR-versus-MLM table is produced by one implementation. This module supplies only the strategy:

  * ``draw(i)``            -- decode to <eos> instead of expanding a mask span
  * ``self_scores(drawn)`` -- a TRUE perplexity, not a pseudo one. A masked LM must re-score each
    position with the others visible; a causal model factorises properly, so this is just the NLL of
    the answer under one forward pass.
  * ``judge_inputs(i, seq)`` -- rebuilt per candidate, because causal lengths vary by construction.
    The judge stays a masked LM, so the prompt is masked — that is the judge's format, not ours.
  * ``payload_length_policy()`` -- reports that length came from <eos>, which is the whole point:
    Findings §10's gap-sentinel head has no counterpart here.

**These numbers are free-running, not teacher-forced.** ``training.causal_compute_metrics`` reports
argmax over the true prefix and is for checkpoint selection; this samples. The README's "Teacher-
forced accuracy is not generation accuracy" gap applies to the causal side too.
"""
import logging
import math

import torch
from cdrseqgen.eval.metrics import blosum_provenance

from cdrseqgen.datasets.dataset_ar import CAUSAL_DECODE_STEPS, build_causal_prompt, layout_for
from cdrseqgen.datasets.dataset_common import framework_for_dataset
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.eval.generative import (Draw, SUMMARY_KEYS, aggregate, build_masked_judge_prompt,
                                       log_report, run_eval_loop, select_indices,
                                       write_confidence_figure, write_payload)
from cdrseqgen.methods.ar.sampling import sample_loops
from cdrseqgen.methods.common import logits_of

logger = logging.getLogger(__name__)


class CausalStrategy:
    """Decode-to-<eos> candidates for the shared evaluation loop."""

    supplies_native_length = False

    def __init__(self, *, model, tokenizer, device, generator, eval_dataset, judge_dataset,
                 judge_tokenizer, num_samples, temperature, top_k, top_p,
                 max_new_tokens=CAUSAL_DECODE_STEPS):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.generator = generator
        self.eval_dataset, self.judge_dataset = eval_dataset, judge_dataset
        self.judge_tokenizer = judge_tokenizer
        self.num_samples, self.temperature = num_samples, temperature
        self.top_k, self.top_p = top_k, top_p
        self.max_new_tokens = max_new_tokens
        self._framework_cache = (None, None)
        self.attempts = 0
        self.complete_nonempty = 0
        self.complete_empty = 0
        self.incomplete = 0

        self.length_matches = 0
        self.length_absolute_error_capped = 0

    # --- framework lookup -------------------------------------------------------------------
    # `framework_for_dataset` already unwraps a Subset, so the eval side passes the SUBSET index
    # and lets it resolve. The judge dataset is the full one, so that side resolves the index first.
    # Doing both is the bug this comment exists to prevent: it indexes the mapping twice.
    def framework(self, i):
        cached_index, cached = self._framework_cache
        if cached_index != i:
            cached = framework_for_dataset(self.eval_dataset, i)
            self._framework_cache = (i, cached)
        return cached

    def _judge_framework(self, i):
        if self.judge_dataset is self.eval_dataset:
            return self.framework(i)
        full_index = self.eval_dataset.indices[i] if hasattr(self.eval_dataset, "indices") else i
        return framework_for_dataset(self.judge_dataset, full_index)

    # --- the strategy protocol --------------------------------------------------------------
    def payload_length_policy(self):
        return {"mode": "inferred", "source": "eos_token"}

    def draw(self, i):
        framework = self.framework(i)
        # No reserve override: the prompt must match the one training built for this framework.
        prompt, _ = build_causal_prompt(self.tokenizer, framework)
        native = framework["target"].upper().strip()
        results = sample_loops(
            self.model, self.tokenizer, prompt, device=self.device,
            num_samples=self.num_samples, temperature=self.temperature,
            top_k=self.top_k, top_p=self.top_p, max_new_tokens=self.max_new_tokens,
            generator=self.generator)

        # Keep quality metrics conditional on a biologically non-empty completed loop, but retain
        # every failed attempt in the run-level validity accounting. Otherwise frameworks where the
        # model never emits EOS disappear and make the remaining sample look artificially healthy.
        complete = [r for r in results if r["complete"]]
        kept = [r for r in complete if r["loop"]]
        incomplete = len(results) - len(complete)
        empty = len(complete) - len(kept)
        self.attempts += len(results)
        length_matches = sum(
            r["complete"] and bool(r["loop"]) and len(r["loop"]) == len(native)
            for r in results)
        # An unterminated draw has no chosen length. Its cutoff length is still useful for sweep
        # ranking as long as it is penalized by at least one residue rather than looking exact when
        # the cutoff happens to equal the native length.
        capped_errors = [max(1, abs(len(r["loop"]) - len(native))) if not r["complete"]
                         else abs(len(r["loop"]) - len(native)) for r in results]
        self.complete_nonempty += len(kept)
        self.complete_empty += empty
        self.incomplete += incomplete
        self.length_matches += length_matches
        self.length_absolute_error_capped += sum(capped_errors)
        if not kept:
            return None
        seqs = [r["loop"].upper() for r in kept]
        return Draw(
            native=native,
            seqs=seqs,
            spans=[(prompt, self.tokenizer.encode(s)) for s in seqs],
            target_len=len(native),
            length_policy=self.payload_length_policy(),
            record_extra={
                "attempts": len(results), "incomplete": incomplete, "empty": empty,
                "length_matches": length_matches,
                "length_absolute_error_sum_capped": sum(capped_errors),
            },
        )

    @torch.no_grad()
    def self_scores(self, drawn):
        """True perplexity of each candidate under the model that produced it.

        exp of the mean negative log-likelihood over the answer tokens (loop + <eos>), from a single
        forward pass. The masked side needs one pass per position for its pseudo-perplexity; a
        causal factorisation does not.
        """
        return causal_perplexities(self.model, self.tokenizer, self.device, drawn.spans)

    def judge_inputs(self, i, seq):
        """Rebuild the judge's masked prompt at THIS candidate's length; causal lengths vary."""
        framework = self._judge_framework(i)
        if not seq:
            return None
        return build_masked_judge_prompt(self.judge_tokenizer, framework, len(seq))


def run_causal_generative_eval(model, tokenizer, eval_dataset, *, device, judge_dataset=None,
                               judge_model=None, judge_tokenizer=None, num_examples=150,
                               num_samples=20, temperature=1.0, top_k=0, top_p=1.0,
                               max_new_tokens=CAUSAL_DECODE_STEPS, stratify=False, seed=42,
                               skip_perplexity=False, output_path=None, title="", judge_name=None,
                               anchor_trim=ANCHOR_TRIM, meta=None):
    """Sample from held-out frameworks and write generation_eval.json, causal side."""
    import random

    model.eval()
    if isinstance(anchor_trim, bool) or not isinstance(anchor_trim, int) or anchor_trim < 0:
        raise ValueError("anchor_trim must be a non-negative integer")

    torch.manual_seed(seed)
    lengths = [framework_for_dataset(eval_dataset, i)["target_length"]
               for i in range(len(eval_dataset))]
    selected = select_indices(lengths, num_examples, stratify, random.Random(seed))
    generator = torch.Generator(device=device).manual_seed(seed)

    strategy = CausalStrategy(
        model=model, tokenizer=tokenizer, device=device, generator=generator,
        eval_dataset=eval_dataset,
        judge_dataset=judge_dataset if judge_dataset is not None else eval_dataset,
        judge_tokenizer=judge_tokenizer, num_samples=num_samples, temperature=temperature,
        top_k=top_k, top_p=top_p, max_new_tokens=max_new_tokens)

    records = run_eval_loop(
        strategy, selected, model=model, tokenizer=tokenizer, device=device,
        skip_perplexity=skip_perplexity, judge_model=judge_model,
        judge_tokenizer=judge_tokenizer, anchor_trim=anchor_trim)

    attempts = strategy.attempts
    validity = {
        "frameworks_selected": len(selected),
        "frameworks_with_valid_candidates": len(records),
        "attempts": attempts,
        "complete_nonempty": strategy.complete_nonempty,
        "complete_empty": strategy.complete_empty,
        "incomplete": strategy.incomplete,
        "valid_generation_rate": strategy.complete_nonempty / attempts if attempts else None,
        "completion_rate": ((strategy.complete_nonempty + strategy.complete_empty) / attempts
                            if attempts else None),
        "length_matches": strategy.length_matches,
        # Unlike summary.length_match_rate/length_absolute_error_mean, these denominators retain
        # incomplete and empty attempts, so a sweep cannot improve by dropping failed decodes.
        "fixed_length_acceptance_rate": strategy.length_matches / attempts if attempts else None,
        "length_absolute_error_mean_capped": (
            strategy.length_absolute_error_capped / attempts if attempts else None),
    }

    payload = dict(meta or {})
    payload.update({
        "schema_version": "cdrseqgen.generation_eval.v1",
        # A BLOSUM number without its matrix and gap penalties is not reproducible.
        "blosum": blosum_provenance(),
        "method": "ar",
        "judge": judge_name,
        "length_policy": strategy.payload_length_policy(),
        "anchor_trim_each_end": anchor_trim,
        "decoding": {
            "mode": "causal", "num_samples": num_samples, "temperature": temperature,
            "top_k": top_k, "top_p": top_p, "max_new_tokens": max_new_tokens, "seed": seed,
        },
        "n_eval_total": len(eval_dataset),
        "n_selected": len(selected),
        "n_generated": len(records),
        "generation_validity": validity,
        "summary": {key: aggregate(records, key)[0] for key, _ in SUMMARY_KEYS},
        "records": records,
    })
    log_report(records, title=title, judge_name=judge_name)
    write_payload(payload, output_path, len(records))
    write_confidence_figure(payload, output_path, title)
    return payload


@torch.no_grad()
def causal_perplexities(model, tokenizer, device, spans):
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
