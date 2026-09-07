"""Seq2seq generative evaluation: sample loops from held-out frameworks and score them.

  * Seq2SeqStrategy          -- the four things the shared loop cannot do for an encoder-decoder
  * run_s2s_generative_eval  -- select examples, run the loop, write generation_eval.json

The loop itself lives in :mod:`cdrseqgen.eval.generative` and is shared with both siblings, so an
s2s-versus-ar-versus-mlm table is produced by one implementation. This module supplies only the
strategy, and it is deliberately near-identical to ``methods/ar/evaluation.py``:

  * ``draw(i)``            -- encode the framework once, decode to <eos>
  * ``self_scores(drawn)`` -- a TRUE perplexity, not a pseudo one. Like the causal model and unlike
    the masked one, the decoder factorises properly, so this is the NLL of the loop under one pass.
  * ``judge_inputs(i, seq)`` -- rebuilt per candidate, because generated lengths vary by
    construction. The judge stays a masked LM, so the prompt is masked — that is the judge's format.
  * ``payload_length_policy()`` -- reports that length came from <eos>, as on the causal side.

**These numbers are free-running, not teacher-forced.** ``training.s2s_compute_metrics`` reports
argmax over the true prefix and is for checkpoint selection; this samples.
"""
import logging
import math

import torch

from cdrseqgen.data.dataset_common import framework_for_dataset
from cdrseqgen.data.dataset_s2s import (S2S_DECODE_STEPS, build_decoder_pair, build_encoder_input)
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.eval.generative import (Draw, SUMMARY_KEYS, aggregate, build_masked_judge_prompt,
                                       log_report, run_eval_loop, select_indices,
                                       write_confidence_figure, write_payload)
from cdrseqgen.eval.metrics import blosum_provenance
from cdrseqgen.methods.common import logits_of
from cdrseqgen.methods.s2s.sampling import sample_loops

logger = logging.getLogger(__name__)


class Seq2SeqStrategy:
    """Decode-to-<eos> candidates for the shared evaluation loop."""

    supplies_native_length = False

    def __init__(self, *, model, tokenizer, device, generator, eval_dataset, judge_dataset,
                 judge_tokenizer, num_samples, temperature, top_k, top_p,
                 max_new_tokens=S2S_DECODE_STEPS):
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
        # No override of any kind: the encoder input must match the one training built for this
        # framework, which is why build_encoder_input takes no generation-time parameter.
        encoder_ids, _ = build_encoder_input(self.tokenizer, framework)
        native = framework["target"].upper().strip()
        results = sample_loops(
            self.model, self.tokenizer, encoder_ids, device=self.device,
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
            spans=[(encoder_ids, self.tokenizer.encode(s)) for s in seqs],
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

        exp of the mean negative log-likelihood over the decoder stream (loop + <eos>), from a
        single forward pass. The masked side needs one pass per position for its pseudo-perplexity;
        a proper autoregressive decoder does not.
        """
        return s2s_perplexities(self.model, self.tokenizer, self.device, drawn.spans)

    def judge_inputs(self, i, seq):
        """Rebuild the judge's masked prompt at THIS candidate's length; generated lengths vary."""
        framework = self._judge_framework(i)
        if not seq:
            return None
        return build_masked_judge_prompt(self.judge_tokenizer, framework, len(seq))


def run_s2s_generative_eval(model, tokenizer, eval_dataset, *, device, judge_dataset=None,
                            judge_model=None, judge_tokenizer=None, num_examples=150,
                            num_samples=20, temperature=1.0, top_k=0, top_p=1.0,
                            max_new_tokens=S2S_DECODE_STEPS, stratify=False, seed=42,
                            skip_perplexity=False, output_path=None, title="", judge_name=None,
                            anchor_trim=ANCHOR_TRIM, meta=None):
    """Sample from held-out frameworks and write generation_eval.json, encoder-decoder side."""
    import random

    model.eval()
    if isinstance(anchor_trim, bool) or not isinstance(anchor_trim, int) or anchor_trim < 0:
        raise ValueError("anchor_trim must be a non-negative integer")

    torch.manual_seed(seed)
    lengths = [framework_for_dataset(eval_dataset, i)["target_length"]
               for i in range(len(eval_dataset))]
    selected = select_indices(lengths, num_examples, stratify, random.Random(seed))
    generator = torch.Generator(device=device).manual_seed(seed)

    strategy = Seq2SeqStrategy(
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
        "method": "s2s",
        "judge": judge_name,
        "length_policy": strategy.payload_length_policy(),
        "anchor_trim_each_end": anchor_trim,
        "decoding": {
            "mode": "seq2seq", "num_samples": num_samples, "temperature": temperature,
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
def s2s_perplexities(model, tokenizer, device, spans):
    """Score (encoder_ids, loop_ids) pairs under the model, including the answer terminator.

    No shift here either: ``build_decoder_pair`` offsets the pair, so the logit at t answers the
    label at t. This mirrors the loss in ``models/transformer.py`` and differs from the causal
    method's ``causal_perplexities``, which must shift because its labels are aligned instead.
    """
    scores = []
    for encoder_ids, loop_ids in spans:
        decoder_input, labels = build_decoder_pair(tokenizer, loop_ids)
        source = torch.tensor([list(encoder_ids)], dtype=torch.long, device=device)
        decoder = torch.tensor([decoder_input], dtype=torch.long, device=device)
        gold = torch.tensor([labels], dtype=torch.long, device=device)
        logits = logits_of(model(input_ids=source, decoder_input_ids=decoder)).float()
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), gold.reshape(-1),
            ignore_index=-100, reduction="mean")
        scores.append(float(math.exp(nll.item())) if torch.isfinite(nll) else None)
    return scores
