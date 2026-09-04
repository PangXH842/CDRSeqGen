"""Core script for retrieving generative evaluation metrics from a trained model.

Two callers: `train.py` runs it post-training on the run it just finished (no judge by default),
and `evaluate.py` is the standalone entrypoint for re-evaluating an existing checkpoint (where the
judge is worth setting up).

  * run_generative_eval -- master function: load eval dataset, select indices, sample from the
                           model, score, then log the summary and write the JSON
  * select_indices      -- randomly select CDRs from eval dataset,
                           stratify by CDR lengths if option selected
  * build_judge         -- optional, set specified third-party model to judge perplexity
  * masked_view         -- one dataset item -> (input_ids, mask_positions, native_ids)
  * bucket_of           -- map a loop length to its short/medium/long bucket
  * aggregate           -- (mean, stdev) of one metric across the sampled records
  * format_report       -- build the summary table as a string, without emitting it
  * log_report          -- emit that table through the logger, as one record

Imports:
  * metrics.calculate               -- retrieve dict of sequence-based metrics for generated samples
  * generation.sample_masked_span   -- shared independent / iterative decoder dispatch
  * sampling.pseudo_log_likelihood  -- calculate PLL and pseudo-perplexity of a filled loop
"""
import logging
import os
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from cdrseqgen.artifacts import atomic_write_json
from cdrseqgen.data.dataset_mlm import AntibodyInfillingDataset
from cdrseqgen.eval.metrics import (
    amino_acid_recovery,
    blosum_provenance,
    calculate,
    confidence_metrics,
    core_anchor_indices,
    recovery_at_indices,
)
from cdrseqgen.methods.mlm.recovery import ANCHOR_TRIM
from cdrseqgen.eval.generative import (  # noqa: F401  (re-exported for callers)
    Draw, build_judge, SUMMARY_KEYS, aggregate, bucket_of, format_report, log_report,
    run_eval_loop, score_with_judge, select_indices, write_confidence_figure,
    write_payload)
from cdrseqgen.methods.mlm.generation import sample_masked_span
from cdrseqgen.methods.mlm.length import (
    native_mask_length,
    resolve_generation_length_policy,
)
from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood

logger = logging.getLogger(__name__)

# Split CDR length into buckets, especially for H3, for summary purposes. Imported rather than
# redeclared so a bucket boundary means the same thing here and in the benchmark scorer.
from cdrseqgen.eval.metrics import LENGTH_BUCKETS  # noqa: E402


def masked_view(item):
    """(input_ids, mask_positions, native_ids) for one dataset item"""
    input_ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()
    mask_positions = [p for p, l in enumerate(labels) if l != -100]
    native_ids = [labels[p] for p in mask_positions]
    return input_ids, mask_positions, native_ids


class FixedNativeStrategy:
    """Original protocol: the dataset's own masked view supplies the mask count and the input.

    The input tensor is taken from the eval dataset rather than reassembled, so partner-chain and
    antigen context are exactly the ones the model trained against. Every recovery number in the
    README came from this path; it is kept byte-for-byte.
    """

    supplies_native_length = True

    def __init__(self, *, model, tokenizer, device, generator, eval_dataset, judge_dataset,
                 judge_tokenizer, decoding, num_samples, temperature, top_k, top_p):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.generator = generator
        self.eval_dataset, self.judge_dataset = eval_dataset, judge_dataset
        self.judge_tokenizer = judge_tokenizer
        self.decoding, self.num_samples = decoding, num_samples
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self._judge_cache = (None, None)

    def payload_length_policy(self):
        return {"mode": "fixed", "source": "native_mask_count"}

    def draw(self, i):
        input_ids, mask_positions, native_ids = masked_view(self.eval_dataset[i])
        if not mask_positions:
            return None
        length_decision = native_mask_length(len(mask_positions))
        native = self.tokenizer.decode(native_ids).upper().strip()
        sampled_ids = sample_masked_span(
            self.model, self.tokenizer, self.device, self.generator,
            input_ids, mask_positions,
            decoding=self.decoding, num_samples=self.num_samples,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
        )
        spans = [(input_ids, mask_positions, ids) for ids in sampled_ids]
        seqs = [self.tokenizer.decode(ids).upper().strip() for ids in sampled_ids]
        return Draw(
            native=native,
            seqs=seqs,
            spans=spans,
            target_len=length_decision.length,
            length_policy=length_decision.as_dict(),
            record_extra={},
        )

    def self_scores(self, drawn):
        """Pseudo-perplexity: a masked LM scores each hidden position with the others visible."""
        return [
            pseudo_log_likelihood(self.model, input_ids, mask_positions, token_ids,
                                  self.tokenizer.special_ids, self.tokenizer.mask_id, self.device)[1]
            for input_ids, mask_positions, token_ids in drawn.spans
        ]

    def judge_inputs(self, i, _seq):
        """Every candidate has the native length here, so one prompt serves them all."""
        cached_index, cached_prompt = self._judge_cache
        if cached_index == i:
            return cached_prompt
        # Map the eval-subset index back to the full dataset so the judge sees the same entry
        full_idx = self.eval_dataset.indices[i] if hasattr(self.eval_dataset, "indices") else i
        judge_input, judge_mask_positions, _ = masked_view(self.judge_dataset[full_idx])
        self._judge_cache = (i, (judge_input, judge_mask_positions))
        return self._judge_cache[1]


def run_generative_eval(model, tokenizer, eval_dataset, *, device,
                        decoding="independent", num_samples=20,
                        temperature=1.0, top_k=0, top_p=1.0,
                        length_mode="fixed_native", length_predictor=None,
                        min_length=None, max_length=None,
                        length_temperature=1.0,
                        num_examples=200, stratify=False, seed=42,
                        skip_perplexity=False,
                        judge_model=None, judge_tokenizer=None,
                        judge_dataset=None, judge_name=None,
                        anchor_trim=ANCHOR_TRIM,
                        output_path=None, meta=None, title=""):
    """Sample + score candidate loops for a subsample of the eval split, returns the payload dict.

    Both length tracks run the same loop, aggregation, reporting, and writer; they differ only in
    the strategy that turns one eval index into candidates, which is the one part that must not be
    shared (the fixed track is told the native mask count, the variable track must never be).
    """
    model.eval()
    if isinstance(anchor_trim, bool) or not isinstance(anchor_trim, int) or anchor_trim < 0:
        raise ValueError("anchor_trim must be a non-negative integer")

    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Native loop lengths, used only to choose and bucket examples — never passed to generation.
    lengths = [eval_dataset[i]["target_len"] for i in range(len(eval_dataset))]
    rng = random.Random(seed)
    selected = select_indices(lengths, num_examples, stratify, rng)

    length_policy = resolve_generation_length_policy(
        length_mode, min_length=min_length, max_length=max_length,
    )
    if length_policy is not None:
        if length_predictor is None:
            raise FileNotFoundError("variable-length evaluation requires a length predictor")
        from cdrseqgen.methods.mlm.variable_length_evaluation import VariableLengthStrategy
        strategy = VariableLengthStrategy(
            model=model, tokenizer=tokenizer, device=device, generator=generator,
            eval_dataset=eval_dataset, judge_dataset=judge_dataset,
            judge_tokenizer=judge_tokenizer, length_predictor=length_predictor,
            length_policy=length_policy, length_temperature=length_temperature,
            decoding=decoding, num_samples=num_samples, temperature=temperature,
            top_k=top_k, top_p=top_p,
        )
    else:
        strategy = FixedNativeStrategy(
            model=model, tokenizer=tokenizer, device=device, generator=generator,
            eval_dataset=eval_dataset, judge_dataset=judge_dataset,
            judge_tokenizer=judge_tokenizer, decoding=decoding, num_samples=num_samples,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

    records = run_eval_loop(
        strategy, selected,
        model=model, tokenizer=tokenizer, device=device,
        skip_perplexity=skip_perplexity,
        judge_model=judge_model, judge_tokenizer=judge_tokenizer,
        anchor_trim=anchor_trim,
    )

    log_report(records, title=title, judge_name=judge_name)

    payload = dict(meta or {})
    payload.update({
        "schema_version": "cdrseqgen.generation_eval.v1",
        # A BLOSUM number without its matrix and gap penalties is not reproducible.
        "blosum": blosum_provenance(),
        "judge": judge_name,
        "length_policy": strategy.payload_length_policy(),
        "anchor_trim_each_end": anchor_trim,
        "decoding": {
            "mode": decoding, "num_samples": num_samples, "temperature": temperature,
            "top_k": top_k, "top_p": top_p, "seed": seed,
        },
        "n_eval_total": len(eval_dataset),
        "n_generated": len(records),
        "summary": {key: aggregate(records, key)[0] for key, _ in SUMMARY_KEYS},
        "records": records,
    })

    write_payload(payload, output_path, len(records))
    write_confidence_figure(payload, output_path, title)
    return payload


