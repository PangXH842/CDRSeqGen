"""Coordinate generated-loop evaluation for MLM models.

Callers:
  * evaluate.py -> mlm.eval_cli.evaluate -> run_generative_eval
  * mlm.training -> common.evaluation.post_training_generation
                 -> mlm.eval_cli.evaluate -> run_generative_eval
  Both paths supply a loaded model, tokenizer, evaluation dataset and optional judge.

Functions/classes defined here:
  * masked_view         -- extract prompt tokens, supervised positions and native tokens
  * FixedNativeStrategy -- fill the dataset's existing masks, exposing the native length
  * run_generative_eval -- select examples, choose a strategy, score and return the payload

Workflow:
  fixed_native     -> FixedNativeStrategy
  bounded/inferred -> variable_length_evaluation.VariableLengthStrategy
                   -> eval.generative.run_eval_loop -> report + JSON/figure output

Shared responsibilities:
  * cdrseqgen.eval.metrics calculates sequence measurements.
  * cdrseqgen.eval.generative owns the scoring loop, aggregation and output helpers.
    Its imported helpers remain available here for existing callers.
  * mlm.generation fills masked spans; mlm.sampling supplies MLM pseudo-perplexity.
  * mlm.recovery supplies Trainer validation metrics, a separate evaluation stage.
"""
import logging
import os
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from cdrseqgen.artifacts import atomic_write_json
from cdrseqgen.datasets.dataset_mlm import MaskedInfillingDataset
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

# Shared native-length reporting buckets; also re-exported for existing callers.
from cdrseqgen.eval.metrics import LENGTH_BUCKETS  # noqa: E402


def masked_view(item):
    """Return prompt IDs, supervised positions and native IDs as three lists.

    Example: labels use -100 for positions excluded from the loss
      input_ids:      [10, 99, 99, 20]    (99 is a mask in this example)
      labels:         [-100, 3, 4, -100]
      mask_positions: [1, 2]
      native_ids:     [3, 4]

    Positions come from labels, without checking the input token at each position.
    Fixed-native generation expects the dataset's fully masked evaluation view.
    """
    input_ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()
    mask_positions = [p for p, l in enumerate(labels) if l != -100]
    native_ids = [labels[p] for p in mask_positions]
    return input_ids, mask_positions, native_ids


class FixedNativeStrategy:
    """Adapt the dataset's native-length masked prompt to the shared scoring loop.

    Example: three supervised loop positions -> three masks -> length-3 candidates.

    The supplied dataset determines partner/antigen context, including any ablation.
    Candidate generation receives the native mask count. Each candidate retains its
    prompt for self-scoring; the judge uses a separate prompt from judge_dataset.
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
        """Describe the native-mask-count policy for the evaluation payload."""
        return {"mode": "fixed", "source": "native_mask_count"}

    def draw(self, i):
        """Return candidates and scoring prompts for index i, or None if no masks exist."""
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
        """Return generator pseudo-perplexities in candidate order.

        Each filled loop is re-masked one position at a time; the other candidate
        residues remain visible. sampling.pseudo_log_likelihood performs the scoring.
        """
        return [
            pseudo_log_likelihood(self.model, input_ids, mask_positions, token_ids,
                                  self.tokenizer.special_ids, self.tokenizer.mask_id, self.device)[1]
            for input_ids, mask_positions, token_ids in drawn.spans
        ]

    def judge_inputs(self, i, _seq):
        """Return and cache the judge's masked prompt for evaluation index i.

        All candidates share the native length, so _seq does not change this prompt.
        A Subset index is mapped back to the corresponding judge_dataset entry.
        """
        cached_index, cached_prompt = self._judge_cache
        if cached_index == i:
            return cached_prompt
        # Resolve one outer Subset layer; the judge dataset uses the full corpus order.
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
    """Generate and score loops from a supplied evaluation dataset; return a payload dict.

    Steps:
      1. Set model.eval(), seed sampling and select evaluation frameworks.
      2. Choose native-length prompts or a bounded/inferred strategy with a length head.
      3. Run shared sequence, optional self-perplexity and optional judge scoring.
      4. Log the report, assemble the payload and call the JSON/figure writers.

    Notes:
      * The caller loads the model, datasets and optional judge before this function.
      * Native lengths support selection/reporting; only fixed_native exposes them
        to generation. Bounded/inferred generation requires length_predictor.
      * skip_perplexity disables generator self-scoring; judge scoring is independent.
      * n_generated counts framework records, each containing its candidate draws.
      * Errors propagate to the caller. Model evaluation mode and the Torch seed
        are set here and are not restored on return.
    """
    model.eval()
    if isinstance(anchor_trim, bool) or not isinstance(anchor_trim, int) or anchor_trim < 0:
        raise ValueError("anchor_trim must be a non-negative integer")

    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # This length list selects examples; each strategy controls its generation inputs.
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


