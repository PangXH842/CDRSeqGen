"""Shared evaluation pipeline for generated CDR candidates.

Workflow:                                      [*] = method-specific strategy hook
  method wrapper                              -- choose split, seed and generation settings
    build_judge (optional)                    -- masked-LM reference scorer
    select_indices                            -- choose evaluation frameworks
    run_eval_loop
      [*] draw(index) -> Draw / None           -- generate and retain candidate sequences
      calculate                               -- sequence metrics from metrics.py
      [*] self_scores(Draw)                    -- generator scores in candidate order
      score_with_judge
        [*] judge_inputs(index, sequence)      -- judge prompt and mask positions
        pseudo_log_likelihood                 -- masked-LM reference scores
      candidate details + confidence metrics  -- one scored record per returned Draw
    log_report -> format_report -> aggregate  -- framework means and population stddevs
    write_payload / write_confidence_figure   -- JSON and optional PNG

Entities and helpers:
  * Draw                       -- one framework's candidates and scoring inputs
  * bucket_of                  -- reference-length bucket used in selection and reports
  * normalize_scores           -- require one scalar score per candidate; exclude NaN/Inf
  * build_masked_judge_prompt  -- framework + candidate length -> judge input and mask positions
  * [*] payload_length_policy  -- strategy metadata read by the method wrapper

Notes:
  * Strategies own decoding, validity accounting and generator self-scoring.
    Wrappers assemble the final payload and choose generator/judge context independently.
  * MLM self-scores are pseudo-perplexities; AR/S2S use their causal factorizations.
    The independent judge is a masked LM for every method.
  * Means are computed per framework; aggregate gives each contributing framework
    equal weight. Each metric can have a different eligible population.
"""
import logging
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from numbers import Number

import torch

from cdrseqgen.artifacts import atomic_write_json
from cdrseqgen.datasets.dataset_common import inject_context
from cdrseqgen.datasets.dataset_mlm import assemble_masked_input
from cdrseqgen.eval.metrics import (LENGTH_BUCKETS, amino_acid_recovery, calculate,
                                    blosum_aligned_ratio, blosum_identity_ratio,
                                    confidence_metrics, core_anchor_indices,
                                    recovery_at_indices)
from cdrseqgen.eval import ANCHOR_TRIM

logger = logging.getLogger(__name__)


SUMMARY_KEYS = [
    # Perplexity
    ("judge_perplexity_mean", "judge-ppl"),
    ("judge_coverage", "judge-cov"),
    ("self_perplexity_mean", "self-ppl"),
    # Diversity
    ("diversity_hamming", "diversity"),
    ("per_position_entropy_bits", "entropy(b)"),
    ("unique_fraction", "unique"),
    # AAR
    ("aar_mean", "recovery"),
    ("aar_length_matched_mean", "recovery@L"),
    ("core_recovery_mean", "core@L"),
    ("anchor_recovery_mean", "anchor@L"),
    # BLOSUM62
    ("blosum62_ratio_mean", "blosum@L"),
    ("blosum62_core_ratio_mean", "blosum-core@L"),
    ("blosum62_anchor_ratio_mean", "blosum-anch@L"),
    ("blosum62_aligned_ratio_mean", "blosum-aln"),
    # Novelty
    ("novelty_mean", "novelty"),
    ("length_match_rate", "length-match"),
    ("length_absolute_error_mean", "length-MAE"),
    ("edit_similarity_mean", "edit-sim"),
    # Spearman correlation
    ("self_conf_spearman", "self-rho"),
    ("self_conf_top1_recovery", "self-top1"),
    ("judge_conf_spearman", "judge-rho"),
    ("judge_conf_top1_recovery", "judge-top1"),
    ("self_conf_baseline_recovery", "rand-base"),
]


@dataclass
class Draw:
    """Carry one framework's retained candidates and their self-scoring inputs.

    Example:
      native:     "ACD"
      seqs:       ["ACD", "AQD", "AC"]
      spans:      [inputs_0, inputs_1, inputs_2]   (same candidate order)
      target_len: 3                              (reference length)

    Fields:
      native / target_len -- ground-truth loop and its length for scoring/reporting
      seqs                -- normalized candidate strings, including repeated draws
      spans               -- MLM: (input_ids, mask_positions, token_ids)
                             AR/S2S: (prompt_or_encoder_input, answer_token_ids)
      length_policy       -- metadata describing how generation length was chosen
      record_extra        -- additional fields such as attempt/failure counts

    Notes:
      * target_len is not an output-length constraint for length-free generation.
      * record_extra is merged last and can overwrite standard record fields.
    """

    native: str
    seqs: list
    spans: list
    target_len: int
    length_policy: dict
    record_extra: dict = field(default_factory=dict)


def bucket_of(length):
    """Return the shared short/medium/long label for a loop length.

    Example:
      length:  8       14       21
      bucket: "short" "medium" "long"

    Notes:
      * Boundaries come from LENGTH_BUCKETS in metrics.py.
      * Out-of-range values also return "long"; inputs are not validated.
    """
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= length <= hi:
            return name
    return "long"


def select_indices(lengths, k, stratify, rng):
    """Return sorted evaluation indices sampled without replacement.

    Example: rng=random.Random(42), stratify=False
      lengths: [8, 14, 21]
      k:       2
      indices: [0, 2]          (frameworks with reference lengths 8 and 21)

    Stratified example: rng=random.Random(42), k=3
      lengths: [8, 9, 14, 15, 21, 22]
      indices: [0,    2,          5]   (one short, one medium, one long)

    Notes:
      * rng is seeded by the caller. Without stratification, selection is uniform.
      * k=None, k<0 or k>=len(lengths) returns all indices; k=0 returns [].
      * Stratification balances non-empty length buckets, then fills any shortfall
        from unselected indices. It does not preserve corpus proportions.
      * Buckets follow first-observed order; final truncation favors earlier buckets
        when k is smaller than the number of buckets.
      * Reference lengths choose frameworks here; they are not sent to the generator.
    """
    n = len(lengths)

    if k is None or k < 0 or k >= n:
        return list(range(n))       # Select all

    if not stratify:
        return sorted(rng.sample(range(n), k))      # Simple randomized select

    # Balanced across length buckets so rare long H3s will not be left out by random draws
    by_bucket = defaultdict(list)
    for i, length in enumerate(lengths):
        by_bucket[bucket_of(length)].append(i)

    per_bucket = max(1, k // len(by_bucket))
    picked = []
    for idxs in by_bucket.values():
        picked += rng.sample(idxs, min(per_bucket, len(idxs)))

    # Fill quotas that small buckets could not satisfy, still without replacement.
    if len(picked) < k:
        taken = set(picked)
        rest = [i for i in range(n) if i not in taken]
        picked += rng.sample(rest, min(k - len(picked), len(rest)))

    return sorted(picked[:k])


def aggregate(records, key):
    """Return (mean, population_stddev) for one metric across frameworks.

    Example: record["metrics"][key] values are [0.0, None, 1.0]
      included: [0.0, 1.0]
      mean:     (0 + 1) / 2 = 0.5
      stddev:   sqrt(((0 - 0.5)**2 + (1 - 0.5)**2) / 2) = 0.5
      return:   (0.5, 0.5)

    Notes:
      * Each contributing framework has equal weight, regardless of candidate count.
      * Missing keys and None are excluded; NaN/Inf are currently not filtered.
      * No values return (None, 0.0); one value has stddev=0.0.
      * Stddev describes variation across frameworks, not a confidence interval
        or variation across training seeds.
    """
    vals = [r["metrics"][key] for r in records if r["metrics"].get(key) is not None]
    if not vals:
        return None, 0.0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def format_report(records, title="", judge_name=None):
    """Return a text report of overall and length-bucket means ± population stddevs.

    Example: two short-loop records with aar_mean values 0.0 and 1.0
      OVERALL  (n=2)
        recovery        0.500  ± 0.500
      length=short  (n=2)
        recovery        0.500  ± 0.500
      (excerpt)

    Notes:
      * SUMMARY_KEYS chooses metric names; target_len chooses length buckets.
      * n counts records in the block; each metric may have fewer contributing records.
      * Missing metrics are omitted; empty records produce only the header/footer.
      * The spread is population stddev, not a confidence interval.
      * judge_name only labels the scorer. Comparable judge scores still require
        matched weights, tokenization, context and scoring populations.
      * Self-perplexity depends on the generator method and model.
    """
    lines = []

    def block(label, subset):
        """Append available metric means and spreads for a non-empty record subset."""
        if not subset:
            return
        lines.append(f"\n  {label}  (n={len(subset)})")
        for key, pretty in SUMMARY_KEYS:
            mean, sd = aggregate(subset, key)
            if mean is None:        # never collected for this subset
                continue
            lines.append(f"    {pretty:12s} {mean:8.3f}  ± {sd:.3f}")

    lines.append("\n" + "=" * 72)
    if title:
        lines.append(f"📄 {title}")
    if judge_name:
        lines.append(f"   judge = {judge_name}  (cross-model-comparable perplexity)")
    else:
        lines.append("   self-ppl only — NOT comparable across models; use a judge for that")
    lines.append("=" * 72)
    block("OVERALL", records)
    for name, _, _ in LENGTH_BUCKETS:
        block(f"length={name}", [r for r in records if bucket_of(r["target_len"]) == name])
    lines.append("=" * 72)
    return "\n".join(lines)


def log_report(records, title="", judge_name=None):
    """Log ``format_report`` as one INFO entry; return ``None``."""
    logger.info(format_report(records, title=title, judge_name=judge_name))


def normalize_scores(scores, expected_count, *, label):
    """Return ordered float/None scores, requiring one entry per candidate.

    Example: [2.0, NaN, None] -> [2.0, None, None].
    Finite numeric scalars retain their values; non-finite values become unavailable.
    A wrong count or nonnumeric/non-scalar entry raises ValueError with the scorer
    label. Scalar tensors are accepted; strings and booleans are not scores.
    """
    try:
        values = list(scores)
    except TypeError as exc:
        raise ValueError(f"{label} must return one score or None per candidate") from exc
    if len(values) != expected_count:
        raise ValueError(
            f"{label} returned {len(values)} scores for {expected_count} candidates"
        )

    normalized = []
    nonfinite = 0
    for index, score in enumerate(values):
        if score is None:
            normalized.append(None)
            continue
        numeric = isinstance(score, Number) and not isinstance(score, bool)
        if isinstance(score, torch.Tensor):
            numeric = score.ndim == 0 and score.dtype != torch.bool
        if not numeric:
            raise ValueError(f"{label} score {index} must be a numeric scalar or None")
        try:
            value = float(score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} score {index} must be a real numeric scalar or None") from exc
        if math.isfinite(value):
            normalized.append(value)
        else:
            normalized.append(None)
            nonfinite += 1
    if nonfinite:
        logger.warning("%s returned %d non-finite score(s); treating them as unavailable",
                       label, nonfinite)
    return normalized


def run_eval_loop(strategy, selected, *, model, tokenizer, device, skip_perplexity,
                  judge_model, judge_tokenizer, anchor_trim=ANCHOR_TRIM):
    """Return scored framework records for the selected evaluation indices.

    Workflow per index:
      strategy.draw -> Draw -> sequence metrics + self/judge scores
                    -> candidate details -> confidence metrics -> framework record

    Example: native="ACD", self-scoring and judge enabled
      candidates:    ["ACD", "AQD", "AC"]
      recovery:      [1.0,   2/3,   None]    (wrong-length AAR is undefined)
      self scores:   [2.0,   3.0,   4.0]
      judge scores:  [1.5,   None,  2.5]
      judge coverage: 2 / 3

    Notes:
      * A None draw produces no record; strategies/wrappers track failed attempts.
      * skip_perplexity disables self-scoring only; an enabled judge still scores.
      * Core/anchor recovery, positional BLOSUM and confidence use length-matched
        candidates; aligned BLOSUM accepts non-empty candidates of any length.
      * Self/judge score lists must match candidate count; mismatches raise ValueError.
      * Non-finite scores become None before means, coverage and candidate records.
      * Means within a record weight eligible draws equally; aggregate handles frameworks.
      * The strategy owns generator execution; model/tokenizer are unused interface args.
        device is used for judging. Errors propagate; persistence happens in the wrapper.
    """
    try:
        from tqdm import tqdm
        data_iterator = tqdm(selected, desc="frameworks", disable=None)  # None -> off when not a tty
    except ImportError:
        data_iterator = selected

    records = []
    for i in data_iterator:
        drawn = strategy.draw(i)
        if drawn is None:
            continue

        metrics = calculate(drawn.seqs, drawn.native)
        metrics["target_len"] = drawn.target_len

        # Let the strategy supply the generator's own scoring factorization.
        self_perplexities = [None] * len(drawn.seqs)
        if not skip_perplexity:
            self_perplexities = normalize_scores(
                strategy.self_scores(drawn), len(drawn.seqs),
                label=f"self scorer at eval index {i}",
            )
            scored_self = [p for p in self_perplexities if p is not None]
            if scored_self:
                metrics["self_perplexity_mean"] = sum(scored_self) / len(scored_self)

        judge_perplexities = [None] * len(drawn.seqs)
        if judge_model is not None:
            judge_perplexities = normalize_scores(
                score_with_judge(strategy, i, drawn.seqs, judge_model, judge_tokenizer, device),
                len(drawn.seqs), label=f"judge scorer at eval index {i}",
            )
            scored = [p for p in judge_perplexities if p is not None]
            if scored:
                metrics["judge_perplexity_mean"] = sum(scored) / len(scored)
            # Report coverage alongside the mean over scored candidates.
            metrics["judge_candidates_scored"] = len(scored)
            metrics["judge_coverage"] = len(scored) / len(drawn.seqs) if drawn.seqs else None
            if len(scored) < len(drawn.seqs):
                logger.warning(
                    f"⚠️  judge scored {len(scored)}/{len(drawn.seqs)} candidates on "
                    f"eval index {i} (unavailable scores); judge_perplexity_mean "
                    "covers only those."
                )

        # Positional scores require a length match; absent core/anchors score None.
        core_indices, anchor_indices = core_anchor_indices(len(drawn.native), anchor_trim)
        candidates = []
        for seq, self_ppl, judge_ppl in zip(drawn.seqs, self_perplexities, judge_perplexities):
            matched = len(seq) == len(drawn.native)
            candidates.append({
                "seq": seq,
                "length": len(seq),
                "length_match": matched,
                "recovery": amino_acid_recovery(seq, drawn.native) if matched else None,
                "core_recovery":
                    recovery_at_indices(seq, drawn.native, core_indices) if matched else None,
                "anchor_recovery":
                    recovery_at_indices(seq, drawn.native, anchor_indices) if matched else None,
                # Only the aligned BLOSUM ratio supports unequal lengths.
                "blosum62_ratio": blosum_identity_ratio(seq, drawn.native) if matched else None,
                "blosum62_core_ratio":
                    blosum_identity_ratio(seq, drawn.native, core_indices) if matched else None,
                "blosum62_anchor_ratio":
                    blosum_identity_ratio(seq, drawn.native, anchor_indices) if matched else None,
                "blosum62_aligned_ratio": blosum_aligned_ratio(seq, drawn.native),
                "self_perplexity": self_ppl,
                "judge_perplexity": judge_ppl,
            })
        for key in ("core_recovery", "anchor_recovery", "blosum62_ratio",
                    "blosum62_core_ratio", "blosum62_anchor_ratio", "blosum62_aligned_ratio"):
            values = [c[key] for c in candidates if c[key] is not None]
            if values:
                metrics[f"{key}_mean"] = sum(values) / len(values)
        recoveries = [c["recovery"] for c in candidates]
        metrics.update(confidence_metrics(
            recoveries, [c["self_perplexity"] for c in candidates], prefix="self_conf"))
        metrics.update(confidence_metrics(
            recoveries, [c["judge_perplexity"] for c in candidates], prefix="judge_conf"))

        records.append({
            "eval_index": i,
            "target_len": drawn.target_len,
            "length_policy": drawn.length_policy,
            "native": drawn.native,
            "samples": drawn.seqs,
            "candidates": candidates,
            "metrics": metrics,
            **drawn.record_extra,
        })
    return records


def score_with_judge(strategy, i, seqs, judge_model, judge_tokenizer, device):
    """Return independent masked-LM pseudo-perplexities in candidate order.

    Example:
      candidate:          "ACD"       "AC"
      encoded token count:  3           2
      judge mask count:     3           3
      result:             ppl_ACD      None    (count mismatch)

    Scoring:
      fill candidate -> re-mask one position per variant -> score its residue
      pseudo-ppl = exp(-mean(log p(residue | rest)))
      p=0.5 at every position -> pseudo-ppl=2.0

    Notes:
      * strategy.judge_inputs(i, seq) supplies (input_ids, mask_positions) or None.
      * Missing prompts and token/mask count mismatches produce None placeholders.
      * Special-token logits are banned; sequence order and list length are preserved.
      * Other errors propagate; run_eval_loop validates count and filters NaN/Inf.
      * The local import loads the MLM scorer only when judging is requested.
    """
    from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood

    perplexities = []
    for seq in seqs:
        prompt = strategy.judge_inputs(i, seq)
        if prompt is None:
            perplexities.append(None)
            continue
        judge_input, judge_mask_positions = prompt
        candidate = judge_tokenizer.encode(seq)
        if len(candidate) != len(judge_mask_positions):
            perplexities.append(None)  # tokenizer disagreement; skip rather than mis-score
            continue
        perplexities.append(pseudo_log_likelihood(
            judge_model, judge_input, judge_mask_positions, candidate,
            judge_tokenizer.special_ids, judge_tokenizer.mask_id, device,
        )[1])
    return perplexities


def write_payload(payload, output_path, n_records):
    """Write the evaluation payload atomically to JSON and log the outcome.

    Example:
      output_path: "results/generation_eval.json"
      n_records:   12
      payload -> atomic JSON write -> log "Saved 12 framework records"

    Notes:
      * A falsey output_path skips writing; return value is always None.
      * n_records labels the log message; it does not validate the payload.
      * OSError becomes a warning; other serialization errors propagate.
        Callers needing persisted results must check the output artifact.
    """
    if not output_path:
        return
    try:
        atomic_write_json(output_path, payload)
        logger.info(f"💾 Saved {n_records} framework records to: {output_path}")
    except OSError as e:
        logger.warning(f"⚠️  Could not write results to {output_path}: {e}")


def write_confidence_figure(payload, output_path, title=""):
    """Save a confidence-selection PNG beside the evaluation JSON path.

    Example:
      output_path: "results/generation_eval.json"
      figure_path: "results/generation_eval_confidence.png"
      payload["records"] -> candidate selection curves -> saved PNG path

    Notes:
      * The label is title, falling back to the JSON basename.
      * Return None for no path, no candidate scores, or plotting failure.
      * Plotting imports are deferred; exceptions become warnings.
    """
    if not output_path:
        return None
    try:
        from cdrseqgen.visualization.confidence_plot import save_run_figure

        figure_path = os.path.splitext(output_path)[0] + "_confidence.png"
        saved = save_run_figure(payload.get("records", []), figure_path,
                                label=title or os.path.basename(output_path))
        if saved:
            logger.info(f"📈 Confidence-selection figure saved to: {saved}")
        return saved
    except Exception as e:
        logger.warning(f"⚠️  Could not draw the confidence figure ({type(e).__name__}): {e}")
        return None


def build_masked_judge_prompt(judge_tokenizer, framework, length):
    """Return (input_ids, mask_positions) for judging a candidate of the given length.

    Example: scratch tokenizer, pt1="AC", pt2="WY", length=3, no optional context
      input:     A C <mask> <mask> <mask> W Y
      positions: 0 1    2      3      4   5 6
      mask_positions: [2, 3, 4]

    Workflow:
      encode flanks -> inject/cap context -> assemble judge-format masked input

    Notes:
      * framework supplies pt1/pt2, optional partner_chain/antigen and max_len (default 256).
      * length is the candidate residue count; native loop residues are not read.
      * Tokenizers with outer wrappers add them and shift mask positions accordingly.
      * score_with_judge checks that candidate token count equals mask count.
    """
    pt1_ids = judge_tokenizer.encode(framework["pt1"])
    pt2_ids = judge_tokenizer.encode(framework["pt2"])
    pt1_ids, pt2_ids = inject_context(
        judge_tokenizer, pt1_ids, pt2_ids, length,
        partner_seq=framework.get("partner_chain"), antigen_seq=framework.get("antigen"),
        max_len=framework.get("max_len", 256))
    input_ids, mask_start = assemble_masked_input(judge_tokenizer, pt1_ids, pt2_ids, length)
    return input_ids, list(range(mask_start, mask_start + length))


def build_judge(model_type, weights_path=None, *, data_path, max_len, target_region,
                include_partner_chain=False, include_antigen=False, device=None):
    """Return (judge_model, judge_tokenizer, judge_dataset) for a masked-LM judge.

    Workflow:
      model_type -> resolve MLM backbone -> build tokenizer -> load model weights
                 -> move to device (optional) -> model.eval()
      data_path + target/context flags -> full MaskedInfillingDataset

    Notes:
      * weights_path overrides model weights; model_type still selects the tokenizer
        and architecture. They must match.
      * Strategies map evaluation-subset indices back to the full judge dataset;
        generator and judge datasets must use corresponding records/order.
      * Judge partner/antigen context can differ from generator context.
      * Keep judge weights, tokenization, context and scoring populations fixed
        for cross-generator comparisons. Scores do not establish binding function.
      * Registry imports are deferred until a judge is requested.
    """
    from cdrseqgen.datasets.dataset_mlm import MaskedInfillingDataset
    from cdrseqgen.methods.mlm.backbones import resolve

    judge_backbone = resolve(model_type)
    judge_tokenizer = judge_backbone.build_tokenizer(model_type)
    judge_model = judge_backbone.load_pretrained_model(weights_path or model_type)
    if device is not None:
        judge_model.to(device)
    judge_model.eval()

    judge_dataset = MaskedInfillingDataset(
        data_path, judge_tokenizer,
        max_len=max_len,
        target_region=target_region,
        include_partner_chain=include_partner_chain,
        include_antigen=include_antigen,
    )
    return judge_model, judge_tokenizer, judge_dataset
