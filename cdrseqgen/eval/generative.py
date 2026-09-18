"""Method-neutral generative evaluation: select examples, score them, judge them, write them.

  * Draw / run_eval_loop  -- the loop every method shares
  * select_indices        -- stratified or plain subsampling of an eval split
  * score_with_judge      -- independent-judge perplexity per candidate
  * format_report / log_report / write_payload / write_confidence_figure

Split out of ``methods/mlm/evaluation.py`` on 2026-09-01, once a second method existed to check the
abstraction against. The loop owns everything that must NOT differ between methods — which examples
are drawn, how metrics are aggregated, which judge scores them, what gets written — and a *strategy*
owns the two things that must:

  * ``draw(i)``          -- turn an eval index into candidates (masked expansion vs causal decode)
  * ``self_scores(drawn)`` -- how surprised the model is by its own draw. A masked LM has only a
    PSEUDO-perplexity; a causal model has a true one. Inlining either would have made the loop
    method-specific.

Also required of a strategy: ``judge_inputs(i, seq)`` and ``payload_length_policy()``.

The reason this is shared rather than copied is comparability. Before the split, two evaluation
tracks in this project had already drifted apart in four ways — progress bar, write guard, summary
statistic, empty-mask guard — and differences like that make two tracks quietly incomparable. An
AR-versus-MLM table has exactly that failure mode, one method wide.
"""
import logging
import os
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from cdrseqgen.artifacts import atomic_write_json
from cdrseqgen.datasets.dataset_common import inject_context
from cdrseqgen.datasets.dataset_mlm import assemble_masked_input
from cdrseqgen.eval.metrics import (LENGTH_BUCKETS, amino_acid_recovery, calculate,
                                    blosum_aligned_ratio, blosum_identity_ratio,
                                    blosum_provenance, confidence_metrics,
                                    core_anchor_indices, recovery_at_indices)
from cdrseqgen.eval import ANCHOR_TRIM

logger = logging.getLogger(__name__)


SUMMARY_KEYS = [
    ("judge_perplexity_mean", "judge-ppl"),
    ("judge_coverage", "judge-cov"),
    ("self_perplexity_mean", "self-ppl"),
    ("diversity_hamming", "diversity"),
    ("per_position_entropy_bits", "entropy(b)"),
    ("unique_fraction", "unique"),
    ("aar_mean", "recovery"),
    ("aar_length_matched_mean", "recovery@L"),
    # Split the length-matched recovery: anchors are largely V/J-determined and easy, the core is
    # the part a design method has to actually get right. Reported separately so a gain in one
    # cannot hide a loss in the other.
    ("core_recovery_mean", "core@L"),
    ("anchor_recovery_mean", "anchor@L"),
    # BLOSUM62: a conservative substitution scores near 1.0, a radical one far below, where
    # exact-match recovery scores both zero. `blosum-aln` is over EVERY attempt (global alignment,
    # so a wrong-length draw still scores); the @L rows share recovery@L's length-matched
    # population. It is a generic evolutionary matrix, not a binding or structural measure.
    ("blosum62_ratio_mean", "blosum@L"),
    ("blosum62_core_ratio_mean", "blosum-core@L"),
    ("blosum62_anchor_ratio_mean", "blosum-anch@L"),
    ("blosum62_aligned_ratio_mean", "blosum-aln"),
    ("novelty_mean", "novelty"),
    ("length_match_rate", "length-match"),
    ("length_absolute_error_mean", "length-MAE"),
    ("edit_similarity_mean", "edit-sim"),
    # Does the model's own confidence rank its candidates? rho ~ 0 means confidence is useless
    # for selection, and top1 vs baseline is the recovery a designer would actually get.
    ("self_conf_spearman", "self-rho"),
    ("self_conf_top1_recovery", "self-top1"),
    ("judge_conf_spearman", "judge-rho"),
    ("judge_conf_top1_recovery", "judge-top1"),
    ("self_conf_baseline_recovery", "rand-base"),
]


@dataclass
class Draw:
    """One example's candidates, in the shape the shared scoring loop consumes.

    ``spans`` carries (input_ids, mask_positions, token_ids) per candidate so self-perplexity is
    scored against the prompt that produced it — the fixed track repeats one prompt, the variable
    track has a different prompt per sampled length.
    """

    native: str
    seqs: list
    spans: list
    target_len: int
    length_policy: dict
    record_extra: dict = field(default_factory=dict)


def bucket_of(length):
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= length <= hi:
            return name
    return "long"


def select_indices(lengths, k, stratify, rng):
    """Pick which eval CDR data examples to generate for (all if k < 0 or k >= n)"""
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

    # Redistribute the shortfall. An under-populated bucket cannot fill its quota, and without this
    # the draw silently comes back short of k: SD2 H1 is 749 short-loop / 1 medium, so a request for
    # 150 returned 75 + 1 = 76 and eight runs of the 07-29 batch had their generative metrics
    # computed on half the intended sample. Top up from whatever is left, still without replacement.
    if len(picked) < k:
        taken = set(picked)
        rest = [i for i in range(n) if i not in taken]
        picked += rng.sample(rest, min(k - len(picked), len(rest)))

    return sorted(picked[:k])


def aggregate(records, key):
    """(mean, stdev) of a per-example metric across the sweep; (None, 0.0) if never present.

    None rather than NaN: this lands in generation_eval.json, and a bare NaN there is not valid
    JSON -- standards-compliant readers reject the file, and tables printed the string "nan" for a
    metric that simply was not collected. Every reader here already treats None as absent.
    """
    vals = [r["metrics"][key] for r in records if r["metrics"].get(key) is not None]
    if not vals:
        return None, 0.0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def format_report(records, title="", judge_name=None):
    """Build the summary block as one string: overall, then per H3-length bucket"""
    lines = []

    def block(label, subset):
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
    """Emit the summary through the logger, as one single record"""
    logger.info(format_report(records, title=title, judge_name=judge_name))


def run_eval_loop(strategy, selected, *, model, tokenizer, device, skip_perplexity,
                  judge_model, judge_tokenizer, anchor_trim=ANCHOR_TRIM):
    """Per-example generate -> score -> record, identical for every length track."""
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

        # Self-scoring is the second thing a method cannot share. A masked LM has only a
        # PSEUDO-perplexity (score each position with the rest visible); a causal model has a true
        # one. Both are "how surprised is the model by its own draw", so the loop asks the strategy
        # rather than assuming either.
        self_perplexities = [None] * len(drawn.seqs)
        if not skip_perplexity:
            self_perplexities = strategy.self_scores(drawn)
            scored_self = [p for p in self_perplexities if p is not None]
            if scored_self:
                metrics["self_perplexity_mean"] = sum(scored_self) / len(scored_self)

        judge_perplexities = [None] * len(drawn.seqs)
        if judge_model is not None:
            judge_perplexities = score_with_judge(
                strategy, i, drawn.seqs, judge_model, judge_tokenizer, device
            )
            scored = [p for p in judge_perplexities if p is not None]
            if scored:
                metrics["judge_perplexity_mean"] = sum(scored) / len(scored)
            # Skipping keeps one bad candidate from voiding a whole eval, but the mean is then over
            # an unknown subset unless the coverage travels with it. (benchmark/scorer.py raises
            # instead — it scores a frozen manifest, where a partial judge population would
            # silently weaken a published comparison.)
            metrics["judge_candidates_scored"] = len(scored)
            metrics["judge_coverage"] = len(scored) / len(drawn.seqs) if drawn.seqs else None
            if len(scored) < len(drawn.seqs):
                logger.warning(
                    f"⚠️  judge scored {len(scored)}/{len(drawn.seqs)} candidates on "
                    f"eval index {i} (tokenizer length disagreement); judge_perplexity_mean "
                    "covers only those."
                )

        # Per-candidate detail. Recovery is defined only where the length matches, so the
        # correlation answers "among candidates of the right length, does confidence pick the
        # accurate one" rather than rediscovering that wrong-length candidates score badly.
        # The core/anchor split needs a positional correspondence, so like recovery it is defined
        # only on length-matched candidates. It is what separates "the model learned the J-gene
        # motif" from "the model learned the hypervariable middle" — the distinction a core-weighted
        # objective is meant to move, and one whole-loop AAR averages away.
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
                # BLOSUM beside recovery at every level, never instead of it: a conservative
                # substitution is invisible to exact-match recovery and a radical one is
                # indistinguishable from it. `blosum62_aligned_ratio` is the only one defined
                # for a candidate of the wrong length, which is most causal H3 draws.
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
    """Judge perplexity per candidate, aligned with ``seqs``; None where it could not be scored.

    Returning a positional list rather than a compacted one is what lets a candidate's judge score
    be paired with its own recovery downstream — a compacted list silently misaligns as soon as
    one candidate is skipped.
    """
    # Deferred, like build_judge's registry import: the judge is always a MASKED LM whatever
    # produced the candidate, so scoring it with a pseudo-log-likelihood is the judge's own
    # requirement — but importing a method package at module level would invert the dependency and
    # make cdrseqgen.eval depend on cdrseqgen.methods.
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
    """Persist atomically, treating an unwritable path as a warning rather than lost work."""
    if not output_path:
        return
    try:
        atomic_write_json(output_path, payload)
        logger.info(f"💾 Saved {n_records} framework records to: {output_path}")
    except OSError as e:
        logger.warning(f"⚠️  Could not write results to {output_path}: {e}")


def write_confidence_figure(payload, output_path, title=""):
    """Draw the confidence-selection figure beside the JSON, as training writes its curves.

    Skipped silently when there is nothing to plot — a run with --skip_perplexity and no judge
    has no per-candidate scores, which is a configuration choice rather than a failure. Guarded
    for the same reason post_training_generation is: a plotting problem must not cost a finished
    evaluation. matplotlib is imported lazily so an eval that plots nothing never loads it.
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
    """A masked prompt for the independent judge at one candidate's length.

    Neutral despite being masked: the judge is a masked LM (IgBert by default) whatever produced the
    candidate, so this is the JUDGE's input format, not the method-under-test's. A causal strategy
    needs it for exactly the same reason the variable-length masked track does — candidate lengths
    vary, so the native-length prompt does not align.

    Returns (input_ids, mask_positions); ``mask_positions`` has ``length`` entries, which is what
    ``score_with_judge`` checks against the tokenized candidate before scoring.
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
    """Load an INDEPENDENT reference LM plus a parallel dataset in its own vocabulary.

    Neutral, and deliberately so. The judge is what makes perplexity comparable ACROSS models —
    self-perplexity measures self-consistency, so an overconfident model rates its own samples
    highly regardless of quality. That argument does not care which method produced the candidate,
    and Findings §11 rests on one judge scoring both CDRSeqGen and IgLM. A per-method copy would be
    the fastest way to make an AR-versus-MLM table meaningless.

    The judge is a MASKED LM (IgBert by default) whatever it is judging, so it is built through the
    masked backbone registry and reads the masked dataset view. That is the judge's own format, not
    an assumption about the model under test.

    Returns (judge_model, judge_tokenizer, judge_dataset).
    """
    # Local imports: a heavy registry and the masked dataset are only needed when a judge is asked
    # for, and importing them at module load would drag transformers into every plotting script.
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
