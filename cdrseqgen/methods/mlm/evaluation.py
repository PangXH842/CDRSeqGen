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
import json
import logging
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from cdrseqgen.data.dataset import AntibodyInfillingDataset
from cdrseqgen.eval.metrics import calculate
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

SUMMARY_KEYS = [
    ("judge_perplexity_mean", "judge-ppl"),
    ("judge_coverage", "judge-cov"),
    ("self_perplexity_mean", "self-ppl"),
    ("diversity_hamming", "diversity"),
    ("per_position_entropy_bits", "entropy(b)"),
    ("unique_fraction", "unique"),
    ("aar_mean", "recovery"),
    ("aar_length_matched_mean", "recovery@L"),
    ("novelty_mean", "novelty"),
    ("length_match_rate", "length-match"),
    ("length_absolute_error_mean", "length-MAE"),
    ("edit_similarity_mean", "edit-sim"),
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


def masked_view(item):
    """(input_ids, mask_positions, native_ids) for one dataset item"""
    input_ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()
    mask_positions = [p for p, l in enumerate(labels) if l != -100]
    native_ids = [labels[p] for p in mask_positions]
    return input_ids, mask_positions, native_ids

def aggregate(records, key):
    """(mean, stdev) of a per-example metric across the sweep; (nan, 0) if never present"""
    vals = [r["metrics"][key] for r in records if r["metrics"].get(key) is not None]
    if not vals:
        return float("nan"), 0.0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def build_judge(model_type, weights_path=None, *, data_path, max_len, target_region,
                include_partner_chain=False, include_antigen=False, device=None):
    """Load an INDEPENDENT reference LM plus a parallel dataset in its own vocabulary.

    Returns (judge_model, judge_tokenizer, judge_dataset).
    """
    from cdrseqgen.methods.mlm.backbones import resolve  # local: avoids a heavy import at module load

    judge_backbone = resolve(model_type)
    judge_tokenizer = judge_backbone.build_tokenizer(model_type)
    judge_model = judge_backbone.load_pretrained_model(weights_path or model_type)
    if device is not None:
        judge_model.to(device)
    judge_model.eval()

    judge_dataset = AntibodyInfillingDataset(
        data_path, judge_tokenizer,
        max_len=max_len,
        target_region=target_region,
        include_partner_chain=include_partner_chain,
        include_antigen=include_antigen,
    )
    return judge_model, judge_tokenizer, judge_dataset


def format_report(records, title="", judge_name=None):
    """Build the summary block as one string: overall, then per H3-length bucket"""
    lines = []

    def block(label, subset):
        if not subset:
            return
        lines.append(f"\n  {label}  (n={len(subset)})")
        for key, pretty in SUMMARY_KEYS:
            mean, sd = aggregate(subset, key)
            if mean != mean:  # nan -> metric not collected
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
                        decoding="iterative", num_samples=20,
                        temperature=1.0, top_k=0, top_p=1.0,
                        length_mode="fixed_native", length_predictor=None,
                        min_length=None, max_length=None,
                        length_temperature=1.0,
                        num_examples=200, stratify=False, seed=42,
                        skip_perplexity=False,
                        judge_model=None, judge_tokenizer=None,
                        judge_dataset=None, judge_name=None,
                        output_path=None, meta=None, title=""):
    """Sample + score candidate loops for a subsample of the eval split, returns the payload dict.

    Both length tracks run the same loop, aggregation, reporting, and writer; they differ only in
    the strategy that turns one eval index into candidates, which is the one part that must not be
    shared (the fixed track is told the native mask count, the variable track must never be).
    """
    model.eval()

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
    )

    log_report(records, title=title, judge_name=judge_name)

    payload = dict(meta or {})
    payload.update({
        "judge": judge_name,
        "length_policy": strategy.payload_length_policy(),
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
    return payload


def run_eval_loop(strategy, selected, *, model, tokenizer, device, skip_perplexity,
                  judge_model, judge_tokenizer):
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

        if not skip_perplexity:
            perplexities = [
                pseudo_log_likelihood(model, input_ids, mask_positions, token_ids,
                                      tokenizer.special_ids, tokenizer.mask_id, device)[1]
                for input_ids, mask_positions, token_ids in drawn.spans
            ]
            if perplexities:
                metrics["self_perplexity_mean"] = sum(perplexities) / len(perplexities)

        if judge_model is not None:
            judge_perplexities = score_with_judge(
                strategy, i, drawn.seqs, judge_model, judge_tokenizer, device
            )
            if judge_perplexities:
                metrics["judge_perplexity_mean"] = (
                    sum(judge_perplexities) / len(judge_perplexities)
                )
            # Skipping keeps one bad candidate from voiding a whole eval, but the mean is then over
            # an unknown subset unless the coverage travels with it. (benchmark/scorer.py raises
            # instead — it scores a frozen manifest, where a partial judge population would
            # silently weaken a published comparison.)
            metrics["judge_candidates_scored"] = len(judge_perplexities)
            metrics["judge_coverage"] = (
                len(judge_perplexities) / len(drawn.seqs) if drawn.seqs else None
            )
            if len(judge_perplexities) < len(drawn.seqs):
                logger.warning(
                    f"⚠️  judge scored {len(judge_perplexities)}/{len(drawn.seqs)} candidates on "
                    f"eval index {i} (tokenizer length disagreement); judge_perplexity_mean "
                    "covers only those."
                )

        records.append({
            "eval_index": i,
            "target_len": drawn.target_len,
            "length_policy": drawn.length_policy,
            "native": drawn.native,
            "samples": drawn.seqs,
            "metrics": metrics,
            **drawn.record_extra,
        })
    return records


def score_with_judge(strategy, i, seqs, judge_model, judge_tokenizer, device):
    """Perplexity of each candidate under the independent judge, skipping ones it cannot encode."""
    perplexities = []
    for seq in seqs:
        prompt = strategy.judge_inputs(i, seq)
        if prompt is None:
            continue
        judge_input, judge_mask_positions = prompt
        candidate = judge_tokenizer.encode(seq)
        if len(candidate) != len(judge_mask_positions):
            continue  # tokenizer disagreement on this loop; skip rather than mis-score
        perplexities.append(pseudo_log_likelihood(
            judge_model, judge_input, judge_mask_positions, candidate,
            judge_tokenizer.special_ids, judge_tokenizer.mask_id, device,
        )[1])
    return perplexities


def write_payload(payload, output_path, n_records):
    """Persist the payload, treating an unwritable path as a warning rather than lost work."""
    if not output_path:
        return
    try:
        with open(output_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(f"💾 Saved {n_records} framework records to: {output_path}")
    except OSError as e:
        logger.warning(f"⚠️  Could not write results to {output_path}: {e}")
