"""Shared core for generative evaluation — used by BOTH callers, one implementation.

  * train.py   — post-training hook: samples from the in-memory best model when a config opts in
                 with "run_generative_eval": true.
  * eval_generative.py — standalone entrypoint: samples from any finished checkpoint on disk, with an
                 optional independent judge for cross-model comparison.

The two differ only in *where the model comes from*, not in what "generative evaluation" means, so
the logic lives here once. Nothing in this module imports train.py (that would be circular): the
callers unpack their own arg dataclasses and pass plain values in.

What it measures, and why it isn't recovery: recovery rewards reproducing the native loop, which is
the opposite of design. Here each held-out framework gets a *library* of sampled candidates, scored
for diversity, novelty vs. the native, and plausibility (pseudo-perplexity). Summaries are also
bucketed by CDR-H3 length, because difficulty scales with it and a single mean hides that.

Cross-model caveat, enforced by the API: self pseudo-perplexity is computed under the model's OWN
distribution, so it measures self-consistency and is NOT comparable across models (empirically it
inflated a scratch-vs-ESM-2 gap from 1.5x to 2.6x). Pass a judge built by build_judge() to score
every candidate under one INDEPENDENT reference LM — that is what makes perplexity comparable.
"""
import os
import json
import random
import statistics
from collections import defaultdict

import torch

from cdrseqgen.data.dataset import AntibodyInfillingDataset
from cdrseqgen.eval.metrics import summarize
from cdrseqgen.eval.sampling import (
    decode_greedy,
    sample_independent,
    sample_iterative,
    pseudo_log_likelihood,
)

# CDR-H3 difficulty scales with length, so every summary is broken out by loop length rather than
# hiding short/long behaviour behind one mean.
LENGTH_BUCKETS = [("short", 0, 11), ("medium", 12, 17), ("long", 18, 10**6)]

SUMMARY_KEYS = [
    ("judge_perplexity_mean", "judge-ppl"),
    ("self_perplexity_mean", "self-ppl"),
    ("diversity_hamming", "diversity"),
    ("per_position_entropy_bits", "entropy(b)"),
    ("unique_fraction", "unique"),
    ("recovery_mean", "recovery"),
    ("novelty_mean", "novelty"),
]


def bucket_of(length):
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= length <= hi:
            return name
    return "long"


def select_indices(lengths, k, stratify, rng):
    """Pick which eval examples to generate for (all of them if k < 0 or k >= n)."""
    n = len(lengths)
    if k is None or k < 0 or k >= n:
        return list(range(n))
    if not stratify:
        return sorted(rng.sample(range(n), k))

    # Balanced across length buckets so rare long H3s aren't washed out by a random draw.
    by_bucket = defaultdict(list)
    for i, length in enumerate(lengths):
        by_bucket[bucket_of(length)].append(i)
    per_bucket = max(1, k // len(by_bucket))
    picked = []
    for idxs in by_bucket.values():
        picked += rng.sample(idxs, min(per_bucket, len(idxs)))
    return sorted(picked[:k])


def masked_view(item):
    """(input_ids, mask_positions, native_ids) for one dataset item.

    The dataset already assembled the context (partner/antigen injection included) and marked the
    target span in `labels`, so the samplers consume it directly — no hand-built pt1/pt2, and no
    chance of drifting out of sync with the injection rules.
    """
    input_ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()
    mask_positions = [p for p, l in enumerate(labels) if l != -100]
    native_ids = [labels[p] for p in mask_positions]
    return input_ids, mask_positions, native_ids


def _decode(model, tokenizer, device, generator, input_ids, mask_positions,
            decoding, num_samples, temperature, top_k, top_p):
    banned = tokenizer.special_ids
    if decoding == "greedy":
        return decode_greedy(model, input_ids, mask_positions, banned, device)
    if decoding == "independent":
        return sample_independent(model, input_ids, mask_positions, banned, num_samples,
                                  temperature, top_k, top_p, device, generator)
    return sample_iterative(model, input_ids, mask_positions, banned, num_samples,
                            temperature, top_k, top_p, device, tokenizer.mask_id, generator)


def aggregate(records, key):
    """(mean, stdev) of a per-example metric across the sweep; (nan, 0) if never present."""
    vals = [r["metrics"][key] for r in records if r["metrics"].get(key) is not None]
    if not vals:
        return float("nan"), 0.0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def build_judge(model_type, weights_path=None, *, data_path, max_len, target_region,
                include_partner_chain=False, include_antigen=False, device=None):
    """Load an INDEPENDENT reference LM plus a parallel dataset in its own vocabulary.

    Same data, same split, same injection — only the tokenizer differs — so eval example i lines up
    with judge example i and candidates can be re-encoded into the judge's vocab and scored.

    Returns (judge_model, judge_tokenizer, judge_dataset).
    """
    from cdrseqgen.models.models import resolve  # local: avoids a heavy import at module load

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


def print_report(records, title="", judge_name=None):
    """Console summary: overall, then per H3-length bucket."""
    def block(label, subset):
        if not subset:
            return
        print(f"\n  {label}  (n={len(subset)})")
        for key, pretty in SUMMARY_KEYS:
            mean, sd = aggregate(subset, key)
            if mean != mean:  # nan -> metric not collected
                continue
            print(f"    {pretty:12s} {mean:8.3f}  ± {sd:.3f}")

    print("\n" + "=" * 72)
    if title:
        print(f"📄 {title}")
    if judge_name:
        print(f"   judge = {judge_name}  (cross-model-comparable perplexity)")
    else:
        print("   self-ppl only — NOT comparable across models; use a judge for that")
    print("=" * 72)
    block("OVERALL", records)
    for name, _, _ in LENGTH_BUCKETS:
        block(f"length={name}", [r for r in records if bucket_of(r["target_len"]) == name])
    print("=" * 72)


def run_generative_eval(model, tokenizer, eval_dataset, *, device,
                        decoding="iterative", num_samples=20,
                        temperature=1.0, top_k=0, top_p=1.0,
                        num_examples=200, stratify=False, seed=42,
                        skip_perplexity=False,
                        judge_model=None, judge_tokenizer=None, judge_dataset=None,
                        judge_name=None, output_path=None, meta=None,
                        title="", progress=True):
    """Sample + score candidate loops for a subsample of the eval split. Returns the payload dict.

    `model` may be mid-training (train.py passes trainer.model), so eval mode is forced here —
    otherwise dropout would silently corrupt every sample.
    """
    model.eval()

    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    lengths = [eval_dataset[i]["target_len"] for i in range(len(eval_dataset))]
    rng = random.Random(seed)
    selected = select_indices(lengths, num_examples, stratify, rng)

    iterator = selected
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(selected, desc="frameworks")
        except ImportError:
            pass

    records = []
    for i in iterator:
        item = eval_dataset[i]
        input_ids, mask_positions, native_ids = masked_view(item)
        if not mask_positions:
            continue
        native = tokenizer.decode(native_ids).upper().strip()

        sampled_ids = _decode(model, tokenizer, device, generator, input_ids, mask_positions,
                              decoding, num_samples, temperature, top_k, top_p)
        seqs = [tokenizer.decode(ids).upper().strip() for ids in sampled_ids]

        metrics = summarize(seqs, native)
        metrics["target_len"] = len(mask_positions)

        if not skip_perplexity:
            ppls = [pseudo_log_likelihood(model, input_ids, mask_positions, ids,
                                          tokenizer.special_ids, tokenizer.mask_id, device)[1]
                    for ids in sampled_ids]
            metrics["self_perplexity_mean"] = sum(ppls) / len(ppls)

        if judge_model is not None:
            # Map the eval-subset index back to the full dataset so the judge sees the same entry.
            full_idx = eval_dataset.indices[i] if hasattr(eval_dataset, "indices") else i
            j_input, j_mask_pos, _ = masked_view(judge_dataset[full_idx])
            j_ppls = []
            for seq in seqs:
                cand = judge_tokenizer.encode(seq)
                if len(cand) != len(j_mask_pos):
                    continue  # tokenizer disagreement on this loop; skip rather than mis-score
                j_ppls.append(pseudo_log_likelihood(judge_model, j_input, j_mask_pos, cand,
                                                    judge_tokenizer.special_ids,
                                                    judge_tokenizer.mask_id, device)[1])
            if j_ppls:
                metrics["judge_perplexity_mean"] = sum(j_ppls) / len(j_ppls)

        records.append({
            "eval_index": i,
            "target_len": len(mask_positions),
            "native": native,
            "samples": seqs,
            "metrics": metrics,
        })

    print_report(records, title=title, judge_name=judge_name)

    payload = dict(meta or {})
    payload.update({
        "judge": judge_name,
        "decoding": {
            "mode": decoding, "num_samples": num_samples, "temperature": temperature,
            "top_k": top_k, "top_p": top_p, "seed": seed,
        },
        "n_eval_total": len(eval_dataset),
        "n_generated": len(records),
        "summary": {key: aggregate(records, key)[0] for key, _ in SUMMARY_KEYS},
        "records": records,
    })

    if output_path:
        try:
            with open(output_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\n💾 Saved {len(records)} framework records to: {output_path}")
        except OSError as e:
            print(f"\n⚠️  Could not write results to {output_path}: {e}")

    return payload
