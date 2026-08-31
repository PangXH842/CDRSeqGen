"""Paired bootstrap confidence intervals for a matched two-model benchmark.

    python benchmark/bootstrap.py --scores results/benchmark/<id>/scores/<track>.json
    python benchmark/bootstrap.py --scores <path> --iterations 20000 --output ci.json

The comparison in Findings §11 reports point estimates with no uncertainty, on 25 examples. This
turns each one into "CDRSeqGen minus IgLM, with a 95% interval", which is the difference between a
number that can leave the repo and one that cannot.

Three choices matter, and all three follow from the experiment being *matched*:

  * **The example is the unit, not the candidate.** Both models generate many candidates per
    framework, but those candidates are not independent — they share a framework, a native loop and
    a difficulty. Resampling candidates would treat 500 draws as 500 observations and produce an
    interval several times too narrow. Resampling the 25 examples is the honest unit, and it is what
    the roadmap specifies.
  * **The resample is paired.** One draw of example ids is applied to *both* models, so the
    per-example difficulty that dominates raw variance cancels in the difference. An unpaired
    bootstrap of two 25-example means would be far wider and would answer a question nobody asked,
    since both models saw the same frozen manifest.
  * **A metric contributes only where both models have it.** Positional AAR is defined only for
    candidates matching the native length, so it is missing on the examples where a model never
    produced one (7 of 25 for CDRSeqGen, 8 for IgLM on the length-free track). Averaging over
    different example subsets for the two models would compare them on different problems, so the
    paired set is the intersection, and its size is reported beside every interval.

The interval is a percentile bootstrap. With n = 25 it is approximate — read a CI that straddles
zero as "not resolved at this sample size", not as evidence of equivalence.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

# Metrics worth an interval, with the direction that counts as better.
DEFAULT_METRICS = (
    ("whole_span_aar_mean", "whole-span AAR", "higher"),
    ("core_aar_mean", "core AAR", "higher"),
    ("anchor_aar_mean", "anchor AAR", "higher"),
    ("length_match_rate", "length match rate", "higher"),
    ("length_absolute_error_mean", "length MAE", "lower"),
    ("edit_similarity_mean", "edit similarity", "higher"),
    ("judge_perplexity_mean", "judge perplexity", "lower"),
    ("diversity_hamming", "diversity (Hamming)", "higher"),
)

CONFIDENCE = 0.95


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def load_runs(scores_path):
    """{label: {example_id: metrics}} for a scores payload with exactly two runs."""
    payload = json.loads(Path(scores_path).read_text())
    runs = payload.get("runs", [])
    if len(runs) != 2:
        raise SystemExit(f"expected exactly 2 runs in {scores_path}, found {len(runs)}")
    out = {}
    for run in runs:
        label = run.get("label") or run.get("adapter") or f"run{len(out)}"
        out[label] = {record["example_id"]: record.get("metrics", {})
                      for record in run.get("records", [])}
    return payload, out


def paired_values(runs, metric):
    """(ids, a_values, b_values) over examples where BOTH runs report a finite value."""
    (label_a, a), (label_b, b) = runs.items()
    shared = [i for i in a if i in b]
    ids, values_a, values_b = [], [], []
    for example_id in shared:
        first, second = a[example_id].get(metric), b[example_id].get(metric)
        if _finite(first) and _finite(second):
            ids.append(example_id)
            values_a.append(float(first))
            values_b.append(float(second))
    return ids, values_a, values_b


def paired_bootstrap(values_a, values_b, iterations=10000, seed=42):
    """Percentile CI for mean(a) - mean(b), resampling examples (not candidates) with replacement."""
    n = len(values_a)
    if n == 0:
        return None
    observed = statistics.mean(values_a) - statistics.mean(values_b)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        # One index draw applied to both arms — this is what makes the interval paired.
        picks = [rng.randrange(n) for _ in range(n)]
        diffs.append(statistics.mean(values_a[i] for i in picks)
                     - statistics.mean(values_b[i] for i in picks))
    diffs.sort()
    lower = diffs[max(0, int((1 - CONFIDENCE) / 2 * iterations) - 1)]
    upper = diffs[min(iterations - 1, int((1 + CONFIDENCE) / 2 * iterations) - 1)]
    # Fraction of resamples on the same side of zero as the observed difference: a directional
    # stability measure, not a p-value.
    same_side = sum(1 for d in diffs if (d > 0) == (observed > 0)) / iterations
    return {
        "n_examples": n,
        "mean_a": statistics.mean(values_a),
        "mean_b": statistics.mean(values_b),
        "difference": observed,
        "ci_low": lower,
        "ci_high": upper,
        "excludes_zero": (lower > 0) or (upper < 0),
        "directional_stability": same_side,
    }


def compare(scores_path, metrics=DEFAULT_METRICS, iterations=10000, seed=42):
    payload, runs = load_runs(scores_path)
    label_a, label_b = list(runs)
    results = []
    for key, pretty, direction in metrics:
        ids, values_a, values_b = paired_values(runs, key)
        stats = paired_bootstrap(values_a, values_b, iterations=iterations, seed=seed)
        if stats is None:
            continue
        better = None
        if stats["excludes_zero"]:
            a_ahead = (stats["difference"] > 0) == (direction == "higher")
            better = label_a if a_ahead else label_b
        results.append({"metric": key, "label": pretty, "direction": direction,
                        "better": better, **stats})
    return {
        "scores_path": str(scores_path),
        "manifest": payload.get("manifest"),
        "judge": payload.get("judge"),
        "runs": [label_a, label_b],
        "iterations": iterations,
        "seed": seed,
        "confidence": CONFIDENCE,
        "results": results,
    }


def format_report(comparison):
    label_a, label_b = comparison["runs"]
    lines = [
        f"Paired bootstrap — {label_a} minus {label_b} "
        f"({comparison['iterations']:,} resamples, seed {comparison['seed']}, "
        f"{int(CONFIDENCE * 100)}% percentile CI)",
        "The example is the resampling unit; a metric uses only examples where both runs report it.",
        "",
        f"  {'metric':<22} {'n':>3}  {label_a:>10}  {label_b:>10}  {'diff':>9}  "
        f"{'95% CI':>20}  verdict",
    ]
    for row in comparison["results"]:
        ci = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        verdict = f"{row['better']} ahead" if row["better"] else "not resolved (CI spans 0)"
        lines.append(
            f"  {row['label']:<22} {row['n_examples']:>3}  {row['mean_a']:>10.4f}  "
            f"{row['mean_b']:>10.4f}  {row['difference']:>+9.4f}  {ci:>20}  {verdict}")

    resolved = [r for r in comparison["results"] if r["better"]]
    lines.append("")
    lines.append(f"  {len(resolved)} of {len(comparison['results'])} metrics resolve at "
                 f"{int(CONFIDENCE * 100)}%; the rest are not separated at this sample size.")
    return "\n".join(lines)


def markdown_table(comparison):
    """The §11-style table, with the interval added."""
    label_a, label_b = comparison["runs"]
    lines = [f"| metric | n | {label_a} | {label_b} | difference | 95% CI | resolved |",
             "|---|---:|---:|---:|---:|---|---|"]
    for row in comparison["results"]:
        verdict = f"**{row['better']}**" if row["better"] else "no"
        lines.append(
            f"| {row['label']} | {row['n_examples']} | {row['mean_a']:.4f} | {row['mean_b']:.4f} | "
            f"{row['difference']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {verdict} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True,
                        help="A scores JSON written by benchmark/scorer.py (exactly two runs).")
    parser.add_argument("--iterations", type=int, default=10000, help="Bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=42, help="Resampling seed.")
    parser.add_argument("--output", default=None, help="Write the full result as JSON here.")
    parser.add_argument("--markdown", action="store_true",
                        help="Also print a Markdown table ready for the README.")
    args = parser.parse_args()

    comparison = compare(args.scores, iterations=args.iterations, seed=args.seed)
    print(format_report(comparison))
    if args.markdown:
        print()
        print(markdown_table(comparison))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(comparison, indent=2))
        print(f"\n✅ Saved as: {args.output}")


if __name__ == "__main__":
    sys.exit(main())
