"""Rank a sweep's settings across seeds: runs.tsv -> summary.tsv plus a printed table.

  * load_trials     -- completed rows of runs.tsv, each scored on the chosen metric
  * rank_settings   -- collapse seeds into one row per setting, ordered best first
  * summarize_sweep -- the whole thing: read, rank, write summary.tsv, print the report

This used to live as a heredoc at the end of ``scripts/sweep.sh``, which meant it ran only if the
trial loop ran to completion. An interrupted sweep therefore produced no summary at all even though
every finished trial's numbers were already on disk — which is exactly when you most want to see
where the sweep had got to. As a module it runs on demand:

    python -m cdrseqgen.visualization.sweep_tables --sweep results/runs/20260825_165553_sweep_loss
    python -m cdrseqgen.visualization.sweep_tables --sweep <dir> --rank_metric judge_ppl

Seeds are the replicate dimension; every other column identifies the setting. A setting's score is
the mean across its seeds, and `sd_score` is the spread that mean sits on — read the gap between
two settings against it before believing either.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict

# (source, key, higher_is_better). Ranking on a metric the objective optimises would only
# rediscover the objective, which is why the loss preset defaults to judge_ppl.
RANKERS = {
    "core":      ("epoch", "eval_core_accuracy", True),
    "whole":     ("epoch", "eval_accuracy", True),
    "judge_ppl": ("generation", "judge_perplexity_mean", False),
    "diversity": ("generation", "diversity_hamming", True),
    # Free-running core recovery. Unlike "core" this is not teacher-forced, so a core-weighted
    # objective does not get to grade its own homework at the token level -- but it is still the
    # region the objective targets, so read it beside judge_ppl rather than instead of it.
    "gen_core":  ("generation", "core_recovery_mean", True),
}

# Seeds are the replicate dimension: everything else identifies the setting.
KEY = ("lr", "warmup", "weight_decay", "epochs", "hidden", "layers", "loss", "core_w", "anchor_w")

SUMMARY_FIELDS = list(KEY) + ["n_seeds", "mean_score", "sd_score", "mean_core", "mean_whole",
                              "mean_best_epoch", "mean_decay"]

# Two settings closer than this on a recovery metric are a tie: it is the seed noise floor measured
# on a per-loop delta in README Findings §4.
SEED_NOISE_FLOOR = 0.006


def load_trials(map_path, rank_metric):
    """Completed trials from runs.tsv, each carrying its best epoch and its ranking score."""
    source, metric_key, _ = RANKERS[rank_metric]
    trials = []
    with open(map_path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["status"] != "completed":
                continue
            metrics_path = os.path.join(row["run_dir"], "epoch_metrics.csv")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as metrics_file:
                metrics = list(csv.DictReader(metrics_file))
            if not metrics:
                continue
            best = max(metrics, key=lambda r: float(r["eval_core_accuracy"]))
            trial = {**row,
                     "best_core": float(best["eval_core_accuracy"]),
                     "best_whole": float(best["eval_accuracy"]),
                     "best_epoch": float(best["epoch"]),
                     "final_core": float(metrics[-1]["eval_core_accuracy"]),
                     "score": None}
            if source == "epoch":
                trial["score"] = (trial["best_core"] if metric_key == "eval_core_accuracy"
                                  else trial["best_whole"])
            else:
                gen_path = os.path.join(row["run_dir"], "generation_eval.json")
                if os.path.exists(gen_path):
                    with open(gen_path) as gen_file:
                        summary = json.load(gen_file).get("summary", {})
                    value = summary.get(metric_key)
                    # summaries write NaN for a metric that was never collected
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        trial["score"] = float(value)
            trials.append(trial)
    return trials


def rank_settings(trials, higher_is_better):
    """One row per setting, seeds collapsed into a mean and a spread, best first."""
    groups = defaultdict(list)
    for trial in trials:
        groups[tuple(trial[k] for k in KEY)].append(trial)

    ranked = []
    for key, members in groups.items():
        values = [member["score"] for member in members]
        entry = dict(zip(KEY, key))
        entry.update({
            "n_seeds": len(values),
            "mean_score": statistics.mean(values),
            "sd_score": statistics.stdev(values) if len(values) > 1 else math.nan,
            "mean_core": statistics.mean(m["best_core"] for m in members),
            "mean_whole": statistics.mean(m["best_whole"] for m in members),
            "mean_best_epoch": statistics.mean(m["best_epoch"] for m in members),
            # best - final exposes overfitting: a large gap means the curve peaked and decayed.
            "mean_decay": statistics.mean(m["best_core"] - m["final_core"] for m in members),
        })
        ranked.append(entry)
    ranked.sort(key=lambda e: (-e["mean_score"] if higher_is_better else e["mean_score"],
                               float(e["lr"])))
    return ranked


def write_summary(ranked, summary_path):
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(ranked)
    return summary_path


def format_report(ranked, rank_metric, higher_is_better):
    """The human-readable ranking, including the caveats that decide whether to believe it."""
    arrow = "higher is better" if higher_is_better else "LOWER is better"
    lines = [f"\nValidation ranking by {rank_metric} ({arrow}), averaged across seeds",
             " lr       wu     wd      ep   loss                  cw/aw     n   score     "
             "sd        core      best_ep"]
    for entry in ranked:
        sd = "n/a" if math.isnan(entry["sd_score"]) else f"{entry['sd_score']:.6f}"
        loss = "—" if entry["loss"] == "__keep__" else entry["loss"]
        weights = "—" if entry["core_w"] == "__keep__" else f"{entry['core_w']}/{entry['anchor_w']}"
        lines.append(
            f" {float(entry['lr']):.0e}  {float(entry['warmup']):>5.2f}  "
            f"{float(entry['weight_decay']):>5.3f}  {float(entry['epochs']):>4.0f}  "
            f"{loss:<21} {weights:<9} {entry['n_seeds']:>2}  {entry['mean_score']:.6f}  "
            f"{sd:>8}  {entry['mean_core']:.6f}  {entry['mean_best_epoch']:>6.1f}")

    if ranked:
        winner = ranked[0]
        lines.append(f"\nWinner by {rank_metric} (validation only): "
                     + ", ".join(f"{k}={winner[k]}" for k in KEY if winner[k] != "__keep__")
                     + f", score={winner['mean_score']:.6f}")
        if winner["n_seeds"] < 2:
            lines.append("WARNING: winner has fewer than two completed seeds; do not freeze it yet.")
        if len(ranked) > 1:
            gap = abs(winner["mean_score"] - ranked[1]["mean_score"])
            if rank_metric in ("core", "whole") and gap < SEED_NOISE_FLOOR:
                lines.append(f"Gap to runner-up: {gap:.6f} — below the ~{SEED_NOISE_FLOOR} "
                             "seed-noise floor; treat as a tie.")
            else:
                lines.append(f"Gap to runner-up: {gap:.6f}")
        if rank_metric in ("core", "whole") and any(e["loss"] != "__keep__" for e in ranked):
            lines.append("WARNING: ranking a loss sweep on recovery is circular — a core-weighted "
                         "objective optimises exactly this number. Re-rank with "
                         "RANK_METRIC=judge_ppl.")
    return "\n".join(lines)


def summarize_sweep(sweep_dir, rank_metric="core", map_path=None, summary_path=None, quiet=False):
    """Read a sweep's runs.tsv, write summary.tsv beside it, and return the ranked settings."""
    if rank_metric not in RANKERS:
        raise ValueError(f"rank_metric must be one of {sorted(RANKERS)}, got {rank_metric!r}")
    map_path = map_path or os.path.join(sweep_dir, "runs.tsv")
    summary_path = summary_path or os.path.join(sweep_dir, "summary.tsv")
    if not os.path.exists(map_path):
        print(f"⚠️ No runs.tsv under {sweep_dir} — nothing to summarize")
        return None

    _, _, higher_is_better = RANKERS[rank_metric]
    trials = load_trials(map_path, rank_metric)
    scored = [trial for trial in trials if trial["score"] is not None]
    if len(scored) < len(trials):
        print(f"NOTE: {len(trials) - len(scored)} completed trial(s) have no {rank_metric} value "
              f"(generative eval off, or the metric was not collected) and are unranked.")
    if not scored:
        print(f"⚠️ No completed trial carries a {rank_metric} score — no summary written")
        return None

    ranked = rank_settings(scored, higher_is_better)
    write_summary(ranked, summary_path)
    if not quiet:
        print(format_report(ranked, rank_metric, higher_is_better))
    print(f"✅ Summary successfully saved as: {summary_path}")
    return ranked


def main():
    parser = argparse.ArgumentParser(
        description="Rank a sweep's settings across seeds and write summary.tsv.")
    parser.add_argument("--sweep", required=True, help="Sweep directory containing runs.tsv.")
    parser.add_argument("--rank_metric", default="core", choices=sorted(RANKERS),
                        help="What to rank settings on (default: core).")
    parser.add_argument("--quiet", action="store_true", help="Write the file without the report.")
    args = parser.parse_args()
    if summarize_sweep(args.sweep, args.rank_metric, quiet=args.quiet) is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
