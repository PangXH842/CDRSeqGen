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

from cdrseqgen.artifacts import load_run_config

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
    # Length, the causal method's own output. The masked model predicts length with a separate head
    # (Findings §10), so length never needed to be a sweep objective; a causal model chooses it by
    # emitting <eos>. Read attempt-level validity rather than the conditional summary: otherwise a
    # checkpoint that emits one exact loop and drops every other decode can beat a healthy one.
    # Capped MAE assigns an incomplete draw its observed cutoff error (at least one residue).
    "length_mae":   ("validity", "length_absolute_error_mean_capped", False),
    "length_match": ("validity", "fixed_length_acceptance_rate", True),
    "valid_rate":   ("validity", "valid_generation_rate", True),
    # These are all free-running -- no teacher-forced counterpart exists, and that is the point:
    # teacher-forced core accuracy conditions on the true prefix, so it can select a checkpoint that
    # continues a loop well and generates one badly (0.400 vs 8% H3 length match on 2026-09-02).
}

# Seeds are the replicate dimension: everything else identifies the setting.
KEY = ("lr", "warmup", "weight_decay", "epochs", "hidden", "layers", "loss", "core_w", "anchor_w")

SUMMARY_FIELDS = list(KEY) + [
    "selection_metric", "n_seeds", "mean_score", "sd_score", "mean_core", "mean_whole",
    "mean_best_epoch", "mean_decay",
]

# Two settings closer than this on a recovery metric are a tie: it is the seed noise floor measured
# on a per-loop delta in README Findings §4.
SEED_NOISE_FLOOR = 0.006


def _finite_value(mapping, key):
    """A finite numeric artifact value, or None for missing/blank/NaN data."""
    try:
        value = float(mapping.get(key))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _selected_epoch(run_dir, metrics):
    """The epoch whose checkpoint Trainer actually selected, plus its metric key."""
    try:
        config, _ = load_run_config(run_dir)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        config = {}

    configured = str(config.get("metric_for_best_model") or "core_accuracy")
    selection_key = configured if configured.startswith("eval_") else f"eval_{configured}"
    available = [
        key for key in (selection_key, "eval_core_accuracy", "eval_accuracy")
        if any(_finite_value(row, key) is not None for row in metrics)
    ]
    if not available:
        return None, selection_key
    selection_key = available[0]
    candidates = [
        (row, _finite_value(row, selection_key)) for row in metrics
        if _finite_value(row, selection_key) is not None
    ]
    higher = config.get("greater_is_better")
    if higher is None:
        higher = not selection_key.endswith("loss")
    chooser = max if higher else min
    return chooser(candidates, key=lambda pair: pair[1])[0], selection_key


def _generation_score(payload, source, metric_key):
    """Read one generation score without letting dropped causal draws improve length ranking."""
    summary = payload.get("summary", {})
    if source == "generation":
        return _finite_value(summary, metric_key)

    validity = payload.get("generation_validity") or {}
    direct = _finite_value(validity, metric_key)
    if direct is not None:
        return direct

    if metric_key == "fixed_length_acceptance_rate":
        attempts = _finite_value(validity, "attempts")
        matches = _finite_value(validity, "length_matches")
        if matches is None and attempts is not None:
            matches = sum(
                candidate.get("length_match") is True
                for record in payload.get("records", [])
                for candidate in record.get("candidates", [])
            )
        if attempts:
            return matches / attempts
        # Masked runs have no invalid-attempt block; all generated candidates enter the summary.
        return _finite_value(summary, "length_match_rate") if not validity else None

    if metric_key == "length_absolute_error_mean_capped":
        valid_rate = _finite_value(validity, "valid_generation_rate")
        # Legacy causal artifacts did not store capped errors. Their conditional MAE is safe only
        # when every attempt was valid; otherwise leave the trial unranked rather than flatter it.
        if not validity or valid_rate == 1.0:
            return _finite_value(summary, "length_absolute_error_mean")
    return None


def load_trials(map_path, rank_metric):
    """Completed trials scored at the checkpoint each run actually selected."""
    source, metric_key, _ = RANKERS[rank_metric]
    trials = []
    with open(map_path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["status"] != "completed":
                continue
            run_dir = row["run_dir"]
            metrics_path = os.path.join(run_dir, "epoch_metrics.csv")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as metrics_file:
                metrics = list(csv.DictReader(metrics_file))
            if not metrics:
                continue

            selected, selection_metric = _selected_epoch(run_dir, metrics)
            if selected is None:
                continue
            selected_core = _finite_value(selected, "eval_core_accuracy")
            selected_whole = _finite_value(selected, "eval_accuracy")
            selected_epoch = _finite_value(selected, "epoch")
            core_values = [
                value for value in (_finite_value(metric, "eval_core_accuracy")
                                    for metric in metrics) if value is not None
            ]
            final_core = next(
                (value for value in reversed([
                    _finite_value(metric, "eval_core_accuracy") for metric in metrics
                ]) if value is not None),
                None,
            )
            if None in (selected_core, selected_whole, selected_epoch, final_core) or not core_values:
                continue

            trial = {
                **row,
                # Kept under the historical names for summary.tsv compatibility. These now describe
                # the selected checkpoint, not an independently maximized core-accuracy epoch.
                "best_core": selected_core,
                "best_whole": selected_whole,
                "best_epoch": selected_epoch,
                "selection_metric": selection_metric,
                "peak_core": max(core_values),
                "final_core": final_core,
                "score": None,
            }
            if source == "epoch":
                trial["score"] = _finite_value(selected, metric_key)
            else:
                gen_path = os.path.join(run_dir, "generation_eval.json")
                if os.path.exists(gen_path):
                    with open(gen_path) as gen_file:
                        trial["score"] = _generation_score(
                            json.load(gen_file), source, metric_key)
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
        selection_metrics = {member["selection_metric"] for member in members}
        entry.update({
            "selection_metric": (selection_metrics.pop()
                                 if len(selection_metrics) == 1 else "mixed"),
            "n_seeds": len(values),
            "mean_score": statistics.mean(values),
            "sd_score": statistics.stdev(values) if len(values) > 1 else math.nan,
            "mean_core": statistics.mean(m["best_core"] for m in members),
            "mean_whole": statistics.mean(m["best_whole"] for m in members),
            "mean_best_epoch": statistics.mean(m["best_epoch"] for m in members),
            # Peak - final exposes core-recovery decay independently of which metric selected the
            # checkpoint (the loss preset deliberately selects on whole-span accuracy).
            "mean_decay": statistics.mean(m["peak_core"] - m["final_core"] for m in members),
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
             "sd        core    select_ep"]
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
        lines.append(f"Checkpoints selected on: {winner['selection_metric']}")
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
        if rank_metric == "length_match":
            lines.append("NOTE: length_match is exact completed non-empty lengths / all attempts; "
                         "incomplete and empty draws count as failures.")
        elif rank_metric == "length_mae":
            lines.append("NOTE: length_mae includes every attempt; incomplete draws use their "
                         "cutoff length with a minimum one-residue penalty.")
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
