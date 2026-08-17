#!/usr/bin/env python3
"""Render a matched CDRSeqGen-versus-baseline dashboard from shared scorer output.

The score JSON, rather than adapter output, is the plotting contract. This keeps validation and
metric definitions in benchmark/scorer.py and makes the figure a pure, reproducible view of an
already-scored comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SCORE_SCHEMA_VERSION = "cdrseqgen.benchmark_scores.v1"
RECOVERY_METRICS = (
    ("whole_span_aar_mean", "Whole"),
    ("core_aar_mean", "Core"),
    ("anchor_aar_mean", "Anchor"),
)
LENGTH_BUCKETS = ("short", "medium", "long")
COLORS = ("#2878B5", "#E07A24", "#3A923A", "#8A60A8", "#C44E52")
SUPPORTED_TRACKS = {"fixed_native_length", "length_free"}


def load_scores(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        payload = json.load(handle)
    validate_scores(payload)
    return payload


def _macro_mean(summary: dict[str, Any], metric: str) -> float:
    value = summary.get(metric)
    if not isinstance(value, dict) or not isinstance(value.get("macro_mean"), (int, float)):
        raise ValueError(f"score artifact lacks {metric}.macro_mean")
    result = float(value["macro_mean"])
    if not math.isfinite(result):
        raise ValueError(f"score artifact has non-finite {metric}.macro_mean")
    return result


def _optional_macro_mean(summary: dict[str, Any], metric: str) -> float | None:
    value = summary.get(metric)
    if not isinstance(value, dict):
        return None
    result = value.get("macro_mean")
    if not isinstance(result, (int, float)) or not math.isfinite(result):
        return None
    return float(result)


def benchmark_track(payload: dict[str, Any]) -> str:
    """Return the one sampling track shared by every scored run."""
    tracks = {
        (run.get("sampling") or {}).get("track", "fixed_native_length")
        for run in payload.get("runs") or []
    }
    if len(tracks) != 1:
        raise ValueError(f"scored runs do not share one benchmark track: {sorted(tracks)}")
    track = tracks.pop()
    if track not in SUPPORTED_TRACKS:
        raise ValueError(f"unsupported benchmark track {track!r}")
    return track


def validate_scores(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCORE_SCHEMA_VERSION:
        raise ValueError(
            f"expected score schema {SCORE_SCHEMA_VERSION!r}, got "
            f"{payload.get('schema_version')!r}"
        )
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValueError("comparison plotting requires at least two scored runs")
    track = benchmark_track(payload)
    for run in runs:
        if not isinstance(run.get("aggregate"), dict):
            raise ValueError(f"run {run.get('label', '<unknown>')!r} lacks aggregate metrics")
        required = (
            [metric for metric, _ in RECOVERY_METRICS]
            if track == "fixed_native_length"
            else ["length_match_rate", "length_absolute_error_mean", "edit_similarity_mean"]
        )
        for metric in required:
            _macro_mean(run["aggregate"], metric)
        _macro_mean(run["aggregate"], "diversity_hamming")
        if not isinstance(run.get("records"), list):
            raise ValueError(f"run {run.get('label', '<unknown>')!r} lacks scored records")


def display_label(run: dict[str, Any]) -> str:
    raw = str(run.get("label") or run.get("adapter") or "model")
    known = {"cdrseqgen": "CDRSeqGen", "iglm": "IgLM"}
    return known.get(raw.lower(), raw.replace("_", " ").title())


def normalized_position_curve(
    run: dict[str, Any], bins: int = 11
) -> tuple[list[float], list[float | None], list[int]]:
    """Macro-average recovery in normalized-position bins, preserving examples as the unit.

    A long loop can contribute multiple residues to one bin. Those residues are averaged inside the
    example first, so longer loops do not silently receive more weight than shorter loops.
    """
    if bins < 2:
        raise ValueError("position bins must be at least 2")
    across_examples: list[list[float]] = [[] for _ in range(bins)]
    for record in run.get("records", []):
        within_example: list[list[float]] = [[] for _ in range(bins)]
        for point in record.get("metrics", {}).get("position_recovery", []):
            relative = point.get("relative_position")
            recovery = point.get("recovery")
            if not isinstance(relative, (int, float)) or not isinstance(recovery, (int, float)):
                continue
            if not math.isfinite(relative) or not math.isfinite(recovery):
                continue
            index = min(bins - 1, max(0, int(float(relative) * (bins - 1) + 0.5)))
            within_example[index].append(float(recovery))
        for index, values in enumerate(within_example):
            if values:
                across_examples[index].append(statistics.mean(values))
    x_values = [index / (bins - 1) for index in range(bins)]
    y_values = [statistics.mean(values) if values else None for values in across_examples]
    counts = [len(values) for values in across_examples]
    return x_values, y_values, counts


def _grouped_bars(
    ax: plt.Axes,
    runs: list[dict[str, Any]],
    categories: list[str],
    values: list[list[float]],
) -> None:
    width = 0.8 / len(runs)
    for run_index, (run, run_values) in enumerate(zip(runs, values)):
        offset = (run_index - (len(runs) - 1) / 2) * width
        positions = [index + offset for index in range(len(categories))]
        bars = ax.bar(
            positions,
            run_values,
            width=width,
            color=COLORS[run_index % len(COLORS)],
            label=display_label(run),
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.2f}" if math.isfinite(value) else "" for value in run_values],
            padding=2,
            fontsize=8,
        )
    ax.set_xticks(range(len(categories)), categories)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def _recovery_panel(ax: plt.Axes, runs: list[dict[str, Any]]) -> None:
    values = [
        [_macro_mean(run["aggregate"], metric) for metric, _ in RECOVERY_METRICS]
        for run in runs
    ]
    _grouped_bars(ax, runs, [label for _, label in RECOVERY_METRICS], values)
    ax.set_title("A. Recovery decomposition")
    ax.set_ylabel("Amino-acid recovery")
    ax.legend(frameon=False, loc="upper center", ncols=len(runs))


def _length_panel(ax: plt.Axes, runs: list[dict[str, Any]]) -> None:
    available = [
        bucket
        for bucket in LENGTH_BUCKETS
        if all(bucket in run.get("by_length_bucket", {}) for run in runs)
    ]
    if not available:
        ax.text(0.5, 0.5, "No length-bucket metrics", ha="center", va="center")
        ax.set_axis_off()
        return
    values = [
        [
            _macro_mean(run["by_length_bucket"][bucket], "core_aar_mean")
            for bucket in available
        ]
        for run in runs
    ]
    category_labels = []
    for bucket in available:
        examples = runs[0]["by_length_bucket"][bucket]["core_aar_mean"].get("examples", "?")
        category_labels.append(f"{bucket.title()}\n(n={examples})")
    _grouped_bars(ax, runs, category_labels, values)
    ax.set_title("B. Core recovery by H3 length")
    ax.set_ylabel("Core amino-acid recovery")


def _position_panel(ax: plt.Axes, runs: list[dict[str, Any]], bins: int) -> None:
    plotted = False
    for run_index, run in enumerate(runs):
        x_values, y_values, _ = normalized_position_curve(run, bins=bins)
        points = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
        if not points:
            continue
        x_present, y_present = zip(*points)
        ax.plot(
            x_present,
            y_present,
            marker="o",
            linewidth=2,
            markersize=4,
            color=COLORS[run_index % len(COLORS)],
            label=display_label(run),
        )
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No position-resolved metrics", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.set_title("C. Recovery across normalized H3 position")
    ax.set_xlabel("Relative position (N-terminus → C-terminus)")
    ax.set_ylabel("Amino-acid recovery")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(linestyle="--", alpha=0.35)


def _tradeoff_panel(ax: plt.Axes, runs: list[dict[str, Any]]) -> None:
    for run_index, run in enumerate(runs):
        diversity = _macro_mean(run["aggregate"], "diversity_hamming")
        recovery = _macro_mean(run["aggregate"], "core_aar_mean")
        ax.scatter(
            diversity,
            recovery,
            s=110,
            color=COLORS[run_index % len(COLORS)],
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
        ax.annotate(
            display_label(run),
            (diversity, recovery),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("D. Recovery–diversity operating point")
    ax.set_xlabel("Pairwise Hamming diversity (higher →)")
    ax.set_ylabel("Core recovery (higher →)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(linestyle="--", alpha=0.35)


def _length_free_summary_panel(ax: plt.Axes, runs: list[dict[str, Any]]) -> None:
    metrics = (
        ("length_match_rate", "Length match"),
        ("edit_similarity_mean", "Edit similarity"),
        ("valid_generation_rate", "Valid"),
    )
    values = [[_macro_mean(run["aggregate"], metric) for metric, _ in metrics] for run in runs]
    _grouped_bars(ax, runs, [label for _, label in metrics], values)
    ax.set_title("A. Length-free generation quality")
    ax.set_ylabel("Rate / similarity")
    ax.legend(frameon=False, loc="upper center", ncols=len(runs))


def _length_error_panel(ax: plt.Axes, runs: list[dict[str, Any]]) -> None:
    categories = [("Overall", None)]
    for bucket in LENGTH_BUCKETS:
        summaries = [run.get("by_length_bucket", {}).get(bucket, {}) for run in runs]
        if all(_optional_macro_mean(summary, "length_absolute_error_mean") is not None
               for summary in summaries):
            categories.append((bucket.title(), bucket))

    values = []
    for run in runs:
        row = []
        for _label, bucket in categories:
            summary = run["aggregate"] if bucket is None else run["by_length_bucket"][bucket]
            row.append(_macro_mean(summary, "length_absolute_error_mean"))
        values.append(row)

    width = 0.8 / len(runs)
    for run_index, (run, run_values) in enumerate(zip(runs, values)):
        offset = (run_index - (len(runs) - 1) / 2) * width
        positions = [index + offset for index in range(len(categories))]
        bars = ax.bar(
            positions,
            run_values,
            width=width,
            color=COLORS[run_index % len(COLORS)],
            label=display_label(run),
        )
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in run_values], padding=2, fontsize=8)
    ax.set_xticks(range(len(categories)), [label for label, _ in categories])
    ax.set_title("B. Absolute length error")
    ax.set_ylabel("Mean error (residues; lower is better)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def _conditional_recovery_panel(ax: plt.Axes, runs: list[dict[str, Any]]) -> None:
    values = [
        [
            _optional_macro_mean(run["aggregate"], metric)
            if _optional_macro_mean(run["aggregate"], metric) is not None
            else float("nan")
            for metric, _ in RECOVERY_METRICS
        ]
        for run in runs
    ]
    if not any(math.isfinite(value) for row in values for value in row):
        ax.text(0.5, 0.5, "No exact-length candidates", ha="center", va="center")
        ax.set_axis_off()
        return
    _grouped_bars(ax, runs, [label for _, label in RECOVERY_METRICS], values)
    ax.set_title("C. Recovery among exact-length candidates only")
    ax.set_ylabel("Conditional amino-acid recovery")


def _length_free_tradeoff_panel(ax: plt.Axes, runs: list[dict[str, Any]]) -> None:
    for run_index, run in enumerate(runs):
        diversity = _macro_mean(run["aggregate"], "diversity_hamming")
        similarity = _macro_mean(run["aggregate"], "edit_similarity_mean")
        ax.scatter(
            diversity,
            similarity,
            s=110,
            color=COLORS[run_index % len(COLORS)],
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
        ax.annotate(
            display_label(run),
            (diversity, similarity),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("D. Diversity–edit-similarity operating point")
    ax.set_xlabel("Pairwise sequence diversity (higher →)")
    ax.set_ylabel("Edit similarity to native (higher →)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(linestyle="--", alpha=0.35)


def is_development_artifact(payload: dict[str, Any]) -> bool:
    benchmark_id = str(payload.get("manifest", {}).get("benchmark_id", "")).lower()
    return "validation" in benchmark_id or "smoke" in benchmark_id


def build_figure(
    payload: dict[str, Any], *, title: str | None = None, position_bins: int = 11
) -> plt.Figure:
    validate_scores(payload)
    runs = payload["runs"]
    track = benchmark_track(payload)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    if track == "length_free":
        _length_free_summary_panel(axes[0][0], runs)
        _length_error_panel(axes[0][1], runs)
        _conditional_recovery_panel(axes[1][0], runs)
        _length_free_tradeoff_panel(axes[1][1], runs)
    else:
        _recovery_panel(axes[0][0], runs)
        _length_panel(axes[0][1], runs)
        _position_panel(axes[1][0], runs, position_bins)
        _tradeoff_panel(axes[1][1], runs)

    benchmark_id = payload.get("manifest", {}).get("benchmark_id", "unknown benchmark")
    examples = runs[0]["aggregate"].get("micro_accounting", {}).get("examples")
    if examples is None:
        examples = runs[0]["aggregate"]["whole_span_aar_mean"].get("examples", "?")
    sample_counts = {run.get("sampling", {}).get("num_samples") for run in runs}
    samples = sample_counts.pop() if len(sample_counts) == 1 else "mixed"
    figure.suptitle(
        title or (
            "CDR-H3 length-free benchmark comparison"
            if track == "length_free"
            else "CDR-H3 fixed-length benchmark comparison"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.945,
        f"{benchmark_id} · {examples} frameworks · {samples} samples/framework",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    if is_development_artifact(payload):
        figure.text(
            0.5,
            0.015,
            "DEVELOPMENT / VALIDATION SMOKE — NOT A FINAL BENCHMARK CLAIM",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#A32121",
        )
    figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.92))
    return figure


def save_comparison_figure(
    payload: dict[str, Any],
    output: str | Path,
    *,
    title: str | None = None,
    position_bins: int = 11,
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(payload, title=title, position_bins=position_bins)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a matched benchmark dashboard from benchmark/scorer.py output."
    )
    parser.add_argument("--scores", required=True, help="Shared benchmark score JSON")
    parser.add_argument("--output", required=True, help="Output figure path (for example .png or .pdf)")
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument(
        "--position-bins",
        type=int,
        default=11,
        help="Number of normalized-position bins (default: 11)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_scores(args.scores)
    output = save_comparison_figure(
        payload,
        args.output,
        title=args.title,
        position_bins=args.position_bins,
    )
    print(f"Saved benchmark comparison figure: {output}")


if __name__ == "__main__":
    main()
