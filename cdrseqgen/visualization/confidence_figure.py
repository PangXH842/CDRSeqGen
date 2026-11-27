"""Plot whether per-candidate confidence scores improve candidate selection.

Reads per-candidate blocks written by ``evaluation.py`` into ``generation_eval.json`` and evaluates
whether selecting the top-k most confident candidates improves recovery over a random draw.

Usage:
    python -m cdrseqgen.visualization.confidence_figure \\
        --eval results/runs/latest/h3_OAS_scratch/generation_eval.json
    python -m cdrseqgen.visualization.confidence_figure \\
        --eval a.json --label "H3 scratch" --eval b.json --label "H3 ESM-2"

Panels (1x2 -> ``<save_dir>/<filename>.png``):
  * A. Top-k selection curve -- mean recovery when keeping the k most confident candidates by
       self-perplexity and judge perplexity, compared against the random-draw baseline.
  * B. Per-framework Spearman rho -- histogram of confidence-vs.-recovery rank correlations across
       frameworks, showing how consistently each score ranks candidates.

Functions:
  * selection_curve   -- mean top-k recovery for ``k = 1..max_k`` under one score key.
  * baseline_recovery -- expected recovery of a random draw across frameworks.
  * rho_distribution  -- per-framework Spearman rho between ``-score`` and candidate recovery.
  * build_figure      -- compose the two-panel matplotlib figure for one or more runs.
  * has_scores        -- check whether records carry any plottable confidence score.
  * save_run_figure   -- single-run helper used after evaluation; returns path or ``None``.
  * main              -- CLI entry point.

Notes:
  * Judge curves require runs evaluated with ``--judge``; self-perplexity curves require
    ``--skip_perplexity`` to have been off.
"""
import argparse
import json
import os
import statistics

from cdrseqgen.visualization import plot_style
import matplotlib.pyplot as plt

from cdrseqgen.eval.metrics import spearman


# --- Score Configuration & Selection/Correlation Metrics -----------------------------------------

SCORES = (
    ("self_perplexity", "self-perplexity", plot_style.color(0)),
    ("judge_perplexity", "judge perplexity", plot_style.color(1)),
)


def _usable(record):
    """Candidates that can enter the analysis: right length, and therefore a defined recovery."""
    return [c for c in record.get("candidates", []) if c.get("recovery") is not None]


def selection_curve(records, score_key, max_k):
    """Mean recovery when keeping the k most confident candidates, for k = 1..max_k.

    Perplexity is lower-is-better, so the sort is ascending. Frameworks with fewer than k usable
    candidates contribute all of theirs rather than being dropped, which keeps the framework
    population identical across k.
    """
    if max_k <= 0:
        return []
    # Sort each framework once and precompute cumulative recovery sums for O(max_k * N) lookup.
    prefix_means = []
    for record in records:
        usable = [c for c in _usable(record) if c.get(score_key) is not None]
        if not usable:
            continue
        ordered = sorted(usable, key=lambda c: c[score_key])
        running = 0.0
        means = []
        for idx, cand in enumerate(ordered, start=1):
            running += cand["recovery"]
            means.append(running / idx)
        prefix_means.append(means)

    if not prefix_means:
        return [None] * max_k
    return [
        statistics.mean(means[min(k, len(means)) - 1] for means in prefix_means)
        for k in range(1, max_k + 1)
    ]


def baseline_recovery(records):
    """Expected recovery of a randomly kept candidate — flat in k, so one number suffices."""
    per_framework = [statistics.mean(c["recovery"] for c in _usable(r))
                     for r in records if _usable(r)]
    return statistics.mean(per_framework) if per_framework else None


def rho_distribution(records, score_key):
    values = []
    for record in records:
        usable = [c for c in _usable(record) if c.get(score_key) is not None]
        if len(usable) < 3:
            continue
        # negate: low perplexity should mean high confidence
        rho = spearman([-c[score_key] for c in usable], [c["recovery"] for c in usable])
        if rho is not None:
            values.append(rho)
    return values


# --- Two-Panel Figure Builder (Panel A: Selection Curve, Panel B: Spearman rho) ------------------

def build_figure(datasets, max_k):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_sel, ax_rho = axes

    for label, records, dash in datasets:
        base = baseline_recovery(records)
        n = max(len(_usable(r)) for r in records)
        ks = list(range(1, min(max_k, n) + 1))
        if base is not None:
            ax_sel.axhline(base, color="#777", ls=":", lw=1.5)
            ax_sel.annotate(f"random draw  {base:.3f}", (ks[-1], base), fontsize=8,
                            color="#555", ha="right", va="bottom")
        for key, name, colour in SCORES:
            curve = selection_curve(records, key, len(ks))
            if all(v is None for v in curve):
                continue
            ax_sel.plot(ks, curve, marker="o", ms=4, lw=2, ls=dash, color=colour,
                        label=f"{name}{'' if len(datasets) == 1 else f' · {label}'}")

        for key, name, colour in SCORES:
            values = rho_distribution(records, key)
            if not values:
                continue
            ax_rho.hist(values, bins=20, alpha=.55, color=colour,
                        label=f"{name}  ρ̄={statistics.mean(values):+.3f}")
            ax_rho.axvline(statistics.mean(values), color=colour, lw=2)

    plot_style.style_axes(
        ax_sel,
        title="A. Does selecting on confidence beat a random draw?",
        xlabel="k — candidates kept, most confident first",
        ylabel="Mean recovery of the kept candidates",
        legend=True,
        legend_fontsize=9,
    )

    ax_rho.axvline(0, color="k", lw=1)
    plot_style.style_axes(
        ax_rho,
        title="B. …and how consistently?",
        xlabel="Spearman ρ (confidence vs recovery), one per framework",
        ylabel="Frameworks",
        legend=True,
        legend_fontsize=9,
    )
    fig.tight_layout()
    return fig


# --- Single-Run Export & CLI Entry Point ---------------------------------------------------------

def has_scores(records):
    """True when at least one candidate carries a score worth plotting."""
    return any(
        c.get(key) is not None
        for record in records for c in record.get("candidates", [])
        for key, _, _ in SCORES
    )


def save_run_figure(records, out_path, *, label=None, max_k=20):
    """Build and save the two-panel figure for one run. Returns the path, or None if there is
    nothing to plot (no per-candidate block, or every score missing)."""
    records = [r for r in records if r.get("candidates")]
    if not records or not has_scores(records):
        return None
    figure = build_figure([(label or "run", records, "-")], max_k)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    figure.savefig(out_path, dpi=plot_style.DEFAULT_DPI, bbox_inches="tight")
    plt.close(figure)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True,
                    help="generation_eval.json with per-candidate scores; repeat to overlay runs")
    ap.add_argument("--label", action="append", default=None, help="Label per --eval, in order")
    ap.add_argument("--max_k", type=int, default=20, help="Largest k on the selection curve")
    ap.add_argument("--save_dir", default="results/figures/training")
    ap.add_argument("--filename", default="confidence_selection")
    args = ap.parse_args()

    labels = args.label or [os.path.basename(os.path.dirname(p)) or p for p in args.eval]
    if len(labels) != len(args.eval):
        raise SystemExit("--label must be given once per --eval, or not at all")

    datasets = []
    for path, label, dash in zip(args.eval, labels, ["-", "--", "-.", ":"] * 4):
        with open(path) as handle:
            payload = json.load(handle)
        records = [r for r in payload.get("records", []) if r.get("candidates")]
        if not records:
            raise SystemExit(
                f"{path} has no per-candidate scores. It predates the confidence block — re-run "
                "evaluate.py (with --judge for the judge curves)."
            )
        datasets.append((label, records, dash))

    fig = build_figure(datasets, args.max_k)
    plot_style.save_figure(fig, args.save_dir, args.filename, dpi=plot_style.DEFAULT_DPI)

    for label, records, _ in datasets:
        base = baseline_recovery(records)
        print(f"\n{label}: random draw {base:.4f}")
        for key, name, _ in SCORES:
            curve = selection_curve(records, key, 3)
            rhos = rho_distribution(records, key)
            if curve[0] is None:
                continue
            print(f"   {name:<18} top1 {curve[0]:.4f} ({curve[0]-base:+.4f})   "
                  f"top3 {curve[2]:.4f}   ρ̄ {statistics.mean(rhos):+.3f} "
                  f"(negative on {sum(1 for r in rhos if r < 0)}/{len(rhos)} frameworks)")


if __name__ == "__main__":
    main()
