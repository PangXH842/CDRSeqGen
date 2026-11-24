#!/usr/bin/env python3
"""Paired uncertainty intervals, capacity accounting, and two-arm framework bootstrap.

Combines three post-hoc statistical utilities that read scored benchmark artifacts without running models:
  - Multi-arm H3-cluster and crossed (framework x training-seed) bootstrap intervals
    (formerly benchmark/analysis.py)
  - Parameter counts and actual training exposure from saved safetensors/trainer states
    (formerly benchmark/capacity.py)
  - Historical two-arm paired framework bootstrap CLI (formerly benchmark/bootstrap.py):
      python benchmark/analysis/stats.py --scores results/benchmark/<id>/scores/<track>.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from statistics import mean
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark._common import stable_example_seed


# ---------------------------------------------------------------------------
# Multi-arm paired intervals over verified H3 groups (formerly benchmark/analysis.py)
# ---------------------------------------------------------------------------

METRICS = ['edit_similarity_mean', 'length_absolute_error_mean', 'length_match_rate',
           'core_aar_mean', 'judge_perplexity_mean', 'diversity_hamming']


def planned_pairs(labels, externals=('iglm',), references=('empirical',)):
    """Keep each neural seed explicit; one pool per external generator; parameter-free references at the bottom.

    Every local arm is compared with each external. Every arm is compared with each reference except those listed
    before it, so with references ('empirical', 'retrieval') the retrieval pool is compared with the empirical one,
    and every system with both. External pools are not compared with each other. The pair count is the Bonferroni
    family size, so each added arm or reference widens the family.
    """
    externals, references = tuple(externals), tuple(references)
    if (len(labels) != len(set(labels)) or not references or not externals
            or not set(references) <= set(labels) or not set(externals) <= set(labels)
            or set(externals) & set(references)):
        raise ValueError(f'Unique labels including every reference {list(references)} and external {list(externals)} '
                         'are required, with no label in both roles')
    local = [label for label in labels if label not in {*externals, *references}]
    pairs = [(label, external) for external in externals for label in local]
    for position, reference in enumerate(references):
        pairs += [(label, reference) for label in labels if label != reference and label not in references[:position]]
    return pairs


def cluster_interval(a, b, groups, iterations=20000, seed=42, family_size=1):
    """Resample whole groups, retaining the framework-weighted estimand.

    Example: group X has 3 frameworks and group Y has 1. A draw [X,Y]
    averages 4 paired differences; [X,X] averages 6. Groups are sampled
    uniformly, but their frameworks retain equal weight within each draw.
    Candidate pools and trained models stay fixed.
    """
    if not (len(a) == len(b) == len(groups)) or len(a) == 0:
        raise ValueError('Nonempty, equal-length paired inputs are required')
    if iterations < 1000 or family_size < 1:
        raise ValueError('Use at least 1,000 resamples and a positive family size')
    if any(not math.isfinite(x) for x in [*a, *b]):
        raise ValueError('Intervals require finite paired values')
    unique = list(dict.fromkeys(groups))
    point = dict(n_examples=len(a), n_groups=len(unique), mean_a=mean(a), mean_b=mean(b), difference=mean(a) - mean(b))
    if len(unique) < 2:
        return {**point, 'ci_low': None, 'ci_high': None, 'family_ci_low': None, 'family_ci_high': None,
                'reason': 'Fewer than two contributing groups'}
    index = {group: i for i, group in enumerate(unique)}
    sums = np.zeros(len(unique))
    sizes = np.zeros(len(unique))
    for first, second, group in zip(a, b, groups):
        sums[index[group]] += first - second
        sizes[index[group]] += 1
    rng = np.random.default_rng(seed)
    # Counts describe uniformly sampled groups; the ratio retains original framework weighting.
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=iterations)
    differences = (weights @ sums) / (weights @ sizes)
    low, high = np.quantile(differences, [.025, .975])
    alpha = .05 / family_size
    family_low, family_high = np.quantile(differences, [alpha / 2, 1 - alpha / 2])
    return {**point, 'ci_low': float(low), 'ci_high': float(high),
            'family_ci_low': float(family_low), 'family_ci_high': float(family_high)}


def crossed_interval(a, b, groups, iterations=20000, seed=42, family_size=1):
    """Paired difference of two arms whose rows are training seeds and whose columns are frameworks.

    Whole H3 groups are resampled as in cluster_interval, and independently for each arm its training seeds, so the
    interval carries framework and training variability together. An arm with a single row is a fixed pool (a frozen
    external model, a reference). A percentile bootstrap over a handful of seeds is slightly too narrow (roughly a
    factor (n-1)/n in variance), so read these intervals as supporting rather than confirmatory.
    """
    a, b = (np.atleast_2d(np.asarray(arm, dtype=float)) for arm in (a, b))
    if a.size == 0 or b.size == 0 or not (a.shape[1] == b.shape[1] == len(groups)):
        raise ValueError('Nonempty arms with one column per framework are required')
    if iterations < 1000 or family_size < 1:
        raise ValueError('Use at least 1,000 resamples and a positive family size')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Intervals require finite values')
    unique = list(dict.fromkeys(groups))
    point = dict(n_examples=a.shape[1], n_groups=len(unique), n_seeds_a=a.shape[0], n_seeds_b=b.shape[0],
                 mean_a=float(a.mean()), mean_b=float(b.mean()), difference=float(a.mean() - b.mean()))
    if len(unique) < 2:
        return {**point, 'ci_low': None, 'ci_high': None, 'family_ci_low': None, 'family_ci_high': None,
                'reason': 'Fewer than two contributing groups'}
    owner = np.array([unique.index(group) for group in groups])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=iterations)
    weights = counts[:, owner].astype(float)
    weights /= weights.sum(axis=1, keepdims=True)          # framework weights of each resample; ratio estimator

    def estimate(arm):
        per_seed = weights @ arm.T                           # resamples x seeds: each seed's weighted framework mean
        if arm.shape[0] == 1:
            return per_seed[:, 0]
        draws = rng.multinomial(arm.shape[0], np.full(arm.shape[0], 1 / arm.shape[0]), size=iterations)
        return (draws * per_seed).sum(axis=1) / arm.shape[0]

    differences = estimate(a) - estimate(b)
    low, high = np.quantile(differences, [.025, .975])
    alpha = .05 / family_size
    family_low, family_high = np.quantile(differences, [alpha / 2, 1 - alpha / 2])
    return {**point, 'ci_low': float(low), 'ci_high': float(high),
            'family_ci_low': float(family_low), 'family_ci_high': float(family_high)}


def analyze(scores, group_by_id, pairs, iterations=20000, seed=42):
    """Analyze matched intersections; report their framework and group populations."""
    runs = {}
    for run in scores['runs']:
        label = run['label']
        if label in runs:
            raise ValueError('Duplicate scored arm label')
        records = run['records']
        values = {r['example_id']: r['metrics'] for r in records}
        if len(values) != len(records):
            raise ValueError(f'Duplicate frameworks in {label}')
        runs[label] = values
    membership = set(next(iter(runs.values())))
    if any(set(records) != membership for records in runs.values()) or set(group_by_id) != membership:
        raise ValueError('Scores and group mapping must cover identical framework IDs')
    if len(pairs) != len(set(pairs)) or any(a == b or a not in runs or b not in runs for a, b in pairs):
        raise ValueError('Invalid or duplicate planned comparisons')
    results = []
    for first, second in pairs:
        for metric in METRICS:
            ids = [i for i in sorted(membership) if all(isinstance(runs[label][i].get(metric), (int, float))
                   and math.isfinite(runs[label][i][metric]) for label in [first, second])]
            if not ids:
                results.append(dict(a=first, b=second, metric=metric, n_examples=0, n_groups=0,
                                    reason='No finite paired population'))
                continue
            row = cluster_interval([runs[first][i][metric] for i in ids], [runs[second][i][metric] for i in ids],
                                   [group_by_id[i] for i in ids], iterations,
                                   stable_example_seed(seed, f'{first}:{second}:{metric}'),
                                   len(pairs) if metric == 'edit_similarity_mean' else 1)
            results.append(dict(a=first, b=second, metric=metric, example_ids=ids, **row))
    return dict(schema_version='cdrseqgen.cluster_comparison.v1', iterations=iterations, seed=seed,
                primary_metric='edit_similarity_mean', primary_comparison_family=len(pairs), pairs=pairs,
                resampling_unit='verified original representative H3 cluster', estimand='equal-framework mean paired difference',
                primary_intervals='95% marginal plus Bonferroni family intervals across planned primary pairs',
                supporting_intervals='95% marginal, unadjusted; exploratory',
                limitations=['Candidate pools and trained checkpoints fixed; generation/training variability excluded',
                             'Comparisons selected after reviewing validation means; exploratory, not confirmatory',
                             'Representative clusters do not guarantee separation of all biological relatives',
                             'Conditional metrics compare only the paired eligible intersection'], results=results)


def report(analysis):
    """Render primary contrasts; a zero-spanning interval is not equivalence."""
    lines = ['# Paired validation analysis', '', '| Comparison | Frameworks / groups | Edit difference | 95% marginal CI | Family CI |',
             '|---|---:|---:|---|---|']
    for row in analysis['results']:
        if row['metric'] != 'edit_similarity_mean':
            continue
        interval = lambda prefix: f"[{row[prefix + 'low']:+.5f}, {row[prefix + 'high']:+.5f}]" if row.get(prefix + 'low') is not None else 'undefined'
        lines.append(f"| {row['a']} − {row['b']} | {row['n_examples']} / {row['n_groups']} | {row['difference']:+.5f} | {interval('ci_')} | {interval('family_ci_')} |")
    return '\n'.join(lines + ['', *analysis['limitations'], '',
                              'Intervals spanning zero do not establish equivalence. Supporting endpoints and paired populations are in analysis.json.', ''])


def plot(analysis, path):
    """Save a primary-difference figure with marginal and family intervals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in analysis['results'] if r['metric'] == 'edit_similarity_mean' and r.get('ci_low') is not None]
    fig, axis = plt.subplots(figsize=(9, max(3, len(rows) * .32 + 1)))
    for i, row in enumerate(rows):
        axis.plot([row['family_ci_low'], row['family_ci_high']], [i, i], color='#abc1d1', lw=5)
        axis.plot([row['ci_low'], row['ci_high']], [i, i], color='#276890', lw=2)
        axis.scatter(row['difference'], i, color='#173e55', s=20, zorder=3)
    axis.set_yticks(range(len(rows)), [r['a'] + ' − ' + r['b'] for r in rows])
    axis.invert_yaxis()
    axis.axvline(0, color='gray', ls='--', lw=1)
    axis.set_xlabel('Paired native edit-similarity difference; trained models and candidate pools fixed')
    axis.set_title('Exploratory validation: dark = marginal 95%; pale = Bonferroni family interval', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Capacity and actual training exposure (formerly benchmark/capacity.py)
# ---------------------------------------------------------------------------

def tensor_parameters(path):
    """Count stored learned tensors for the supported scratch checkpoints/heads.

    Their saved tensors are parameters: MLM positions are learned; rotary
    buffers in AR/S2S are not persisted. Reject unrecognized saved buffers.
    """
    with Path(path).open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(length))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    if any('inv_freq' in k or 'running_' in k for k in tensors):
        raise ValueError('Unrecognized persisted buffer; do not count it as a parameter')
    return sum(math.prod(v['shape']) for v in tensors.values())


def gpt2_parameters(config):
    """Count GPT-2 learned parameters, counting its tied output embedding once."""
    if config.get('model_type') != 'gpt2' or config.get('add_cross_attention', False):
        raise ValueError('Unsupported external architecture for configuration-derived count')
    width = config['n_embd']
    inner = config.get('n_inner') or 4 * width
    embeddings = config['vocab_size'] * width + config['n_positions'] * width
    blocks = config['n_layer'] * (4 * width * width + 2 * width * inner + 9 * width + inner)
    output = 0 if config.get('tie_word_embeddings', True) else config['vocab_size'] * width
    return embeddings + blocks + 2 * width + output


def extract_capacity(root, source_plan, iglm_config):
    """Keep actual stopping epochs/steps and timing scopes; absent hardware stays unknown."""
    rows = []
    for arm in source_plan['arms']:
        run = Path(root) / arm['checkpoint']
        config = json.loads((run / 'run_config.json').read_text())
        metrics = json.loads((run / 'train_results.json').read_text())
        states = [json.loads(p.read_text()) for p in run.glob('checkpoint-*/trainer_state.json')]
        if (run / 'trainer_state.json').exists():
            states.append(json.loads((run / 'trainer_state.json').read_text()))
        state = max(states, key=lambda d: d['global_step'])
        if not math.isclose(state['epoch'], metrics['epoch'], abs_tol=1e-6):
            raise ValueError(f'Final state/epoch mismatch for {arm["label"]}')
        parameters = tensor_parameters(run / 'model.safetensors')
        head = tensor_parameters(run / 'length_head.safetensors') if arm['method'] == 'mlm' else 0
        best = Path(state['best_model_checkpoint']).name
        selected = json.loads((run / best / 'trainer_state.json').read_text())
        rows.append(dict(arm=arm['label'], method=arm['method'], training_seed=config['seed'],
                         sequence_parameters=parameters, length_head_parameters=head, system_parameters=parameters + head,
                         trained_sequence_parameters=parameters, actual_epochs=state['epoch'], selected_epoch=selected['epoch'],
                         optimizer_steps=state['global_step'], training_runtime_seconds=metrics['train_runtime'],
                         maximum_epochs=config['num_train_epochs'], training_hardware_recorded=None,
                         source_checkpoint=str(run), timing_scope='Trainer sequence-training runtime; excludes MLM length fitting and generative evaluation'))
    rows.append(dict(arm='iglm', sequence_parameters=gpt2_parameters(iglm_config), length_head_parameters=0,
                     system_parameters=gpt2_parameters(iglm_config), count_source='released GPT-2 config; tied weights counted once',
                     trained_sequence_parameters=0, actual_epochs=None, optimizer_steps=None, training_runtime_seconds=None,
                     training_hardware_recorded=None, timing_scope='Frozen released baseline; pretraining compute unavailable'))
    rows.append(dict(arm='empirical', sequence_parameters=0, length_head_parameters=0, system_parameters=0,
                     actual_epochs=0, optimizer_steps=0, training_runtime_seconds=0, timing_scope='No neural training'))
    return dict(rows=rows, limitations=['Unequal parameter counts and tuning exposure; no isolated architecture effect',
                'Historical GPU model is unrecorded here; current hardware is not backfilled',
                'Trainer samples/steps-per-second may use the planned epoch budget; not used to infer actual exposure',
                'Timing is not an end-to-end generation or compute-normalized comparison'])


# ---------------------------------------------------------------------------
# Historical two-arm paired framework bootstrap (formerly benchmark/bootstrap.py)
# ---------------------------------------------------------------------------

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
        "uncertainty_scope": {
            "resampling_unit": "example",
            "candidate_pools": "fixed_observed_draws",
            "multiplicity_adjustment": "none",
        },
        "results": results,
    }


def format_report(comparison):
    label_a, label_b = comparison["runs"]
    lines = [
        f"Paired bootstrap — {label_a} minus {label_b} "
        f"({comparison['iterations']:,} resamples, seed {comparison['seed']}, "
        f"{int(CONFIDENCE * 100)}% percentile CI)",
        "The example is the resampling unit; a metric uses only examples where both runs report it.",
        "Candidate pools are fixed: generation-seed uncertainty is not included.",
        "Intervals are per metric and unadjusted for examining multiple endpoints.",
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
