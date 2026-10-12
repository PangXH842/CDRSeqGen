"""Paired multi-arm intervals over verified H3 groups and fixed candidate pools."""
import math
from statistics import mean

import numpy as np

from benchmark._common import stable_example_seed

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
    if not (len(a) == len(b) == len(groups)) or len(a)==0:
        raise ValueError('Nonempty, equal-length paired inputs are required')
    if iterations < 1000 or family_size < 1:
        raise ValueError('Use at least 1,000 resamples and a positive family size')
    if any(not math.isfinite(x) for x in [*a,*b]):
        raise ValueError('Intervals require finite paired values')
    unique = list(dict.fromkeys(groups))
    point = dict(n_examples=len(a), n_groups=len(unique), mean_a=mean(a), mean_b=mean(b), difference=mean(a)-mean(b))
    if len(unique) < 2:
        return {**point, 'ci_low': None, 'ci_high': None, 'family_ci_low': None, 'family_ci_high': None,
                'reason': 'Fewer than two contributing groups'}
    index = {group:i for i,group in enumerate(unique)}
    sums = np.zeros(len(unique)); sizes = np.zeros(len(unique))
    for first,second,group in zip(a,b,groups):
        sums[index[group]] += first-second
        sizes[index[group]] += 1
    rng = np.random.default_rng(seed)
    # Counts describe uniformly sampled groups; the ratio retains original framework weighting.
    weights = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=iterations)
    differences = (weights @ sums) / (weights @ sizes)
    low,high = np.quantile(differences, [.025,.975])
    alpha = .05/family_size
    family_low,family_high = np.quantile(differences, [alpha/2,1-alpha/2])
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
    if len(pairs) != len(set(pairs)) or any(a == b or a not in runs or b not in runs for a,b in pairs):
        raise ValueError('Invalid or duplicate planned comparisons')
    results = []
    for first,second in pairs:
        for metric in METRICS:
            ids = [i for i in sorted(membership) if all(isinstance(runs[label][i].get(metric),(int,float))
                   and math.isfinite(runs[label][i][metric]) for label in [first,second])]
            if not ids:
                results.append(dict(a=first,b=second,metric=metric,n_examples=0,n_groups=0,reason='No finite paired population'))
                continue
            row = cluster_interval([runs[first][i][metric] for i in ids], [runs[second][i][metric] for i in ids],
                                   [group_by_id[i] for i in ids], iterations,
                                   stable_example_seed(seed,f'{first}:{second}:{metric}'),
                                   len(pairs) if metric == 'edit_similarity_mean' else 1)
            results.append(dict(a=first,b=second,metric=metric,example_ids=ids,**row))
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
        interval = lambda prefix: f"[{row[prefix+'low']:+.5f}, {row[prefix+'high']:+.5f}]" if row.get(prefix+'low') is not None else 'undefined'
        lines.append(f"| {row['a']} − {row['b']} | {row['n_examples']} / {row['n_groups']} | {row['difference']:+.5f} | {interval('ci_')} | {interval('family_ci_')} |")
    return '\n'.join(lines+['', *analysis['limitations'], '', 'Intervals spanning zero do not establish equivalence. Supporting endpoints and paired populations are in analysis.json.', ''])


def plot(analysis, path):
    """Save a primary-difference figure with marginal and family intervals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in analysis['results'] if r['metric']=='edit_similarity_mean' and r.get('ci_low') is not None]
    fig,axis = plt.subplots(figsize=(9, max(3,len(rows)*.32+1)))
    for i,row in enumerate(rows):
        axis.plot([row['family_ci_low'],row['family_ci_high']],[i,i],color='#abc1d1',lw=5)
        axis.plot([row['ci_low'],row['ci_high']],[i,i],color='#276890',lw=2)
        axis.scatter(row['difference'],i,color='#173e55',s=20,zorder=3)
    axis.set_yticks(range(len(rows)),[r['a']+' − '+r['b'] for r in rows]);axis.invert_yaxis()
    axis.axvline(0,color='gray',ls='--',lw=1)
    axis.set_xlabel('Paired native edit-similarity difference; trained models and candidate pools fixed')
    axis.set_title('Exploratory validation: dark = marginal 95%; pale = Bonferroni family interval',fontsize=10)
    fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)
