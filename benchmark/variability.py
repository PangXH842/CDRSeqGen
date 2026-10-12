#!/usr/bin/env python3
"""Seed variability, same-seed determinism, capacity and sampling sensitivity from the saved batches (CPU only).

The 9-12 October queue trained further seeds, retrained seed 42, scanned model size and reran IgCraft at temperature
1.0. This turns those batches into the numbers the manuscript quotes, from saved artifacts alone:

  seeds        mean and spread of each method over seven training seeds, and contrasts whose intervals resample H3 groups
               and each arm's seeds together (analysis.crossed_interval), so training variability is counted
  determinism  seed 42 trained three times with an identical config: scores, candidate identity, weight differences,
               training trajectory
  capacity     the size scan beside the small models of the same seeds
  sensitivity  IgCraft at its default temperature against temperature 1.0
  endpoints    every arm again on the loop core (two residues trimmed from each end), a supporting endpoint defined
               after the primary one and computed from the saved candidates
  ladder       optional: how much of the learned gain over unconditional sampling a retrieval reference captures

Every input is hash-checked and frozen, and inputs are only read. Contrasts here are supporting analyses on the validation
split, chosen after the data were seen; none is confirmatory.

  scripts/run.sh benchmark variability plan
  scripts/run.sh benchmark variability run [--references results/benchmark/oas4_h3_retrieval_20261012]
"""
from __future__ import annotations

import argparse
from datetime import datetime
from functools import lru_cache
import json
import os
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark._common import atomic_write_json, execution_source_identity, sha256_file, stable_example_seed
from benchmark.analysis import cluster_interval, crossed_interval
from benchmark.capacity import tensor_parameters
from benchmark.extend import base_context, verify_complete
from cdrseqgen.eval.metrics import edit_similarity

SCHEMA_VERSION = 'cdrseqgen.variability.v1'
METHODS = ('mlm', 'ar', 's2s')
SIZES = ('M', 'L', 'XL')
ORIGINAL_SEEDS, NEW_SEEDS, SCAN_SEEDS = (42, 43), (44, 45, 46, 47, 48), (42, 43)
ALL_SEEDS = ORIGINAL_SEEDS + NEW_SEEDS
METRICS = {'edit': 'edit_similarity_mean', 'length_mae': 'length_absolute_error_mean', 'length_match': 'length_match_rate',
           'judge': 'judge_perplexity_mean', 'diversity': 'diversity_hamming'}
REFERENCE_LABELS = ('retrieval', 'retrieval_j', 'retrieval_fine')
ARTIFACTS = ['variability.json', 'variability.md', 'capacity.png', 'capacity.pdf', 'seed_spread.png', 'seed_spread.pdf']


def read(path):
    return json.loads(Path(path).read_text())


def spread(values):
    values = [float(v) for v in values]
    return dict(n=len(values), mean=statistics.fmean(values), sd=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values), max=max(values), values=values)


def aggregate(run, metric):
    return float(run['aggregate'][metric]['macro_mean'])


def core(loop, trim):
    """The loop without `trim` residues at each end; a loop too short to have a core has none."""
    return loop[trim:-trim] if len(loop) > 2 * trim else ''


def core_similarity(candidate, native):
    """Edit similarity of two cores. Two empty cores agree; an empty one against a non-empty one scores zero."""
    return 1.0 if candidate == native else 0.0 if not candidate or not native else edit_similarity(candidate, native)


@lru_cache(maxsize=None)
def generation(path):
    return read(path)


class Study:
    """The frameworks, their H3 groups and the saved candidates that every analysis below reads."""

    def __init__(self, manifest, clusters, settings):
        records = manifest['records']
        self.ids = [r['example_id'] for r in records]
        self.natives = {r['example_id']: r['native'] for r in records}
        self.groups = [clusters['cluster_ids'][r['source_index']] for r in records]
        self.trim, self.iterations, self.seed = settings['anchor_trim'], settings['iterations'], settings['seed']
        self._vectors = {}

    def samples(self, run):
        return {r['example_id']: r['samples'] for r in generation(run['input']['path'])['records']}

    def vector(self, run, endpoint):
        """Per-framework score of one arm: 'edit' is the scorer's native edit similarity, 'core' the trimmed one."""
        key = (run['input']['sha256'], endpoint)
        if key not in self._vectors:
            if endpoint == 'edit':
                values = {r['example_id']: r['metrics']['edit_similarity_mean'] for r in run['records']}
            else:
                samples = self.samples(run)
                values = {i: statistics.fmean(core_similarity(core(s, self.trim), core(self.natives[i], self.trim))
                                              for s in samples[i]) for i in self.ids}
            self._vectors[key] = np.array([values[i] for i in self.ids], dtype=float)
        return self._vectors[key]

    def matrix(self, runs, endpoint):
        return np.stack([self.vector(run, endpoint) for run in runs])

    def contrast(self, a_runs, b_runs, endpoint, name, family):
        a, b = self.matrix(a_runs, endpoint), self.matrix(b_runs, endpoint)
        derived = stable_example_seed(self.seed, f'{name}:{endpoint}')
        crossed = crossed_interval(a, b, self.groups, self.iterations, derived, family)
        fixed = cluster_interval(a.mean(axis=0), b.mean(axis=0), self.groups, self.iterations, derived, family)
        return dict(a=name.split(':')[0], b=name.split(':')[1], endpoint=endpoint, **crossed,
                    frameworks_only={k: fixed[k] for k in ('ci_low', 'ci_high', 'family_ci_low', 'family_ci_high')})


def contrast_pairs(has_retrieval):
    pairs = [('mlm', 'ar'), ('mlm', 's2s'), ('s2s', 'ar'), ('mlm', 'iglm'), ('ar', 'iglm'), ('s2s', 'iglm'),
             ('mlm', 'igcraft'), ('ar', 'igcraft'), ('s2s', 'igcraft'), ('iglm', 'igcraft')]
    if has_retrieval:
        pairs += [(system, 'retrieval') for system in ('mlm', 'ar', 's2s', 'iglm', 'igcraft')] + [('retrieval', 'empirical')]
    return pairs


def seeds_block(study, arms, has_retrieval):
    pairs = contrast_pairs(has_retrieval)
    out = dict(seeds=list(ALL_SEEDS), new_seeds=list(NEW_SEEDS), methods={}, pools={}, contrasts=[], new_seed_contrasts=[])
    for method in METHODS:
        runs = arms[method]
        out['methods'][method] = dict(
            metrics={name: spread([aggregate(run, key) for run in runs]) for name, key in METRICS.items()},
            core_edit=spread([study.vector(run, 'core').mean() for run in runs]),
            new_seed_edit=spread([aggregate(run, METRICS['edit']) for run in runs[len(ORIGINAL_SEEDS):]]))
    for name, runs in arms.items():
        if name not in METHODS:
            out['pools'][name] = dict(metrics={m: aggregate(runs[0], key) for m, key in METRICS.items()},
                                      core_edit=float(study.vector(runs[0], 'core').mean()))
    for endpoint in ('edit', 'core'):
        out['contrasts'] += [study.contrast(arms[a], arms[b], endpoint, f'{a}:{b}', len(pairs)) for a, b in pairs]
    methods = [(a, b) for a, b in pairs if a in METHODS and b in METHODS]
    newest = {m: arms[m][len(ORIGINAL_SEEDS):] for m in METHODS}
    out['new_seed_contrasts'] = [study.contrast(newest[a], newest[b], 'edit', f'{a}:{b}', len(methods)) for a, b in methods]
    return out


def checkpoint_of(run):
    """The checkpoint directory of a locally trained arm, from the plan of the batch that generated it."""
    batch = Path(run['input']['path']).parents[1]
    arm = next(a for a in read(batch / 'plan.json')['arms'] if a['label'] == run['label'])
    return ROOT / arm['checkpoint']


def trajectory(directory):
    states = [read(p) for p in Path(directory).glob('checkpoint-*/trainer_state.json')]
    if not states:
        return None
    last = max(states, key=lambda state: state['global_step'])
    return dict(epochs=last['epoch'], steps=last['global_step'], best_checkpoint=Path(last['best_model_checkpoint']).name,
                best_metric=last.get('best_metric'))


def weight_difference(first, second):
    from safetensors import safe_open
    with safe_open(str(first / 'model.safetensors'), 'np') as a, safe_open(str(second / 'model.safetensors'), 'np') as b:
        names = sorted(a.keys())
        if names != sorted(b.keys()):
            raise ValueError('checkpoints have different tensors')
        gaps = [float(np.abs(a.get_tensor(n).astype(np.float64) - b.get_tensor(n).astype(np.float64)).max()) for n in names]
    return dict(tensors=len(names), tensors_differing=sum(g > 0 for g in gaps), max_abs_difference=max(gaps))


def determinism_block(study, main_runs, repeat_runs):
    rows = {}
    for method in METHODS:
        runs = {'s42': main_runs[f'{method}_s42'], 'r1': repeat_runs[f'{method}_s42_r1'], 'r2': repeat_runs[f'{method}_s42_r2']}
        samples = {k: study.samples(run) for k, run in runs.items()}
        pairs = (('s42', 'r1'), ('r1', 'r2'), ('s42', 'r2'))
        directories = {k: checkpoint_of(run) for k, run in runs.items()}
        rows[method] = dict(
            runs={k: dict(label=run['label'], **{m: aggregate(run, key) for m, key in METRICS.items()},
                          trajectory=trajectory(directories[k])) for k, run in runs.items()},
            edit_range=max(aggregate(r, METRICS['edit']) for r in runs.values()) - min(aggregate(r, METRICS['edit']) for r in runs.values()),
            identical_frameworks={f'{a}_{b}': sum(samples[a][i] == samples[b][i] for i in study.ids) for a, b in pairs},
            weights={f'{a}_{b}': weight_difference(directories[a], directories[b]) for a, b in pairs})
    return dict(frameworks=len(study.ids), methods=rows)


def capacity_block(study, main_runs, scan_runs, seeds):
    rows = []
    for method in METHODS:
        small = [main_runs[f'{method}_s{s}'] for s in SCAN_SEEDS]
        sd = seeds['methods'][method]['metrics']['edit']['sd']
        cells = [('S', small)] + [(size, [scan_runs[f'{method}_{size}_s{s}'] for s in SCAN_SEEDS]) for size in SIZES]
        baseline = statistics.fmean(aggregate(r, METRICS['edit']) for r in small)
        for size, runs in cells:
            edits = [aggregate(r, METRICS['edit']) for r in runs]
            rows.append(dict(
                method=method, size=size, parameters=tensor_parameters(checkpoint_of(runs[0]) / 'model.safetensors'),
                seeds=list(SCAN_SEEDS), edit_by_seed=edits, edit=statistics.fmean(edits),
                core_edit=float(np.mean([study.vector(r, 'core').mean() for r in runs])),
                **{m: statistics.fmean(aggregate(r, key) for r in runs) for m, key in METRICS.items() if m != 'edit'},
                edit_minus_small=statistics.fmean(edits) - baseline, edit_minus_small_in_seed_sd=(statistics.fmean(edits) - baseline) / sd))
    return dict(rows=rows, small_model_seed_sd={m: seeds['methods'][m]['metrics']['edit']['sd'] for m in METHODS},
                note='S is the small model of the original batch. Its mean and parameter count use seeds 42 and 43 like the other sizes.')


def length_summary(study, run):
    payload = generation(run['input']['path'])
    lengths = [len(s) for r in payload['records'] for s in r['samples']]
    raw = [g for r in payload['records'] for g in r.get('raw_generations', [])]
    natives = [len(study.natives[i]) for i in study.ids]
    return dict(mean=statistics.fmean(lengths), sd=statistics.pstdev(lengths), min=min(lengths), max=max(lengths),
                native_mean=statistics.fmean(natives), native_sd=statistics.pstdev(natives), native_max=max(natives),
                interior_gap_slots=sum(bool(g.get('interior_gap')) for g in raw) if raw and 'interior_gap' in raw[0] else None,
                attempts=len(lengths))


def sensitivity_block(study, default_run, sensitivity_run):
    arms = {'default': default_run, 'sensitivity': sensitivity_run}
    first, second = (study.vector(arms[k], 'edit') for k in ('default', 'sensitivity'))
    return dict(
        arms={k: dict(label=run['label'], **{m: aggregate(run, key) for m, key in METRICS.items()},
                      core_edit=float(study.vector(run, 'core').mean()), core_frameworks=run['aggregate']['core_aar_mean'].get('examples'),
                      lengths=length_summary(study, run)) for k, run in arms.items()},
        edit_difference=cluster_interval(second, first, study.groups, study.iterations,
                                         stable_example_seed(study.seed, 'sensitivity'), 1),
        note='sensitivity minus default; one exploratory contrast outside any family')


def ladder_block(study, arms, references):
    """Where each reference sits between unconditional sampling and the learned systems, per endpoint."""
    out = {}
    for endpoint in ('edit', 'core'):
        level = lambda runs: float(study.matrix(runs, endpoint).mean())
        floor = level(arms['empirical'])
        learned = {name: level(arms[name]) for name in (*METHODS, 'iglm', 'igcraft')}
        refs = {label: level([run]) for label, run in references.items()}
        out[endpoint] = dict(empirical=floor, learned=learned, references=refs, captured={
            label: {name: (value - floor) / (learned[name] - floor) if learned[name] > floor else None for name in learned}
            for label, value in refs.items()})
    return out


# ---------------------------------------------------------------------------------------------------------------
# inputs: verification and freezing

def freeze(frozen, path):
    frozen[str(Path(path).resolve())] = sha256_file(path)


def seal_runs(scores, frozen):
    """Prove each scored run describes the finished generation file it names, then freeze that file."""
    for run in scores['runs']:
        path = Path(run['input']['path'])
        if sha256_file(path) != run['input']['sha256']:
            raise ValueError(f"{run['label']}: generations differ from what was scored")
        if not generation(str(path)).get('finished_at'):
            raise ValueError(f"{run['label']}: generations are unfinished")
        freeze(frozen, path)


def load_batch(directory, frozen):
    directory = ROOT / directory
    if not (directory / 'scores/comparison.json').is_file():
        raise ValueError(f'No scored comparison in {directory}. That batch must run to completion first: its last stage '
                         'writes scores/comparison.json (see the batch recipe\'s protocol for the commands).')
    scores = read(directory / 'scores/comparison.json')
    for name in ('plan.json', 'scores/comparison.json'):
        freeze(frozen, directory / name)
    seal_runs(scores, frozen)
    return scores, {run['label']: run for run in scores['runs']}


def require(runs, labels, where):
    missing = [label for label in labels if label not in runs]
    if missing:
        raise ValueError(f'{where} lacks {missing}')


def build_plan(recipe_path, references=None):
    """Verify and freeze every input; nothing is written."""
    recipe_path = Path(recipe_path).resolve()
    recipe = read(recipe_path)
    frozen = {}
    freeze(frozen, recipe_path)
    main = (ROOT / recipe['main']).resolve()
    verify_complete(main)
    context = base_context(main)
    manifest_path = Path(context['plan']['manifest'])
    for path in (main / 'plan.json', main / 'scores/comparison.json', main / 'capacity.json', context['clusters'], manifest_path):
        freeze(frozen, path)
    main_scores = read(main / 'scores/comparison.json')
    seal_runs(main_scores, frozen)
    main_runs = {run['label']: run for run in main_scores['runs']}
    require(main_runs, [f'{m}_s{s}' for m in METHODS for s in ORIGINAL_SEEDS] + ['iglm', 'empirical', 'igcraft'], 'the main comparison')
    batches = {}
    for key, labels in (('seeds', [f'{m}_s{s}' for m in METHODS for s in NEW_SEEDS]),
                        ('repeats', [f'{m}_s42_r{k}' for m in METHODS for k in (1, 2)]),
                        ('capacity', [f'{m}_{size}_s{s}' for m in METHODS for size in SIZES for s in SCAN_SEEDS]),
                        ('igcraft_t1', ['igcraft_t1'])):
        scores, runs = load_batch(recipe[key], frozen)
        require(runs, labels, f'the {key} batch')
        batches[key] = scores
    reference_dir = references or recipe.get('references')
    if reference_dir:
        scores, runs = load_batch(reference_dir, frozen)
        require(runs, ['retrieval'], 'the references batch')
        batches['references'] = scores
    digests = {main_scores['manifest']['sha256']} | {scores['manifest']['sha256'] for scores in batches.values()}
    if len(digests) != 1:
        raise ValueError('The batches were scored against different manifests')
    return dict(schema_version=SCHEMA_VERSION, recipe=str(recipe_path), name=recipe['name'], main=str(main),
                inputs={key: str((ROOT / recipe[key]).resolve()) for key in ('seeds', 'repeats', 'capacity', 'igcraft_t1')},
                references=str((ROOT / reference_dir).resolve()) if reference_dir else None,
                manifest=str(manifest_path), clusters=str(Path(context['clusters']).resolve()),
                analysis=recipe['analysis'], frozen_files=frozen, source=execution_source_identity())


def validate_plan(plan):
    if plan['source'] != execution_source_identity():
        raise ValueError('Benchmark source changed; preserve this analysis and prepare a new one')
    for path, digest in plan['frozen_files'].items():
        if sha256_file(path) != digest:
            raise ValueError(f'Frozen input changed: {path}')


def runs_of(directory):
    return {run['label']: run for run in read(Path(directory) / 'scores/comparison.json')['runs']}


def compute(plan):
    """All analyses, as one JSON-able dictionary."""
    main_runs = runs_of(plan['main'])
    seed_runs, repeat_runs = runs_of(plan['inputs']['seeds']), runs_of(plan['inputs']['repeats'])
    scan_runs, t1_runs = runs_of(plan['inputs']['capacity']), runs_of(plan['inputs']['igcraft_t1'])
    reference_runs = runs_of(plan['references']) if plan['references'] else {}
    study = Study(read(plan['manifest']), read(plan['clusters']), plan['analysis'])
    pool = {**main_runs, **seed_runs}
    arms = {method: [pool[f'{method}_s{s}'] for s in ALL_SEEDS] for method in METHODS}
    arms.update(iglm=[main_runs['iglm']], igcraft=[main_runs['igcraft']], empirical=[main_runs['empirical']],
                igcraft_t1=[t1_runs['igcraft_t1']])
    references = {label: reference_runs[label] for label in REFERENCE_LABELS if label in reference_runs}
    for label, run in references.items():
        arms[label] = [run]
    seeds = seeds_block(study, arms, 'retrieval' in references)
    capacity_references = {'iglm': next(r for r in read(Path(plan['main']) / 'capacity.json')['rows'] if r['arm'] == 'iglm'),
                           'igcraft': next(r for r in read(Path(plan['main']) / 'capacity.json')['rows'] if r['arm'] == 'igcraft')}
    result = dict(
        schema_version=SCHEMA_VERSION, name=plan['name'], frameworks=len(study.ids), groups=len(set(study.groups)),
        analysis=plan['analysis'], seeds=seeds, determinism=determinism_block(study, main_runs, repeat_runs),
        capacity=capacity_block(study, main_runs, scan_runs, seeds), sensitivity=sensitivity_block(study, main_runs['igcraft'], t1_runs['igcraft_t1']),
        external_parameters={k: dict(parameters=row['system_parameters'], by_module=row.get('parameters_by_module'))
                             for k, row in capacity_references.items()},
        ladder=ladder_block(study, arms, references) if 'retrieval' in references else None,
        inputs=plan['frozen_files'],
        definitions=dict(
            edit='native edit similarity: one minus Levenshtein distance over the longer length, averaged over a framework\'s 20 candidates, then over frameworks',
            core=f"the same on the loop core: {plan['analysis']['anchor_trim']} residues trimmed from each end of candidate and native; a candidate with no core scores 0",
            contrasts='H3 groups and each arm\'s training seeds are resampled together; family intervals are Bonferroni over the planned contrasts of that endpoint; frameworks_only keeps each arm\'s seed mean fixed',
            seeds='seeds 42 and 43 come from the original batch; 44-48 were trained later on the current source tree'),
        limitations=[
            'Validation split only; the contrasts were chosen after the data were seen and none is confirmatory.',
            'Percentile bootstrap over seven seeds is slightly too narrow; read the crossed intervals as supporting evidence.',
            'Candidate pools are drawn once (20 per framework); that Monte-Carlo noise is not in the intervals.',
            'Core edit similarity is a supporting endpoint defined after the primary one was reported.',
            'The scan changes only model size; the learning rate was not retuned for the larger models.'])
    return result


# ---------------------------------------------------------------------------------------------------------------
# report

def fmt(value, digits=4):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def interval(low, high, digits=4):
    return 'undefined' if low is None else f'[{low:+.{digits}f}, {high:+.{digits}f}]'


def report(result):
    out = ['# Seed variability, determinism, capacity and sensitivity', '',
           f"{result['frameworks']} frameworks in {result['groups']} H3 groups. " + result['definitions']['edit'], '']
    out += ['## Training seeds (7 per method)', '', '| Method | Edit (mean ± sd) | min-max | Length MAE | Match % | Judge | Diversity | Core edit |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
    for method, block in result['seeds']['methods'].items():
        m = block['metrics']
        out.append(f"| {method} | {m['edit']['mean']:.4f} ± {m['edit']['sd']:.4f} | {m['edit']['min']:.4f}-{m['edit']['max']:.4f} | "
                   f"{m['length_mae']['mean']:.3f} | {100 * m['length_match']['mean']:.2f} | {m['judge']['mean']:.3f} | "
                   f"{m['diversity']['mean']:.3f} | {block['core_edit']['mean']:.4f} ± {block['core_edit']['sd']:.4f} |")
    for name, pool in result['seeds']['pools'].items():
        m = pool['metrics']
        out.append(f"| {name} | {m['edit']:.4f} | single pool | {m['length_mae']:.3f} | {100 * m['length_match']:.2f} | {m['judge']:.3f} | {m['diversity']:.3f} | {pool['core_edit']:.4f} |")
    for endpoint in ('edit', 'core'):
        rows = [c for c in result['seeds']['contrasts'] if c['endpoint'] == endpoint]
        out += ['', f'### Seed-aware contrasts, {"native edit similarity" if endpoint == "edit" else "core edit similarity"}', '',
                '| Contrast | Difference | 95% (groups + seeds) | Family | 95% (groups only) |', '|---|---:|---|---|---|']
        for c in rows:
            out.append(f"| {c['a']} − {c['b']} | {c['difference']:+.4f} | {interval(c['ci_low'], c['ci_high'])} | "
                       f"{interval(c['family_ci_low'], c['family_ci_high'])} | {interval(c['frameworks_only']['ci_low'], c['frameworks_only']['ci_high'])} |")
    out += ['', '### Method contrasts using only the five seeds trained after the refactor', '', '| Contrast | Difference | 95% | Family |', '|---|---:|---|---|']
    for c in result['seeds']['new_seed_contrasts']:
        out.append(f"| {c['a']} − {c['b']} | {c['difference']:+.4f} | {interval(c['ci_low'], c['ci_high'])} | {interval(c['family_ci_low'], c['family_ci_high'])} |")
    out += ['', '## Same seed, identical config (seed 42 trained three times)', '',
            '| Method | Edit s42 / r1 / r2 | Identical frameworks (of 150) s42-r1 / r1-r2 / s42-r2 | Max weight difference s42-r1 / r1-r2 | Epochs run | Best metric |',
            '|---|---|---|---|---|---|']
    for method, row in result['determinism']['methods'].items():
        runs, same, weights = row['runs'], row['identical_frameworks'], row['weights']
        epochs = ' / '.join(fmt(runs[k]['trajectory']['epochs'], 0) if runs[k]['trajectory'] else 'n/a' for k in ('s42', 'r1', 'r2'))
        best = ' / '.join(fmt(runs[k]['trajectory']['best_metric'], 5) if runs[k]['trajectory'] else 'n/a' for k in ('s42', 'r1', 'r2'))
        out.append(f"| {method} | {runs['s42']['edit']:.5f} / {runs['r1']['edit']:.5f} / {runs['r2']['edit']:.5f} | "
                   f"{same['s42_r1']} / {same['r1_r2']} / {same['s42_r2']} | {weights['s42_r1']['max_abs_difference']:.1e} / {weights['r1_r2']['max_abs_difference']:.1e} | {epochs} | {best} |")
    out += ['', '## Capacity (seeds 42 and 43)', '', '| Method | Size | Parameters (M) | Edit | Δ vs small | Δ in seed sd | Length MAE | Judge | Core edit |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['capacity']['rows']:
        out.append(f"| {r['method']} | {r['size']} | {r['parameters'] / 1e6:.2f} | {r['edit']:.4f} | {r['edit_minus_small']:+.4f} | "
                   f"{r['edit_minus_small_in_seed_sd']:+.1f} | {r['length_mae']:.3f} | {r['judge']:.3f} | {r['core_edit']:.4f} |")
    s = result['sensitivity']
    out += ['', '## IgCraft temperature (default 0.1 against 1.0)', '', '| Arm | Edit | Core edit | Length MAE | Match % | Judge | Diversity | Loop length mean (sd, max) | Native |', '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for key, arm in s['arms'].items():
        n = arm['lengths']
        out.append(f"| {arm['label']} | {arm['edit']:.4f} | {arm['core_edit']:.4f} | {arm['length_mae']:.3f} | {100 * arm['length_match']:.2f} | "
                   f"{arm['judge']:.3f} | {arm['diversity']:.3f} | {n['mean']:.2f} ({n['sd']:.2f}, {n['max']}) | {n['native_mean']:.2f} ({n['native_sd']:.2f}, {n['native_max']}) |")
    d = s['edit_difference']
    out += ['', f"Edit similarity, sensitivity minus default: {d['difference']:+.4f} {interval(d['ci_low'], d['ci_high'])} (exploratory, outside any family).", '']
    if result['ladder']:
        out += ['## Reference ladder', '']
        for endpoint, ladder in result['ladder'].items():
            out += [f"**{endpoint}**: empirical {ladder['empirical']:.4f}; " + '; '.join(f'{k} {v:.4f}' for k, v in ladder['references'].items())
                    + '; learned ' + ', '.join(f'{k} {v:.4f}' for k, v in ladder['learned'].items()), '']
            for label, share in ladder['captured'].items():
                out.append(f"- {label} captures " + ', '.join(f'{100 * v:.0f}% ({k})' for k, v in share.items() if v is not None) + ' of the learned gain over unconditional sampling')
            out.append('')
    out += ['## Limitations', ''] + [f'- {line}' for line in result['limitations']] + ['']
    return '\n'.join(out)


# ---------------------------------------------------------------------------------------------------------------
# figures

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
# Categorical slots 1-3 of the reference palette, validated all-pairs on this surface; colour follows the entity.
SERIES = {'mlm': ('#2a78d6', 'o', 'MLM'), 'ar': ('#eb6834', 's', 'AR'), 's2s': ('#1baf7a', '^', 'S2S')}


def style(axis):
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(AXIS)
        axis.spines[side].set_linewidth(0.8)
    axis.set_facecolor(SURFACE)
    axis.tick_params(colors=SECONDARY, labelsize=8, length=3, width=0.8)
    axis.grid(True, color=GRID, linewidth=0.6, linestyle='-')
    axis.set_axisbelow(True)


def mark(axis, x, y, color, marker, size=7.5, **kw):
    """A marker at least 8 px across with a surface-coloured ring so it stays legible where marks overlap."""
    axis.plot([x], [y], marker=marker, markersize=size, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.4, linestyle='none', zorder=4, **kw)


def capacity_figure(result, path, references=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.6, 3.7), dpi=200, facecolor=SURFACE)
    style(axis)
    rows = result['capacity']['rows']
    axis.set_xscale('log')
    axis.set_xlim(0.6e6, 7e8)
    for method, (color, marker, name) in SERIES.items():
        own = [r for r in rows if r['method'] == method]
        xs = [r['parameters'] for r in own]
        seven = result['seeds']['methods'][method]['metrics']['edit']
        # The small size shows all seven seeds (their mean and range); the larger sizes have the two scan seeds.
        ys = [seven['mean'] if r['size'] == 'S' else r['edit'] for r in own]
        axis.plot(xs, ys, color=color, linewidth=1.5, solid_capstyle='round', zorder=3)
        for r, y in zip(own, ys):
            values = seven['values'] if r['size'] == 'S' else r['edit_by_seed']
            axis.plot([r['parameters']] * 2, [min(values), max(values)], color=color, linewidth=1.5, alpha=0.45, zorder=2)
            mark(axis, r['parameters'], y, color, marker)
        axis.annotate(name, (xs[-1], ys[-1]), xytext=(9, 0), textcoords='offset points', va='center', fontsize=8.5, color=INK)
    shown = [r['edit'] for r in rows] + [result['seeds']['methods'][m]['metrics']['edit'][k] for m in SERIES for k in ('min', 'max')]
    # IgLM lands on the large S2S model and IgCraft far to the right; their labels hang off leader lines.
    for key, label, offset in (('iglm', 'IgLM', (16, 26)), ('igcraft', 'IgCraft', (-12, 14))):
        pool = result['seeds']['pools'][key]['metrics']['edit']
        x = result['external_parameters'][key]['parameters']
        shown.append(pool)
        mark(axis, x, pool, MUTED, 'D', size=7)
        axis.annotate(label, (x, pool), xytext=offset, textcoords='offset points', ha='left' if offset[0] > 0 else 'right', va='center',
                      fontsize=8.5, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.8, shrinkA=0, shrinkB=5))
    for label, value in (references or {}).items():
        shown.append(value)
        axis.axhline(value, color=MUTED, linewidth=1.0, zorder=1)
        axis.annotate(label, (axis.get_xlim()[0], value), xytext=(4, 3), textcoords='offset points', fontsize=8, color=SECONDARY)
    axis.set_ylim(min(shown) - 0.003, max(shown) + 0.0025)
    axis.set_xlabel('Parameters (log scale)', fontsize=9, color=SECONDARY)
    axis.set_ylabel('Native edit similarity', fontsize=9, color=SECONDARY)
    axis.set_xticks([1e6, 3e6, 1e7, 3e7, 1e8, 3e8])
    axis.set_xticklabels(['1M', '3M', '10M', '30M', '100M', '300M'])
    axis.minorticks_off()
    for method, (color, marker, name) in SERIES.items():
        axis.plot([], [], color=color, marker=marker, markersize=6, linewidth=1.5, label=name)
    axis.legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc='lower left', ncols=3)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    figure.savefig(Path(path).with_suffix('.pdf'), facecolor=SURFACE)
    plt.close(figure)


def seed_figure(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pools = result['seeds']['pools']
    names = {'iglm': 'IgLM', 'igcraft': 'IgCraft', 'retrieval': 'retrieval', 'retrieval_j': 'retrieval, FR4 only',
             'retrieval_fine': 'retrieval, finer key', 'empirical': 'unconditional'}
    methods = sorted(SERIES, key=lambda m: -result['seeds']['methods'][m]['metrics']['edit']['mean'])
    singles = sorted((k for k in names if k in pools), key=lambda k: -pools[k]['metrics']['edit'])
    ladder = [(m, None) for m in methods] + [(k, names[k]) for k in singles]
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), dpi=200, facecolor=SURFACE, gridspec_kw={'width_ratios': [1.25, 1]}, sharey=True)
    full, zoom = axes
    for axis in axes:
        style(axis)
        axis.grid(False, axis='y')
    rows = list(range(len(ladder)))[::-1]
    low, high = zoom.get_xlim() if False else (0.384, 0.396)
    for row, (key, label) in zip(rows, ladder):
        if key in SERIES:
            color, marker, name = SERIES[key]
            block = result['seeds']['methods'][key]['metrics']['edit']
            full.plot([block['min'], block['max']], [row, row], color=color, linewidth=2, alpha=0.5, solid_capstyle='round', zorder=2)
            mark(full, block['mean'], row, color, marker)
            for v in block['values']:
                mark(zoom, v, row, color, marker, size=6.5)
            zoom.plot([block['mean']] * 2, [row - 0.3, row + 0.3], color=INK, linewidth=1.6, zorder=5)
        else:
            value = pools[key]['metrics']['edit']
            mark(full, value, row, MUTED, 'D', size=6.5)
            if low <= value <= high:
                mark(zoom, value, row, MUTED, 'D', size=6.5)
            else:
                zoom.annotate(f'{value:.3f}, off scale', (low, row), xytext=(6, 0), textcoords='offset points', va='center', fontsize=7.5, color=MUTED)
    full.set_yticks(rows)
    full.set_yticklabels([SERIES[k][2] if k in SERIES else label for k, label in ladder], fontsize=8.5, color=INK)
    full.set_xlim(0.27, 0.405)
    zoom.set_xlim(0.384, 0.396)
    zoom.set_xticks([0.384, 0.388, 0.392, 0.396])
    for axis, title in ((full, 'All arms'), (zoom, 'Zoom on the learned systems')):
        axis.set_title(title, fontsize=9, color=SECONDARY, loc='left')
        axis.set_xlabel('Native edit similarity', fontsize=9, color=SECONDARY)
    figure.text(0.5, 0.005, 'Left: seed range and mean. Right: each seed, mean marked by a bar. Diamonds: single candidate pools.', ha='center', fontsize=7.5, color=MUTED)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(path, facecolor=SURFACE)
    figure.savefig(Path(path).with_suffix('.pdf'), facecolor=SURFACE)
    plt.close(figure)


# ---------------------------------------------------------------------------------------------------------------

def execute(plan, output):
    validate_plan(plan)
    output.mkdir(parents=True)
    atomic_write_json(output / 'plan.json', plan)
    result = compute(plan)
    atomic_write_json(output / 'variability.json', result)
    (output / 'variability.md').write_text(report(result))
    references = result['ladder']['edit']['references'] if result['ladder'] else {}
    capacity_figure(result, output / 'capacity.png', {'retrieval (flank key)': references['retrieval']} if 'retrieval' in references else None)
    seed_figure(result, output / 'seed_spread.png')
    validate_plan(plan)
    atomic_write_json(output / 'complete.json', dict(
        completed_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(), artifacts={n: sha256_file(output / n) for n in ARTIFACTS}))
    summary(output)


def summary(output):
    print((output / 'variability.md').read_text() if (output / 'variability.md').exists() else 'No analysis yet')
    if (output / 'complete.json').exists():
        complete = read(output / 'complete.json')
        changed = [n for n, d in complete['artifacts'].items() if sha256_file(output / n) != d]
        print('complete: ' + ('artifacts match their hashes' if not changed else f'CHANGED: {changed}'))
    else:
        print('not complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['plan', 'run', 'status'])
    parser.add_argument('batch', nargs='?', type=Path, help='status: the analysis directory')
    parser.add_argument('--recipe', type=Path, default=ROOT / 'benchmark/recipes/oas4_h3_variability.json')
    parser.add_argument('--references', help='Benchmark batch holding the retrieval reference arms')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'status':
        if args.batch is None:
            parser.error('status requires an analysis directory')
        summary(args.batch.resolve())
        return
    plan = build_plan(args.recipe, args.references)
    print(f"{plan['name']}: {len(plan['frozen_files'])} frozen inputs; references {'yes' if plan['references'] else 'no'}")
    if args.command == 'plan':
        return
    stamp = datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S')
    execute(plan, args.output.resolve() if args.output else ROOT / f"results/benchmark/{stamp}_{plan['name']}")


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
