#!/usr/bin/env python3
"""Add arms scored in a later batch to a finished comparison: merge, pair and extend capacity (CPU only).

run.py generates and judges the new arm(s) in a batch of their own (a recipe with externals and no arms). This
merges their scores into a completed comparison, computes the paired H3-group intervals over the family that now
includes them, and appends their capacity rows. Both input batches are read only: the base's own analysis stays as
published, and this writes a separate, wider family into a new directory. The base is a finished follow-up or an
earlier extension, so extensions chain and each inherits the roles the previous ones assigned.

  scripts/run.sh benchmark extend plan --base results/benchmark/oas4_h3_followup_20261008 --added BATCH
  scripts/run.sh benchmark extend run  --base results/benchmark/EARLIER_EXTENSION --added BATCH \
      --reference retrieval --exclude retrieval_j --exclude retrieval_fine

Roles of the added arms. By default each is an external generator, compared with every local arm and with the
references. `--reference LABEL` makes it a parameter-free reference pool (like the empirical sampler), compared with
every system. `--exclude LABEL` merges and scores it but keeps it out of the contrast family, for sensitivity checks.

The wider family is Bonferroni-adjusted over more contrasts, so its intervals are wider than the base's for the
same pairs. By default every added candidate must be retained and judged, as for the base arms; pass
--allow-incomplete for a generator that can return invalid attempts (the shortfall stays in the accounting).
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark._common import atomic_write_json, execution_source_identity, sha256_file
from benchmark.analysis import analyze, planned_pairs, plot, report
from benchmark.externals import plan_externals
from benchmark.followup import merge_scores

ARTIFACTS = ['scores/comparison.json', 'capacity.json', 'analysis.json', 'analysis.md', 'analysis.png']
EXTENSION = 'cdrseqgen.benchmark_extension.v1'


def read(path):
    return json.loads(Path(path).read_text())


def verify_complete(batch):
    """A finished follow-up whose sealed artifacts still match the hashes recorded when it completed."""
    for name, digest in read(batch / 'complete.json')['artifacts'].items():
        if sha256_file(batch / name) != digest:
            raise ValueError(f'Base artifact changed since it completed: {name}')


def root_followup(directory):
    """The finished follow-up an extension chain started from."""
    directory = Path(directory)
    while read(directory / 'plan.json').get('schema_version') == EXTENSION:
        directory = Path(read(directory / 'plan.json')['base'])
    return directory


def base_context(base):
    """What the base contributes: its H3 groups, externals, reference pools and excluded arms.

    Plans written before roles existed default to the empirical sampler as the only reference and nothing excluded.
    """
    plan = read(base / 'plan.json')
    if plan.get('schema_version') == EXTENSION:
        return dict(plan=plan, clusters=Path(plan.get('clusters') or root_followup(base) / 'clusters.json'),
                    externals=list(plan['externals']), references=list(plan.get('references', ['empirical'])),
                    excluded=list(plan.get('excluded', [])))
    return dict(plan=plan, clusters=base / 'clusters.json', references=['empirical'], excluded=[],
                externals=[entry['label'] for entry in plan_externals(read(Path(plan['source_batch']) / 'plan.json'))])


def build_plan(base, added, allow_incomplete=False, references=(), exclude=()):
    """Validate the two batches and freeze every input; nothing is written."""
    base, added = Path(base).resolve(), Path(added).resolve()
    verify_complete(base)
    context = base_context(base)
    base_plan, added_plan = context['plan'], read(added / 'plan.json')
    base_scores, added_scores = read(base / 'scores/comparison.json'), read(added / 'scores/comparison.json')
    labels = [run['label'] for run in added_scores['runs']]
    if added_plan.get('arms') or added_plan.get('stage') != 'full':
        raise ValueError('The added batch must be a full run of external generators only')
    if sorted(entry['label'] for entry in plan_externals(added_plan)) != sorted(labels):
        raise ValueError('Added scores do not match the externals the added batch planned')
    references, exclude = list(references), list(exclude)
    unknown = [label for label in references + exclude if label not in labels]
    if unknown or set(references) & set(exclude):
        raise ValueError(f'--reference/--exclude must name added arms ({labels}), each in one role only; got {unknown or "overlap"}')
    # A dry merge, so a mismatched manifest, judge, scoring policy or population fails here and not in `run`.
    merge_scores(base_scores, added_scores, labels, require_full=not allow_incomplete)
    frozen = {}

    def freeze(path):
        frozen[str(Path(path).resolve())] = sha256_file(path)

    for path in [base / 'plan.json', base / 'scores/comparison.json', context['clusters'], base / 'capacity.json',
                 added / 'plan.json', added / 'scores/comparison.json']:
        freeze(path)
    freeze(base_plan['manifest'])
    for run in added_scores['runs']:
        generations = added / 'generations' / f"{run['label']}.json"
        if not read(generations).get('finished_at') or sha256_file(generations) != run['input']['sha256']:
            raise ValueError(f"{run['label']} generations are unfinished or differ from what was scored")
        freeze(generations)
    generators = [label for label in labels if label not in references and label not in exclude]
    return dict(schema_version=EXTENSION, base=str(base), added=str(added), labels=labels,
                externals=context['externals'] + generators,
                references=context['references'] + references, excluded=context['excluded'] + exclude,
                clusters=str(Path(context['clusters']).resolve()), allow_incomplete=allow_incomplete,
                analysis=base_plan['analysis'], manifest=base_plan['manifest'], frozen_files=frozen,
                source=execution_source_identity())


def validate_plan(plan):
    if plan['source'] != execution_source_identity():
        raise ValueError('Benchmark source changed; preserve this extension and prepare a new one')
    for path, digest in plan['frozen_files'].items():
        if sha256_file(path) != digest:
            raise ValueError(f'Frozen input changed: {path}')


def capacity_rows(plan):
    """One row per added generator, from the parameter counts its adapter recorded."""
    rows = []
    for label in plan['labels']:
        model = read(Path(plan['added']) / 'generations' / f'{label}.json')['model']
        parameters = model.get('parameters_network')
        if parameters == 0:  # a reference pool with no neural network, like the empirical sampler
            rows.append(dict(arm=label, sequence_parameters=0, length_head_parameters=0, system_parameters=0, actual_epochs=0,
                             optimizer_steps=0, training_runtime_seconds=0, timing_scope='No neural training'))
            continue
        rows.append(dict(arm=label, sequence_parameters=parameters, length_head_parameters=0, system_parameters=parameters,
                         parameters_by_module=model.get('parameters_by_module'),
                         count_source='adapter provenance (model.parameters_network); an EMA copy is not counted',
                         trained_sequence_parameters=0, actual_epochs=None, optimizer_steps=None,
                         training_runtime_seconds=None, training_hardware_recorded=None,
                         timing_scope='Frozen released baseline; pretraining compute unavailable'))
    return rows


def execute(plan, output):
    validate_plan(plan)
    base, added = Path(plan['base']), Path(plan['added'])
    output.mkdir(parents=True)
    atomic_write_json(output / 'plan.json', plan)
    (output / 'scores').mkdir()
    merged = merge_scores(read(base / 'scores/comparison.json'), read(added / 'scores/comparison.json'),
                          plan['labels'], require_full=not plan['allow_incomplete'])
    atomic_write_json(output / 'scores/comparison.json', merged)
    clusters, manifest = read(plan['clusters']), read(plan['manifest'])
    group_by_id = {r['example_id']: clusters['cluster_ids'][r['source_index']] for r in manifest['records']}
    family = [run['label'] for run in merged['runs'] if run['label'] not in plan['excluded']]
    pairs = planned_pairs(family, externals=plan['externals'], references=plan['references'])
    analysis = analyze(merged, group_by_id, pairs, plan['analysis']['iterations'], plan['analysis']['seed'])
    analysis['inputs'] = dict(scores_sha256=sha256_file(output / 'scores/comparison.json'),
                              clusters_sha256=sha256_file(plan['clusters']), manifest_sha256=sha256_file(plan['manifest']))
    analysis['roles'] = dict(externals=plan['externals'], references=plan['references'], excluded=plan['excluded'])
    atomic_write_json(output / 'analysis.json', analysis)
    (output / 'analysis.md').write_text(report(analysis))
    plot(analysis, output / 'analysis.png')
    previous = read(base / 'capacity.json')
    atomic_write_json(output / 'capacity.json', {**previous, 'rows': previous['rows'] + capacity_rows(plan)})
    validate_plan(plan)
    atomic_write_json(output / 'complete.json', dict(
        completed_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(), added=plan['labels'],
        contrasts=len(pairs), artifacts={name: sha256_file(output / name) for name in ARTIFACTS}))
    summary(output)


def summary(output):
    """Print the table the manuscript will quote; reads saved files only."""
    print((output / 'analysis.md').read_text() if (output / 'analysis.md').exists() else 'No analysis yet')
    if (output / 'complete.json').exists():
        complete = read(output / 'complete.json')
        changed = [n for n, d in complete['artifacts'].items() if sha256_file(output / n) != d]
        print(f"complete: {complete['contrasts']} contrasts; added {complete['added']}; "
              f"{'artifacts match their hashes' if not changed else f'CHANGED: {changed}'}")
    else:
        print('not complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['plan', 'run', 'status'])
    parser.add_argument('batch', nargs='?', type=Path, help='status: the extension directory')
    parser.add_argument('--base', type=Path, default=ROOT / 'results/benchmark/oas4_h3_followup_20261008')
    parser.add_argument('--added', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--reference', action='append', default=[], metavar='LABEL',
                        help='Added arm that is a parameter-free reference pool; compared with every system')
    parser.add_argument('--exclude', action='append', default=[], metavar='LABEL',
                        help='Added arm that is merged and scored but kept out of the contrast family')
    args = parser.parse_args()
    if args.command == 'status':
        if args.batch is None:
            parser.error('status requires an extension directory')
        summary(args.batch.resolve())
        return
    if args.added is None:
        parser.error('--added is required')
    plan = build_plan(args.base, args.added, args.allow_incomplete, args.reference, args.exclude)
    print(f"extend {Path(plan['base']).name} with {plan['labels']}: externals {plan['externals']}, "
          f"references {plan['references']}, excluded {plan['excluded']}, {len(plan['frozen_files'])} frozen inputs")
    if args.command == 'plan':
        return
    stamp = datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S')
    execute(plan, args.output.resolve() if args.output else ROOT / f"results/benchmark/{stamp}_{Path(plan['added']).name}_extension")


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
