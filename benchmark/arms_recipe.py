#!/usr/bin/env python3
"""Write a benchmark recipe whose arms are the checkpoints of finished training batches.

The OAS4 recipe lists its six checkpoints by hand. For a queue that trains new seeds or sizes, the run
directories do not exist until training ends, so this reads each batch's saved plan and run table instead,
and copies every other setting (manifest, sampling, judge, endpoint) from a reference recipe. The new arms are
therefore generated and scored under exactly the protocol the original six were.

  python benchmark/arms_recipe.py --name oas4_h3_seeds --batch results/runs/BATCH [--batch ...] \
      --output benchmark/recipes/generated/oas4_h3_seeds.json

An arm is labelled by its trial name without `_scratch_no_partner` (mlm_s44, ar_L_s42, s2s_s42_r1). Only completed
scratch trials qualify; anything else stops the script unless a flag names the exception it may skip:
--skip-pretrained for non-scratch trials, --skip-incomplete for trials that did not finish (listed on stderr, so an
unattended queue can salvage a partly finished batch without hiding the shortfall).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'benchmark/recipes/oas4_h3_length_free.json'
KEPT = ('manifest', 'smoke_manifest', 'sampling', 'judge', 'endpoint')
STRIP = '_scratch_no_partner'


def read(path):
    return json.loads(Path(path).read_text())


def batch_arms(batch, skip_pretrained=False, skip_incomplete=False):
    """One arm per completed scratch trial of a training batch, in plan order."""
    batch = Path(batch).resolve()
    plan = read(batch / 'plan.json')
    with (batch / 'runs.tsv').open() as handle:
        rows = {row['trial_id']: row for row in csv.DictReader(handle, delimiter='\t')}  # a retry replaces its first row
    arms = []
    for trial in plan['trials']:
        config, row = trial['config'], rows.get(trial['trial_id'])
        if config.get('model_type') != 'scratch':
            if skip_pretrained:
                continue
            raise ValueError(f"{trial['suffix']} is not a scratch model; the benchmark adapter serves scratch checkpoints")
        if row is None or row['status'] != 'completed':
            message = f"{trial['suffix']} has not completed ({row['status'] if row else 'no row'}) in {batch.name}"
            if not skip_incomplete:
                raise ValueError(message)
            print(f'SKIPPED: {message}', file=sys.stderr)
            continue
        arms.append(dict(label=trial['suffix'].replace(STRIP, ''), method=config['method'],
                         training_seed=config['seed'], checkpoint=str(Path(row['run_dir']).resolve().relative_to(ROOT))))
    return arms


def build(name, batches, reference=REFERENCE, skip_pretrained=False, skip_incomplete=False):
    recipe = {'name': name, **{key: read(reference)[key] for key in KEPT}}
    recipe['arms'] = [arm for batch in batches for arm in batch_arms(batch, skip_pretrained, skip_incomplete)]
    labels = [arm['label'] for arm in recipe['arms']]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f'Arms need unique labels; got {labels}')
    recipe['limits'] = read(reference)['limits'] + [
        f'Arms are checkpoints of {len(batches)} later training batch(es), trained on the current source tree; '
        'the original six were trained before the 7 October command-interface refactor.']
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True)
    parser.add_argument('--batch', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--skip-pretrained', action='store_true')
    parser.add_argument('--skip-incomplete', action='store_true')
    args = parser.parse_args()
    recipe = build(args.name, args.batch, args.reference, args.skip_pretrained, args.skip_incomplete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(recipe, indent=2) + '\n')
    print(f"Wrote {len(recipe['arms'])} arms to {args.output}: {', '.join(a['label'] for a in recipe['arms'])}")


if __name__ == '__main__':
    main()
