"""Plan/run the inference-only comparison; scripts/run.sh benchmark exposes this file.

  plan   -- validate inputs and show the budget without model execution
  smoke  -- five frameworks, five draws, training seed 42 plus every external generator
  run    -- 150 frameworks, twenty draws, both training seeds plus every external generator
  status -- inspect saved outputs
  resume -- reuse a frozen batch plan and compatible adapter outputs
  recipe -- write a benchmark recipe whose arms are checkpoints of finished training batches

Scientific choices live in benchmark/recipes; trained checkpoints are read only. External generators
(benchmark/adapters/registry.py) run in their own environments and meet the scorer only through
adapter-output JSON, so a recipe may also name externals without any local arms to add a generator
to a finished batch.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark._common import atomic_write_json, execution_source_identity, sha256_file, validate_manifest_record
from benchmark.adapters.cdrseqgen import (
    TOKENIZER_ASSETS, load_run_config, pretrained_input_files, resolve_decoding,
    validate_checkpoint_contract,
)
from benchmark.adapters.registry import (
    DEFAULT_PACKAGES, EXTERNALS, RESERVED_LABELS, default_python, plan_externals, recipe_externals)


# EOS-terminating methods use EOSSampler; all three registered methods serve both scratch and
# pretrained backbones (pretrained MLM also pins its saved tokenizer and fitted length head).
EOS_METHODS = ('ar', 's2s')
PRETRAINED_METHODS = ('mlm', 'ar', 's2s')
# A pretrained model's vocabulary and span layout are saved beside its weights (a scratch model's are code).
SAVED_VOCAB_FILES = TOKENIZER_ASSETS

REFERENCE_RECIPE = ROOT / 'benchmark/recipes/oas4_h3_length_free.json'
RECIPE_KEPT_KEYS = ('manifest', 'smoke_manifest', 'sampling', 'judge', 'endpoint')
TRIAL_LABEL_STRIP = '_scratch_no_partner'


def read(path):
    return json.loads(Path(path).read_text())


def environment_identity(python, packages=DEFAULT_PACKAGES):
    """Read interpreter/package versions; do not import models or inspect GPUs.

    A package the environment lacks is recorded as null rather than aborting, since externals need not
    share the training stack. Resume passes the names a plan stored, so its comparison stays like-for-like.
    """
    script = ("import sys,json,importlib.metadata as m\n"
              "def version(name):\n"
              "    try: return m.version(name)\n"
              "    except m.PackageNotFoundError: return None\n"
              f"print(json.dumps(dict(python=sys.version,packages={{n:version(n) for n in {list(packages)!r}}})))")
    return json.loads(subprocess.check_output([str(python), '-c', script], text=True))


def freeze_external_inputs(entry, frozen_files):
    """Hash an external's input files into the plan, refusing any that is not its pinned digest."""
    for path, pin in EXTERNALS[entry['adapter']].inputs(entry['options']).items():
        digest = sha256_file(ROOT / path)
        if pin and digest != pin:
            raise ValueError(f"{entry['label']} input {path} has digest {digest}, not its pinned {pin}")
        frozen_files[str(path)] = digest


def arm_paths(arm, config, checkpoint, config_path):
    """Files that identify one local arm; raise if the benchmark cannot serve it.

    Scratch and registered pretrained backbones are served for MLM, AR, and S2S;
    pretrained checkpoints pin their saved vocabulary and layout files beside their weights,
    and MLM checkpoints also pin their fitted length head.
    """
    paths = [checkpoint / 'model.safetensors', checkpoint / 'config.json', config_path]
    if config['model_type'] != 'scratch':
        if arm['method'] not in PRETRAINED_METHODS:
            raise ValueError(f"{arm['label']}: pretrained {arm['method']} checkpoints are not served by the "
                             f"benchmark adapter (supported pretrained methods: {PRETRAINED_METHODS})")
        from cdrseqgen.methods.registry import checkpoint_method
        checkpoint_method(config).module('backbones').resolve(config['model_type'])  # an unknown name fails here
        paths.extend(pretrained_input_files(checkpoint))
    if arm['method'] == 'mlm':
        paths.extend([checkpoint / 'length_config.json', checkpoint / 'length_head.safetensors'])
        length_seed = read(checkpoint / 'length_config.json').get('seed')
        if length_seed != arm['training_seed']:
            raise ValueError(
                f"{arm['label']}: length_config seed {length_seed!r} does not match training_seed {arm['training_seed']!r}"
            )
    return paths


def build_plan(recipe_path, stage, cdr_python, external_pythons=None):
    """Validate a recipe into a frozen plan; `external_pythons` maps an external's label to its interpreter."""
    recipe = read(recipe_path)
    external_pythons = external_pythons or {}
    externals = [dict(label=entry['label'], adapter=entry['adapter'], options=entry['options'],
                      python=str(Path(external_pythons.get(entry['label']) or default_python(entry['adapter'])).resolve()))
                 for entry in recipe_externals(recipe)]
    unknown = set(external_pythons) - {entry['label'] for entry in externals}
    if unknown:
        raise ValueError(f'Interpreters given for externals this recipe does not name: {sorted(unknown)}')
    arm_labels = [arm['label'] for arm in recipe.get('arms', [])]
    labels = arm_labels + [entry['label'] for entry in externals]
    if not labels or len(labels) != len(set(labels)) or RESERVED_LABELS.intersection(labels):
        raise ValueError(f'Recipe needs at least one arm or external, with unique non-reserved labels: {labels}')
    for entry in externals:
        if not Path(entry['python']).is_file():
            raise ValueError(f"No interpreter for external {entry['label']!r} at {entry['python']}; "
                             f"create its environment or pass --external-python {entry['label']}=PATH")
    manifest_path = ROOT / recipe['smoke_manifest' if stage == 'smoke' else 'manifest']
    manifest = read(manifest_path)
    digest = sha256_file(manifest_path)
    expected_digest = manifest_path.with_suffix('.json.sha256').read_text().split()[0]
    if digest != expected_digest:
        raise ValueError(f'Manifest SHA-256 mismatch for {manifest_path}: {digest} != {expected_digest}')
    track = manifest['task']['track']
    if track != 'length_free':
        raise ValueError(f'Benchmark runner requires length_free track, got {track!r}')
    split = manifest['selection']['split']
    if split not in {'validation', 'test'}:
        raise ValueError(f"Manifest split must be 'validation' or 'test', got {split!r}")
    if recipe.get('split') and split != recipe['split']:
        raise ValueError(f"Manifest split {split!r} does not match recipe split {recipe['split']!r}")
    records = manifest['records']
    if not records:
        raise ValueError(f'Manifest {manifest_path} contains no records')
    expected_counts = recipe.get('expected_frameworks')
    if isinstance(expected_counts, dict) and stage in expected_counts:
        if len(records) != expected_counts[stage]:
            raise ValueError(f"Stage {stage!r} expects {expected_counts[stage]} frameworks, got {len(records)}")
    elif isinstance(expected_counts, int) and stage == 'full':
        if len(records) != expected_counts:
            raise ValueError(f"Full stage expects {expected_counts} frameworks, got {len(records)}")
    if len({r['example_id'] for r in records}) != len(records):
        raise ValueError(f'Manifest {manifest_path} contains duplicate example_id values')
    for record in records:
        validate_manifest_record(record)
        for entry in externals:
            EXTERNALS[entry['adapter']].check_record(record, recipe['sampling'], entry['options'])
    source = ROOT / 'data/processed' / manifest['source']['processed_file']
    source_digest = sha256_file(source)
    if source_digest != manifest['source']['processed_file_sha256']:
        raise ValueError(
            f"Processed corpus hash mismatch for {source}: {source_digest} != {manifest['source']['processed_file_sha256']}"
        )
    frozen_files = {str(manifest_path.relative_to(ROOT)): digest,
                    str(recipe_path.relative_to(ROOT)): sha256_file(recipe_path)}
    for entry in externals:
        freeze_external_inputs(entry, frozen_files)
    arms = []
    for arm in recipe.get('arms', []):
        if stage == 'smoke' and arm['training_seed'] != 42:
            continue
        checkpoint = ROOT / arm['checkpoint']
        config, config_path = load_run_config(checkpoint)
        validate_checkpoint_contract(config, manifest)
        if config['method'] != arm['method'] or config['seed'] != arm['training_seed']:
            raise ValueError(
                f"{arm['label']}: checkpoint config ({config['method']!r}, seed {config['seed']!r}) "
                f"differs from recipe ({arm['method']!r}, seed {arm['training_seed']!r})"
            )
        decoding = resolve_decoding(config)
        for path in arm_paths(arm, config, checkpoint, config_path):
            frozen_files[str(path.relative_to(ROOT))] = sha256_file(path)
        arms.append({**arm, 'decoding': decoding})
    samples = recipe['sampling']['smoke_num_samples' if stage == 'smoke' else 'num_samples']
    # One interpreter may serve several roles, so fingerprint the union of what each role needs.
    packages = {str(cdr_python): list(DEFAULT_PACKAGES)}
    for entry in externals:
        names = packages.setdefault(entry['python'], [])
        names.extend(n for n in EXTERNALS[entry['adapter']].packages if n not in names)
    return dict(schema_version='cdrseqgen.benchmark_batch.v2', name=recipe['name'], stage=stage,
                manifest=str(manifest_path.relative_to(ROOT)), frameworks=len(records),
                arms=arms, externals=externals, sampling={**recipe['sampling'], 'num_samples': samples},
                judge=recipe['judge'], endpoint=recipe['endpoint'], limits=recipe['limits'],
                requested_attempts=len(records) * samples * (len(arms) + len(externals)),
                cdr_python=str(cdr_python),
                environments={python: environment_identity(python, names) for python, names in packages.items()},
                execution_source=execution_source_identity(), frozen_files=frozen_files)


def validate_frozen_plan(plan):
    if plan['execution_source'] != execution_source_identity():
        raise ValueError('Benchmark source changed; preserve this batch and plan a new source-versioned run')
    for path, digest in plan['frozen_files'].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f'Frozen benchmark input changed: {path}')
    for python, expected in plan['environments'].items():
        if environment_identity(python, tuple(expected['packages'])) != expected:
            raise ValueError(f'Benchmark environment changed: {python}')


def generation_commands(plan, output):
    """Return explicit argument lists; adapters own tokenization and attempt accounting."""
    manifest = str(ROOT / plan['manifest'])
    s = plan['sampling']
    common = ['--manifest', manifest, '--num-samples', str(s['num_samples']),
              '--temperature', str(s['temperature']), '--top-p', str(s['top_p']),
              '--seed', str(s['seed']), '--device', 'cuda']
    commands = []
    for arm in plan['arms']:
        path = output / 'generations' / (arm['label'] + '.json')
        command = [plan['cdr_python'], str(ROOT / 'benchmark/adapters/cdrseqgen.py'), *common,
                   '--checkpoint', str(ROOT / arm['checkpoint']), '--output', str(path),
                   '--decoding', arm['decoding'], '--top-k', str(s['top_k']),
                   '--length-temperature', str(s['length_temperature'])]
        if arm['method'] != 'mlm':
            command += ['--max-new-tokens', str(s['max_new_tokens'])]
        commands.append((arm['label'], path, command))
    for entry in plan_externals(plan):
        spec = EXTERNALS[entry['adapter']]
        path = output / 'generations' / (entry['label'] + '.json')
        commands.append((entry['label'], path, [entry['python'], str(ROOT / spec.script), *common,
                                                *spec.arguments(path, s, entry['options'])]))
    return commands


def status(output):
    plan = read(output / 'plan.json')
    for label, path, _ in generation_commands(plan, output):
        payload = read(path) if path.exists() else {}
        records = payload.get('records', [])
        print(f"{label}: {len(records)}/{plan['frameworks']} frameworks; finished={bool(payload.get('finished_at'))}; "
              f"returned={sum(len(r['samples']) for r in records)}; "
              f"invalid={sum(r['invalid_attempts'] for r in records)}")
    print(f"Shared scores: {(output / 'scores/comparison.json').exists()}")


def execute(plan, output, gpu, resume=False):
    output.mkdir(parents=True, exist_ok=resume)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume:
            atomic_write_json(output / 'plan.json', plan)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'HF_HUB_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        commands = generation_commands(plan, output)
        for label, path, command in commands:
            validate_frozen_plan(plan)
            if path.exists():
                command.append('--resume')
            print(f'Generate {label}: {path}', flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        validate_frozen_plan(plan)
        score_path = output / 'scores/comparison.json'
        if not score_path.exists():
            command = [plan['cdr_python'], str(ROOT / 'benchmark/scorer.py'),
                       '--manifest', str(ROOT / plan['manifest']), '--output', str(score_path),
                       '--judge', plan['judge']['name'], '--judge-max-len', str(plan['judge']['max_len']),
                       '--device', 'cuda']
            for label, path, _ in commands:
                command += ['--input', f'{label}={path}']
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        # The figure compares runs, so a batch that only adds one generator to a finished comparison has none yet.
        if len(commands) > 1:
            figure = output / 'comparison.png'
            subprocess.run([plan['cdr_python'], str(ROOT / 'benchmark/analysis/plot_comparison.py'),
                            '--scores', str(score_path), '--output', str(figure)], cwd=ROOT, env=env, check=True)
        status(output)


# ---------------------------------------------------------------------------
# Recipe builder from finished training batches (formerly benchmark/arms_recipe.py)
# ---------------------------------------------------------------------------

def batch_arms(batch, skip_pretrained=False, skip_incomplete=False):
    """One arm per completed eligible trial of a training batch, in plan order.

    Completed scratch trials and completed pretrained MLM/AR/S2S trials qualify;
    untrained (`do_train` false) trials do not.
    """
    batch = Path(batch).resolve()
    plan = read(batch / 'plan.json')
    with (batch / 'runs.tsv').open() as handle:
        rows = {row['trial_id']: row for row in csv.DictReader(handle, delimiter='\t')}  # a retry replaces its first row
    arms = []
    for trial in plan['trials']:
        config, row = trial['config'], rows.get(trial['trial_id'])
        pretrained = config.get('model_type') != 'scratch'
        if pretrained and skip_pretrained:
            continue
        if pretrained and config['method'] not in PRETRAINED_METHODS:
            raise ValueError(f"{trial['suffix']} is a pretrained {config['method']} model; the benchmark adapter serves "
                             f"pretrained checkpoints for {PRETRAINED_METHODS}")
        if not config.get('do_train', True):
            raise ValueError(f"{trial['suffix']} was not trained (do_train false) and keeps no checkpoint to benchmark; "
                             'pass only the batches that trained')
        if row is None or row['status'] != 'completed':
            message = f"{trial['suffix']} has not completed ({row['status'] if row else 'no row'}) in {batch.name}"
            if not skip_incomplete:
                raise ValueError(message)
            print(f'SKIPPED: {message}', file=sys.stderr)
            continue
        arms.append(dict(label=trial['suffix'].replace(TRIAL_LABEL_STRIP, ''), method=config['method'],
                         training_seed=config['seed'], checkpoint=str(Path(row['run_dir']).resolve().relative_to(ROOT))))
    return arms


def build_arms_recipe(name, batches, reference=REFERENCE_RECIPE, skip_pretrained=False, skip_incomplete=False):
    """Build a benchmark recipe from completed training batches using `reference`'s protocol settings."""
    ref = read(reference)
    recipe = {'name': name, **{key: ref[key] for key in RECIPE_KEPT_KEYS}}
    recipe['arms'] = [arm for batch in batches for arm in batch_arms(batch, skip_pretrained, skip_incomplete)]
    labels = [arm['label'] for arm in recipe['arms']]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f'Arms need unique labels; got {labels}')
    recipe['limits'] = ref['limits'] + [
        f'Arms are checkpoints of {len(batches)} later training batch(es), trained on the current source tree; '
        'the original six were trained before the 7 October command-interface refactor.']
    return recipe


def label_path(value):
    label, separator, path = value.partition('=')
    if not (label and separator and path):
        raise argparse.ArgumentTypeError('expected LABEL=PATH')
    return label, Path(path)


def _recipe_main(argv):
    parser = argparse.ArgumentParser(
        prog='benchmark/run.py recipe',
        description='Write a benchmark recipe whose arms are the checkpoints of finished training batches.',
    )
    parser.add_argument('--name', required=True)
    parser.add_argument('--batch', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=REFERENCE_RECIPE)
    parser.add_argument('--skip-pretrained', action='store_true')
    parser.add_argument('--skip-incomplete', action='store_true')
    args = parser.parse_args(argv)
    recipe = build_arms_recipe(args.name, args.batch, args.reference, args.skip_pretrained, args.skip_incomplete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(recipe, indent=2) + '\n')
    print(f"Wrote {len(recipe['arms'])} arms to {args.output}: {', '.join(a['label'] for a in recipe['arms'])}")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ['recipe']:
        _recipe_main(arguments[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['plan', 'smoke', 'run', 'resume', 'status'])
    parser.add_argument('batch', nargs='?', type=Path)
    parser.add_argument('--recipe', type=Path, default=ROOT / 'benchmark/recipes/oas4_h3_length_free.json')
    parser.add_argument('--stage', choices=['smoke', 'full'], default='full', help='Plan stage only')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cdr-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--external-python', type=label_path, action='append', default=[], metavar='LABEL=PATH',
                        help="Interpreter for one external generator; default is its adapter's conda environment")
    parser.add_argument('--iglm-python', type=Path, help="Shorthand for --external-python iglm=PATH")
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(arguments)
    if not args.gpu.isdigit():
        parser.error('--gpu must be one GPU index, such as 0')
    if args.command in ['resume', 'status']:
        if args.batch is None:
            parser.error('resume/status requires a batch directory')
        output = args.batch.resolve()
        if args.command == 'status':
            status(output)
        else:
            plan = read(output / 'plan.json')
            validate_frozen_plan(plan)
            execute(plan, output, args.gpu, resume=True)
        return
    if args.batch is not None:
        parser.error('Use --output for a new batch')
    stage = args.stage if args.command == 'plan' else 'smoke' if args.command == 'smoke' else 'full'
    external_pythons = dict(args.external_python)
    if args.iglm_python:
        external_pythons['iglm'] = args.iglm_python
    plan = build_plan(args.recipe.resolve(), stage, args.cdr_python.resolve(), external_pythons)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        arm_labels = [a['label'] for a in plan['arms']] + [e['label'] for e in plan['externals']]
        print(f"{plan['name']} {stage}: {len(arm_labels)} arms, {plan['frameworks']} frameworks, "
              f"{plan['sampling']['num_samples']} draws = {plan['requested_attempts']:,} requested attempts")
        print('Arms: ' + ', '.join(arm_labels))
        print('Generation: heavy-chain flanks; inferred MLM length or natural EOS. Judge: heavy chain only.')
        print('Source: ' + plan['execution_source']['sha256'])
        print('Primary endpoint: ' + plan['endpoint'])
    if args.command == 'plan':
        return
    stamp = datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S_%f')
    output = args.output.resolve() if args.output else ROOT / f"results/benchmark/{stamp}_{plan['name']}_{stage}"
    execute(plan, output, args.gpu)


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
