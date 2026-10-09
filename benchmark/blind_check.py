#!/usr/bin/env python3
"""Check that an adapter's candidates do not depend on the hidden native loop.

A length-free adapter may read the flanks (pt1, pt2) and the conditioning its manifest declares; it must not
see the answer. The scorer validates what an adapter wrote, not what it read, so each adapter's prompt
builder would otherwise be trusted. This runs the adapter under identical seeds on

  A  the manifest,
  A' the manifest again (control: the adapter must be reproducible for the comparison to mean anything),
  B  a copy whose hidden fields are replaced: a native loop of another length and other residues, and
     source_index / sequence_id swapped between frameworks so the answer cannot be looked up by index.

A == A' with A == B means nothing hidden reached the candidates. A != A' is inconclusive, not a pass.
Limits: a lookup keyed on something this does not change (say, matching pt1 against the corpus) is invisible
to it, and only length-free manifests qualify, since fixed-length tracks hand over the native length by design.

  python benchmark/blind_check.py --manifest M.json -- PYTHON ADAPTER.py
      --manifest {manifest} --output {output} --num-samples 3 --seed 42 --device cuda ...
  (one command line; {manifest} and {output} are replaced for each of the three runs)

or take the command from a saved batch plan, which checks the adapter exactly as that batch ran it:

  python benchmark/blind_check.py --manifest M.json --plan results/benchmark/BATCH/plan.json --label mlm_s42 [--device cpu]

Exit status: 0 pass, 1 hidden-field dependence found, 2 inconclusive. Model execution is user-run.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark._common import AMINO_ACIDS, atomic_write_json, validate_manifest_record

IDENTIFIERS = ('source_index', 'sequence_id')


def perturb_manifest(manifest):
    """Copy of the manifest with every hidden field replaced by consistent, different content."""
    records = manifest['records']
    if len(records) < 2:
        raise ValueError('Need at least two frameworks to swap identifiers between them')
    changed = []
    for position, record in enumerate(records):
        rng = random.Random(f"blind:{record['example_id']}")
        length = len(record['native'])
        other = rng.choice([n for n in range(max(1, length - 4), length + 5) if n != length])
        native = ''.join(rng.choice(sorted(AMINO_ACIDS)) for _ in range(other))
        donor = records[(position + 1) % len(records)]
        swapped = {key: donor[key] for key in IDENTIFIERS if key in record}
        new = {**record, **swapped, 'native': native, 'native_length': other,
               'parent_sequence': record['pt1'] + native + record['pt2'],
               'infill_range': [len(record['pt1']), len(record['pt1']) + other]}
        validate_manifest_record(new)
        changed.append(new)
    return {**manifest, 'records': changed}


def observable(payload):
    """What a reader of the candidates sees, without fields defined by the native loop or the clock."""
    return {record['example_id']: dict(
                samples=record['samples'], attempts=record['attempts'], valid_attempts=record['valid_attempts'],
                raw=[{k: v for k, v in raw.items() if k != 'length_match'} for raw in record.get('raw_generations', [])],
                failures=record.get('failures', []))
            for record in payload['records']}


def differing(first, second):
    """Example IDs whose observable output differs, or that only one run produced."""
    a, b = observable(first), observable(second)
    return sorted(i for i in set(a) | set(b) if a.get(i) != b.get(i))


def run_adapter(command, manifest, output):
    argv = [part.replace('{manifest}', str(manifest)).replace('{output}', str(output)) for part in command]
    subprocess.run(argv, check=True)
    return json.loads(Path(output).read_text())


def plan_command(plan, label, num_samples=5, device=None):
    """The adapter command a saved plan used for `label`, with the manifest and output left as placeholders."""
    from benchmark.run import generation_commands
    matches = [command for name, _, command in generation_commands(plan, Path('/unused')) if name == label]
    if len(matches) != 1:
        raise ValueError(f'{label!r} is not an arm of this plan')
    command = list(matches[0])
    for flag, value in (('--manifest', '{manifest}'), ('--output', '{output}'), ('--num-samples', str(num_samples))) + \
            ((('--device', device),) if device else ()):
        command[command.index(flag) + 1] = value
    return command


def check(manifest, command, workdir, limit=4, control=True):
    """Run the three-way comparison and return the verdict record."""
    if manifest['task']['track'] != 'length_free':
        raise ValueError('Blind checks need a length_free manifest; fixed-length tracks reveal the native length')
    if not any('{manifest}' in part for part in command) or not any('{output}' in part for part in command):
        raise ValueError('The adapter command needs {manifest} and {output} placeholders')
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    original = {**manifest, 'records': manifest['records'][:limit]}
    blind = perturb_manifest(original)
    paths = {}
    for name, content in (('original', original), ('blind', blind)):
        paths[name] = workdir / f'{name}_manifest.json'
        atomic_write_json(paths[name], content)
    reference = run_adapter(command, paths['original'], workdir / 'original.json')
    repeat = run_adapter(command, paths['original'], workdir / 'control.json') if control else None
    hidden = run_adapter(command, paths['blind'], workdir / 'blind.json')
    control_differs = differing(reference, repeat) if control else None
    blind_differs = differing(reference, hidden)
    verdict = ('inconclusive' if control_differs else 'leak' if blind_differs else 'pass')
    report = dict(verdict=verdict, frameworks=len(original['records']), control_run=control,
                  control_differs=control_differs, hidden_field_differs=blind_differs, command=command)
    atomic_write_json(workdir / 'blind_check.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=4, help='Frameworks to run per pass (at least 2)')
    parser.add_argument('--workdir', type=Path, help='Where to keep manifests, outputs and the report')
    parser.add_argument('--no-control', action='store_true', help='Skip the repeat run; a pass is then weaker evidence')
    parser.add_argument('--plan', type=Path, help='Saved run.py plan.json to take the adapter command from')
    parser.add_argument('--label', help='Arm label within --plan')
    parser.add_argument('--num-samples', type=int, default=5, help='With --plan: attempts per framework')
    parser.add_argument('--device', help='With --plan: override the adapter device, e.g. cpu')
    parser.add_argument('command', nargs=argparse.REMAINDER, help="After '--': the adapter command")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.plan:
        if command or not args.label:
            parser.error('--plan takes --label and no explicit command')
        command = plan_command(json.loads(args.plan.read_text()), args.label, args.num_samples, args.device)
    if not command:
        parser.error("Give the adapter command after '--', or --plan and --label")
    manifest = json.loads(args.manifest.read_text())
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix='blind_check_'))
    report = check(manifest, command, workdir, args.limit, not args.no_control)
    print(f"blind check {report['verdict']}: {report['frameworks']} frameworks; "
          f"control differs: {report['control_differs']}; hidden-field differs: {report['hidden_field_differs']}")
    print(f'Report: {workdir / "blind_check.json"}')
    raise SystemExit({'pass': 0, 'leak': 1, 'inconclusive': 2}[report['verdict']])


if __name__ == '__main__':
    main()
