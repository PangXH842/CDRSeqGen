"""Shared batch machinery: identity, completion checks, frozen-plan reading and batch status.

Nothing here knows about suites, recipes or the command line, so the benchmark tools can adopt it later.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

import cdrseqgen
from cdrseqgen.artifacts import (
    EPOCH_METRICS_FILENAME, GENERATION_EVAL_FILENAME, CONTEXT_ABLATION_FILENAME, RUN_CONFIG_FILENAME,
    sha256_file,
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def flag(value):
    if str(value).lower() in {"1", "true", "yes"}:
        return True
    if str(value).lower() in {"0", "false", "no", ""}:
        return False
    raise ValueError(f"Expected true/false, got {value!r}")


def source_identity():
    """Changes to executable sources or installed training packages invalidate resume."""
    package = Path(cdrseqgen.__file__).resolve().parent   # the cdrseqgen package, as before
    sources = {str(p.relative_to(package)): sha256_file(p) for p in sorted(package.rglob('*.py'))}
    sources["train.py"] = sha256_file("train.py")
    packages = {}
    for name in ("torch", "transformers", "accelerate", "safetensors"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(source_sha256=fingerprint(sources), python=sys.version, packages=packages)


def needs_generation(config):
    return config.get("run_generative_eval", True) and config.get("gen_num_examples", 150) != 0


def completed_artifacts(run_dir, config):
    if not run_dir:
        return False
    root = Path(run_dir)
    try:
        with (root / RUN_CONFIG_FILENAME).open() as handle:
            saved = json.load(handle)
        for key, value in config.items():
            # Called under a matching trial_id, which already pins the corpus content hash.
            if value is None or key in {"output_dir", "data_path"}:
                continue
            actual = saved.get(key)
            if key == "warmup_ratio" and key not in saved:
                # New TrainingArguments removes warmup_ratio and stores the fraction in steps.
                expected = config.get("warmup_steps") or value
                if saved.get("warmup_steps") == expected:
                    continue
            # Transformers 5 converts the deprecated warmup_ratio into warmup_steps.
            # Older versions leave steps=0; both represent the same requested ratio.
            if key == "warmup_steps" and config.get("warmup_ratio") is not None:
                allowed = (value,) if value else (value, config["warmup_ratio"])
                if actual in allowed:
                    continue
            # TrainingArguments normalizes report_to from a string to a list.
            if key == "report_to" and isinstance(value, str):
                value = [] if value == "none" else [value]
            if actual != value:
                return False
        with (root / EPOCH_METRICS_FILENAME).open() as handle:
            if not next(csv.DictReader(handle), None):
                return False
        if needs_generation(config):
            filenames = [GENERATION_EVAL_FILENAME]
            if config.get('gen_eval_context_ablation'):
                filenames.append(CONTEXT_ABLATION_FILENAME)
            for filename in filenames:
                with (root / filename).open() as handle:
                    payload = json.load(handle)
                if not payload.get("records") or not payload.get("summary"):
                    return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def read_plan(batch):
    """Read a planned batch; historical unplanned folders cannot be resumed."""
    path = batch / 'plan.json'
    if not path.is_file():
        raise ValueError(f'{batch} has no plan.json; historical results remain in RESULTS.md')
    plan = json.loads(path.read_text())
    if (not isinstance(plan, dict) or plan.get('schema_version') != 1
            or plan.get('mode') not in {'suite', 'sweep'}
            or not isinstance(plan.get('trials'), list) or not plan['trials']):
        raise ValueError(f'Invalid frozen plan: {path}')
    return plan


def batch_active(batch):
    """Read the existing batch lock without creating or changing any files."""
    path = batch / '.lock'
    if not path.exists():
        return False
    with path.open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def batch_status(batch):
    """Treat a completed journal row as complete only when its artifacts agree."""
    plan = read_plan(batch)
    ledger = batch / 'runs.tsv'
    states = {}
    if ledger.is_file():
        with ledger.open() as handle:
            states = {row['trial_id']: row for row in csv.DictReader(handle, delimiter='\t')}
    rows = []
    for trial in plan['trials']:
        row = states.get(trial['trial_id'], {})
        status = row.get('status', 'pending')
        if status == 'completed' and not completed_artifacts(
                row.get('run_dir'), trial['config']):
            status = 'incomplete'
        rows.append(dict(trial=trial['suffix'] or Path(trial['source']).stem,
                         seed=trial['config'].get('seed', 42), status=status,
                         run_dir=row.get('run_dir', ''), log=row.get('log', ''),
                         error=row.get('error', '')))
    return dict(batch=str(batch), name=plan['name'], active=batch_active(batch),
                completed=sum(row['status'] == 'completed' for row in rows),
                total=len(rows), trials=rows)
