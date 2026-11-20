"""Turn trials into a frozen plan, and re-check a stored plan before it is resumed."""
from __future__ import annotations

import copy
import re

from .core import fingerprint, read_plan, source_identity
from .spec import spec_trials, suite_trials, sweep_trials, validate_datasets


def build_plan(arguments, env):
    gpu_arg = arguments[0] if arguments else "0"
    devices = re.split(r"[,\s]+", gpu_arg.strip())
    if not devices or any(not re.fullmatch(r"[0-9]+", d) for d in devices):
        raise ValueError("GPU ids must be nonnegative integers, e.g. 0 or 0,1,2")
    if len(set(devices)) != len(devices):
        raise ValueError("Duplicate GPU ids would oversubscribe the same device")
    rest = arguments[1:] or ["all"]
    if rest[0] == "spec":
        if len(rest) != 2:
            raise ValueError("Usage: run.sh <gpus> spec <experiment.json>")
        trials, name = spec_trials(rest[1], env)
        mode, rank = "suite", None  # Reuse suite validation, execution, reports and resume.
    elif rest[0] == "sweep":
        if len(rest) > 3:
            raise ValueError("Usage: run.sh <gpus> sweep [preset] [base_config]")
        mode, name = "sweep", rest[1] if len(rest) > 1 else "h3"
        base = rest[2] if len(rest) > 2 else "configs/mlm/heavy/h3_scratch.json"
        trials, rank = sweep_trials(name, base, env)
    else:
        mode, name = "suite", "_".join(dict.fromkeys(rest))
        trials, rank = suite_trials(rest, env), None
        if env.get("SEED"):
            name += f"_s{int(env['SEED'])}"
    if not trials:
        raise ValueError("The experiment contains no trials")
    datasets = validate_datasets(trials, mode)
    runtime = source_identity()
    for trial in trials:
        c = trial["config"]
        dataset = datasets[c["data_path"]]["sha256"]
        # The corpus hash pins content; its resolved location remains provenance, not identity.
        identity = {k: v for k, v in c.items() if k != "data_path"}
        trial["trial_id"] = fingerprint(dict(config=identity, dataset=dataset, runtime=runtime))
        setting = {k: v for k, v in identity.items() if k not in {"seed", "data_seed"}}
        trial["setting_id"] = fingerprint(dict(config=setting, dataset=dataset, runtime=runtime))
    # Devices/output are execution choices, not part of scientific identity.
    plan = dict(schema_version=1, mode=mode, name=name, rank_metric=rank,
                runtime=runtime, datasets=datasets, trials=trials)
    return plan, devices


def frozen_resume_plan(batch):
    """Validate stored settings against current code/data, without rereading recipes.

    Recompute trial identities as well as corpus hashes so a modified frozen
    config cannot be executed under an old trial ID or skip an incompatible run.
    """
    plan = read_plan(batch)
    if source_identity() != plan['runtime']:
        raise ValueError('Source/runtime differs from the frozen plan; use the original '
                         'checkout/environment or start a new batch')
    checked = copy.deepcopy(plan)
    actual_data = validate_datasets(checked['trials'], checked['mode'])
    if actual_data != plan['datasets'] or checked != plan:
        raise ValueError('Dataset differs from the frozen plan; restore the original corpus')
    seen = set()
    for trial in plan['trials']:
        config = {k: v for k, v in trial['config'].items() if k != 'data_path'}
        dataset = actual_data[trial['config']['data_path']]['sha256']
        identity = fingerprint(dict(config=config, dataset=dataset, runtime=plan['runtime']))
        setting = {k: v for k, v in config.items() if k not in {'seed', 'data_seed'}}
        setting_id = fingerprint(dict(config=setting, dataset=dataset, runtime=plan['runtime']))
        if (identity != trial['trial_id'] or setting_id != trial['setting_id']
                or identity in seen):
            raise ValueError('Frozen trial identity differs from its config; restore plan.json')
        seen.add(identity)
    return plan


def gpu_devices(value):
    """Parse an explicit device list with the same rules as the shared planner."""
    devices = re.split(r'[,\s]+', value.strip())
    if (any(not re.fullmatch(r'[0-9]+', device) for device in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError('Use distinct nonnegative GPU IDs, e.g. --gpus 0,1')
    return devices
