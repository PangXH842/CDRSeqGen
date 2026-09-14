"""Plan and execute experiments: suites and sweeps share everything after trial expansion.

The shell scripts only locate the repository and interpreter. A plan freezes merged input configs,
dataset hashes, and source/runtime identity; train.py still owns argument defaults and training.
One executor handles GPU sharding, logs, completion, resume, and reporting for either plan.
No model libraries are imported here, so orchestration tests need no GPU or downloaded weights.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

from cdrseqgen.artifacts import (
    EPOCH_METRICS_FILENAME, GENERATION_EVAL_FILENAME, RUN_CONFIG_FILENAME,
    atomic_write_json, sha256_file,
)
from cdrseqgen.methods.registry import resolve
from cdrseqgen.visualization.sweep_tables import KEY, RANKERS


def _configs(loops, models, contexts):
    return [f"configs/{loop}_{model}{context}.json"
            for model in models for loop in loops for context in contexts]


SUITES = {
    "mlm_heavy": _configs(("h1", "h2", "h3"), ("scratch", "esm2"), ("", "_lc")),
    "mlm_light": _configs(("l1", "l2", "l3"), ("scratch", "esm2"), ("", "_hc")),
    "mlm_antigen": [f"configs/h{i}_{model}{ctx}.json" for model in ("scratch", "esm2")
                    for ctx in ("_ag_lconly", "_ag") for i in (1, 2, 3)],
    "mlm_igbert": _configs(("h1", "h2", "h3"), ("igbert",), ("", "_lc")),
    "mlm_protbert": _configs(("h1", "h2", "h3"), ("protbert",), ("", "_lc")),
    "ar_heavy": _configs(("h1", "h2", "h3"), ("ar_scratch",), ("", "_lc")),
    "s2s_heavy": _configs(("h3",), ("s2s_scratch",), ("", "_lc")),
    "s2s_esm2": _configs(("h3",), ("s2s_esm2",), ("", "_lc")),
    "s2s_igbert": _configs(("h3",), ("s2s_igbert",), ("", "_lc")),
    "s2s_protbert": _configs(("h3",), ("s2s_protbert",), ("", "_lc")),
    "compare": [f"configs/h3_{model}{ctx}.json" for ctx in ("", "_lc")
                for model in ("scratch", "ar_scratch", "s2s_scratch")],
}
SUITES["s2s_plm"] = SUITES["s2s_esm2"] + SUITES["s2s_igbert"] + SUITES["s2s_protbert"]
SUITES["mlm_plm"] = SUITES["mlm_igbert"] + SUITES["mlm_protbert"]
SUITES["all"] = SUITES["mlm_heavy"] + SUITES["mlm_antigen"] + SUITES["mlm_light"]
RENAMED = dict(oas="mlm_heavy", lcdr="mlm_light", ar="ar_heavy", antigen="mlm_antigen",
               igbert="mlm_igbert", protbert="mlm_protbert", plm="mlm_plm")

# Adding a preset changes a grid, not another execution loop.
PRESETS = {
    "h3": dict(LRS="1e-4 3e-4 1e-3 3e-3", WARMUPS="0.0 0.06",
               WEIGHT_DECAYS="0.0 0.01", SEEDS="42 43", EPOCH_LIST="30"),
    "lr": dict(LRS="1e-4 3e-4 1e-3", WARMUPS="0.06", WEIGHT_DECAYS="0.01",
               SEEDS="42 43", EPOCH_LIST="20"),
    "pilot": dict(LRS="1e-4 3e-4 1e-3", WARMUPS="0.0", WEIGHT_DECAYS="0.0",
                  SEEDS="42", EPOCH_LIST="10 20"),
    "arch": dict(LRS="3e-4", WARMUPS="0.06", WEIGHT_DECAYS="0.01", SEEDS="42",
                 EPOCH_LIST="30", HIDDEN_SIZES="128 192 256", NUM_LAYERS="4 6"),
    "loss": dict(LRS="1e-3", WARMUPS="0.06", WEIGHT_DECAYS="0.01", SEEDS="42 43 44",
                 EPOCH_LIST="30", LOSS_TYPES="cross_entropy weighted_cross_entropy",
                 CORE_WEIGHTS="1.0 3.0", ANCHOR_WEIGHTS="1.0 0.0", GEN_EVAL="true",
                 RANK_METRIC="judge_ppl", SELECT_METRIC="accuracy",
                 GEN_EXAMPLES="100", GEN_SAMPLES="10"),
    "quick": dict(LRS="1e-4 1e-3", WARMUPS="0.0", WEIGHT_DECAYS="0.0",
                  SEEDS="42", EPOCH_LIST="2"),
}
DIMENSIONS = dict(lr="LRS", warmup="WARMUPS", weight_decay="WEIGHT_DECAYS",
                  epochs="EPOCH_LIST", hidden="HIDDEN_SIZES", layers="NUM_LAYERS",
                  loss="LOSS_TYPES", core_w="CORE_WEIGHTS", anchor_w="ANCHOR_WEIGHTS",
                  seed="SEEDS")
RUN_FIELDS = [*KEY, "seed", "scheduler", "status", "run_dir", "trial_id", "setting_id",
              "config", "gpu", "log", "error"]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def flag(value):
    if str(value).lower() in {"1", "true", "yes"}:
        return True
    if str(value).lower() in {"0", "false", "no", ""}:
        return False
    raise ValueError(f"Expected true/false, got {value!r}")


def load_config(path, env, overrides=None):
    with Path(path).open() as handle:
        config = json.load(handle)
    config.update(overrides or {})
    config["method"] = resolve(config.get("method", "mlm")).name
    if env.get("ENCODER_INIT") or env.get("FREEZE_ENCODER"):
        if config["method"] != "s2s" or config.get("model_type") == "scratch":
            raise ValueError("ENCODER_INIT/FREEZE_ENCODER require an S2S PLM config")
        if env.get("ENCODER_INIT"):
            if env["ENCODER_INIT"] not in {"pretrained", "random"}:
                raise ValueError("ENCODER_INIT must be pretrained or random")
            config["encoder_init"] = env["ENCODER_INIT"]
        if env.get("FREEZE_ENCODER"):
            config["freeze_encoder"] = flag(env["FREEZE_ENCODER"])
    if env.get("DATA_PATH"):
        config["data_path"] = env["DATA_PATH"]
    if not config.get("data_path"):
        raise ValueError(f"{path}: specify data_path in the config or DATA_PATH")
    return config


def suite_trials(names, env):
    paths = []
    for name in names:
        if name not in SUITES:
            hint = f" (renamed to {RENAMED[name]})" if name in RENAMED else ""
            raise ValueError(f"Unknown suite {name!r}{hint}. Choose: {', '.join(SUITES)}")
        paths.extend(SUITES[name])
    trials = []
    for path in dict.fromkeys(paths):
        config = load_config(path, env)
        if env.get("SEED"):
            config.update(seed=int(env["SEED"]), data_seed=int(env["SEED"]))
        if "compare" in names and path in SUITES["compare"]:
            # Apply the generation protocol to every arm; do not rely on coinciding defaults.
            config.update(max_len=384, gen_top_p=0.9, gen_top_k=0, gen_temperature=1.0,
                          gen_num_examples=150, gen_num_samples=20, gen_seed=42,
                          gen_stratify=False, gen_judge="Exscientia/IgBert",
                          run_generative_eval=True)
            if config["method"] == "mlm":
                config["gen_length_mode"] = "inferred"
        trials.append(dict(source=path, config=config, grid={}, suffix=""))
    return trials


def spec_trials(path, env):
    """Expand a declarative recipe into the ordinary suite execution format.

    Paths use the repository root, like suite configs. Merge base < shared < arm;
    DATA_PATH may relocate the corpus, but arm controls must be explicit in the spec.
    """
    for key in ("SEED", "ENCODER_INIT", "FREEZE_ENCODER"):
        if env.get(key):
            raise ValueError(f"Spec mode: set {key} in the spec overrides instead of the environment")
    with Path(path).open() as handle:
        spec = json.load(handle)

    def fields(value, allowed, where):
        if not isinstance(value, dict) or value.keys() - allowed:
            raise ValueError(f"{where}: expected an object with fields {sorted(allowed)}")

    def name(value, where):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            raise ValueError(f"{where}: use a nonempty name containing letters, digits, '_' or '-'")
        return value

    fields(spec, {"name", "base_config", "overrides", "trials"}, "Experiment spec")
    experiment = name(spec.get("name"), "Experiment name")
    shared = spec.get("overrides", {})
    if not isinstance(shared, dict):
        raise ValueError("Spec overrides must be an object")
    arms = spec.get("trials")
    if not isinstance(arms, list) or not arms:
        raise ValueError("Spec trials must be a nonempty list")
    trials, labels, identities = [], set(), set()
    for arm in arms:
        fields(arm, {"name", "base_config", "overrides"}, "Trial")
        label = name(arm.get("name"), "Trial name")
        if label in labels:
            raise ValueError(f"Duplicate trial name: {label}")
        labels.add(label)
        base = arm.get("base_config", spec.get("base_config"))
        if not isinstance(base, str) or not base:
            raise ValueError(f"{label}: specify base_config on the spec or trial")
        overrides = arm.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"{label}: overrides must be an object")
        config = load_config(base, env, {**shared, **overrides})
        identity = fingerprint(config)
        if identity in identities:
            raise ValueError(f"{label}: duplicate resolved config; use distinct settings or seeds")
        identities.add(identity)
        trials.append(dict(source=base, config=config, grid={}, suffix=label))
    return trials, experiment


def sweep_trials(preset, base, env):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose: {', '.join(PRESETS)}")
    settings = dict(PRESETS[preset])
    settings.update({k: v for k, v in env.items() if v})
    config = load_config(base, env)
    rank = settings.get("RANK_METRIC", "core")
    select = settings.get("SELECT_METRIC", "core_accuracy")
    generate = flag(settings.get("GEN_EVAL", "false"))
    if rank not in RANKERS:
        raise ValueError(f"RANK_METRIC must be one of {', '.join(RANKERS)}")
    if select not in {"core_accuracy", "accuracy"}:
        raise ValueError("SELECT_METRIC must be core_accuracy or accuracy")
    if RANKERS[rank][0] != "epoch" and not generate:
        raise ValueError(f"RANK_METRIC={rank} needs GEN_EVAL=true")
    grids = [settings.get(name, "__keep__").split() for name in DIMENSIONS.values()]
    trials, seen = [], set()
    for values in itertools.product(*grids):
        grid = dict(zip(DIMENSIONS, values))
        c = dict(config)
        c.update(eval_split="validation", learning_rate=float(grid["lr"]),
                 warmup_ratio=float(grid["warmup"]), warmup_steps=0,
                 weight_decay=float(grid["weight_decay"]), num_train_epochs=float(grid["epochs"]),
                 lr_scheduler_type=settings.get("SCHEDULER", "linear"),
                 early_stopping_patience=int(settings.get("PATIENCE", "5")),
                 early_stopping_threshold=float(settings.get("STOP_THRESHOLD", "0.0001")),
                 metric_for_best_model=select, greater_is_better=True,
                 load_best_model_at_end=True, save_total_limit=1,
                 seed=int(grid["seed"]), data_seed=int(grid["seed"]),
                 run_generative_eval=generate)
        if grid["hidden"] != "__keep__" or grid["layers"] != "__keep__":
            if c.get("model_type") != "scratch":
                raise ValueError("Architecture sweeps currently require a scratch backbone")
        if grid["hidden"] != "__keep__":
            c.update(hidden_size=int(grid["hidden"]), intermediate_size=None)
        if grid["layers"] != "__keep__":
            c["num_layers"] = int(grid["layers"])
            if c["method"] == "s2s":
                c.update(num_encoder_layers=int(grid["layers"]),
                         num_decoder_layers=int(grid["layers"]))
        if grid["loss"] != "__keep__":
            if c["method"] != "mlm":
                raise ValueError("Loss sweeps require MLM; AR/S2S do not implement weighted loss")
            loss = grid["loss"]
            if loss not in {"cross_entropy", "weighted_cross_entropy"}:
                raise ValueError(f"Unsupported loss: {loss}")
            cw, aw = ((1., 1.) if loss == "cross_entropy" else
                      (float(grid["core_w"]), float(grid["anchor_w"])))
            if min(cw, aw) < 0 or cw + aw <= 0:
                raise ValueError("Loss weights must be nonnegative with a positive total")
            # Only the ratio changes the objective; equal weights are plain CE.
            if cw == aw:
                loss, cw, aw = "cross_entropy", 1., 1.
            else:
                cw, aw = (cw / aw, 1.) if aw else (1., 0.)
            c.update(loss_type=loss, core_loss_weight=cw, anchor_loss_weight=aw)
            grid.update(loss=loss, core_w=str(cw), anchor_w=str(aw))
        for name, key in (("GEN_EXAMPLES", "gen_num_examples"), ("GEN_SAMPLES", "gen_num_samples")):
            if settings.get(name, "__keep__") != "__keep__":
                c[key] = int(settings[name])
        identity = fingerprint(c)
        if identity in seen:
            continue
        seen.add(identity)
        suffix = f"lr{grid['lr']}_wu{grid['warmup']}_wd{grid['weight_decay']}_ep{grid['epochs']}_s{grid['seed']}"
        if grid["hidden"] != "__keep__":
            suffix += f"_h{grid['hidden']}"
        if grid["layers"] != "__keep__":
            suffix += f"_l{grid['layers']}"
        if grid["loss"] != "__keep__":
            suffix += f"_{grid['loss']}_{grid['core_w']}-{grid['anchor_w']}"
        trials.append(dict(source=base, config=c, grid=grid, suffix=suffix))
    return trials, rank


def validate_datasets(trials, mode):
    """Inspect each effective corpus once, using the same gate for both entry points."""
    datasets = {}
    for trial in trials:
        config = trial["config"]
        path = str(Path(config["data_path"]).expanduser().resolve())
        config["data_path"] = path
        if path not in datasets:
            with Path(path).open() as handle:
                entries = json.load(handle)
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"Empty or invalid corpus: {path}")
            splits, antigen = {}, {}
            for row in entries:
                split = row.get("split")
                if not isinstance(split, str):
                    raise ValueError(f"{path}: every entry must declare a named split")
                splits[split] = splits.get(split, 0) + 1
                if row.get("antigen"):
                    antigen[split] = antigen.get(split, 0) + 1
            del entries
            datasets[path] = dict(sha256=sha256_file(path), splits=splits, antigen=antigen)
        info = datasets[path]
        evaluation = config.get("eval_split")
        needed = {"train", "validation", "test"} if mode == "sweep" else {"train", evaluation}
        if not evaluation or needed - info["splits"].keys():
            raise ValueError(f"{trial['source']}: {path} must contain {sorted(needed, key=str)} "
                             "and declare an explicit eval_split")
        if config.get("include_antigen") and any(
                not info["antigen"].get(split) for split in ("train", evaluation)):
            raise ValueError(f"{trial['source']}: antigen context requested but missing from "
                             f"train/{evaluation} in {path}")
    return datasets


def source_identity():
    """Changes to executable sources or installed training packages invalidate resume."""
    package = Path(__file__).resolve().parent
    sources = {str(p.relative_to(package)): sha256_file(p) for p in sorted(package.rglob('*.py'))}
    sources["train.py"] = sha256_file("train.py")
    packages = {}
    for name in ("torch", "transformers", "accelerate", "safetensors"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(source_sha256=fingerprint(sources), python=sys.version, packages=packages)


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
        base = rest[2] if len(rest) > 2 else "configs/h3_scratch.json"
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
        trial["trial_id"] = fingerprint(dict(config=c, dataset=dataset, runtime=runtime))
        setting = {k: v for k, v in c.items() if k not in {"seed", "data_seed"}}
        trial["setting_id"] = fingerprint(dict(config=setting, dataset=dataset, runtime=runtime))
    # Devices/output are execution choices, not part of scientific identity.
    plan = dict(schema_version=1, mode=mode, name=name, rank_metric=rank,
                runtime=runtime, datasets=datasets, trials=trials)
    return plan, devices


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
            if value is None or key == "output_dir":
                continue
            actual = saved.get(key)
            # Transformers 5 converts the deprecated warmup_ratio into warmup_steps.
            # Older versions leave steps=0; both represent the same requested ratio.
            if key == "warmup_steps" and config.get("warmup_ratio") is not None:
                if actual in (value, config["warmup_ratio"]):
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
            with (root / GENERATION_EVAL_FILENAME).open() as handle:
                payload = json.load(handle)
            if not payload.get("records") or not payload.get("summary"):
                return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def report(plan, output):
    jobs = ([('batch_tables', '--batch', str(output), '--output'),
             ('batch_plots', '--batch', str(output))] if plan["mode"] == "suite" else
            [('sweep_tables', '--sweep', str(output), '--rank_metric', plan["rank_metric"]),
             ('sweep_plots', '--sweep', str(output))])
    for module, *args in jobs:
        code = subprocess.call([sys.executable, '-m', f'cdrseqgen.visualization.{module}', *args])
        if code:
            print(f"Reporting failed ({module}); training artifacts are in {output}", flush=True)


def execute(plan, devices, output, env):
    """One lifecycle for all trials, with one sequential shard per device."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another runner is using {output}") from None
        plan_path = output / 'plan.json'
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError(f"Experiment differs from {plan_path}; use a new EXPERIMENT_DIR. "
                                 "Resume requires unchanged configs, data, source, and protocol.")
        elif any(p.name != '.lock' for p in output.iterdir()):
            raise ValueError(f"{output} has no matching plan.json; use a fresh EXPERIMENT_DIR. "
                             "Legacy results remain readable but cannot be auto-resumed.")
        else:
            atomic_write_json(plan_path, plan)
        _execute_locked(plan, devices, output, env)
        with (output / 'runs.tsv').open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        failed = sum(r['status'] != 'completed' for r in rows)
        report(plan, output)
        if plan['mode'] == 'suite' and not failed:
            try:
                link = Path('results/runs/latest')
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    link.unlink()
                if not link.exists():
                    link.symlink_to(os.path.relpath(output, link.parent.resolve()))
            except OSError as exc:
                print(f"Could not update latest: {exc}", file=sys.stderr)
        print(f"{'FAILED' if failed else 'Done'}: {len(rows)-failed}/{len(rows)} completed; {output}")
        return 1 if failed else 0


def _execute_locked(plan, devices, output, env):
    configs_dir, logs_dir = output / '.configs', output / 'logs'
    configs_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    map_path = output / 'runs.tsv'
    states = {}
    if map_path.exists():
        with map_path.open() as handle:
            states = {r['trial_id']: r for r in csv.DictReader(handle, delimiter='\t')}
    mutex = threading.Lock()
    stop = threading.Event()
    active = set()

    def persist():
        pending = map_path.with_suffix('.tsv.tmp')
        with pending.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS, delimiter='\t')
            writer.writeheader()
            writer.writerows(states[t['trial_id']] for t in plan['trials'] if t['trial_id'] in states)
        pending.replace(map_path)

    def shard(gpu, trials):
        for trial in trials:
            if stop.is_set():
                break
            identity, config = trial['trial_id'], trial['config']
            with mutex:
                old = states.get(identity, {})
                if old.get('status') == 'completed' and completed_artifacts(old.get('run_dir'), config):
                    print(f"SKIP completed: {trial['source']} {trial['suffix']}", flush=True)
                    continue
            config_path = configs_dir / f'{identity}.json'
            atomic_write_json(config_path, config)
            log = logs_dir / f"{Path(trial['source']).stem}_{identity[:12]}.log"
            child_env = dict(env, CUDA_VISIBLE_DEVICES=gpu, CDRSEQGEN_RESULTS_ROOT=str(output),
                             CDRSEQGEN_SKIP_LATEST='1', CDRSEQGEN_RUN_SUFFIX=trial['suffix'])
            row = {k: trial['grid'].get(k, '__keep__') for k in KEY}
            row.update(seed=config.get('seed', 42), scheduler=config.get('lr_scheduler_type', 'linear'),
                       status='running', run_dir='', trial_id=identity, setting_id=trial['setting_id'],
                       config=trial['source'], gpu=gpu, log=str(log), error='')
            with mutex:
                states[identity] = dict(row)
                persist()
            print(f"RUN gpu {gpu}: {trial['source']} {trial['suffix']} | {log}", flush=True)
            try:
                with log.open('a', buffering=1) as handle:
                    handle.write(f"\n--- attempt {datetime.now().isoformat()} ---\n")
                    with subprocess.Popen([sys.executable, 'train.py', str(config_path)], env=child_env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, bufsize=1) as process:
                        with mutex:
                            active.add(process)
                            if stop.is_set():
                                process.terminate()
                        for line in process.stdout:
                            handle.write(line)
                            if len(devices) == 1:
                                print(line, end='', flush=True)
                            marker = 'Workspace initialized at: '
                            if marker in line:
                                row['run_dir'] = line.split(marker, 1)[1].strip()
                        code = process.wait()
                        with mutex:
                            active.discard(process)
                row['status'] = ('completed' if code == 0 and completed_artifacts(row['run_dir'], config)
                                 else 'failed' if code else 'incomplete')
                if row['status'] != 'completed':
                    row['error'] = f'exit {code}' if code else 'Missing or incompatible run artifacts'
            except OSError as exc:
                row.update(status='failed', error=str(exc))
            with mutex:
                states[identity] = row
                persist()
            print(f"{row['status'].upper()}: {trial['source']} {trial['suffix']}", flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(shard, gpu, plan['trials'][i::len(devices)])
                   for i, gpu in enumerate(devices)]
        try:
            for future in futures:
                future.result()
        except BaseException:
            stop.set()
            with mutex:
                for process in active:
                    if process.poll() is None:
                        process.terminate()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Suites: {', '.join(SUITES)}\nPresets: {', '.join(PRESETS)}\n\n"
               "run.sh <gpus> [suite ...]\nrun.sh <gpus> sweep [preset] [base_config]\nrun.sh <gpus> spec <experiment.json>\n\n"
               "Shared overrides: PYTHON, DATA_PATH, DRY_RUN=1, EXPERIMENT_DIR (exact resume).\n"
               "S2S PLM controls (suites/sweeps): ENCODER_INIT=pretrained|random, FREEZE_ENCODER=true|false.\n"
               "Spec: base_config + shared/per-trial overrides; put seed/encoder controls in the file.\n"
               "Suite: SEED. Sweep: LRS, WARMUPS, WEIGHT_DECAYS, SEEDS, EPOCH_LIST, HIDDEN_SIZES,\n"
               "NUM_LAYERS, LOSS_TYPES, CORE_WEIGHTS, ANCHOR_WEIGHTS, GEN_EVAL, RANK_METRIC,\n"
               "SELECT_METRIC, GEN_EXAMPLES, GEN_SAMPLES, PATIENCE, STOP_THRESHOLD, SCHEDULER.\n"
               "SWEEP_DIR remains an alias for EXPERIMENT_DIR in sweep mode.")
    parser.add_argument('arguments', nargs='*', metavar='ARG')
    args = parser.parse_args(argv)
    try:
        plan, devices = build_plan(args.arguments, os.environ)
        print(f"{plan['mode']} {plan['name']}: {len(plan['trials'])} trials on GPU(s) {','.join(devices)}")
        if flag(os.environ.get('DRY_RUN', 'false')):
            print(json.dumps(plan, indent=2))
            return 0
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        tag = ('sweep_' if plan['mode'] == 'sweep' else '') + plan['name']
        output = os.environ.get('EXPERIMENT_DIR')
        legacy = os.environ.get('SWEEP_DIR') if plan['mode'] == 'sweep' else None
        if output and legacy and Path(output).resolve() != Path(legacy).resolve():
            raise ValueError('EXPERIMENT_DIR and SWEEP_DIR disagree')
        return execute(plan, devices, output or legacy or f'results/runs/{stamp}_{tag}', os.environ)
    except (OSError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
