"""Experiment definitions: built-in suites, sweep presets, spec files, and their expansion into trials.

A trial is a resolved training config plus where it came from; planning, identity and execution live elsewhere.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import re

from cdrseqgen.artifacts import sha256_file
from cdrseqgen.methods.registry import resolve
from cdrseqgen.visualization.sweep_table import RANKERS

from .core import fingerprint, flag


def _configs(loops, models, contexts, directory="mlm/heavy"):
    return [f"configs/{directory}/{loop}_{model}{context}.json"
            for model in models for loop in loops for context in contexts]


SUITES = {
    "mlm_heavy": _configs(("h1", "h2", "h3"), ("scratch", "esm2"), ("", "_lc")),
    "mlm_light": _configs(("l1", "l2", "l3"), ("scratch", "esm2"), ("", "_hc"), "mlm/light"),
    "mlm_antigen": [f"configs/mlm/antigen/h{i}_{model}{ctx}.json" for model in ("scratch", "esm2")
                    for ctx in ("_ag_lconly", "_ag") for i in (1, 2, 3)],
    "mlm_igbert": _configs(("h1", "h2", "h3"), ("igbert",), ("", "_lc")),
    "mlm_protbert": _configs(("h1", "h2", "h3"), ("protbert",), ("", "_lc")),
    "ar_heavy": _configs(("h1", "h2", "h3"), ("ar_scratch",), ("", "_lc"), "ar"),
    "s2s_heavy": _configs(("h3",), ("s2s_scratch",), ("", "_lc"), "s2s"),
    "s2s_esm2": _configs(("h3",), ("s2s_esm2",), ("", "_lc"), "s2s"),
    "s2s_igbert": _configs(("h3",), ("s2s_igbert",), ("", "_lc"), "s2s"),
    "s2s_protbert": _configs(("h3",), ("s2s_protbert",), ("", "_lc"), "s2s"),
    "compare": [f"configs/{directory}/h3_{model}{ctx}.json" for ctx in ("", "_lc")
                for directory, model in (("mlm/heavy", "scratch"), ("ar", "ar_scratch"),
                                         ("s2s", "s2s_scratch"))],
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
            chains = {"heavy": {}, "light": {}}
            for row in entries:
                split = row.get("split")
                if not isinstance(split, str):
                    raise ValueError(f"{path}: every entry must declare a named split")
                splits[split] = splits.get(split, 0) + 1
                if row.get("antigen"):
                    antigen[split] = antigen.get(split, 0) + 1
                # Counted per chain because an unpaired corpus carries only one of them, and the
                # partner chain a run needs depends on which loop it targets.
                for chain, seen in chains.items():
                    if row.get(chain):
                        seen[split] = seen.get(split, 0) + 1
            del entries
            datasets[path] = dict(sha256=sha256_file(path), splits=splits, antigen=antigen,
                                  chains=chains)
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
        # The antigen gate's twin. Without it an unpaired corpus turns include_partner_chain into
        # a silent no-op: the run trains with no context, is still tagged _LC in its directory
        # name, and lands in the partner-context column of every table -- invalidating the
        # comparison rather than failing. `chains` is absent from plans frozen before this check.
        target = str(config.get("target_region", "h3")).lower()
        partner = "heavy" if target.startswith("l") else "light"
        if config.get("include_partner_chain") and any(
                not info.get("chains", {}).get(partner, {}).get(split)
                for split in ("train", evaluation)):
            raise ValueError(f"{trial['source']}: partner-chain context requested for target "
                             f"{target} but {path} has no {partner} chain in train/{evaluation}. "
                             "An unpaired corpus can only run without partner context.")
    return datasets
