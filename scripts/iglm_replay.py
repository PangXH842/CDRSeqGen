#!/usr/bin/env python3
"""Replay a stored external-IgLM generation draw for draw with the in-process `iglm` backbone.

The benchmark's IgLM row was produced by the upstream package in its own environment; its output keeps
every attempt and each example's seed. This script rebuilds each prompt from the same frozen manifest,
seeds the random generators exactly as that adapter did, samples with `ar.sampling.sample_loops` and
compares the attempts one by one. Agreement supports parity for the recorded inputs and runtime;
it does not establish equivalence for every possible input. Valid loop strings are compared exactly.
Two failed attempts match as outcomes, since historical failure records may omit raw generated tokens.

Seeds map to the same draws only on the same kind of device (CPU and CUDA random streams differ), so a
stored GPU batch must be replayed on a GPU. Investigate mismatches in prompts, seeds, sampling settings,
runtime and device. Floating-point differences can change a sampled token and subsequent draws, but
the pattern of divergence alone does not identify its cause.

  python scripts/iglm_replay.py --batch results/benchmark/oas4_h3_length_free_20261007 --device cuda
  python scripts/iglm_replay.py --generations OUT.json --manifest M.json --device cpu     # a CPU reference

Reads only; writes the report file named by --output. Select the GPU with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RELEASES = Path.home() / "miniconda3/envs/iglm/lib/python3.11/site-packages/iglm/trained_models"


def stored_attempts(record):
    """{attempt: ('ok', loop) | ('fail', code)} from one stored adapter record."""
    outcome = {}
    for draw in record.get("raw_generations", []):
        outcome[draw["attempt"]] = ("ok", draw["loop"]) if draw.get("valid_generation", True) else ("fail", "invalid")
    for failure in record.get("failures", []):
        if "attempt" in failure:
            outcome.setdefault(failure["attempt"], ("fail", failure.get("code", "failed")))
    return outcome


def ours_attempts(draws):
    """The same shape from `sample_loops` results; an empty loop is a failure, as in the adapter."""
    return {number: ("ok", draw["loop"]) if draw["complete"] and draw["loop"] and draw.get("valid", True)
            else ("fail", draw.get("failure") or "invalid")
            for number, draw in enumerate(draws, start=1)}


def compare(stored, ours, attempts):
    """First differing attempt and matching outcomes; missing attempts never count as agreement."""
    same, first = 0, None
    for number in range(1, attempts + 1):
        a, b = stored.get(number), ours.get(number)
        if a is not None and b is not None and (a == b or (a[0] == "fail" and b[0] == "fail")):
            same += 1
        elif first is None:
            first = number
    return first, same


def seed_everything(seed):
    """The adapter's `iglm.utils.seed.set_seeds`: python, numpy and torch (and, through transformers, CUDA)."""
    import numpy as np
    import torch
    import transformers
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(generations, manifest, model_dir, device, max_len, limit=None):
    import torch
    import transformers
    from benchmark._common import stable_example_seed
    from cdrseqgen.datasets.dataset_ar import build_causal_prompt
    from cdrseqgen.methods.ar.backbones import resolve
    from cdrseqgen.methods.ar.sampling import sample_loops

    sampling = generations["sampling"]
    if sampling.get("length_policy") != "natural":
        raise ValueError("Only natural-length (length-free) batches can be replayed")
    attempts = sampling["num_samples"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("Replay needs a positive num_samples attempt budget")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("Replay limit must be a positive integer")
    if not generations.get("records"):
        raise ValueError("Replay needs at least one stored framework record")
    by_id = {record["example_id"]: record for record in manifest["records"]}
    region = manifest["task"]["target_region"]

    stored_hash = generations["model"].get("checkpoint_sha256")
    weights = Path(model_dir) / "model.safetensors"
    if not weights.is_file():
        weights = Path(model_dir) / "pytorch_model.bin"
    weights_match = bool(stored_hash) and weights.is_file() and sha256(weights) == stored_hash

    source = f"iglm:{model_dir}"
    backbone = resolve(source)
    tokenizer = backbone.build_tokenizer(source, checkpoint_path=str(model_dir))
    model = backbone.load_pretrained_model(str(model_dir)).to(device).eval()

    rows, started = [], time.perf_counter()
    for stored in generations["records"][:limit]:
        record = by_id[stored["example_id"]]
        seed = stable_example_seed(sampling["base_seed"], record["example_id"])
        if "seed" in stored and stored["seed"] != seed:
            raise ValueError(f"{record['example_id']}: stored seed {stored['seed']} is not the derived seed {seed}")
        framework = dict(pt1=record["pt1"], pt2=record["pt2"], target_length=0, max_len=max_len,
                         partner_chain=None, antigen=None, target_region=region,
                         chain_token=record["chain_token"], species_token=record["species_token"])
        prompt, meta = build_causal_prompt(tokenizer, framework)
        if meta["pt1_len"] + meta["pt2_len"] != len(record["pt1"]) + len(record["pt2"]):
            raise ValueError(f"{record['example_id']}: flanks were truncated at max_len={max_len}; raise it")
        seed_everything(seed)
        draws = sample_loops(model, tokenizer, prompt, device=device, num_samples=attempts,
                             temperature=sampling["temperature"], top_k=0, top_p=sampling["top_p"],
                             max_new_tokens=sampling["max_new_tokens"], generator=None)
        first, same = compare(stored_attempts(stored), ours_attempts(draws), attempts)
        rows.append(dict(example_id=record["example_id"], attempts_identical=same, first_difference=first))
    identical = [row for row in rows if row["first_difference"] is None]
    return dict(
        examples=len(rows), examples_identical=len(identical), attempts=len(rows) * attempts,
        attempts_identical=sum(row["attempts_identical"] for row in rows),
        different=[row for row in rows if row["first_difference"] is not None],
        weights_match=weights_match, stored_weights_sha256=stored_hash,
        seconds=round(time.perf_counter() - started, 1),
        environment=dict(device=str(device), torch=torch.__version__, transformers=transformers.__version__,
                         gpu=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None),
        stored_environment={key: generations["model"].get(key) for key in
                            ("device", "torch_version", "transformers_version", "package_version")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=Path, help="benchmark batch directory (plan.json, generations/)")
    parser.add_argument("--label", default="iglm", help="generations/<label>.json inside --batch")
    parser.add_argument("--generations", type=Path, help="adapter output to replay (instead of --batch)")
    parser.add_argument("--manifest", type=Path, help="its manifest (default: the batch plan's)")
    parser.add_argument("--model-dir", type=Path, help="released IgLM directory (default: the stored model name)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-len", type=int, default=256, help="prompt budget; must not truncate any flank")
    parser.add_argument("--limit", type=int, default=None, help="replay only the first N examples")
    parser.add_argument("--min-example-match", type=float, default=0.985,
                        help="fraction of examples that must reproduce every attempt (default 0.985)")
    parser.add_argument("--output", type=Path, help="write the report here")
    args = parser.parse_args(argv)
    if not 0 < args.min_example_match <= 1:
        parser.error("--min-example-match must be in (0, 1]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.batch:
        plan = json.loads((args.batch / "plan.json").read_text())
        args.generations = args.generations or args.batch / "generations" / f"{args.label}.json"
        args.manifest = args.manifest or ROOT / plan["manifest"]
    if not (args.generations and args.manifest):
        parser.error("give --batch, or both --generations and --manifest")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda but no CUDA device is visible; use CUDA_VISIBLE_DEVICES or --device cpu")
    generations = json.loads(args.generations.read_text())
    manifest = json.loads(args.manifest.read_text())
    model_dir = args.model_dir or RELEASES / generations["model"]["name"]
    report = replay(generations, manifest, model_dir, torch.device(args.device), args.max_len, args.limit)
    ratio = report["examples_identical"] / max(report["examples"], 1)
    report["verdict"] = "pass" if ratio >= args.min_example_match and report["weights_match"] else "FAIL"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['examples_identical']}/{report['examples']} examples and "
          f"{report['attempts_identical']}/{report['attempts']} attempts reproduced exactly on "
          f"{report['environment']['gpu'] or 'CPU'}; weights {'match' if report['weights_match'] else 'DIFFER from'} "
          f"the stored hash; {report['seconds']} s -> {report['verdict']}")
    for row in report["different"][:10]:
        print(f"  differs: {row['example_id']} from attempt {row['first_difference']} "
              f"({row['attempts_identical']} of its attempts identical)")
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
