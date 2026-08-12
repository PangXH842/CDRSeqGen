#!/usr/bin/env python3
"""Generate CDRSeqGen outputs for a frozen benchmark manifest and checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Direct execution puts benchmark/adapters first on sys.path, where this file would shadow the
# `cdrseqgen` package. Replace that entry with the repository root before lazy package imports.
ADAPTER_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == ADAPTER_DIR:
    sys.path.pop(0)
    sys.path.insert(0, str(ADAPTER_DIR.parents[1]))

SCHEMA_VERSION = "cdrseqgen.adapter_output.v1"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
RUN_CONFIG_FILENAME = "run_config.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_example_seed(base_seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def git_state(repo: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True
            ).strip()
        )
        return {"repository_commit": commit, "repository_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"repository_commit": None, "repository_dirty": None}


def validate_manifest_record(record: dict[str, Any]) -> None:
    required = {
        "example_id", "parent_sequence", "pt1", "native", "pt2", "native_length",
        "infill_range",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"manifest record is missing fields: {missing}")
    if record["parent_sequence"] != record["pt1"] + record["native"] + record["pt2"]:
        raise ValueError(f"{record['example_id']} fails parent-sequence reconstruction")
    if record["native_length"] != len(record["native"]):
        raise ValueError(f"{record['example_id']} has an incorrect native_length")
    expected_range = [len(record["pt1"]), len(record["pt1"]) + len(record["native"])]
    if record["infill_range"] != expected_range:
        raise ValueError(f"{record['example_id']} has an incorrect infill_range")


def load_run_config(checkpoint: Path) -> tuple[dict[str, Any], Path]:
    path = checkpoint / RUN_CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"checkpoint lacks {RUN_CONFIG_FILENAME}: {checkpoint}")
    with path.open() as handle:
        return json.load(handle), path


def validate_checkpoint_contract(run_config: dict[str, Any], manifest: dict[str, Any]) -> None:
    task = manifest.get("task") or {}
    expected_target = task.get("target_region")
    if run_config.get("target_region") != expected_target:
        raise ValueError(
            f"checkpoint targets {run_config.get('target_region')!r}, manifest targets "
            f"{expected_target!r}"
        )
    if task.get("track") != "fixed_native_length":
        raise ValueError("CDRSeqGen adapter currently supports only fixed_native_length manifests")
    for config_key, task_key in (
        ("include_partner_chain", "partner_chain_context"),
        ("include_antigen", "antigen_context"),
    ):
        if bool(run_config.get(config_key, False)) != bool(task.get(task_key, False)):
            raise ValueError(
                f"checkpoint {config_key}={run_config.get(config_key)!r} does not match "
                f"manifest {task_key}={task.get(task_key)!r}"
            )
    processed_name = (manifest.get("source") or {}).get("processed_file")
    if processed_name and Path(run_config.get("data_path", "")).name != processed_name:
        raise ValueError(
            f"checkpoint data file {run_config.get('data_path')!r} does not match manifest "
            f"source {processed_name!r}"
        )


class CDRSeqGenSampler:
    def __init__(self, checkpoint: Path, run_config: dict[str, Any], device: str):
        import torch

        from cdrseqgen.data.dataset import assemble_masked_input, inject_context
        from cdrseqgen.eval.sampling import sample_independent, sample_iterative
        from cdrseqgen.models.registry import resolve

        self.torch = torch
        self.assemble_masked_input = assemble_masked_input
        self.inject_context = inject_context
        self.sample_independent = sample_independent
        self.sample_iterative = sample_iterative
        self.run_config = run_config
        self.max_len = int(run_config["max_len"])

        self.backbone = resolve(run_config["model_type"])
        self.tokenizer = self.backbone.build_tokenizer(run_config["model_type"])
        self.model = self.backbone.load_pretrained_model(str(checkpoint))
        selected_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.device = selected_device
        self.model.to(self.device)
        self.model.eval()

    def prepare(self, record: dict[str, Any]) -> None:
        pt1_ids = self.tokenizer.encode(record["pt1"])
        pt2_ids = self.tokenizer.encode(record["pt2"])
        original_pt1_len = len(pt1_ids)
        original_pt2_len = len(pt2_ids)
        target_len = record["native_length"]
        pt1_ids, pt2_ids = self.inject_context(
            self.tokenizer,
            pt1_ids,
            pt2_ids,
            target_len,
            max_len=self.max_len,
        )
        self.input_ids, mask_start = self.assemble_masked_input(
            self.tokenizer, pt1_ids, pt2_ids, target_len
        )
        self.mask_positions = list(range(mask_start, mask_start + target_len))
        self.prompt_info = {
            "max_len": self.max_len,
            "input_tokens": len(self.input_ids),
            "pt1_tokens": len(pt1_ids),
            "pt2_tokens": len(pt2_ids),
            "pt1_truncated_tokens": original_pt1_len - len(pt1_ids),
            "pt2_truncated_tokens": original_pt2_len - len(pt2_ids),
        }
        if len(self.input_ids) > self.max_len:
            raise ValueError(
                f"{record['example_id']} assembled length {len(self.input_ids)} exceeds "
                f"checkpoint max_len {self.max_len}"
            )

    def draw_many(
        self,
        *,
        num_samples: int,
        decoding: str,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int,
    ) -> list[str]:
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(seed)

        kwargs = dict(
            model=self.model,
            input_ids=self.input_ids,
            mask_positions=self.mask_positions,
            banned_ids=self.tokenizer.special_ids,
            device=self.device,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        if decoding == "independent":
            sampled_ids = self.sample_independent(**kwargs)
        else:
            sampled_ids = self.sample_iterative(
                **kwargs,
                mask_id=self.tokenizer.mask_id,
            )
        return [self.tokenizer.decode(ids).upper().strip() for ids in sampled_ids]


def run_example(
    record: dict[str, Any],
    sampler,
    *,
    num_samples: int,
    decoding: str,
    temperature: float,
    top_k: int,
    top_p: float,
    base_seed: int,
) -> dict[str, Any]:
    validate_manifest_record(record)
    seed = stable_example_seed(base_seed, record["example_id"])
    sampler.prepare(record)
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    try:
        samples = sampler.draw_many(
            num_samples=num_samples,
            decoding=decoding,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
    except Exception as exc:
        samples = []
        failures.append({
            "code": "generation_exception",
            "detail": f"{type(exc).__name__}: {exc}",
        })

    valid_samples = []
    raw_generations = []
    for attempt, loop in enumerate(samples, start=1):
        invalid = sorted(set(loop) - AMINO_ACIDS)
        valid_generation = not invalid
        length_match = len(loop) == record["native_length"]
        raw_generations.append({
            "attempt": attempt,
            "full_sequence": record["pt1"] + loop + record["pt2"],
            "loop": loop,
            "length": len(loop),
            "valid_generation": valid_generation,
            "length_match": length_match,
        })
        if invalid:
            failures.append({
                "attempt": attempt,
                "code": "invalid_amino_acid",
                "detail": f"generated loop contains unsupported residues: {invalid}",
            })
        elif not length_match:
            failures.append({
                "attempt": attempt,
                "code": "length_mismatch",
                "detail": f"generated {len(loop)} residues, expected {record['native_length']}",
            })
        if valid_generation and length_match:
            valid_samples.append(loop)

    attempts = len(samples)
    valid_attempts = sum(item["valid_generation"] for item in raw_generations)
    length_matches = sum(
        item["valid_generation"] and item["length_match"] for item in raw_generations
    )
    return {
        "example_id": record["example_id"],
        "native": record["native"],
        "native_length": record["native_length"],
        "samples": valid_samples,
        "attempts": attempts,
        "valid_attempts": valid_attempts,
        "invalid_attempts": attempts - valid_attempts,
        "length_matches": length_matches,
        "length_mismatches": valid_attempts - length_matches,
        "fixed_length_acceptance_rate": (len(valid_samples) / attempts) if attempts else 0.0,
        "complete": len(valid_samples) == num_samples,
        "raw_generations": raw_generations,
        "failures": failures,
        "prompt": getattr(sampler, "prompt_info", None),
        "seed": seed,
        "runtime_seconds": time.perf_counter() - started,
    }


def build_payload(
    args,
    manifest: dict[str, Any],
    manifest_hash: str,
    run_config: dict[str, Any],
    run_config_path: Path,
    sampler,
) -> dict[str, Any]:
    import torch
    import transformers

    weights_path = args.checkpoint / "model.safetensors"
    config_path = args.checkpoint / "config.json"
    if not weights_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"checkpoint lacks model.safetensors/config.json: {args.checkpoint}")
    repo = Path(__file__).resolve().parents[2]
    return {
        "schema_version": SCHEMA_VERSION,
        "adapter": "cdrseqgen",
        "benchmark_id": manifest["benchmark_id"],
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest_hash,
            "schema_version": manifest.get("schema_version"),
        },
        "model": {
            "name": run_config["model_type"],
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(weights_path),
            "config_sha256": sha256_file(config_path),
            "run_config_sha256": sha256_file(run_config_path),
            "run_config": run_config,
            "backbone": sampler.backbone.id,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(sampler.device),
            **git_state(repo),
        },
        "sampling": {
            "track": "fixed_native_length",
            "length_policy": "native_mask_count",
            "num_samples": args.num_samples,
            "decoding": args.decoding,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "base_seed": args.seed,
            "record_limit": args.limit,
            "duplicates_preserved": True,
            "raw_generations_preserved": True,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "records": [],
    }


def assert_resume_compatible(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in ("schema_version", "adapter", "benchmark_id", "manifest", "model", "sampling"):
        if payload.get(key) != expected.get(key):
            raise ValueError(f"existing output is incompatible for resume: {key} differs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--decoding", choices=("independent", "iterative"), default="iterative")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.temperature == 0 and args.num_samples != 1:
        raise ValueError("greedy decoding (--temperature 0) supports exactly one sample")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

    manifest_hash = sha256_file(args.manifest)
    with args.manifest.open() as handle:
        manifest = json.load(handle)
    records = manifest.get("records") or []
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[:args.limit]
    if not records:
        raise ValueError("manifest contains no selected records")

    run_config, run_config_path = load_run_config(args.checkpoint)
    validate_checkpoint_contract(run_config, manifest)
    sampler = CDRSeqGenSampler(args.checkpoint, run_config, args.device)
    expected = build_payload(
        args, manifest, manifest_hash, run_config, run_config_path, sampler
    )
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"output already exists; pass --resume to continue: {args.output}")
        with args.output.open() as handle:
            payload = json.load(handle)
        assert_resume_compatible(payload, expected)
    else:
        payload = expected

    completed = {record["example_id"] for record in payload["records"]}
    total = len(records)
    for position, record in enumerate(records, start=1):
        if record["example_id"] in completed:
            print(f"[{position}/{total}] skip {record['example_id']} (already complete)")
            continue
        print(f"[{position}/{total}] sample {record['example_id']}", flush=True)
        result = run_example(
            record,
            sampler,
            num_samples=args.num_samples,
            decoding=args.decoding,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            base_seed=args.seed,
        )
        payload["records"].append(result)
        atomic_write_json(args.output, payload)
        print(
            f"  retained={len(result['samples'])}/{args.num_samples} "
            f"failures={len(result['failures'])} runtime={result['runtime_seconds']:.2f}s",
            flush=True,
        )

    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    payload["summary"] = {
        "records_requested": total,
        "records_completed": len(payload["records"]),
        "records_with_full_sample_count": sum(r["complete"] for r in payload["records"]),
        "total_attempts": sum(r["attempts"] for r in payload["records"]),
        "total_valid_attempts": sum(r["valid_attempts"] for r in payload["records"]),
        "total_invalid_attempts": sum(r["invalid_attempts"] for r in payload["records"]),
        "total_length_matches": sum(r["length_matches"] for r in payload["records"]),
        "total_length_mismatches": sum(r["length_mismatches"] for r in payload["records"]),
        "total_runtime_seconds": sum(r["runtime_seconds"] for r in payload["records"]),
    }
    atomic_write_json(args.output, payload)
    print(f"Saved {len(payload['records'])} records to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed records remain saved and can be resumed.", file=sys.stderr)
        raise SystemExit(130)
