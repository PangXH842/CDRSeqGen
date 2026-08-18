#!/usr/bin/env python3
"""Generate CDRSeqGen outputs for a frozen benchmark manifest and checkpoint."""

from __future__ import annotations

import argparse
import json
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

from cdrseqgen.artifacts import (
    LENGTH_CONFIG_FILENAME,
    LENGTH_HEAD_FILENAME,
    RUN_CONFIG_FILENAMES,
    load_run_config as shared_load_run_config,
)
from benchmark._common import (
    ADAPTER_SCHEMA_VERSION as SCHEMA_VERSION,
    AMINO_ACIDS,
    assert_resume_compatible,
    atomic_write_json,
    git_state,
    sha256_file,
    stable_example_seed,
    validate_manifest_record,
)


def load_run_config(checkpoint: Path) -> tuple[dict[str, Any], Path]:
    """Adapter-side alias so the benchmark reads checkpoints exactly as the entrypoints do."""
    return shared_load_run_config(checkpoint)


def validate_checkpoint_contract(run_config: dict[str, Any], manifest: dict[str, Any]) -> None:
    task = manifest.get("task") or {}
    expected_target = task.get("target_region")
    if run_config.get("target_region") != expected_target:
        raise ValueError(
            f"checkpoint targets {run_config.get('target_region')!r}, manifest targets "
            f"{expected_target!r}"
        )
    if task.get("track") not in {"fixed_native_length", "length_free"}:
        raise ValueError("manifest track must be fixed_native_length or length_free")
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
    def __init__(self, checkpoint: Path, run_config: dict[str, Any], device: str,
                 track: str = "fixed_native_length"):
        import torch

        from cdrseqgen.methods.mlm.generation import (
            prepare_fixed_length_prompt,
            sample_length_conditioned_spans,
        )
        from cdrseqgen.methods.mlm.backbones import resolve
        from cdrseqgen.methods.mlm.length import (
            fixed_policy,
            native_mask_length,
            resolve_length_policy,
        )
        from cdrseqgen.methods.mlm.length_predictor import LengthPredictor

        self.torch = torch
        self.prepare_fixed_length_prompt = prepare_fixed_length_prompt
        self.sample_length_conditioned_spans = sample_length_conditioned_spans
        self.resolve_length_policy = resolve_length_policy
        self.fixed_policy = fixed_policy
        self.track = track
        self.native_mask_length = native_mask_length
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
        self.length_predictor = (
            LengthPredictor.load(checkpoint, device=self.device)
            if track == "length_free" else None
        )

    def prepare(self, record: dict[str, Any]) -> None:
        # Clear per-example state before anything can raise: run_example reads length_metadata and
        # prompt_info even when this call fails, and the previous example's values would then be
        # recorded against this one.
        self.record = record
        self.length_metadata = None
        self.length_policy = None
        self.length_decision = None
        self.prompt_info = None
        self.input_ids = None
        self.mask_positions = None
        if self.track == "length_free":
            self.length_policy = self.resolve_length_policy()
            self.length_metadata = self.length_policy.as_dict()
            return
        # The native mask count is a length that is known rather than predicted, so it becomes an
        # ordinary fixed policy and takes the same generation path as the length-free track.
        self.length_decision = self.native_mask_length(record["native_length"])
        self.length_policy = self.fixed_policy(
            self.length_decision.length, source=self.length_decision.source
        )
        # Built once here purely to enforce the checkpoint's max_len before any GPU work; the
        # prompt used for sampling is rebuilt inside sample_length_conditioned_spans.
        prompt = self.prepare_fixed_length_prompt(
            self.tokenizer,
            record["pt1"],
            record["pt2"],
            self.length_decision.length,
            max_len=self.max_len,
        )
        self.input_ids = prompt.input_ids
        self.mask_positions = prompt.mask_positions
        self.prompt_info = prompt.metadata()
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
        length_temperature: float,
        seed: int,
    ) -> list[str]:
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(seed)

        generated, self.length_metadata = self.sample_length_conditioned_spans(
            self.model,
            self.tokenizer,
            self.device,
            generator,
            pt1=self.record["pt1"],
            pt2=self.record["pt2"],
            policy=self.length_policy,
            length_predictor=self.length_predictor,
            max_len=self.max_len,
            decoding=decoding,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            length_temperature=length_temperature,
        )
        self.prompt_info = self.length_metadata.get("prompts_by_length")
        return [
            self.tokenizer.decode(span.token_ids).upper().strip()
            for span in generated
        ]


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
    length_temperature: float = 1.0,
) -> dict[str, Any]:
    validate_manifest_record(record)
    seed = stable_example_seed(base_seed, record["example_id"])
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    generation_failed = False
    try:
        # prepare() is inside the guard so an over-long prompt is recorded as one failed example
        # rather than aborting the whole benchmark run.
        sampler.prepare(record)
        samples = sampler.draw_many(
            num_samples=num_samples,
            decoding=decoding,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            length_temperature=length_temperature,
            seed=seed,
        )
    except Exception as exc:
        samples = []
        generation_failed = True
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
        elif not length_match and getattr(sampler, "track", "fixed_native_length") == "fixed_native_length":
            failures.append({
                "attempt": attempt,
                "code": "length_mismatch",
                "detail": f"generated {len(loop)} residues, expected {record['native_length']}",
            })
        retain = length_match or getattr(sampler, "track", "fixed_native_length") == "length_free"
        if valid_generation and retain:
            valid_samples.append(loop)

    attempts = len(samples)
    valid_attempts = sum(item["valid_generation"] for item in raw_generations)
    length_matches = sum(
        item["valid_generation"] and item["length_match"] for item in raw_generations
    )
    length_policy = getattr(sampler, "length_metadata", None)
    if generation_failed:
        length_policy = {"mode": "unavailable", "source": "generation_failed"}
    elif length_policy is None:
        length_policy = {
            "mode": "fixed",
            "source": "native_mask_count",
            "minimum": record["native_length"],
            "maximum": record["native_length"],
            "selected_length": record["native_length"],
        }
    return {
        "example_id": record["example_id"],
        "native": record["native"],
        "native_length": record["native_length"],
        "length_policy": length_policy,
        "generation_failed": generation_failed,
        "samples": valid_samples,
        "attempts": attempts,
        "valid_attempts": valid_attempts,
        "invalid_attempts": attempts - valid_attempts,
        "length_matches": length_matches,
        "length_mismatches": valid_attempts - length_matches,
        "fixed_length_acceptance_rate": (length_matches / attempts) if attempts else 0.0,
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
    track = (manifest.get("task") or {}).get("track", "fixed_native_length")
    length_artifacts = None
    if track == "length_free":
        length_config_path = args.checkpoint / LENGTH_CONFIG_FILENAME
        length_weights_path = args.checkpoint / LENGTH_HEAD_FILENAME
        length_artifacts = {
            "config_sha256": sha256_file(length_config_path),
            "weights_sha256": sha256_file(length_weights_path),
        }
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
            "length_predictor": length_artifacts,
            **git_state(repo),
        },
        "sampling": {
            "track": track,
            "length_policy": "native_mask_count" if track == "fixed_native_length" else "learned",
            "num_samples": args.num_samples,
            "decoding": args.decoding,
            "temperature": args.temperature,
            "length_temperature": args.length_temperature,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--decoding", choices=("independent", "iterative"), default="iterative")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--length-temperature", type=float, default=1.0)
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
    if args.length_temperature < 0:
        raise ValueError("--length-temperature must be non-negative")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[:args.limit]
    if not records:
        raise ValueError("manifest contains no selected records")

    run_config, run_config_path = load_run_config(args.checkpoint)
    validate_checkpoint_contract(run_config, manifest)
    track = (manifest.get("task") or {}).get("track", "fixed_native_length")
    sampler = CDRSeqGenSampler(args.checkpoint, run_config, args.device, track=track)
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
            length_temperature=args.length_temperature,
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
    failed_records = [r for r in payload["records"] if r.get("generation_failed")]
    payload["summary"] = {
        "records_requested": total,
        "records_completed": len(payload["records"]),
        "records_with_full_sample_count": sum(r["complete"] for r in payload["records"]),
        "records_with_generation_failure": len(failed_records),
        "failed_example_ids": [r["example_id"] for r in failed_records],
        "total_attempts": sum(r["attempts"] for r in payload["records"]),
        "total_valid_attempts": sum(r["valid_attempts"] for r in payload["records"]),
        "total_invalid_attempts": sum(r["invalid_attempts"] for r in payload["records"]),
        "total_length_matches": sum(r["length_matches"] for r in payload["records"]),
        "total_length_mismatches": sum(r["length_mismatches"] for r in payload["records"]),
        "total_runtime_seconds": sum(r["runtime_seconds"] for r in payload["records"]),
    }
    atomic_write_json(args.output, payload)
    print(f"Saved {len(payload['records'])} records to {args.output}")
    if failed_records:
        print(
            f"WARNING: {len(failed_records)}/{len(payload['records'])} examples produced no "
            f"candidates: {[r['example_id'] for r in failed_records][:5]}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed records remain saved and can be resumed.", file=sys.stderr)
        raise SystemExit(130)
