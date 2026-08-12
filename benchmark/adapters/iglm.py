#!/usr/bin/env python3
"""Generate attempt-bounded IgLM outputs for a CDRSeqGen benchmark manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Direct execution puts benchmark/adapters first on sys.path, where this file would shadow the
# installed `iglm` package. Replace that entry with the repository root before any lazy IgLM import.
ADAPTER_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == ADAPTER_DIR:
    sys.path.pop(0)
    sys.path.insert(0, str(ADAPTER_DIR.parents[1]))

SCHEMA_VERSION = "cdrseqgen.adapter_output.v1"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


class InvalidGeneration(ValueError):
    """A sampled sequence could not be interpreted as an infilled loop."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_example_seed(base_seed: int, example_id: str) -> int:
    """Derive a resume-safe seed that is stable for each example."""
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def extract_infilled_loop(generated: str, pt1: str, pt2: str) -> str:
    """Extract the inserted span while proving that IgLM preserved both flanks."""
    generated = generated.upper().strip()
    if not generated.startswith(pt1):
        raise InvalidGeneration("prefix_mismatch", "generated sequence does not preserve pt1")
    if pt2 and not generated.endswith(pt2):
        raise InvalidGeneration("suffix_mismatch", "generated sequence does not preserve pt2")

    end = len(generated) - len(pt2) if pt2 else len(generated)
    if end < len(pt1):
        raise InvalidGeneration("overlapping_flanks", "generated sequence is shorter than its flanks")
    loop = generated[len(pt1):end]
    invalid = sorted(set(loop) - AMINO_ACIDS)
    if invalid:
        raise InvalidGeneration(
            "invalid_amino_acid", f"infilled loop contains unsupported residues: {invalid}"
        )
    return loop


class IgLMSampler:
    """One-draw wrapper around IgLM internals; unlike IgLM.infill, it never deduplicates or loops."""

    def __init__(self, model_name: str, device: str, max_length: int, length_policy: str):
        import numpy as np
        import torch
        from iglm import IgLM
        from iglm.model.tokens import BAD_WORD_IDS
        from iglm.model.utils import iglm_to_infilled, mask_span, validate_generated_seq

        self.np = np
        self.torch = torch
        self.bad_word_ids = BAD_WORD_IDS
        self.iglm_to_infilled = iglm_to_infilled
        self.mask_span = mask_span
        self.validate_generated_seq = validate_generated_seq
        self.max_length = max_length
        self.length_policy = length_policy

        self.iglm = IgLM(model_name=model_name)
        if device != "auto":
            self.iglm.device = torch.device(device)
            self.iglm.model.to(self.iglm.device)
        self.iglm.model.eval()

    @property
    def device(self) -> str:
        return str(self.iglm.device)

    def set_seed(self, seed: int) -> None:
        from iglm.utils.seed import set_seeds

        set_seeds(seed)
        if self.iglm.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)

    def _starting_tokens(self, record: dict[str, Any]):
        start, end = record["infill_range"]
        masked = self.mask_span(list(record["parent_sequence"]), start, end)
        tokens = [record["chain_token"], record["species_token"], *masked]
        token_ids = self.iglm.tokenizer.convert_tokens_to_ids(tokens)
        if self.iglm.tokenizer.unk_token_id in token_ids:
            raise ValueError(f"{record['example_id']} contains a token unknown to IgLM")
        starting = self.torch.tensor([token_ids], dtype=self.torch.long, device=self.iglm.device)
        if starting.shape[-1] >= self.max_length:
            raise ValueError(
                f"{record['example_id']} prompt length {starting.shape[-1]} must be below "
                f"--max-length {self.max_length}"
            )
        return starting

    def prepare(self, record: dict[str, Any]) -> None:
        """Validate and cache one example's prompt before its attempt loop."""
        self.starting_tokens = self._starting_tokens(record)
        self.target_total_length = self.starting_tokens.shape[-1] + record["native_length"] + 1
        if self.target_total_length > self.max_length:
            raise ValueError(
                f"{record['example_id']} needs generation length {self.target_total_length}, "
                f"above --max-length {self.max_length}"
            )

    def draw(self, *, temperature: float, top_p: float) -> str:
        """Return exactly one decoded full-sequence draw or raise InvalidGeneration."""
        with self.torch.inference_mode():
            generation_lengths = (
                {
                    "min_length": self.target_total_length,
                    "max_length": self.target_total_length,
                }
                if self.length_policy == "constrain"
                else {"max_length": self.max_length}
            )
            ids = self.iglm.model.generate(
                self.starting_tokens,
                pad_token_id=self.iglm.tokenizer.pad_token_id,
                eos_token_id=self.iglm.tokenizer.cls_token_id,
                forced_eos_token_id=self.iglm.tokenizer.cls_token_id,
                bad_words_ids=self.bad_word_ids,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                **generation_lengths,
            )[0].detach().cpu().numpy()

        if not self.validate_generated_seq(ids, self.iglm.tokenizer):
            raise InvalidGeneration("invalid_token_layout", "IgLM special tokens are malformed")
        decoded_ids = self.iglm_to_infilled(ids, self.iglm.tokenizer)
        decoded = self.iglm.tokenizer.decode(decoded_ids)
        return "".join(decoded).replace(" ", "").upper()


def validate_manifest_record(record: dict[str, Any]) -> None:
    required = {
        "example_id",
        "parent_sequence",
        "pt1",
        "native",
        "pt2",
        "native_length",
        "infill_range",
        "chain_token",
        "species_token",
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


def run_example(
    record: dict[str, Any],
    sampler,
    *,
    num_samples: int,
    max_attempts: int,
    temperature: float,
    top_p: float,
    base_seed: int,
    max_errors: int,
) -> dict[str, Any]:
    """Sample one framework, preserving all raw valid loops and exact attempt accounting."""
    validate_manifest_record(record)
    seed = stable_example_seed(base_seed, record["example_id"])
    sampler.set_seed(seed)
    sampler.prepare(record)

    started = time.perf_counter()
    samples: list[str] = []
    raw_generations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    attempts = 0
    consecutive_errors = 0

    while attempts < max_attempts and len(samples) < num_samples:
        attempts += 1
        try:
            full_sequence = sampler.draw(temperature=temperature, top_p=top_p)
            loop = extract_infilled_loop(full_sequence, record["pt1"], record["pt2"])
            length_match = len(loop) == record["native_length"]
            raw_generations.append({
                "attempt": attempts,
                "full_sequence": full_sequence,
                "loop": loop,
                "length": len(loop),
                "valid_generation": True,
                "length_match": length_match,
            })
            if length_match:
                # Keep raw draws, including duplicates. Diversity belongs in the scorer.
                samples.append(loop)
            consecutive_errors = 0
        except InvalidGeneration as exc:
            failures.append({"attempt": attempts, "code": exc.code, "detail": exc.detail})
            consecutive_errors += 1
        except Exception as exc:  # preserve unexpected model/runtime failures in the artifact
            failures.append({
                "attempt": attempts,
                "code": "generation_exception",
                "detail": f"{type(exc).__name__}: {exc}",
            })
            consecutive_errors += 1

        if consecutive_errors >= max_errors:
            failures.append({
                "attempt": attempts,
                "code": "max_consecutive_errors",
                "detail": f"stopped after {max_errors} consecutive failed attempts",
            })
            break

    valid_attempts = len(raw_generations)
    length_matches = sum(item["length_match"] for item in raw_generations)
    return {
        "example_id": record["example_id"],
        "native": record["native"],
        "native_length": record["native_length"],
        "samples": samples,
        "attempts": attempts,
        "valid_attempts": valid_attempts,
        "invalid_attempts": attempts - valid_attempts,
        "length_matches": length_matches,
        "length_mismatches": valid_attempts - length_matches,
        "fixed_length_acceptance_rate": (length_matches / attempts) if attempts else 0.0,
        "complete": len(samples) == num_samples,
        "raw_generations": raw_generations,
        "failures": failures,
        "seed": seed,
        "runtime_seconds": time.perf_counter() - started,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def repository_commit(path: Path | None) -> str | None:
    if path is None or not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_payload(args, manifest: dict[str, Any], manifest_hash: str, sampler) -> dict[str, Any]:
    try:
        import iglm
        import torch
        import transformers
        from iglm.model.IgLM import CHECKPOINT_DICT

        package_path = Path(iglm.__file__).resolve().parent
        checkpoint = Path(CHECKPOINT_DICT[args.model_name]) / "pytorch_model.bin"
        provenance = {
            "package_version": importlib.metadata.version("iglm"),
            "package_path": str(package_path),
            "repository_commit": repository_commit(args.iglm_repo),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "checkpoint_file": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "device": sampler.device,
        }
    except Exception as exc:
        raise RuntimeError(f"could not collect IgLM provenance: {exc}") from exc

    return {
        "schema_version": SCHEMA_VERSION,
        "adapter": "iglm",
        "benchmark_id": manifest["benchmark_id"],
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest_hash,
            "schema_version": manifest.get("schema_version"),
        },
        "model": {"name": args.model_name, **provenance},
        "sampling": {
            "track": "fixed_native_length",
            "length_policy": args.length_policy,
            "num_samples": args.num_samples,
            "max_attempts": args.max_attempts,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_length": args.max_length,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", choices=("IgLM", "IgLM-S"), default="IgLM")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument(
        "--length-policy",
        choices=("constrain", "filter"),
        default="constrain",
        help="Constrain EOS to native length, or sample variable lengths and filter matches",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-errors", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N manifest records")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    default_repo = Path(__file__).resolve().parents[3] / "IgLM"
    parser.add_argument("--iglm-repo", type=Path, default=default_repo)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.max_attempts <= 0 or args.max_errors <= 0:
        raise ValueError("sample, attempt, and error limits must be positive")
    if args.max_attempts < args.num_samples:
        raise ValueError("--max-attempts must be at least --num-samples")
    if args.temperature <= 0:
        raise ValueError("IgLM sampling requires --temperature > 0")
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

    sampler = IgLMSampler(args.model_name, args.device, args.max_length, args.length_policy)
    expected = build_payload(args, manifest, manifest_hash, sampler)
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
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            top_p=args.top_p,
            base_seed=args.seed,
            max_errors=args.max_errors,
        )
        payload["records"].append(result)
        atomic_write_json(args.output, payload)
        print(
            f"  retained={len(result['samples'])}/{args.num_samples} "
            f"attempts={result['attempts']} acceptance={result['fixed_length_acceptance_rate']:.1%} "
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
