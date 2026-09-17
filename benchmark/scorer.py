#!/usr/bin/env python3
"""Score one or more benchmark adapter outputs through one shared metric implementation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


# Direct execution puts benchmark/ rather than the repository root on sys.path, where neither the
# cdrseqgen package nor benchmark._common is importable. Normalize it before those imports.
BENCHMARK_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == BENCHMARK_DIR:
    sys.path.pop(0)
    sys.path.insert(0, str(BENCHMARK_DIR.parent))


# The sequence metrics come from the project rather than a private copy: a matched benchmark that
# defines "diversity" twice can report two different numbers for the same candidates. (The judge in
# this file already imports cdrseqgen, so the old "no project imports" rationale no longer held.)
from cdrseqgen.eval.metrics import (
    LENGTH_BUCKETS,
    core_anchor_indices,
    edit_similarity,
    hamming_fraction,
    per_position_entropy as position_entropy,
    recovery_at_indices,
)
from benchmark._common import (
    ADAPTER_SCHEMA_VERSION,
    AMINO_ACIDS,
    atomic_write_json,
    git_state,
    sha256_file,
)

SCHEMA_VERSION = "cdrseqgen.benchmark_scores.v1"
# Above this share of examples with zero generation attempts, the run is a failed run rather than a
# result with gaps, and scoring it would report metrics over a silently reduced population.
MAX_DEAD_RECORD_FRACTION = 0.05
SUMMARY_METRICS = (
    "whole_span_aar_mean",
    "core_aar_mean",
    "anchor_aar_mean",
    "diversity_hamming",
    "novelty_to_native_mean",
    "unique_fraction",
    "per_position_entropy_bits",
    "judge_perplexity_mean",
    "valid_generation_rate",
    "length_match_rate",
    "length_absolute_error_mean",
    "edit_similarity_mean",
    "fixed_length_acceptance_rate",
    "runtime_seconds_per_retained_sample",
)


def write_scores_json(path: Path, payload: dict[str, Any]) -> None:
    """Scores are the published artifact, so a NaN is a write-time error, not a bare NaN literal."""
    atomic_write_json(path, payload, allow_nan=False)


def mean_or_none(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def population_sd_or_none(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return statistics.pstdev(present) if len(present) > 1 else 0.0


def validate_adapter_record(record: dict[str, Any], manifest_record: dict[str, Any],
                            *, track: str = "fixed_native_length") -> None:
    required = {
        "example_id", "native", "native_length", "samples", "attempts", "valid_attempts",
        "invalid_attempts", "length_matches", "length_mismatches", "runtime_seconds", "complete",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"{record.get('example_id', '<unknown>')} lacks adapter fields: {missing}")
    example_id = record["example_id"]
    if record["native"] != manifest_record["native"]:
        raise ValueError(f"{example_id} native sequence differs from manifest")
    if record["native_length"] != manifest_record["native_length"]:
        raise ValueError(f"{example_id} native length differs from manifest")
    for field in (
        "native_length", "attempts", "valid_attempts", "invalid_attempts", "length_matches",
        "length_mismatches",
    ):
        if not isinstance(record[field], int) or isinstance(record[field], bool) or record[field] < 0:
            raise ValueError(f"{example_id} has invalid non-negative integer field {field}")
    if not isinstance(record["runtime_seconds"], (int, float)) or record["runtime_seconds"] < 0:
        raise ValueError(f"{example_id} has invalid runtime_seconds")
    if record["attempts"] != record["valid_attempts"] + record["invalid_attempts"]:
        raise ValueError(f"{example_id} attempt accounting is inconsistent")
    if record["valid_attempts"] != record["length_matches"] + record["length_mismatches"]:
        raise ValueError(f"{example_id} valid/length accounting is inconsistent")
    if track == "fixed_native_length" and len(record["samples"]) > record["length_matches"]:
        raise ValueError(f"{example_id} retained more samples than matching attempts")
    for sample in record["samples"]:
        if track == "fixed_native_length" and len(sample) != record["native_length"]:
            raise ValueError(f"{example_id} contains a retained sample with the wrong length")
        invalid = sorted(set(sample) - AMINO_ACIDS)
        if invalid:
            raise ValueError(f"{example_id} contains unsupported sample residues: {invalid}")


def score_record(
    record: dict[str, Any],
    *,
    anchor_trim: int,
    judge_perplexities: list[float] | None = None,
) -> dict[str, Any]:
    native = record["native"]
    samples = record["samples"]
    length = len(native)
    recovery_samples = [sample for sample in samples if len(sample) == length]
    core_indices, anchor_indices = core_anchor_indices(length, anchor_trim)

    whole_values = [recovery_at_indices(sample, native, list(range(length))) for sample in recovery_samples]
    core_values = [recovery_at_indices(sample, native, core_indices) for sample in recovery_samples]
    anchor_values = [recovery_at_indices(sample, native, anchor_indices) for sample in recovery_samples]
    pairwise = [hamming_fraction(first, second) for first, second in combinations(samples, 2)]

    attempts = record["attempts"]
    valid_attempts = record["valid_attempts"]
    retained = len(samples)
    runtime = float(record["runtime_seconds"])
    position_recovery = [
        {
            "position": position,
            "relative_position": (position / (length - 1)) if length > 1 else 0.5,
            "recovery": mean_or_none(
                float(sample[position] == native[position]) for sample in recovery_samples
            ),
        }
        for position in range(length)
    ]

    whole_mean = mean_or_none(whole_values)
    return {
        "n_samples": retained,
        "n_unique": len(set(samples)),
        "whole_span_aar_mean": whole_mean,
        "core_aar_mean": mean_or_none(core_values),
        "anchor_aar_mean": mean_or_none(anchor_values),
        "core_available": bool(core_indices),
        "core_length": len(core_indices),
        "anchor_length": len(anchor_indices),
        "diversity_hamming": mean_or_none(pairwise) if pairwise else 0.0,
        "novelty_to_native_mean": (1.0 - whole_mean) if whole_mean is not None else None,
        "unique_fraction": (len(set(samples)) / retained) if retained else None,
        "per_position_entropy_bits": position_entropy(samples),
        "judge_perplexity_mean": mean_or_none(judge_perplexities or []),
        "judge_candidates_scored": len(judge_perplexities or []),
        "valid_generation_rate": (valid_attempts / attempts) if attempts else None,
        "length_match_rate": (record["length_matches"] / valid_attempts) if valid_attempts else None,
        "length_absolute_error_mean": mean_or_none(abs(len(sample) - length) for sample in samples),
        "edit_similarity_mean": mean_or_none(edit_similarity(sample, native) for sample in samples),
        "fixed_length_acceptance_rate": (record["length_matches"] / attempts) if attempts else None,
        "runtime_seconds": runtime,
        "runtime_seconds_per_attempt": (runtime / attempts) if attempts else None,
        "runtime_seconds_per_retained_sample": (runtime / retained) if retained else None,
        "position_recovery": position_recovery,
    }


def length_bucket(length: int) -> str:
    return next(name for name, low, high in LENGTH_BUCKETS if low <= length <= high)


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        metric: {
            "macro_mean": mean_or_none(record["metrics"][metric] for record in records),
            "between_example_sd": population_sd_or_none(
                record["metrics"][metric] for record in records
            ),
            "examples": sum(record["metrics"][metric] is not None for record in records),
        }
        for metric in SUMMARY_METRICS
    }
    attempts = sum(record["accounting"]["attempts"] for record in records)
    valid = sum(record["accounting"]["valid_attempts"] for record in records)
    matches = sum(record["accounting"]["length_matches"] for record in records)
    retained = sum(record["metrics"]["n_samples"] for record in records)
    runtime = sum(record["metrics"]["runtime_seconds"] for record in records)
    summary["micro_accounting"] = {
        "examples": len(records),
        # Examples that generated nothing still count above but enter no metric, so report them
        # alongside rather than leaving the gap to be inferred from per-metric "examples" counts.
        "examples_without_attempts": sum(
            1 for record in records if not record["accounting"]["attempts"]
        ),
        # Attempted but retained nothing — e.g. IgLM's `filter` policy exhausting its budget
        # without ever matching the native length. These enter no metric either, but for a
        # different reason, and they silently shrink the population a macro mean is taken over.
        "examples_without_samples": sum(
            1 for record in records if not record["metrics"]["n_samples"]
        ),
        "attempts": attempts,
        "valid_attempts": valid,
        "length_matches": matches,
        "retained_samples": retained,
        "valid_generation_rate": (valid / attempts) if attempts else None,
        "length_match_rate": (matches / valid) if valid else None,
        "fixed_length_acceptance_rate": (matches / attempts) if attempts else None,
        "runtime_seconds": runtime,
        "runtime_seconds_per_attempt": (runtime / attempts) if attempts else None,
        "runtime_seconds_per_retained_sample": (runtime / retained) if retained else None,
    }
    return summary


class Judge:
    """Independent masked-LM judge applied identically to every adapter's candidates."""

    def __init__(self, model_name: str, model_path: str | None, device: str, max_len: int):
        import torch

        from cdrseqgen.datasets.dataset_mlm import assemble_masked_input, inject_context
        from cdrseqgen.methods.mlm.backbones import resolve
        from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood

        self.torch = torch
        self.inject_context = inject_context
        self.assemble_masked_input = assemble_masked_input
        self.pseudo_log_likelihood = pseudo_log_likelihood
        self.model_name = model_name
        self.model_source = model_path or model_name
        self.max_len = max_len
        self.backbone = resolve(model_name)
        self.tokenizer = self.backbone.build_tokenizer(model_name)
        self.model = self.backbone.load_pretrained_model(self.model_source)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )
        self.model.to(self.device)
        self.model.eval()

    def prepare(self, record: dict[str, Any], target_len: int | None = None) -> tuple[list[int], list[int]]:
        pt1_ids = self.tokenizer.encode(record["pt1"])
        pt2_ids = self.tokenizer.encode(record["pt2"])
        target_len = record["native_length"] if target_len is None else target_len
        pt1_ids, pt2_ids = self.inject_context(
            self.tokenizer, pt1_ids, pt2_ids, target_len, max_len=self.max_len
        )
        input_ids, mask_start = self.assemble_masked_input(
            self.tokenizer, pt1_ids, pt2_ids, target_len
        )
        if len(input_ids) > self.max_len:
            raise ValueError(
                f"judge prompt for {record['example_id']} exceeds max length {self.max_len}"
            )
        return input_ids, list(range(mask_start, mask_start + target_len))

    def score(self, record: dict[str, Any], samples: list[str]) -> list[float]:
        perplexities = []
        for sample in samples:
            input_ids, mask_positions = self.prepare(record, len(sample))
            token_ids = self.tokenizer.encode(sample)
            if len(token_ids) != len(mask_positions):
                raise ValueError(
                    f"judge tokenizer maps {record['example_id']} sample to {len(token_ids)} "
                    f"tokens, expected {len(mask_positions)}"
                )
            _, perplexity = self.pseudo_log_likelihood(
                self.model,
                input_ids,
                mask_positions,
                token_ids,
                self.tokenizer.special_ids,
                self.tokenizer.mask_id,
                self.device,
            )
            perplexities.append(perplexity)
        return perplexities

    def provenance(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        source_path = Path(self.model_source)
        checkpoint_hash = None
        if source_path.is_file():
            checkpoint_hash = sha256_file(source_path)
        elif source_path.is_dir():
            for filename in ("model.safetensors", "pytorch_model.bin"):
                candidate = source_path / filename
                if candidate.exists():
                    checkpoint_hash = sha256_file(candidate)
                    break
        return {
            "name": self.model_name,
            "source": self.model_source,
            "checkpoint_sha256": checkpoint_hash,
            "backbone": self.backbone.id,
            "max_len": self.max_len,
            "device": str(self.device),
            "model_commit": getattr(config, "_commit_hash", None),
            "torch_version": self.torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
        }


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("inputs must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("inputs must use non-empty LABEL=PATH")
    return label, Path(raw_path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def validate_payload(
    payload: dict[str, Any],
    *,
    label: str,
    manifest: dict[str, Any],
    manifest_hash: str,
    allow_partial: bool,
    allow_dead_records: bool = False,
) -> list[str]:
    if payload.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        raise ValueError(f"{label} has unsupported adapter schema {payload.get('schema_version')!r}")
    if payload.get("benchmark_id") != manifest.get("benchmark_id"):
        raise ValueError(f"{label} benchmark_id differs from manifest")
    if (payload.get("manifest") or {}).get("sha256") != manifest_hash:
        raise ValueError(f"{label} manifest hash differs from supplied manifest")
    expected_track = (manifest.get("task") or {}).get("track", "fixed_native_length")
    actual_track = (payload.get("sampling") or {}).get("track", "fixed_native_length")
    if expected_track not in {"fixed_native_length", "length_free"}:
        raise ValueError(f"manifest has unsupported benchmark track {expected_track!r}")
    if actual_track != expected_track:
        raise ValueError(
            f"{label} sampling track {actual_track!r} differs from manifest {expected_track!r}"
        )
    records = payload.get("records") or []
    if not records:
        raise ValueError(f"{label} contains no completed records")
    ids = [record.get("example_id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate example IDs")
    manifest_ids = [record["example_id"] for record in manifest.get("records") or []]
    unknown = sorted(set(ids) - set(manifest_ids))
    if unknown:
        raise ValueError(f"{label} contains IDs absent from manifest: {unknown[:3]}")
    if not allow_partial and set(ids) != set(manifest_ids):
        missing = sorted(set(manifest_ids) - set(ids))
        raise ValueError(f"{label} is partial; missing {len(missing)} manifest records")
    # A record whose generation crashed carries attempts == 0, which is self-consistent and so
    # survives validate_adapter_record. It then contributes None to every macro mean while still
    # counting toward micro_accounting.examples, making a half-dead run look complete.
    # An absent 'attempts' is a schema problem for validate_adapter_record to report; only an
    # explicit zero means the adapter ran this example and got nothing back.
    dead = [record["example_id"] for record in records if record.get("attempts") == 0]
    if dead and not allow_dead_records and len(dead) / len(records) > MAX_DEAD_RECORD_FRACTION:
        raise ValueError(
            f"{label} has {len(dead)}/{len(records)} examples with zero generation attempts "
            f"(limit {MAX_DEAD_RECORD_FRACTION:.0%}): {dead[:5]}. Re-run those examples or pass "
            "--allow-dead-records to score the survivors only."
        )
    if dead:
        print(
            f"WARNING: {label} has {len(dead)}/{len(records)} examples with zero generation "
            f"attempts; they contribute to no metric: {dead[:5]}",
            file=sys.stderr,
        )
    return ids


def assert_comparable(payloads: list[tuple[str, dict[str, Any], list[str]]]) -> None:
    if len(payloads) < 2:
        return
    first_label, first, first_ids = payloads[0]
    matched_sampling = (
        "track", "num_samples", "temperature", "top_p", "base_seed")
    def scored_ids(payload):
        return {r["example_id"] for r in (payload.get("records") or []) if r.get("samples")}

    first_scored = scored_ids(first)
    for label, payload, ids in payloads[1:]:
        if set(ids) != set(first_ids):
            raise ValueError(f"{label} and {first_label} do not contain the same framework IDs")
        # Containing the same IDs is not the same as scoring them. A run that attempted an
        # example and retained nothing contributes to no macro mean, so two runs can silently
        # be averaged over different populations while appearing matched.
        other_scored = scored_ids(payload)
        if other_scored != first_scored:
            only_first = sorted(first_scored - other_scored)
            only_other = sorted(other_scored - first_scored)
            print(
                f"WARNING: {label} and {first_label} score different populations — "
                f"{len(first_scored)} vs {len(other_scored)} examples with retained samples. "
                f"Macro means are NOT over the same frameworks. "
                f"Only {first_label}: {only_first[:3]}; only {label}: {only_other[:3]}",
                file=sys.stderr,
            )
        for field in matched_sampling:
            if (payload.get("sampling") or {}).get(field) != (first.get("sampling") or {}).get(field):
                raise ValueError(
                    f"unmatched sampling: {label} and {first_label} differ on {field}"
                )


def score_payload(
    label: str,
    path: Path,
    payload: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
    anchor_trim: int,
    judge: Judge | None,
) -> dict[str, Any]:
    records = []
    track = (payload.get("sampling") or {}).get("track", "fixed_native_length")
    total = len(payload["records"])
    for index, source_record in enumerate(payload["records"], start=1):
        example_id = source_record["example_id"]
        manifest_record = manifest_by_id[example_id]
        validate_adapter_record(source_record, manifest_record, track=track)
        judge_values = None
        if judge is not None:
            print(f"[{label} {index}/{total}] judge {example_id}", flush=True)
            judge_values = judge.score(manifest_record, source_record["samples"])
        metrics = score_record(
            source_record, anchor_trim=anchor_trim, judge_perplexities=judge_values
        )
        records.append({
            "example_id": example_id,
            "native": source_record["native"],
            "native_length": source_record["native_length"],
            "length_bucket": length_bucket(source_record["native_length"]),
            "samples": source_record["samples"],
            "accounting": {
                field: source_record[field]
                for field in (
                    "attempts", "valid_attempts", "invalid_attempts", "length_matches",
                    "length_mismatches", "complete",
                )
            },
            "metrics": metrics,
        })
    return {
        "label": label,
        "adapter": payload["adapter"],
        "input": {"path": str(path.resolve()), "sha256": sha256_file(path)},
        "model": payload.get("model"),
        "sampling": payload.get("sampling"),
        "aggregate": aggregate_records(records),
        "by_length_bucket": {
            name: aggregate_records([record for record in records if record["length_bucket"] == name])
            for name, _, _ in LENGTH_BUCKETS
            if any(record["length_bucket"] == name for record in records)
        },
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--input", action="append", type=parse_input, required=True, metavar="LABEL=PATH",
        help="Adapter output to score; repeat for a checked comparison",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        # argparse interpolates '%' in help text, so the threshold is spelled as a fraction.
        "--allow-dead-records", action="store_true",
        help="Score a run even when more than a "
             f"{MAX_DEAD_RECORD_FRACTION:g} fraction of its examples produced no generation "
             "attempts.",
    )
    parser.add_argument("--judge", default=None, help="Independent masked-LM model, e.g. Exscientia/IgBert")
    parser.add_argument("--judge-path", default=None)
    parser.add_argument("--judge-max-len", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    labels = [label for label, _ in args.input]
    if len(labels) != len(set(labels)):
        raise ValueError("--input labels must be unique")
    if args.judge_max_len <= 0:
        raise ValueError("--judge-max-len must be positive")

    manifest_hash = sha256_file(args.manifest)
    manifest = load_json(args.manifest)
    manifest_records = manifest.get("records") or []
    if not manifest_records:
        raise ValueError("manifest contains no records")
    manifest_by_id = {record["example_id"]: record for record in manifest_records}
    if len(manifest_by_id) != len(manifest_records):
        raise ValueError("manifest contains duplicate example IDs")

    loaded = []
    for label, path in args.input:
        payload = load_json(path)
        ids = validate_payload(
            payload,
            label=label,
            manifest=manifest,
            manifest_hash=manifest_hash,
            allow_partial=args.allow_partial,
            allow_dead_records=args.allow_dead_records,
        )
        loaded.append((label, path, payload, ids))
    assert_comparable([(label, payload, ids) for label, _, payload, ids in loaded])

    judge = Judge(args.judge, args.judge_path, args.device, args.judge_max_len) if args.judge else None
    anchor_trim = int((manifest.get("task") or {}).get("anchor_trim_each_end", 2))
    if anchor_trim < 0:
        raise ValueError("manifest anchor_trim_each_end must be non-negative")

    runs = [
        score_payload(label, path, payload, manifest_by_id, anchor_trim, judge)
        for label, path, payload, _ in loaded
    ]
    repo = Path(__file__).resolve().parents[1]
    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest_hash,
            "schema_version": manifest.get("schema_version"),
            "benchmark_id": manifest.get("benchmark_id"),
        },
        "scoring": {
            "aggregation_unit": "example",
            "anchor_trim_each_end": anchor_trim,
            "aar_population": "retained candidates whose length matches the native loop",
            "novelty_definition": "one minus length-matched whole-span amino-acid recovery",
            "diversity_definition": "mean normalized Hamming distance over duplicate-preserved draws",
            "length_match_rate_denominator": "valid generation attempts",
            "length_error_population": "all retained syntactically valid candidates",
            "edit_similarity_definition": "one minus Levenshtein distance normalized by longer length",
            **git_state(repo),
        },
        "judge": judge.provenance() if judge else None,
        "runs": runs,
    }
    write_scores_json(args.output, output)
    print(f"Saved {len(runs)} scored run(s) to {args.output}")
    for run in runs:
        aggregate = run["aggregate"]
        whole = aggregate["whole_span_aar_mean"]["macro_mean"]
        core = aggregate["core_aar_mean"]["macro_mean"]
        diversity = aggregate["diversity_hamming"]["macro_mean"]
        whole_text = f"{whole:.4f}" if whole is not None else "n/a"
        core_text = f"{core:.4f}" if core is not None else "n/a"
        diversity_text = f"{diversity:.4f}" if diversity is not None else "n/a"
        print(
            f"  {run['label']}: n={aggregate['micro_accounting']['examples']} "
            f"whole={whole_text} core={core_text} diversity={diversity_text}"
        )


if __name__ == "__main__":
    main()
