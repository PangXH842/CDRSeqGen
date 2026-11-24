#!/usr/bin/env python3
"""Export frozen CDR-H3 benchmark manifests and their IMGT flank-region sidecars.

A manifest is the common, model-independent input contract shared by CDRSeqGen and external
baselines: antibody strings rather than token IDs, so every adapter sees the same examples while
owning its own tokenization and sampling. Generators that consume IMGT regions directly (such as
IgCraft) also read a flank-only region sidecar built from the same processed corpus and verified
to reconstruct `pt1` and `pt2` without exposing CDR3.

Examples:
    python benchmark/manifest.py export \\
        --input data/processed/processed_oas_tuning.json \\
        --output benchmark/manifests/oas_h3_test_sample_v1.json \\
        --sample-size 25

    python benchmark/manifest.py regions \\
        --manifest benchmark/manifests/X.json \\
        --corpus data/processed/Y.json \\
        --output benchmark/manifests/X_regions.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Direct execution puts benchmark/ rather than the repository root on sys.path, where the shared
# helper package would be invisible. Normalize before importing it.
BENCHMARK_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == BENCHMARK_DIR:
    sys.path.pop(0)
    sys.path.insert(0, str(BENCHMARK_DIR.parent))

from benchmark._common import atomic_write_json, sha256_file
from cdrseqgen.artifacts import processed_metadata_path

SCHEMA_VERSION = "cdrseqgen.benchmark_manifest.v1"
REGIONS_SCHEMA_VERSION = "cdrseqgen.flank_regions.v1"
REGION_ORDER = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4")
PT1_REGIONS = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3")
FLANK_REGIONS = PT1_REGIONS + ("fwr4",)
IGLM_SPECIES_TOKENS = {
    "human": "[HUMAN]",
    "mouse": "[MOUSE]",
    "rat": "[RAT]",
    "rabbit": "[RABBIT]",
    "rhesus": "[RHESUS]",
    "camel": "[CAMEL]",
}


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def h3_record(entry: dict, source_index: int) -> dict:
    """Convert one processed OAS entry to the shared H3 benchmark schema."""
    heavy = entry["heavy"]
    species = str(entry.get("species") or "").strip().lower()
    if species not in IGLM_SPECIES_TOKENS:
        raise ValueError(
            f"source index {source_index} has missing/unsupported IgLM species {species!r}"
        )
    source = entry.get("source") or {}
    if source.get("dataset") != "OAS" or source.get("run") is None:
        raise ValueError(f"source index {source_index} lacks stable OAS source provenance")
    missing = [key for key in (*REGION_ORDER, "full_aa") if key not in heavy]
    if missing:
        raise ValueError(f"source index {source_index} is missing heavy-chain fields: {missing}")

    pt1 = "".join(heavy[key] for key in REGION_ORDER[:5])
    native = heavy["cdr3"]
    pt2 = heavy["fwr4"]
    parent = pt1 + native + pt2
    if parent != heavy["full_aa"]:
        raise ValueError(
            f"source index {source_index} does not reconstruct heavy.full_aa from IMGT regions"
        )
    if not native:
        raise ValueError(f"source index {source_index} has an empty CDR-H3")

    sequence_id = str(entry["sequence_id"])
    source_run = int(source["run"])
    return {
        "example_id": f"oas:{source_run}:{sequence_id}",
        "source_index": source_index,
        "sequence_id": sequence_id,
        "split": entry["split"],
        "chain": "heavy",
        "species": species,
        "chain_token": "[HEAVY]",
        "species_token": IGLM_SPECIES_TOKENS[species],
        "pt1": pt1,
        "native": native,
        "pt2": pt2,
        "native_length": len(native),
        "infill_range": [len(pt1), len(pt1) + len(native)],
        "parent_sequence": parent,
    }


def selection_key(record: dict, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{record['example_id']}".encode("utf-8")).hexdigest()


def validate_records(records: list[dict], split: str) -> None:
    ids = [record["example_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest example_id values are not unique")

    for record in records:
        if record["split"] != split:
            raise ValueError(f"{record['example_id']} is not in requested split {split!r}")
        if record["parent_sequence"] != record["pt1"] + record["native"] + record["pt2"]:
            raise ValueError(f"{record['example_id']} failed parent-sequence reconstruction")
        if record["native_length"] != len(record["native"]):
            raise ValueError(f"{record['example_id']} has an incorrect native_length")
        expected_range = [len(record["pt1"]), len(record["pt1"]) + len(record["native"])]
        if record["infill_range"] != expected_range:
            raise ValueError(f"{record['example_id']} has an incorrect infill_range")


def build_manifest(
    input_path: Path,
    *,
    split: str,
    sample_size: int | None,
    selection_seed: str,
    benchmark_id: str,
    track: str = "fixed_native_length",
) -> dict:
    if track not in {"fixed_native_length", "length_free"}:
        raise ValueError(f"unsupported benchmark track {track!r}")
    with input_path.open() as handle:
        entries = json.load(handle)

    eligible = [
        h3_record(entry, index)
        for index, entry in enumerate(entries)
        if entry.get("split") == split
    ]
    if not eligible:
        raise ValueError(f"no records found for split {split!r} in {input_path}")

    eligible.sort(key=lambda record: selection_key(record, selection_seed))
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("--sample-size must be positive")
        if sample_size > len(eligible):
            raise ValueError(
                f"--sample-size {sample_size} exceeds {len(eligible)} eligible records"
            )
        records = eligible[:sample_size]
    else:
        records = eligible

    # Make output order human-readable without changing the deterministic selected membership.
    records.sort(key=lambda record: record["example_id"])
    validate_records(records, split)

    processed_hash = sha256_file(input_path)
    metadata_path = processed_metadata_path(input_path)
    if not metadata_path.exists():
        raise ValueError(f"missing preprocessing metadata sidecar: {metadata_path}")
    with metadata_path.open() as handle:
        processed_metadata = json.load(handle)
    if processed_metadata.get("processed_file_sha256") != processed_hash:
        raise ValueError(
            f"metadata sidecar hash does not match processed input: {metadata_path}"
        )

    species_values = sorted({record["species"] for record in records})
    species_tokens = sorted({record["species_token"] for record in records})

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "task": {
            "name": "cdr_h3_infilling",
            "target_region": "h3",
            "numbering_scheme": "IMGT",
            "track": track,
            "anchor_trim_each_end": 2,
            "partner_chain_context": False,
            "antigen_context": False,
        },
        "source": {
            "dataset": processed_metadata["dataset"],
            "processed_file": input_path.name,
            "processed_file_sha256": processed_hash,
            "metadata_file": metadata_path.name,
            "preprocessing_metadata": processed_metadata,
        },
        "selection": {
            "split": split,
            "eligible_records": len(eligible),
            "selected_records": len(records),
            "strategy": "lowest SHA-256(selection_seed + ':' + example_id)",
            "selection_seed": selection_seed,
        },
        "conditioning": {
            "chain": "heavy",
            "species_values": species_values,
            "iglm_chain_token": "[HEAVY]",
            "iglm_species_tokens": species_tokens,
        },
        "records": records,
    }


# ---------------------------------------------------------------------------
# Flank-region sidecar (for IMGT-region generators such as IgCraft)
# ---------------------------------------------------------------------------

def flank_regions(record: dict, regions: dict) -> dict[str, str]:
    """The flank regions as a dict, after proving they rebuild the record's pt1 and pt2."""
    if "".join(regions[name] for name in PT1_REGIONS) != record["pt1"] or regions["fwr4"] != record["pt2"]:
        raise ValueError(f"{record['example_id']} flank regions do not rebuild pt1/pt2")
    return {name: regions[name] for name in FLANK_REGIONS}


def build_regions(manifest: dict, corpus: list[dict], corpus_sha256: str) -> dict:
    """The flank-region sidecar payload for `manifest`, built from its source corpus."""
    if corpus_sha256 != manifest["source"]["processed_file_sha256"]:
        raise ValueError("Corpus differs from the one the manifest was built from")
    records = {}
    for record in manifest["records"]:
        entry = corpus[record["source_index"]]
        if entry["sequence_id"] != record["sequence_id"]:
            raise ValueError(f"{record['example_id']} points at a different corpus entry")
        records[record["example_id"]] = flank_regions(record, entry["heavy"])
    return dict(
        schema_version=REGIONS_SCHEMA_VERSION,
        benchmark_id=manifest["benchmark_id"],
        source=dict(
            processed_file=manifest["source"]["processed_file"],
            processed_file_sha256=corpus_sha256,
        ),
        regions=list(FLANK_REGIONS),
        records=records,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_export(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    manifest = build_manifest(
        input_path,
        split=args.split,
        sample_size=args.sample_size,
        track=args.track,
        selection_seed=args.selection_seed,
        benchmark_id=args.benchmark_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    manifest_hash = sha256_file(args.output)
    hash_path = args.output.with_suffix(args.output.suffix + ".sha256")
    with hash_path.open("w") as handle:
        handle.write(f"{manifest_hash}  {args.output.name}\n")

    print(
        f"Exported {manifest['selection']['selected_records']}/"
        f"{manifest['selection']['eligible_records']} {args.split} records to {args.output}"
    )
    print(f"Validated {len(manifest['records'])}/{len(manifest['records'])} records")
    print(f"Manifest SHA-256: {manifest_hash}")


def _cmd_regions(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    payload = build_regions(manifest, json.loads(args.corpus.read_text()), sha256_file(args.corpus))
    atomic_write_json(args.output, payload)
    args.output.with_name(args.output.name + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n"
    )
    print(f'Wrote {len(payload["records"])} frameworks to {args.output}')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export a frozen CDR-H3 manifest from a processed OAS JSON")
    exp.add_argument("--input", type=Path, required=True, help="Processed OAS JSON")
    exp.add_argument("--output", type=Path, required=True, help="Manifest JSON to create")
    exp.add_argument("--split", default="test", help="Named source split (default: test)")
    exp.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Deterministic subset size; omit to export the complete split",
    )
    exp.add_argument("--selection-seed", default="oas-h3-test-v1")
    exp.add_argument("--benchmark-id", default="oas_h3_test_v1")
    exp.add_argument(
        "--track", choices=("fixed_native_length", "length_free"), default="fixed_native_length"
    )
    exp.set_defaults(func=_cmd_export)

    reg = sub.add_parser("regions", help="Export IMGT flank regions for an existing manifest")
    reg.add_argument("--manifest", type=Path, required=True)
    reg.add_argument("--corpus", type=Path, required=True)
    reg.add_argument("--output", type=Path, required=True)
    reg.set_defaults(func=_cmd_regions)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
