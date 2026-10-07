"""Helpers shared by the benchmark manifest exporter, both adapters, and the scorer.

These are provenance and IO primitives, not metrics: hashing an input so a result names the exact
file it came from, deriving a per-example seed that survives --resume, writing output atomically so
an interrupted run leaves a readable file, and recording the repository state a result was produced
under. They were copied into four modules, where sha256_file alone existed five times; a benchmark
whose whole purpose is a matched comparison cannot afford two definitions of "the same" anything.

Note on sys.path: each entrypoint still repairs sys.path itself before importing this module. That
bootstrap cannot live here, because importing this module is what it makes possible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cdrseqgen.artifacts import atomic_write_json, git_state, sha256_file

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
ADAPTER_SCHEMA_VERSION = "cdrseqgen.adapter_output.v1"

# The fields that must match for an existing output file to be extended rather than mixed. Model or
# sampling drift mid-run would produce one file describing two experiments.
RESUME_IDENTITY_KEYS = ("schema_version", "adapter", "benchmark_id", "manifest", "model", "sampling")

# Every manifest record carries these; adapters may require more (IgLM also needs conditioning
# tokens), which they pass as `extra_required`.
MANIFEST_RECORD_FIELDS = (
    "example_id", "parent_sequence", "pt1", "native", "pt2", "native_length", "infill_range",
)


def execution_source_identity() -> dict:
    """Hash local Python sources so dirty-checkout edits cannot mix resumed outputs."""
    root = Path(__file__).resolve().parents[1]
    paths = [p for directory in ('cdrseqgen', 'benchmark')
             for p in (root / directory).rglob('*.py')]
    sources = {str(p.relative_to(root)): sha256_file(p) for p in sorted(paths)}
    return {'sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            'files': sources}


def stable_example_seed(base_seed: int, example_id: str) -> int:
    """Derive a resume-safe seed that is stable for each example.

    Keyed on the example ID rather than its position, so resuming a partial run reproduces the same
    draws regardless of how many examples were completed before the interruption.
    """
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def repository_commit(path: Path | None) -> str | None:
    """Commit of an external dependency's checkout (IgLM's package tree), or None."""
    if path is None or not (Path(path) / ".git").exists():
        return None
    return git_state(Path(path))["repository_commit"]


def validate_manifest_record(record: dict[str, Any], *, extra_required: tuple[str, ...] = ()) -> None:
    """Check one manifest record reconstructs its parent sequence exactly.

    The adapters must agree on where the loop sits, so this verifies pt1/native/pt2 rebuild the
    parent and that native_length and infill_range describe the same span.
    """
    required = set(MANIFEST_RECORD_FIELDS) | set(extra_required)
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


def assert_resume_compatible(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    """Refuse to append to an output produced under different settings."""
    for key in RESUME_IDENTITY_KEYS:
        if payload.get(key) != expected.get(key):
            raise ValueError(f"existing output is incompatible for resume: {key} differs")
