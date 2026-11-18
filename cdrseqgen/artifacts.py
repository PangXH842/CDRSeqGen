"""Shared names and provenance/IO helpers for persisted CDRSeqGen artifacts."""

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

# New runs write the canonical name; readers try the names in order, so old checkpoints still load.
# On a rename, move the old name into RUN_CONFIG_LEGACY_FILENAMES.
RUN_CONFIG_FILENAME = "run_config.json"
RUN_CONFIG_LEGACY_FILENAMES = ("run_meta.json",)
RUN_CONFIG_FILENAMES = (
    RUN_CONFIG_FILENAME,
    *RUN_CONFIG_LEGACY_FILENAMES,
)

# Length predictor files: a separate artifact, needed only for bounded/inferred generation.
LENGTH_CONFIG_FILENAME = "length_config.json"
LENGTH_HEAD_FILENAME = "length_head.safetensors"

# Files a finished run leaves beside its weights, named once so every reader uses one spelling.
GENERATION_EVAL_FILENAME = "generation_eval.json"
CONTEXT_ABLATION_FILENAME = "generation_eval_without_partner.json"
# The schema_version every method's evaluator writes into its payload.
GENERATION_EVAL_SCHEMA = "cdrseqgen.generation_eval.v1"
EPOCH_METRICS_FILENAME = "epoch_metrics.csv"

# A processed corpus carries its provenance in a sibling file: data.json -> data.metadata.json.
PROCESSED_METADATA_SUFFIX = ".metadata.json"


def processed_metadata_path(data_path: str | Path) -> Path:
    """The metadata file beside a processed corpus (data.json -> data.metadata.json).

    Called:
      * cdrseqgen/datasets/preprocess/common.py
      * cdrseqgen/methods/common/evaluation.py
      * cdrseqgen/methods/common/run_setup.py
      * benchmark/export_manifest.py
    """
    return Path(data_path).with_suffix(PROCESSED_METADATA_SUFFIX)


def load_processed_metadata(data_path: str | Path) -> dict[str, Any] | None:
    """Return the processed corpus's metadata (source, build settings, SHA-256) if present.

    Called:
      * none yet; get_evaluation_source_info (cdrseqgen/methods/common/evaluation.py) reads the
        same file itself
    """
    path = processed_metadata_path(data_path)
    if not path.is_file():
        return None         # pre-2026-07 test runs
    with path.open() as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading a potentially multi-gigabyte artifact into memory.

    Called:
      * cdrseqgen/experiments.py
      * benchmark/_common.py
          -> benchmark/adapters/cdrseqgen.py
          -> benchmark/adapters/igcraft.py
          -> benchmark/adapters/iglm.py
          -> benchmark/adapters/retrieval.py
          -> benchmark/export_manifest.py
          -> benchmark/export_regions.py
          -> benchmark/extend.py
          -> benchmark/followup.py
          -> benchmark/leakage.py
          -> benchmark/run.py
          -> benchmark/scorer.py
          -> benchmark/variability.py
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    """Returns (path, size, sha256 digest) for a regular file if present.

    Called:
      * cdrseqgen/methods/common/evaluation.py
      * cdrseqgen/methods/common/run_setup.py
    """
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with None so the result is strict, portable JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None      # NaN, inf -> None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}  # recursion for dicts
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]                # recursion for lists/tuples
    return value


def atomic_write_json(
    path: str | Path,
    payload: dict[str, Any],
    *,
    allow_nan: bool = False,
) -> None:
    """Robust JSON writing function to prevent truncated artifacts when interrupted.

    Called:
      * cdrseqgen/eval/generative.py
      * cdrseqgen/experiments.py
      * cdrseqgen/methods/ar/backbones.py
      * cdrseqgen/methods/common/generation.py
      * cdrseqgen/methods/common/run_setup.py
      * benchmark/_common.py
          -> benchmark/adapters/cdrseqgen.py
          -> benchmark/adapters/igcraft.py
          -> benchmark/adapters/iglm.py
          -> benchmark/adapters/retrieval.py
          -> benchmark/blind_check.py
          -> benchmark/export_regions.py
          -> benchmark/extend.py
          -> benchmark/followup.py
          -> benchmark/leakage.py
          -> benchmark/run.py
          -> benchmark/scorer.py
          -> benchmark/variability.py
    """
    if not allow_nan:
        payload = json_safe(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=allow_nan)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        # Remove the temporary file so a failed write leaves nothing that looks resumable.
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def git_state(repo: str | Path) -> dict[str, Any]:
    """Commit and dirty flag for the checkout that produced an artifact, when available.

    Called:
      * cdrseqgen/methods/common/evaluation.py
      * cdrseqgen/methods/common/run_setup.py
      * benchmark/_common.py
          -> benchmark/adapters/cdrseqgen.py
          -> benchmark/scorer.py
    """
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


def find_run_config(model_dir) -> Path | None:
    """Path of a run's config file (current name first, then legacy names), or None.

    Called:
      * fit_mlm_length.py
      * cdrseqgen/methods/ar/backbones.py
      * cdrseqgen/methods/common/evaluation.py
      * cdrseqgen/visualization/batch_plots.py
      * cdrseqgen/visualization/batch_tables.py
      * cdrseqgen/visualization/sweep_plots.py
      * benchmark/adapters/cdrseqgen.py
    """
    for name in RUN_CONFIG_FILENAMES:
        path = Path(model_dir) / name
        if path.is_file():
            return path
    return None


def read_run_config(model_dir) -> dict | None:
    """The run's saved configuration, or None when the checkpoint predates run configs.

    A corrupt config still raises: only *absence* is a legitimate legacy state.

    Called:
      * evaluate.py
      * fit_mlm_length.py
      * generate.py
      * cdrseqgen/methods/common/generation.py
      * cdrseqgen/methods/mlm/evaluation.py
      * cdrseqgen/visualization/sweep_tables.py
    """
    path = find_run_config(model_dir)
    if path is None:
        return None
    with path.open() as handle:
        return json.load(handle)
