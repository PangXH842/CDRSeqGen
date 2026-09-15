"""Shared names and provenance/IO helpers for persisted CDRSeqGen artifacts."""

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

# New runs write the canonical name. Readers use the ordered tuple so current artifacts win while
# older checkpoints remain readable. On a future rename, move the previous canonical name into
# RUN_CONFIG_LEGACY_FILENAMES; consumers do not need to change.
RUN_CONFIG_FILENAME = "run_config.json"
RUN_CONFIG_LEGACY_FILENAMES = ("run_meta.json",)
RUN_CONFIG_FILENAMES = (
    RUN_CONFIG_FILENAME,
    *RUN_CONFIG_LEGACY_FILENAMES,
)

# The length predictor is a deliberately separate artifact: old sequence checkpoints remain valid
# for fixed-length generation, while bounded/inferred generation fails clearly when these are absent.
LENGTH_CONFIG_FILENAME = "length_config.json"
LENGTH_HEAD_FILENAME = "length_head.safetensors"

# What a finished run leaves beside its weights. Named here rather than at each reader because both
# were bare literals in eight and seven files respectively -- a rename would have had to be found by
# grep, and a reader that missed it degrades silently (a missing generation_eval.json is simply a
# trial with no generative metrics, not an error).
GENERATION_EVAL_FILENAME = "generation_eval.json"
CONTEXT_ABLATION_FILENAME = "generation_eval_without_partner.json"
EPOCH_METRICS_FILENAME = "epoch_metrics.csv"

# A processed corpus carries its provenance in a sibling file: data.json -> data.metadata.json.
PROCESSED_METADATA_SUFFIX = ".metadata.json"


def processed_metadata_path(data_path: str | Path) -> Path:
    """The provenance sidecar beside a processed corpus.

    `with_suffix` rather than string concatenation, so `a.b/data.json` cannot become
    `a.metadata.json`; four call sites were each spelling this out.
    """
    return Path(data_path).with_suffix(PROCESSED_METADATA_SUFFIX)


def load_processed_metadata(data_path: str | Path) -> dict[str, Any] | None:
    """The corpus's provenance sidecar, or None when it has none.

    Absent is normal -- a hand-made or pre-2026-07 corpus has no sidecar -- so this returns None
    rather than raising. A sidecar that exists but is corrupt still raises: unlike a missing run
    config, silently ignoring it would drop the hash every downstream integrity check compares to.
    """
    path = processed_metadata_path(data_path)
    if not path.is_file():
        return None
    with path.open() as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading a potentially multi-gigabyte artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    """Path, size, and digest for a regular file; None when no file is present."""
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
    """Replace non-finite floats with None so the result is strict, portable JSON.

    ``json.dump`` emits bare ``NaN`` / ``Infinity`` by default. Python reads those back, but they
    are not JSON: any standards-compliant reader rejects the file, and a metric that was simply
    never collected rendered as the string "nan" in tables. ``None`` round-trips everywhere and is
    what every reader here already treats as "absent".
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def atomic_write_json(
    path: str | Path,
    payload: dict[str, Any],
    *,
    allow_nan: bool = False,
) -> None:
    """Write JSON via ``os.replace`` so interruption cannot leave a truncated artifact.

    Non-finite floats become ``null`` unless ``allow_nan`` is set, so artifacts stay strict JSON.
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
        # A failed serialization should not leave a confusing sibling that looks resumable.
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def git_state(repo: str | Path) -> dict[str, Any]:
    """Commit and dirty flag for the checkout that produced an artifact, when available."""
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
    """Path to the run config a training run wrote in model_dir, or None if it wrote none."""
    for name in RUN_CONFIG_FILENAMES:
        path = Path(model_dir) / name
        if path.is_file():
            return path
    return None


def load_run_config(model_dir) -> tuple[dict, Path]:
    """Load a run's saved configuration, returning it with the path it came from.

    Raises FileNotFoundError when no config exists and json.JSONDecodeError when one exists but is
    corrupt. Callers must not conflate the two: an absent config means "this checkpoint predates
    run configs", while an unreadable one means the provenance every downstream compatibility check
    depends on is gone, and treating that as absent silently disables those checks.
    """
    path = find_run_config(model_dir)
    if path is None:
        accepted = ", ".join(RUN_CONFIG_FILENAMES)
        raise FileNotFoundError(f"no run config in {model_dir} (expected one of: {accepted})")
    with path.open() as handle:
        return json.load(handle), path


def read_run_config(model_dir) -> dict | None:
    """The run's saved configuration, or None when the checkpoint predates run configs.

    The tolerant counterpart to :func:`load_run_config`, for callers that degrade gracefully on an
    older checkpoint. A corrupt config still raises: only *absence* is a legitimate legacy state.
    """
    try:
        config, _ = load_run_config(model_dir)
    except FileNotFoundError:
        return None
    return config
