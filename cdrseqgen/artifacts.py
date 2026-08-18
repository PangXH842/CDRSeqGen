"""Shared names for files persisted in a CDRSeqGen run directory, and the reader for them."""

import json
from pathlib import Path

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
