"""Run naming, output directories and resolved-configuration provenance.

Functions:
  * results_root -- select the batch directory from the environment or timestamp
  * unique_run_dir -- choose a new run path without overwriting an existing run
  * update_latest_link -- point results/runs/latest at the selected batch
  * write_run_config -- save flat resolved arguments plus dataset/runtime/Git identity
  * dataset_tag -- map a corpus path to the run-name tag used by reports

The shared argument setup uses these helpers, creates the run directory and
attaches file logging. Each method updates latest after dataset validation.
This module does not start training or create directories merely by importing.
"""
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cdrseqgen
from cdrseqgen.artifacts import (
    RUN_CONFIG_FILENAME,
    atomic_write_json,
    file_identity,
    git_state,
    processed_metadata_path,
)

# First match wins, so a more specific corpus must precede the general one. The tag is what keeps
# incomparable test sets off one axis (see visualization/batch_plots), and a corpus built from a
# different donor set has a different test split — so it needs its own tag even though it is still
# OAS. Without the `oas_multi` entry ahead of `oas`, a 4-donor run would be named `h3_OAS_scratch`,
# indistinguishable from a single-donor one, and the batch tables would silently merge the two.
DATASET_TAGS = [
    ("sabdab2",   "SD2"),
    ("oas_multi", "OAS4"),   # Jaffe 2022, all four donors, naive B cells (2026-09-03)
    ("oas",       "OAS"),
]
UNKNOWN_DATASET_TAG = "DS"

RESULTS_ROOT_ENV = "CDRSEQGEN_RESULTS_ROOT"
RUN_SUFFIX_ENV = "CDRSEQGEN_RUN_SUFFIX"
SKIP_LATEST_ENV = "CDRSEQGEN_SKIP_LATEST"
DEFAULT_RESULTS_ROOT = "./results/runs"
LATEST_LINK = "latest"

logger = logging.getLogger(__name__)


def unique_run_dir(root: str, run_name: str) -> str:
    """<root>/<run_name>, suffixed with _2, _3, ... if that path is already taken.

    Run folders carry no timestamp (the batch folder has one), which is only safe if a name clash
    can never silently overwrite an earlier run. Two cases genuinely clash:
      * sweep.sh reuses ONE base config for its whole grid — every grid point resolves to
        the same name (mitigated further by CDRSEQGEN_RUN_SUFFIX, which names the grid point);
      * re-running a config into an existing batch folder, e.g. retrying a failed run.
    Both would otherwise land on an existing directory and overwrite real results, so the clash is
    resolved by renaming and logged loudly rather than passing silently.
    """
    candidate = os.path.join(root, run_name)
    if not os.path.exists(candidate):
        return candidate
    for n in range(2, 1000):
        alternative = os.path.join(root, f"{run_name}_{n}")
        if not os.path.exists(alternative):
            logger.warning(f"⚠️ {candidate} already exists — writing this run to {alternative} instead.")
            return alternative
    raise RuntimeError(f"Could not find a free run directory for {candidate!r} (tried _2.._999).")


def results_root() -> str:
    """Batch directory that this run's folder is created inside (see RESULTS_ROOT_ENV)."""
    from_env = os.environ.get(RESULTS_ROOT_ENV)
    if from_env:
        return from_env
    # Standalone run: give it its own batch folder so every run sits at the same depth.
    return os.path.join(DEFAULT_RESULTS_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))


def update_latest_link(run_dir: str) -> None:
    """Point results/runs/latest at the newest batch unless a batch runner explicitly opts out.

    A run.sh round trains many configs into one results/<batch-timestamp>/ folder; latest tracks that
    whole round, so results/runs/latest/<config>/ addresses any run in it. Sweeps set
    CDRSEQGEN_SKIP_LATEST=1 because their trial folders should not replace the main usable batch.

    Refuses to touch results/runs/latest if it is a real directory rather than a symlink — that would
    mean deleting real results, which is never worth a convenience shortcut.
    """
    if os.environ.get(SKIP_LATEST_ENV, "").strip().lower() in {"1", "true", "yes"}:
        logger.info(f"Skipping results/runs/latest update ({SKIP_LATEST_ENV} is set).")
        return
    batch_dir = os.path.dirname(os.path.abspath(run_dir))
    # Never self-link results/runs/latest -> results (would happen only if a run dir sat directly in
    # results/ with no batch folder; results_root() always interposes one, so this is a guard).
    if batch_dir == os.path.abspath(DEFAULT_RESULTS_ROOT):
        return
    link_path = os.path.join(DEFAULT_RESULTS_ROOT, LATEST_LINK)
    try:
        os.makedirs(DEFAULT_RESULTS_ROOT, exist_ok=True)
        if os.path.islink(link_path):
            os.remove(link_path)
        elif os.path.exists(link_path):
            logger.warning(f"⚠️ {link_path} exists and is not a symlink — leaving it alone.")
            return
        target = os.path.relpath(batch_dir, os.path.abspath(DEFAULT_RESULTS_ROOT))
        os.symlink(target, link_path)
    except OSError as e:
        logger.warning(f"⚠️ Could not update {link_path}: {e}")


def _package_versions():
    packages = ("torch", "transformers", "accelerate", "safetensors", "pandas", "scikit-learn")
    resolved = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = None
    return resolved


def _training_provenance(model_args, data_args):
    """Lightweight run identity without re-hashing a multi-gigabyte processed dataset per trial."""
    data_path = Path(data_args.data_path).expanduser()
    metadata_path = processed_metadata_path(data_path)
    dataset = {
        "path": str(data_path.resolve()),
        "size_bytes": data_path.stat().st_size if data_path.is_file() else None,
        "metadata": file_identity(metadata_path),
        "declared_processed_file_sha256": None,
    }
    if metadata_path.is_file():
        with metadata_path.open() as handle:
            dataset["declared_processed_file_sha256"] = json.load(handle).get(
                "processed_file_sha256"
            )

    # Anchor to the package so moving this module cannot change the checkout.
    repo = Path(cdrseqgen.__file__).resolve().parent.parent
    return {
        "dataset": dataset,
        "model_source": model_args.model_type,
        "runtime": {
            "python": sys.version.split()[0],
            "packages": _package_versions(),
        },
        **git_state(repo),
    }


def write_run_config(output_dir: str, model_args, data_args, gen_args, training_args) -> dict:
    """Persist every resolved argument so a checkpoint and sweep trial are self-describing.

    The file remains flat for backward compatibility with generate.py and evaluate.py, which read
    target_region and context flags directly from the top level.
    """
    meta = {}
    for args in (model_args, data_args, gen_args):
        meta.update(asdict(args))
    meta.update(training_args.to_dict())
    meta["provenance"] = _training_provenance(model_args, data_args)
    atomic_write_json(os.path.join(output_dir, RUN_CONFIG_FILENAME), meta)
    return meta


def dataset_tag(data_path: str) -> str:
    """Resolve the short dataset tag for a data path via DATASET_TAGS (first match wins)."""
    lowered = data_path.lower()
    for substring, tag in DATASET_TAGS:
        if substring in lowered:
            return tag
    logger.warning(
        f"⚠️ Data path '{data_path}' matched no entry in DATASET_TAGS. "
    )
    return UNKNOWN_DATASET_TAG
