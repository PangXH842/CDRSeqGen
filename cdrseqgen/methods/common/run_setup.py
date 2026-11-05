"""Organize run names, output directories and saved configuration/source information.

Functions:
  * results_root        -- select the batch directory from the environment or timestamp
  * unique_run_dir      -- choose a new run path without overwriting an existing run
  * update_latest_link  -- point results/runs/latest at the selected batch
  * write_run_config    -- save flat resolved arguments plus dataset/runtime/Git identity
  * dataset_tag         -- map a corpus path to the run-name tag used by reports

Notes:
  * common.arguments.initialize_run() calls the naming/config helpers, creates the
    run directory and attaches file logging.
  * mlm/ar/s2s.training.prepare_training() calls update_latest_link() after dataset
    and split validation, before model construction.
  * The source record identifies config/data/runtime/Git settings; the declared
    corpus hash is copied from metadata, not recomputed here.
  * Importing this module creates no run directories and starts no training.
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

# Keep specific path fragments before broader matches; see dataset_tag().
DATASET_TAGS = [
    ("sabdab2",   "SD2"),
    ("oas_multi", "OAS4"),
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
    """Return an available run path, adding a numeric suffix when the name exists.

    Example: root/trial and root/trial_2 exist -> return root/trial_3 and warn.
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
    """Return the batch directory under which initialize_run() places its run folder.

    Selection:
      CDRSEQGEN_RESULTS_ROOT is nonempty -> use that directory unchanged
      otherwise -> ./results/runs/YYYYmmdd_HHMMSS (local timestamp)

    The timestamp gives standalone runs the same batch/run directory depth as
    scheduled batches. This function returns a path; it creates no directory.
    """
    from_env = os.environ.get(RESULTS_ROOT_ENV)
    if from_env:
        return from_env
    return os.path.join(DEFAULT_RESULTS_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))


def update_latest_link(run_dir: str) -> None:
    """Point results/runs/latest at the parent batch of the supplied run directory.

    Example: run_dir=results/runs/batch/trial
      -> results/runs/latest points to batch, making latest/trial accessible.

    Called by each method's prepare_training() after dataset/split validation.
    Skip when CDRSEQGEN_SKIP_LATEST is 1/true/yes, or when the parent is the runs
    root itself. The latter prevents latest from linking back to its own parent.

    Replace an existing symlink, but preserve any existing nonsymlink path.
    Store a relative link target; log filesystem errors as warnings so training
    can continue. latest tracks a batch, not an individual trial or selected model.
    """
    if os.environ.get(SKIP_LATEST_ENV, "").strip().lower() in {"1", "true", "yes"}:
        logger.info(f"Skipping results/runs/latest update ({SKIP_LATEST_ENV} is set).")
        return
    batch_dir = os.path.dirname(os.path.abspath(run_dir))
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
    """Return selected installed package versions, using None for missing distributions."""
    packages = ("torch", "transformers", "accelerate", "safetensors", "pandas", "scikit-learn")
    resolved = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = None
    return resolved


def _training_provenance(model_args, data_args):
    """Return dataset/model source information, runtime versions and Git state.

    Called by write_run_config(). Record the corpus path/size and hash its metadata
    sidecar; copy the declared corpus SHA from that sidecar without rehashing or
    verifying the corpus. model_source is the configured identifier/path, not a
    checksum of model weights.

    Locate the repository from the cdrseqgen package so moving this helper does
    not change the checkout recorded. Git information includes the commit and
    dirty flag, without a snapshot of uncommitted source changes.
    """
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
    """Write run_config.json from resolved argument groups and return the saved dict.

    Called by common.arguments.initialize_run() after argument validation.
    Merge model -> data -> generation -> TrainingArguments; later groups win if
    keys overlap. Add the source/runtime record under provenance and write atomically.

    Arguments stay flat so generation/evaluation can read target_region and
    context flags directly. These are resolved settings, including CLI overrides,
    rather than a copy of the original input JSON.
    """
    meta = {}
    for args in (model_args, data_args, gen_args):
        meta.update(asdict(args))
    meta.update(training_args.to_dict())
    meta["provenance"] = _training_provenance(model_args, data_args)
    atomic_write_json(os.path.join(output_dir, RUN_CONFIG_FILENAME), meta)
    return meta


def dataset_tag(data_path: str) -> str:
    """Return the first DATASET_TAGS label matching a case-insensitive path substring.

    Examples:
      processed_oas_multi.json -> OAS4 (four-donor OAS corpus)
      processed_oas.json       -> OAS  (historical single-donor corpus)
      unrecognized.json        -> DS, with a warning

    Called by initialize_run() to name runs; reports also use DATASET_TAGS.
    Ordering matters: oas_multi must precede oas so their runs retain distinct
    labels in report grouping. The tag describes the path, without inspecting data.
    """
    lowered = data_path.lower()
    for substring, tag in DATASET_TAGS:
        if substring in lowered:
            return tag
    logger.warning(
        f"⚠️ Data path '{data_path}' matched no entry in DATASET_TAGS. "
    )
    return UNKNOWN_DATASET_TAG
