"""Setup directory for training batch and runs, and logging.

  * configure_logging   -- stdout logging + quieting the noisy hub libraries (entrypoints call this)
  * attach_file_logging -- tee the root logger into <run_dir>/train.log
  * LogMetricsCallback  -- route the Trainer's metric dicts through logging so train.log records them
  * results_root        -- the batch directory this run's folder is created inside
  * unique_run_dir      -- <batch>/<run_name>, suffixed _2, _3 ... rather than overwriting
  * update_latest_link  -- point results/runs/latest at the newest batch
  * write_run_config      -- persist data/model provenance so a checkpoint is self-describing
  * dataset_tag         -- data path -> short dataset tag (OAS / SD2)

Notes:
  * Every run lives at results/<batch>/<run>/ — always two levels, so there is one shape to reason
    about and results/ never collects loose run folders. Batch runners export CDRSEQGEN_RESULTS_ROOT
    so a whole round shares a batch folder; a standalone train.py gets a batch folder of its own.
  * The run folder is named from the RESOLVED arguments, not the config filename, so a CLI override
    (`--target_region l3` on h3_scratch.json) can't be misfiled. See arguments.setup_args.
  * Logging config is a function, not import-time side effect, because three entrypoints need it
    (train.py, generate.py, evaluate.py) and evaluate.py used to get it only as an accident of
    importing train.py.
"""
import os
import sys
import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from transformers import TrainerCallback

from cdrseqgen.artifacts import (
    RUN_CONFIG_FILENAME,
    atomic_write_json,
    file_identity,
    git_state,
)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILENAME = "train.log"

# These libraries log every HTTP request at INFO and propagate to root, so they land in train.log —
# 15 of 24 lines in a typical ESM-2 run were hub chatter. Their WARNINGs still get through, so a real
# problem (rate limiting, a failed download) is not hidden; only the per-request noise is.
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub")

DATASET_TAGS = [
    ("sabdab2", "SD2"),
    ("oas",     "OAS"),
]
UNKNOWN_DATASET_TAG = "DS"

RESULTS_ROOT_ENV = "CDRSEQGEN_RESULTS_ROOT"
RUN_SUFFIX_ENV = "CDRSEQGEN_RUN_SUFFIX"
DEFAULT_RESULTS_ROOT = "./results/runs"
LATEST_LINK = "latest"

logger = logging.getLogger(__name__)


def configure_logging(level=logging.INFO):
    """Send logging to stdout and quiet the per-request hub chatter. Call once, from an entrypoint."""
    logging.basicConfig(level=level, format=LOG_FORMAT,
                        handlers=[logging.StreamHandler(sys.stdout)])
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


class LogMetricsCallback(TrainerCallback):
    """Route the Trainer's metric dicts through logging, so they reach <run_dir>/train.log.

    HF reports metrics via PrinterCallback (print) or ProgressCallback (tqdm.write) — neither goes
    through the logging module, so the file handler never saw them. A 15-minute run left train.log
    with a silent gap between "Starting Training" and "Starting Evaluation" and no numbers at all.
    `on_log` is the documented hook that receives the same dicts those callbacks print.
    """

    def on_log(self, args, training_state, control, logs=None, **kwargs):
        if not logs:
            return
        epoch = logs.get("epoch")
        body = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in logs.items() if k != "epoch"
        )
        logger.info(f"📉 epoch {epoch:6.2f} │ {body}" if isinstance(epoch, (int, float)) else f"📉 {body}")


def attach_file_logging(output_dir: str) -> str:
    """Tee the log stream into <output_dir>/train.log, so every run keeps its own record.

    Until this existed, logger output went ONLY to stdout: a run launched by scripts/run.sh or bare
    `python train.py` left no readable log at all — the run dir held numbers (epoch_metrics.csv,
    *_results.json) and binary TensorBoard events, but nothing you could open to see what happened.

    Attached to the ROOT logger so the transformers/Trainer loggers are captured too (they propagate
    to root), and called after output_dir exists — configure_logging() runs long before the run
    directory is known. Re-attaching is guarded so repeated calls in one process (tests, notebooks)
    don't duplicate every line.
    """
    log_path = os.path.join(output_dir, LOG_FILENAME)
    root = logging.getLogger()

    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == os.path.abspath(log_path):
            return log_path

    file_handler = logging.FileHandler(log_path, mode="a+", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    # A crash otherwise goes to stderr only — and scripts/run.sh deliberately continues past a failed
    # run ("|| true"), so the run dir would be left with no explanation of what went wrong. Route
    # uncaught exceptions through the logger first so the traceback lands in train.log too.
    def log_uncaught(exc_type, exc_value, exc_tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            logger.error("💥 Run failed with an uncaught exception:", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = log_uncaught
    return log_path


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
    if os.environ.get("CDRSEQGEN_SKIP_LATEST", "").strip().lower() in {"1", "true", "yes"}:
        logger.info("Skipping results/runs/latest update (CDRSEQGEN_SKIP_LATEST is set).")
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
    metadata_path = data_path.with_suffix(".metadata.json")
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

    repo = Path(__file__).resolve().parents[2]
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
