"""Where a run writes and how it logs — everything that sets up a run's workspace.

  * configure_logging   -- stdout logging + quieting the noisy hub libraries (entrypoints call this)
  * attach_file_logging -- tee the root logger into <run_dir>/train.log
  * LogMetricsCallback  -- route the Trainer's metric dicts through logging so train.log records them
  * results_root        -- the batch directory this run's folder is created inside
  * unique_run_dir      -- <batch>/<run_name>, suffixed _2, _3 ... rather than overwriting
  * update_latest_link  -- point results/latest at the newest batch
  * write_run_meta      -- persist data/model provenance so a checkpoint is self-describing
  * dataset_tag         -- data path -> short dataset tag (OAS / SD2)

Notes:
  * Every run lives at results/<batch>/<run>/ — always two levels, so there is one shape to reason
    about and results/ never collects loose run folders. Batch runners export CDRSEQGEN_RESULTS_ROOT
    so a whole round shares a batch folder; a standalone train.py gets a batch folder of its own.
  * The run folder is named from the RESOLVED arguments, not the config filename, so a CLI override
    (`--target_region l3` on h3_scratch.json) can't be misfiled. See arguments.setup_args.
  * Logging config is a function, not import-time side effect, because three entrypoints need it
    (train.py, predict.py, gen_eval.py) and gen_eval.py used to get it only as an accident of
    importing train.py.
"""
import os
import sys
import json
import logging
from datetime import datetime

from transformers import TrainerCallback

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
DEFAULT_RESULTS_ROOT = "./results"
LATEST_LINK = "latest"
RUN_META_FILENAME = "run_meta.json"

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
      * sweep_scratch.sh reuses ONE base config for its whole grid — every grid point resolves to
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
    """Point results/latest at the newest *batch* directory (the parent of this run's folder).

    A run.sh round trains many configs into one results/<batch-timestamp>/ folder; latest tracks that
    whole round, so results/latest/<config>/ addresses any run in it — e.g. results/latest/h3_OAS_esm2/.
    Point predict.py at a specific config: --model_path ./results/latest/<config>.
    The link is relative, so moving/renaming the results tree doesn't break it.

    Refuses to touch results/latest if it is a real directory rather than a symlink — that would
    mean deleting real results, which is never worth a convenience shortcut.
    """
    batch_dir = os.path.dirname(os.path.abspath(run_dir))
    # Never self-link results/latest -> results (would happen only if a run dir sat directly in
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


def write_run_meta(output_dir: str, model_args, data_args) -> dict:
    """Persist the data/model provenance so a checkpoint is self-describing.

    target_region (which CDR loop the model was trained to fill) otherwise lives ONLY in the config
    and the run-dir name. Saving it into the run dir lets gen_eval.py report what the model was
    actually trained for — and warn if it's later evaluated with a config targeting a different loop,
    which would silently mask the wrong CDR and produce confident, meaningless numbers. predict.py
    reads the same file to warn when a checkpoint's expected context isn't being supplied.
    """
    meta = {
        "target_region": data_args.target_region,
        "model_type": model_args.model_type,
        "include_partner_chain": bool(data_args.include_partner_chain),
        "include_antigen": bool(data_args.include_antigen),
        "data_path": data_args.data_path,
        "max_len": data_args.max_len,
    }
    with open(os.path.join(output_dir, RUN_META_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)
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
