"""Configure console/file logging for training, generation and evaluation.

Functions:
  * configure_logging -- configure stdout and quiet noisy dependency loggers
  * attach_file_logging -- append to a run's train.log and log uncaught exceptions

Call configure_logging() from a command, then attach_file_logging() after run
setup creates its output directory. Importing this module configures nothing.
Run directories and provenance live in run_setup.py; Trainer metric logging
lives in callbacks.py.
"""
import logging
import os
import sys

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILENAME = "train.log"

# These libraries log every HTTP request at INFO and propagate to root, so they land in train.log —
# 15 of 24 lines in a typical ESM-2 run were hub chatter. Their WARNINGs still get through, so a real
# problem (rate limiting, a failed download) is not hidden; only the per-request noise is.
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub")

logger = logging.getLogger(__name__)


def configure_logging(level=logging.INFO):
    """Send logging to stdout and quiet the per-request hub chatter. Call once, from an entrypoint."""
    logging.basicConfig(level=level, format=LOG_FORMAT,
                        handlers=[logging.StreamHandler(sys.stdout)])
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


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
