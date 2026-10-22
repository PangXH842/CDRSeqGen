"""Trainer progress logging and callback construction shared across methods.

Classes:
  * LogMetricsCallback -- send Trainer log events through the run's logger

Functions:
  * build_callbacks -- always include metric logging; add requested early stopping
"""
import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


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


def build_callbacks(model_args, training_args, logger=None):
    """Metric logging ALWAYS, early stopping only when a config asks for it.

    ``LogMetricsCallback`` is not optional: without it ``train.log`` records only "Starting
    Training..." and "Starting Evaluation..." with no per-epoch numbers — the exact regression fixed
    on 2026-07-29. Sharing this is what stops a second entrypoint quietly reintroducing it, which is
    precisely what the first causal entrypoint did.
    """
    callbacks = [LogMetricsCallback()]
    if model_args.early_stopping_patience and training_args.load_best_model_at_end:
        from transformers import EarlyStoppingCallback
        if logger:
            logger.info(f"⏳ Early stopping after {model_args.early_stopping_patience} epochs "
                        f"without improvement on {training_args.metric_for_best_model}")
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=model_args.early_stopping_patience,
            early_stopping_threshold=model_args.early_stopping_threshold))
    return callbacks
