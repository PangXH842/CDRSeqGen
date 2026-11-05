"""Build shared callbacks that react to Hugging Face Trainer events.

Classes/Functions:
  * LogMetricsCallback  -- forward Trainer metric events to Python logging
  * build_callbacks     -- return callback instances for the method's Trainer builder
                         -> called by: mlm/ar/s2s.training.prepare_training()

Event flow:
  Trainer logs metrics -> LogMetricsCallback.on_log() -> console/train.log handlers
  Trainer evaluates    -> optional EarlyStoppingCallback.on_evaluate() -> stop decision

Notes:
  * A callback is an object whose event methods Trainer calls during its lifecycle.
    This module supplies metric logging and optional early stopping for MLM, AR and S2S.
  * Logging handlers are configured by the CLI and run setup. Other callbacks, such as
    tokenizer saving and S2S diagnostics, are registered in their owning modules.
"""
import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class LogMetricsCallback(TrainerCallback):
    """Forward nonempty Trainer log events to this module's Python logger.

    Example: logs={"epoch": 2.0, "loss": 0.375}
      -> "📉 epoch   2.00 │ loss=0.375"

    Trainer invokes on_log() when it emits training/evaluation metrics. Configured
    handlers send these messages to the console and, when attached, train.log.
    This callback formats messages; it does not create handlers or modify metrics.
    """

    def on_log(self, args, training_state, control, logs=None, **kwargs):
        """Log one metric dictionary, including its epoch when numeric; ignore empty logs."""
        if not logs:
            return
        epoch = logs.get("epoch")
        body = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in logs.items() if k != "epoch"
        )
        logger.info(f"📉 epoch {epoch:6.2f} │ {body}" if isinstance(epoch, (int, float)) else f"📉 {body}")


def build_callbacks(model_args, training_args, logger=None):
    """Return callback instances for MLM, AR and S2S Trainer construction.

    Called by each method's training.prepare_training(), which passes the returned
    list into build_trainer(callbacks=...). Trainer invokes the callbacks later.

    Returned callbacks:
      * LogMetricsCallback -- always included, to record Trainer metric events.
      * EarlyStoppingCallback -- included when early_stopping_patience is nonzero
        and load_best_model_at_end is enabled. Monitors metric_for_best_model using
        the configured patience and improvement threshold; patience counts evaluations.

    Shared argument validation rejects a requested patience without best-model
    loading before this function is called. The optional logger writes a setup
    message when early stopping is enabled.
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
