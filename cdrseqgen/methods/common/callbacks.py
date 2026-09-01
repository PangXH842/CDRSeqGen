"""Trainer callbacks shared by every method's entrypoint."""


def build_callbacks(model_args, training_args, logger=None):
    """Metric logging ALWAYS, early stopping only when a config asks for it.

    ``LogMetricsCallback`` is not optional: without it ``train.log`` records only "Starting
    Training..." and "Starting Evaluation..." with no per-epoch numbers — the exact regression fixed
    on 2026-07-29. Sharing this is what stops a second entrypoint quietly reintroducing it, which is
    precisely what the first causal entrypoint did.
    """
    from cdrseqgen.train.logging import LogMetricsCallback

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
