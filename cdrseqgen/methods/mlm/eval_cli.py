"""MLM evaluation entrypoint called by root evaluate.py."""

from cdrseqgen.methods.mlm.evaluation import (
    evaluation_options,
    prepare_evaluation,
    resolve_decoding,
    run_generative_eval as evaluate,
)


def main(args, config, run_config=None):
    """Run standalone evaluation through the shared lifecycle and MLM evaluator."""
    from cdrseqgen.methods.common.evaluation import run_evaluation_pipeline

    return run_evaluation_pipeline(
        "mlm", args, config, run_config,
        prepare_evaluation=prepare_evaluation,
        evaluation_options=evaluation_options,
        evaluate=evaluate,
    )


__all__ = [
    "main",
    "prepare_evaluation", "evaluation_options", "evaluate", "resolve_decoding",
]
