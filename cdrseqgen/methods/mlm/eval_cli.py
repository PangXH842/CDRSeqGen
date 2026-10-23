"""MLM evaluation entrypoint called by root evaluate.py.

Execution:
  evaluate.py -> main(args, config, run_config)
              -> common.evaluation.run_evaluation_pipeline()
              -> evaluation.run_generative_eval()

Functions supplied to the shared pipeline (defined in mlm.evaluation):
  * prepare_evaluation -- resolve/check args before resources are loaded;
                         calls resolve_decoding to choose the MLM decoder.
  * evaluation_options -- load the optional length head and return evaluator kwargs.
  * evaluate           -- alias for run_generative_eval; generate, score and report.

resolve_decoding is also re-exported for callers using setup stages separately.
Post-training evaluation calls evaluate directly with already loaded resources.
See mlm.evaluation's module docstring for the strategy and scoring call graph.
"""

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
