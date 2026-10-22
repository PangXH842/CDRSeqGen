"""MLM evaluation entrypoint called by root evaluate.py.

Execution:
  evaluate.py -> main(args, config, run_config) -> evaluation.run_evaluation()

The workflow loads resources through common.evaluation, then runs the MLM
candidate evaluator. The evaluate alias accepts already loaded resources for
post-training evaluation; setup helpers remain exported for existing callers.
"""

from cdrseqgen.methods.mlm.evaluation import (
    run_evaluation,
    evaluation_options,
    prepare_evaluation,
    resolve_decoding,
    run_generative_eval as evaluate,
)


def main(args, config, run_config=None):
    """Hand the standalone command to evaluation.run_evaluation()."""
    return run_evaluation(args, config, run_config)


__all__ = [
    "main",
    "prepare_evaluation", "evaluation_options", "evaluate", "resolve_decoding",
]
