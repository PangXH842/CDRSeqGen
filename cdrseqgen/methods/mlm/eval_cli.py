"""MLM evaluation entrypoint called by root evaluate.py."""

from cdrseqgen.methods.common.evaluation import run_evaluation_pipeline
from cdrseqgen.methods.common.logging import configure_logging
from cdrseqgen.methods.mlm.evaluation import (
    evaluation_options,
    prepare_evaluation,
    run_generative_eval as evaluate,
)


def main(args, config, run_config=None):
    """Connect MLM setup and scoring to the shared evaluation workflow."""
    configure_logging()
    return run_evaluation_pipeline(
        "mlm", args, config, run_config,
        prepare_evaluation=prepare_evaluation,
        evaluation_options=evaluation_options,
        evaluate=evaluate,
    )
