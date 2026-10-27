"""MLM training entrypoint called by root train.py."""

from cdrseqgen.methods.common.logging import configure_logging
from cdrseqgen.methods.common.training import run_training_pipeline
from cdrseqgen.methods.mlm.arguments import setup_args
from cdrseqgen.methods.mlm.training import finish_training, prepare_training


def main():
    """Parse training arguments and connect MLM hooks to the shared workflow."""
    configure_logging()
    model_args, data_args, gen_args, training_args = setup_args()
    run_training_pipeline(
        model_args, data_args, gen_args, training_args,
        prepare_training=prepare_training,
        finish_training=finish_training,
    )
