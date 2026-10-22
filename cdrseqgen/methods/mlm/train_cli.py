"""MLM training command: parse arguments and hand off to training.run_training()."""

from cdrseqgen.methods.mlm.arguments import setup_args
from cdrseqgen.methods.mlm.training import run_training
from cdrseqgen.methods.common.logging import configure_logging


def main():
    configure_logging()
    model_args, data_args, gen_args, training_args = setup_args()
    run_training(model_args, data_args, gen_args, training_args)
