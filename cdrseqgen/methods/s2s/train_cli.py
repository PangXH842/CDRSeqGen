"""Encoder-decoder training entrypoint called by root train.py."""
import functools

from cdrseqgen.methods.common.logging import configure_logging
from cdrseqgen.methods.common.training import run_training_pipeline
from cdrseqgen.methods.s2s.arguments import setup_args
from cdrseqgen.methods.s2s.training import finish_training, prepare_training


def main():
    """Parse training arguments and connect S2S hooks to the shared workflow."""
    configure_logging()
    model_args, data_args, gen_args, training_args = setup_args()
    run_training_pipeline(
        model_args, data_args, gen_args, training_args,
        prepare_training=prepare_training,
        finish_training=functools.partial(finish_training, anchor_trim=model_args.anchor_trim),
    )
