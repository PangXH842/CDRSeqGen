"""Autoregressive training entrypoint called by root train.py."""
import functools

from cdrseqgen.methods.ar.arguments import setup_args
from cdrseqgen.methods.ar.training import finish_training, prepare_training
from cdrseqgen.methods.common.logging import configure_logging
from cdrseqgen.methods.common.training import run_training_pipeline


def main():
    """Parse training arguments and connect AR hooks to the shared workflow.

    A plain ``Trainer`` (the model returns its own loss), no length head, and a decode-to-``<eos>``
    generative evaluation after training.
    """
    configure_logging()
    model_args, data_args, gen_args, training_args = setup_args()
    run_training_pipeline(
        model_args, data_args, gen_args, training_args,
        prepare_training=prepare_training,
        finish_training=functools.partial(finish_training, anchor_trim=model_args.anchor_trim),
    )
