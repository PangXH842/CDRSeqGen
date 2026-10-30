"""S2S generation entrypoint called by root generate.py."""

from cdrseqgen.methods.common.generation import run_generation_pipeline
from cdrseqgen.methods.common.logging import configure_logging
from cdrseqgen.methods.s2s.generation import (
    generate_case,
    generation_state,
    prepare_generation,
)


def main(args, config):
    """Connect S2S setup and case production to the shared generation workflow."""
    configure_logging()
    return run_generation_pipeline(
        "s2s", args, config,
        prepare_generation=prepare_generation,
        generation_state=generation_state,
        generate_case=generate_case,
    )
