"""MLM generation entrypoint called by root generate.py."""

from cdrseqgen.methods.mlm.generation import (
    generate_case,
    generation_state,
    prepare_generation,
)
from cdrseqgen.methods.mlm.length import resolve_case_length


def main(args, config):
    """Run generation through the shared lifecycle and MLM case producer."""
    from cdrseqgen.methods.common.generation import run_generation_pipeline

    return run_generation_pipeline(
        "mlm", args, config,
        prepare_generation=prepare_generation,
        generation_state=generation_state,
        generate_case=generate_case,
    )


__all__ = [
    "main",
    "prepare_generation", "generation_state", "generate_case", "resolve_case_length",
]
