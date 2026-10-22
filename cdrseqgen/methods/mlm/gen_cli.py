"""MLM generation entrypoint called by root generate.py.

Execution:
  generate.py -> main(args, config) -> generation.run_generation()

The workflow resolves policies, loads resources through common.generation and
produces scored case records. Lower-level generation helpers remain exported
for existing callers.
"""

from cdrseqgen.methods.mlm.generation import (
    run_generation,
    generate_case,
    generation_state,
    prepare_generation,
    resolve_case_length,
)


def main(args, config):
    """Hand the standalone command to generation.run_generation()."""
    return run_generation(args, config)


__all__ = [
    "main",
    "prepare_generation", "generation_state", "generate_case", "resolve_case_length",
]
