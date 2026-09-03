"""Causal run configuration: only what differs from the shared bases.

The 28 fields both methods declare identically live in
:mod:`cdrseqgen.methods.common.arguments`, along with config/CLI parsing and run-directory setup.

**What is deliberately absent**, because it has no causal meaning: ``masking_policy`` /
``mixed_full_span_probability`` (nothing is masked), the loss registry (§13 settled that
re-weighting core against anchor moves nothing), ``gen_decoding`` (one causal decoder, not an
independent/iterative choice), and the length policy plus ``train_length_predictor`` — the model
ends its own loop with ``<eos>``, which is the whole reason §10's gap-sentinel head has no
counterpart here.
"""
import logging
from dataclasses import dataclass, field
from typing import Tuple

from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.data.dataset_ar import CAUSAL_DECODE_STEPS, validate_causal_decode_steps
from cdrseqgen.methods.common.arguments import (BaseDataArguments, BaseGenerationArguments,
                                                BaseModelArguments, initialise_run,
                                                parse_config_and_cli,
                                                validate_common_args)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments(BaseModelArguments):
    method: str = field(default="ar", metadata={"help": "Selects the causal pipeline."})


@dataclass
class DataArguments(BaseDataArguments):
    """Causal training needs nothing the base does not already carry."""


@dataclass
class GenerationArguments(BaseGenerationArguments):
    gen_max_new_tokens: int = field(
        default=CAUSAL_DECODE_STEPS,
        metadata={"help": "Decode steps including <eos>. A draw that never emits <eos> within "
                          "the cap is incomplete and is dropped rather than scored as a loop."})


def validate_args(model_args, data_args, gen_args):
    """Validate causal-only limits in addition to the shared method contract."""
    validate_common_args(model_args, data_args, gen_args, expected_method="ar")
    validate_causal_decode_steps(gen_args.gen_max_new_tokens)


def setup_args() -> Tuple[ModelArguments, DataArguments, GenerationArguments, TrainingArguments]:
    """Parse config/CLI, validate, then create and log the run directory."""
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, gen_args, training_args = parse_config_and_cli(parser)

    validate_args(model_args, data_args, gen_args)

    initialise_run(model_args, data_args, gen_args, training_args, marker="ar")
    return model_args, data_args, gen_args, training_args
