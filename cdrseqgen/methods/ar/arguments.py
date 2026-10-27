"""Causal run configuration: only what differs from the shared bases.

The fields every method declares identically live in
:mod:`cdrseqgen.methods.common.arguments`, along with config/CLI parsing and run-directory setup.

**What is deliberately absent**, because it has no causal meaning: ``masking_policy`` /
``mixed_full_span_probability`` (nothing is masked), the loss registry (§13 settled that
re-weighting core against anchor moves nothing), ``gen_decoding`` (one causal decoder, not an
independent/iterative choice), and the length policy plus ``train_length_predictor`` — the model
ends its own loop with ``<eos>``, which is the whole reason §10's gap-sentinel head has no
counterpart here.
"""
from dataclasses import dataclass, field
from typing import Tuple

from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.datasets.dataset_ar import CAUSAL_DECODE_STEPS, validate_causal_decode_steps
from cdrseqgen.methods.ar.backbones import resolve as resolve_backbone
from cdrseqgen.methods.common.arguments import (BaseDataArguments, BaseGenerationArguments,
                                                BaseModelArguments, initialize_run,
                                                parse_config_and_cli,
                                                validate_attention_heads,
                                                validate_common_args,
                                                validate_early_stopping)


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


def validate_inference_args(args):
    """Validate saved-checkpoint decoder/length options and set args.decoding.

    Called by eval_cli.prepare_evaluation and gen_cli.prepare_generation.
    """
    if args.decoding not in (None, 'causal'):
        raise ValueError('AR checkpoints require causal decoding')
    if args.min_length is not None or args.max_length is not None:
        raise ValueError('AR chooses length via EOS; bounded/fixed length is unsupported')
    if args.length_temperature != 1.0:
        raise ValueError('AR has no separate length-temperature control')
    args.decoding = 'causal'


def validate_args(model_args, data_args, gen_args):
    """Validate the decode cap, backbone and attention heads on top of the shared method contract."""
    validate_common_args(model_args, data_args, gen_args, expected_method="ar")
    validate_causal_decode_steps(gen_args.gen_max_new_tokens)
    resolve_backbone(model_args.model_type)
    if model_args.model_type == "scratch":  # a pretrained backbone brings its own width and heads
        validate_attention_heads(model_args, even_head_width=True)


def setup_args() -> Tuple[ModelArguments, DataArguments, GenerationArguments, TrainingArguments]:
    """Parse config/CLI, validate, then create and log the run directory."""
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, gen_args, training_args = parse_config_and_cli(parser)

    validate_args(model_args, data_args, gen_args)
    validate_early_stopping(model_args, training_args)

    initialize_run(model_args, data_args, gen_args, training_args, marker="ar")
    return model_args, data_args, gen_args, training_args
