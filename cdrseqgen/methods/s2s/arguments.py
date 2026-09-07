"""Seq2seq run configuration: only what differs from the shared bases.

The 28 fields every method declares identically live in
:mod:`cdrseqgen.methods.common.arguments`, along with config/CLI parsing and run-directory setup.

**What is deliberately absent**, because it has no encoder-decoder meaning: ``masking_policy`` /
``mixed_full_span_probability`` (nothing is masked out of a supervised stream), the loss registry
(§13 settled that re-weighting core against anchor moves nothing), ``gen_decoding`` (one decoder,
not an independent/iterative choice), and the length policy plus ``train_length_predictor`` — the
decoder ends its own loop with ``<eos>``, so §10's gap-sentinel head has no counterpart, exactly as
on the causal side.

**What is added**: separate encoder and decoder depths. ``num_layers`` alone would make an `s2s`
run and an `ar` run with the same config look comparable while the former carries 2.3x the
parameters (800k vs 1.86M at the shipped defaults), and Findings §9 records what one un-audited
hyperparameter cost across four findings at once.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.data.dataset_s2s import S2S_DECODE_STEPS, validate_s2s_decode_steps
from cdrseqgen.methods.common.arguments import (BaseDataArguments, BaseGenerationArguments,
                                                BaseModelArguments, initialise_run,
                                                parse_config_and_cli,
                                                validate_common_args)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments(BaseModelArguments):
    method: str = field(default="s2s", metadata={"help": "Selects the encoder-decoder pipeline."})
    num_encoder_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Encoder depth; None = num_layers. Set both this and num_decoder_layers "
                          "to compare against an `ar` run at matched parameter count."})
    num_decoder_layers: Optional[int] = field(
        default=None, metadata={"help": "Decoder depth; None = num_layers."})


@dataclass
class DataArguments(BaseDataArguments):
    """Encoder-decoder training needs nothing the base does not already carry.

    Note ``max_len`` BUYS more here than on the causal side at the same value: it caps the encoder
    stream alone, which holds only framework and context, while the causal prompt must additionally
    reserve 96 positions for its answer. At max_len 128 that is 128 tokens of context here against
    31 there. The shipped configs still pin 384, matching the causal arm numerically — neither
    truncates a heavy chain plus partner at that cap, and an identical number removes the question
    from any comparison table. It is a ceiling, not a padding target, so the extra headroom is free.
    """


@dataclass
class GenerationArguments(BaseGenerationArguments):
    gen_max_new_tokens: int = field(
        default=S2S_DECODE_STEPS,
        metadata={"help": "Decode steps including <eos>. A draw that never emits <eos> within "
                          "the cap is incomplete and is dropped rather than scored as a loop."})


def validate_args(model_args, data_args, gen_args):
    """Validate seq2seq-only limits in addition to the shared method contract."""
    validate_common_args(model_args, data_args, gen_args, expected_method="s2s")
    validate_s2s_decode_steps(gen_args.gen_max_new_tokens)
    for name in ("num_encoder_layers", "num_decoder_layers"):
        value = getattr(model_args, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer or unset (got {value!r})")


def setup_args() -> Tuple[ModelArguments, DataArguments, GenerationArguments, TrainingArguments]:
    """Parse config/CLI, validate, then create and log the run directory."""
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, gen_args, training_args = parse_config_and_cli(parser)

    validate_args(model_args, data_args, gen_args)

    initialise_run(model_args, data_args, gen_args, training_args, marker="s2s")
    return model_args, data_args, gen_args, training_args
