"""Generate S2S CDR candidates and case records for supplied frameworks.

Entry path:
  generate.py -> gen_cli.main -> common.generation.run_generation_pipeline
              -> prepare_generation -> generation_state -> generate_case

Functions:
  * prepare_generation -- validate EOS options and resolve the decode cap
  * generation_state   -- return None; no separate method state needs loading
  * generate_case      -- sample one case, calculate measurements and return its record

Notes:
  * gen_cli owns logging; common.generation loads cases/model/tokenizer, seeds
    sampling and writes the output. This module supplies the three method hooks.
  * eos_case_record shares case reporting with AR. S2S supplies its encoder input,
    autoregressive decoder sampler and self-scorer.
  * EOS selects length. Native truth is optional and used only for measurements.
  * Dataset selection, judge scoring and aggregation across evaluation frameworks
    belong to the evaluation workflow.
"""
from cdrseqgen.datasets.dataset_s2s import (
    S2S_DECODE_STEPS, build_encoder_input, validate_s2s_decode_steps,
)
from cdrseqgen.methods.common.eos_decoding import eos_case_record
from cdrseqgen.methods.s2s.arguments import validate_inference_args
from cdrseqgen.methods.s2s.evaluation import perplexities
from cdrseqgen.methods.s2s.sampling import sample_loops


def prepare_generation(args, cases, config):
    """Validate S2S generation, resolve settings in args and return None.

    Steps:
      inference checks -> reject case/CLI length controls -> resolve/check decode cap

    max_new_tokens comes from args, then saved gen_max_new_tokens, then the S2S
    default. EOS selects the actual loop length. No model loading or sampling
    happens here.
    """
    validate_inference_args(args)
    for case in cases:
        for key in ('min_length', 'max_length', 'target_length', 'use_native_length'):
            value = case.get(key, getattr(args, key, None))
            if value is not None and value is not False:
                raise ValueError(f'S2S chooses length via EOS; {key} is unsupported')
    args.max_new_tokens = getattr(args, 'max_new_tokens', None)
    if args.max_new_tokens is None:
        args.max_new_tokens = config.get('gen_max_new_tokens', S2S_DECODE_STEPS)
    validate_s2s_decode_steps(args.max_new_tokens)
    return None


def generation_state(args, policies, device):
    """Return None; S2S needs no separately loaded length head or method state.

    The shared pipeline calls this after loading the model and passes its return
    value to generate_case(). The arguments follow the same hook contract as MLM.
    """
    return None


def generate_case(model, tokenizer, device, args, generator, case, state, print_callback):
    """Return candidates, measurements and attempt accounting for one supplied case.

    eos_case_record() coordinates:
      build_encoder_input -> sample_loops -> optional generator perplexities
                          -> sequence metrics -> print_callback -> case record

    state is unused. Native truth/length stays out of the generation framework;
    truth supports reference measurements after sampling. Failed attempts remain
    in accounting; completed nonempty candidates are retained with duplicates.
    """
    return eos_case_record(model, tokenizer, device, args, generator, case, print_callback,
                           mode='seq2seq', build_context=build_encoder_input,
                           sample_loops=sample_loops, perplexities=perplexities)
