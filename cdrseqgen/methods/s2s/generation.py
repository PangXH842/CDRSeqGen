"""Configure S2S generation and build scored records for supplied frameworks.

Entry path:
  generate.py -> gen_cli.main -> common.generation.run_generation_pipeline
              -> prepare_generation -> generation_state -> generate_case

Callbacks:
  * prepare_generation -- reject length controls (EOS chooses length) and fix the decode cap.
  * generation_state   -- S2S has no length head, so there is no state to load.
  * generate_case      -- sample, score and record one supplied framework.

Each case uses the same encoder input, sampler and self-scorer as evaluation; the record itself is
built by :func:`common.eos_decoding.eos_case_record`, shared with the causal method.
"""
from cdrseqgen.datasets.dataset_s2s import (
    S2S_DECODE_STEPS, build_encoder_input, validate_s2s_decode_steps,
)
from cdrseqgen.methods.common.eos_decoding import eos_case_record
from cdrseqgen.methods.s2s.arguments import validate_inference_args
from cdrseqgen.methods.s2s.evaluation import perplexities
from cdrseqgen.methods.s2s.sampling import sample_loops


def prepare_generation(args, cases, config):
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
    return None


def generate_case(model, tokenizer, device, args, generator, case, state, print_callback):
    """Generate, score and serialize one supplied framework; ``state`` is unused (no length head)."""
    return eos_case_record(model, tokenizer, device, args, generator, case, print_callback,
                           mode='seq2seq', build_context=build_encoder_input,
                           sample_loops=sample_loops, perplexities=perplexities)
