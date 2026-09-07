"""Lazy method dispatch shared by training, generation, and held-out evaluation.

A method owns arguments, backbones, training, evaluation, and a CLI adapter. Importing this registry
loads no model library. Future methods can use different model inputs/tokenizers behind that boundary.
"""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Method:
    name: str
    package: str
    evaluator: str
    decoders: tuple = ()        # decoder names --decoding accepts for this method

    def module(self, component):
        return import_module(f'{self.package}.{component}')

    def evaluate(self, *args, **kwargs):
        return getattr(self.module('evaluation'), self.evaluator)(*args, **kwargs)


METHODS = {
    'mlm': Method('mlm', 'cdrseqgen.methods.mlm', 'run_generative_eval',
                  ('independent', 'iterative')),
    'ar': Method('ar', 'cdrseqgen.methods.ar', 'run_causal_generative_eval', ('causal',)),
    's2s': Method('s2s', 'cdrseqgen.methods.s2s', 'run_s2s_generative_eval', ('seq2seq',)),
}
DEFAULT_METHOD = 'mlm'


def decoding_choices():
    """Every decoder name --decoding accepts, across all methods, in registration order.

    generate.py and evaluate.py each used to hard-code this list, so a method added later was
    rejected by argparse before its own cli could ever validate the request -- an enumeration that
    has to be edited in two more places is one a new method silently falls out of.
    """
    return [decoder for method in METHODS.values() for decoder in method.decoders]


def resolve(name=DEFAULT_METHOD):
    name = str(name).strip().lower()
    if name not in METHODS:
        raise ValueError(f'Unknown method {name!r}. Choose one of: {", ".join(METHODS)}')
    return METHODS[name]


def checkpoint_method(config, requested=None):
    method = resolve(config.get('method', DEFAULT_METHOD))
    if requested is not None and resolve(requested).name != method.name:
        raise ValueError(f'Checkpoint method is {method.name!r}, requested {requested!r}; '
                         'a different method cannot reinterpret its weights.')
    return method
