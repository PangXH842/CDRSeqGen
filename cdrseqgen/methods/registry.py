"""Lazy method dispatch shared by training, generation, and held-out evaluation.

A method exposes train_cli, eval_cli and gen_cli for the three root commands.
Its backbones, training, evaluation and sampling modules implement the stages
those commands use. Importing this registry loads no model library.
"""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Method:
    name: str
    package: str
    decoders: tuple = ()        # decoder names --decoding accepts for this method

    def module(self, component):
        return import_module(f'{self.package}.{component}')

    def evaluate(self, *args, **kwargs):
        """Use the same evaluation interface as root evaluate.py, including after training."""
        return self.module('eval_cli').evaluate(*args, **kwargs)


METHODS = {
    'mlm': Method('mlm', 'cdrseqgen.methods.mlm', ('independent', 'iterative')),
    'ar': Method('ar', 'cdrseqgen.methods.ar', ('causal',)),
    's2s': Method('s2s', 'cdrseqgen.methods.s2s', ('seq2seq',)),
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
