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

    def module(self, component):
        return import_module(f'{self.package}.{component}')

    def evaluate(self, *args, **kwargs):
        return getattr(self.module('evaluation'), self.evaluator)(*args, **kwargs)


METHODS = {
    'mlm': Method('mlm', 'cdrseqgen.methods.mlm', 'run_generative_eval'),
    'ar': Method('ar', 'cdrseqgen.methods.ar', 'run_causal_generative_eval'),
    's2s': Method('s2s', 'cdrseqgen.methods.s2s', 'run_s2s_generative_eval'),
}
DEFAULT_METHOD = 'mlm'


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
