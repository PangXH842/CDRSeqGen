"""Register the three methods and import a method's modules only when a caller asks for one.

Callers:
  * train.py, evaluate.py, generate.py -- pick the method, then call its train_cli, eval_cli
    or gen_cli
  * fit_mlm_length.py -- require an MLM checkpoint
  * benchmark/run.py, benchmark/adapters/cdrseqgen.py -- read a checkpoint's method, backbone
    and sampler
  * cdrseqgen/experiments.py -- normalize each spec's method name
  * methods/common/evaluation.py, methods/common/generation.py -- load the method's backbones,
    training and arguments modules; evaluation also calls Method.evaluate after training
  * methods/common/arguments.py, benchmark/adapters/cdrseqgen.py -- the --decoding choices
  * methods/mlm/arguments.py -- the MLM decoder names
  * visualization/batch_tables.py -- recognise a method prefix in run names

Functions:
  * Method.module     -- import one component of the method: train_cli, backbones, sampling, ...
  * Method.evaluate   -- evaluate loaded resources through eval_cli, including after training
  * METHODS           -- the registered methods: mlm, ar, s2s (DEFAULT_METHOD is mlm)
  * decoding_choices  -- every decoder name any method accepts, in registration order
  * resolve           -- return a method by name; an unknown name raises ValueError
  * checkpoint_method -- return the method a run_config names; refuse a different requested one

Notes:
  * Importing this module loads no model library; Method.module imports a component on first use.
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
        """Evaluate already loaded resources, including after training."""
        return self.module('eval_cli').evaluate(*args, **kwargs)


METHODS = {
    'mlm': Method('mlm', 'cdrseqgen.methods.mlm', ('independent', 'iterative')),
    'ar': Method('ar', 'cdrseqgen.methods.ar', ('causal',)),
    's2s': Method('s2s', 'cdrseqgen.methods.s2s', ('seq2seq',)),
}
DEFAULT_METHOD = 'mlm'


def decoding_choices():
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
