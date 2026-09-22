"""Training entrypoint — dispatches to the method a config declares.

    python train.py configs/mlm/heavy/h3_scratch.json                 # masked-LM (the default)
    python train.py configs/ar/h3_ar_scratch.json              # autoregressive
    python train.py configs/mlm/heavy/h3_scratch.json --learning_rate 3e-4 --num_train_epochs 20
    python train.py --method ar --target_region h3          # bare CLI, no config file

The method comes from the config's ``"method"`` key, or a ``--method`` flag, and defaults to ``mlm``
so every pre-existing config keeps working untouched.

**This file only routes.** Each method owns its own parser and its own ``entrypoint.main()``, in
``cdrseqgen/methods/<method>/entrypoint.py``. That is the point: a single shared parser would have to
accept ``masking_policy`` and ``gen_length_mode`` from a causal run and ``gen_max_new_tokens`` from a
masked one, and silently ignore whichever half did not apply. The 28 arguments the methods DO share
are inherited from ``methods/common/arguments.py``, so they cannot drift — they already had, with
the same field called ``gen_judge_path`` on one side and ``gen_judge_weights`` on the other.

Trailing flags override the JSON rather than replacing it. Batch rounds go through scripts/run.sh,
which exports CDRSEQGEN_RESULTS_ROOT so every run of the round shares one batch folder.
"""
import json
import os
import sys

from cdrseqgen.methods.registry import METHODS, DEFAULT_METHOD, resolve


def resolve_method(argv):
    """Resolve the selected method while leaving its flag for the method-owned parser.

    A ``--method`` flag wins over the config, so a config can be run through the other pipeline
    deliberately; without it the config's own declaration decides, and a config that declares
    nothing is masked — which is every config written before 2026-09-01.
    """
    argv = list(argv)
    method_flags = []
    for index, value in enumerate(argv[1:], start=1):
        if value == "--method":
            if index + 1 >= len(argv):
                raise SystemExit("--method needs a value: " + " | ".join(METHODS))
            method_flags.append(argv[index + 1])
        elif value.startswith("--method="):
            method_flags.append(value.split("=", 1)[1])
    if method_flags:
        return method_flags[-1].strip().lower(), argv
    if len(argv) >= 2 and argv[1].endswith(".json") and os.path.exists(argv[1]):
        with open(argv[1]) as handle:
            declared = json.load(handle).get("method", DEFAULT_METHOD)
            return str(declared).strip().lower(), argv
    return DEFAULT_METHOD, argv


def main():
    method, argv = resolve_method(sys.argv)
    if method not in METHODS:
        raise SystemExit(f"Unknown method {method!r}. Choose one of: {', '.join(METHODS)}")
    # The method's own parser also consumes --method. Keeping it here makes a CLI override replace
    # the config value in run_config.json instead of routing one way while recording another.
    sys.argv = argv
    resolve(method).module('entrypoint').main()


if __name__ == "__main__":
    main()
