"""Training entrypoint — dispatches to the method a config declares.

    python train.py configs/h3_scratch.json                 # masked-LM (the default)
    python train.py configs/h3_ar_scratch.json              # autoregressive
    python train.py configs/h3_scratch.json --learning_rate 3e-4 --num_train_epochs 20
    python train.py --method ar --target_region h3          # bare CLI, no config file

The method comes from the config's ``"method"`` key, or a ``--method`` flag, and defaults to ``mlm``
so every pre-existing config keeps working untouched.

**This file only routes.** Each method owns its own parser and its own ``entrypoint.main()``, in
``cdrseqgen/methods/<method>/entrypoint.py``. That is the point: a single shared parser would have to
accept ``masking_policy`` and ``gen_length_mode`` from a causal run and ``gen_max_new_tokens`` from a
masked one, and silently ignore whichever half did not apply. The 25 arguments the methods DO share
are inherited from ``methods/common/arguments.py``, so they cannot drift — they already had, with
the same field called ``gen_judge_path`` on one side and ``gen_judge_weights`` on the other.

Trailing flags override the JSON rather than replacing it. Batch rounds go through scripts/run.sh,
which exports CDRSEQGEN_RESULTS_ROOT so every run of the round shares one batch folder.
"""
import json
import os
import sys

METHODS = {
    "mlm": "cdrseqgen.methods.mlm.entrypoint",
    "ar": "cdrseqgen.methods.ar.entrypoint",
}
DEFAULT_METHOD = "mlm"


def resolve_method(argv):
    """(method, argv-without-the-method-flag).

    A ``--method`` flag wins over the config, so a config can be run through the other pipeline
    deliberately; without it the config's own declaration decides, and a config that declares
    nothing is masked — which is every config written before 2026-09-01.
    """
    argv = list(argv)
    if "--method" in argv:
        index = argv.index("--method")
        if index + 1 >= len(argv):
            raise SystemExit("--method needs a value: " + " | ".join(METHODS))
        method = argv[index + 1]
        del argv[index:index + 2]
        return method, argv
    if len(argv) >= 2 and argv[1].endswith(".json") and os.path.exists(argv[1]):
        with open(argv[1]) as handle:
            return json.load(handle).get("method", DEFAULT_METHOD), argv
    return DEFAULT_METHOD, argv


def main():
    method, argv = resolve_method(sys.argv)
    if method not in METHODS:
        raise SystemExit(f"Unknown method {method!r}. Choose one of: {', '.join(METHODS)}")
    # The method's own setup_args reads sys.argv, so hand it the argv with --method removed.
    sys.argv = argv
    import importlib
    importlib.import_module(METHODS[method]).main()


if __name__ == "__main__":
    main()
