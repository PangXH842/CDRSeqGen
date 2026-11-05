"""Select a method and pass the training command to its CLI.

To run:
  * python train.py configs/mlm/heavy/h3_scratch.json       # MLM (the default)
  * python train.py configs/ar/h3_ar_scratch.json           # AR
  * python train.py configs/mlm/heavy/h3_scratch.json --learning_rate 3e-4 --num_train_epochs 20
  * python train.py --method ar --eval_split validation --do_train --do_eval

Functions:
  * resolve_method -- choose the method name and return a copy of the argument list
  * main           -- validate that name and call <method>.train_cli.main()

Workflow:
  1. Choose the method: explicit --method > config's method > default MLM.
  2. Keep all arguments, including --method, for the method's own parser.
  3. <method>.train_cli.main() -> logging + arguments.setup_args()
       -> common.training.run_training_pipeline()
       -> prepare_training -> train_and_save -> finish_training.

The method CLI and argument setup handle full parsing, validation and run setup.
Model/data preparation and Trainer execution happen in the training modules.
A method override still requires config fields accepted by the selected method.
"""
import json
import os
import sys

from cdrseqgen.methods.registry import METHODS, DEFAULT_METHOD, resolve


def resolve_method(argv):
    """Return (normalized method name, argv copy) without consuming any arguments.

    Selection order:
      --method ar / --method=ar -> last explicit occurrence wins
      train.py config.json     -> read its method field (config must be first)
      neither specifies one    -> DEFAULT_METHOD (mlm)

    argv includes the program name. Only the method name is normalized; flags
    remain unchanged so the method parser records the same override it executes.
    Registration is checked by main(); full config parsing belongs to setup_args().
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


# ---------- MAIN TRAINING CALLER ----------
def main():
    """Dispatch sys.argv to the selected method's training CLI."""
    # ------ 1. Resolve and validate the method before importing its CLI. ------
    method, argv = resolve_method(sys.argv)
    if method not in METHODS:
        raise SystemExit(f"Unknown method {method!r}. Choose one of: {', '.join(METHODS)}")

    # ------ 2. Pass the complete argument list to the method's parser. ------
    sys.argv = argv

    # ------ 3. Let the method CLI set up the run and call the shared pipeline. ------
    resolve(method).module('train_cli').main()


if __name__ == "__main__":
    main()
