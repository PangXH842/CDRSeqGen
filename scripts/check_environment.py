"""Exercise dependencies and tensor execution without downloads or research data."""
import argparse
import importlib
from pathlib import Path
import sys
import tempfile

# Make imports independent of the launcher's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_training_logging():
    """Construct the configured logging callback and verify a readable scalar event."""
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from transformers import Trainer, TrainingArguments
    from transformers.integrations import TensorBoardCallback

    with tempfile.TemporaryDirectory() as directory:
        args = TrainingArguments(output_dir=directory, use_cpu=True,
                                 report_to="tensorboard", disable_tqdm=True)
        # No datasets or train() call: only exercise Trainer initialization and logging.
        trainer = Trainer(model=torch.nn.Linear(1, 1), args=args)
        callbacks = [callback for callback in trainer.callback_handler.callbacks
                     if isinstance(callback, TensorBoardCallback)]
        if len(callbacks) != 1:
            raise RuntimeError("Trainer did not configure exactly one TensorBoard callback")
        callback = callbacks[0]
        # Keep smoke events local even if the caller set TENSORBOARD_LOGGING_DIR.
        callback.logging_dir = str(Path(directory) / "tensorboard")
        try:
            callback.on_train_begin(args, trainer.state, trainer.control, model=trainer.model)
            callback.on_log(args, trainer.state, trainer.control, logs={"loss": 0.5})
            callback.tb_writer.flush()
            events = EventAccumulator(callback.logging_dir).Reload()
            values = events.Scalars("train/loss")
            if len(values) != 1 or values[0].value != 0.5:
                raise RuntimeError("TensorBoard scalar write/read failed")
        finally:
            callback.on_train_end(args, trainer.state, trainer.control)
    print("Trainer initialization and TensorBoard scalar write/read OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=("cdrprj", "iglm"))
    parser.add_argument("--cpu", action="store_true", help="Skip GPU validation explicitly")
    args = parser.parse_args()

    import torch
    import transformers

    print(f"{args.environment}: Python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"transformers {transformers.__version__}, CUDA build {torch.version.cuda}")
    if args.cpu:
        devices = [torch.device("cpu")]
        print("CPU-only check: GPU readiness is NOT validated.")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --cpu only for intentional CPU validation")
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    for device in devices:
        x = torch.ones((32, 32), device=device, requires_grad=True)
        loss = (x @ x).mean()
        loss.backward()
        if loss.item() != 32 or not torch.isfinite(x.grad).all().item():
            raise RuntimeError(f"Tensor forward/backward failed on {device}")
        print(f"Tensor forward/backward OK: {device}")

    if args.environment == "cdrprj":
        for module in ("accelerate", "pandas", "sklearn", "sentencepiece", "safetensors",
                       "cdrseqgen.methods.mlm.training", "cdrseqgen.methods.ar.training",
                       "cdrseqgen.methods.s2s.training", "cdrseqgen.visualization.sweep_plots",
                       "benchmark.plot_comparison"):
            importlib.import_module(module)
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            fig.savefig(Path(directory) / "smoke.png")
            plt.close(fig)
        print("Training imports and headless plotting OK")
        check_training_logging()
    else:
        from iglm import IgLM
        from iglm.model.tokens import BAD_WORD_IDS
        from iglm.model.utils import iglm_to_infilled, mask_span, validate_generated_seq
        print("IgLM adapter dependencies OK")


if __name__ == "__main__":
    main()
