"""Exercise dependencies and tensor execution without downloads or research data."""
import argparse
import importlib
from pathlib import Path
import sys
import tempfile

# Make imports independent of the launcher's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
    else:
        from iglm import IgLM
        from iglm.model.tokens import BAD_WORD_IDS
        from iglm.model.utils import iglm_to_infilled, mask_span, validate_generated_seq
        print("IgLM adapter dependencies OK")


if __name__ == "__main__":
    main()
