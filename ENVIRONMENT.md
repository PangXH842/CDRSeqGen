# Environment setup and server migration

The migration baseline was captured on 2026-09-21 from Linux on the NVIDIA B200
server. Keep training/scoring (`cdrprj`) separate from the IgLM baseline (`iglm`).
These are the versions observed on this server, not a claim that other versions
are incompatible:

| Environment | Python | PyTorch | Transformers |
| --- | --- | --- | --- |
| cdrprj | 3.11 (capture: 3.11.16) | 2.11.0+cu128 | 5.17.0 |
| iglm | 3.11 (capture: 3.11.14) | 2.9.1 (CUDA 12.8 dependencies) | 4.57.6 |

`requirements/*.txt` preserves the captured Python package pins, with these
intentional corrections: retain only SymPy 1.14.0, add matplotlib 3.10.7 for
plotting, and explicitly select the cu128 PyTorch wheel for IgLM. Unused
`torchvision`, `torchaudio`, and `torchcodec` are omitted. This is a migration
baseline, not a platform-independent or fully hashed lockfile; dependencies
introduced by matplotlib are resolved during installation.

## Create fresh environments

Install Conda and put it on PATH, then run from the checkout:

```bash
bash env.sh             # both environments
# Or: bash env.sh cdrprj / bash env.sh iglm
bash scripts/verify_env.sh
conda run --no-capture-output -n cdrprj bash scripts/unit_tests.sh
```

The installer refuses existing environment names and stops on any installation
or dependency-check error. It does not upgrade this server's existing environments.
If installation fails partway through, inspect the error and the newly created
environment before removing it or retrying its pip install manually.

Conda supplies Python 3.11 and pip; pip installs the pinned application stack,
including PyTorch. The destination needs compatible Linux/CPU architecture and an
NVIDIA driver supporting the selected CUDA 12.8 wheels. Inspect `nvidia-smi` and
`uname -m` on the destination before installing. Do not reuse the former CUDA 12.1
Conda recipe for B200. For different hardware, validate a separate requirements
variant rather than silently upgrading this baseline. See the
[PyTorch Blackwell/CUDA 12.8 release notes](https://pytorch.org/blog/pytorch-2-7/).

The verifier runs `pip check`, exercises tensor forward/backward on every visible
GPU in both environments, imports the training and baseline dependencies, and
renders a headless plot. It requires CUDA by default. For a deliberate CPU-only
check use `bash scripts/verify_env.sh all --cpu`; this does not certify GPU readiness.
Checks use the repository root regardless of the shell's working directory.
They do not download models or prove real checkpoint/data compatibility: after
migration, also run a small saved-checkpoint evaluation and the validation smoke
benchmark documented in [benchmark/README.md](benchmark/README.md).

Use the existing unittest runner above rather than introducing pytest as a setup
dependency. For launchers, `scripts/run.sh` uses `$HOME/miniconda3` when available;
otherwise set `PYTHON=/absolute/path/to/env/bin/python`.

## Validation on the source server (2026-09-21)

Both requirements files installed successfully into fresh temporary Python 3.11
virtual environments, without inheriting the existing environments' packages.
Both passed `pip check` and tensor forward/backward on all eight visible B200 GPUs.
The training environment passed all 462 unit tests, training-module imports, and
headless plotting. The baseline environment passed the IgLM adapter import checks.
The Conda verification wrapper was also checked from outside the repository in
explicit CPU mode, and the installer correctly refused the existing env names.

The old `cdrprj` environment was left unchanged. Its unused torchvision install
breaks Transformers imports, and matplotlib is absent; `pip check` alone does not
catch these runtime issues. Use the clean setup for migration. Destination-server
checks and a real saved-model evaluation are still required after transfer.

## Transfer separately from Git

- The results archive, checksum, manifest, and file list. Verify the checksum before
  extraction into the checkout. The 20260921 bundle generally omits optimizer,
  scheduler, and RNG checkpoint state: it preserves saved models but does not
  generally support exact training continuation. See its manifest for exceptions.
- All of `data/processed/` (about 838 MB), including split/provenance metadata.
  Copy `data/raw/` (about 5.7 GB) as well to preserve preprocessing inputs.
- The cached model directories for `Exscientia/IgBert`, `Rostlab/prot_bert`, and
  `facebook/esm2_t12_35M_UR50D` (about 4.9 GB in the migration inventory).
  Preserve complete `blobs`, `refs`, `snapshots`, and symlinks. Copy only the
  relevant model directories under `~/.cache/huggingface/hub`, or set `HF_HOME`
  to the transferred cache root. See the [HF cache layout](https://huggingface.co/docs/hub/local-cache).
- The original environment exports in `CDRSeqGen_env_migration_20260921/` as
  provenance. The old from-history export records stale CUDA intent, and raw pip
  freezes include machine-local build paths; neither is the installation recipe.
- Any separate research scripts or outputs you want to retain. In particular,
  the sibling `IgLM/src/` directory contains untracked work even though the
  baseline itself imports the installed `iglm` package.

Git does not carry the ignored `dataset/data -> ../data` compatibility symlink.
For historical configs, recreate it in a fresh clone:

```bash
mkdir -p dataset
ln -s ../data dataset/data   # only if this path does not already exist
```

Historical configs and plans can contain absolute `/raid/...` paths. Preserve
original records, and use relocated copies/config overrides for execution.
Record the source Git commit and verify transferred data checksums before
retiring the old server. Authentication should be configured separately on the
new server; it is not part of the environment files.
