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
plotting and TensorBoard 2.20.0 with its dependency pins for configured training
logging, and explicitly select the cu128 PyTorch wheel for IgLM. Unused
`torchvision`, `torchaudio`, and `torchcodec` are omitted. This is a migration
baseline, not a platform-independent or fully hashed lockfile; dependencies
introduced by matplotlib are resolved during installation. TensorBoard and its
added dependencies were verified on the destination server on 2026-10-06.

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
renders a headless plot. It also constructs a CPU Trainer with TensorBoard reporting
and verifies a scalar event can be written and read, without calling `train()`.
It requires CUDA by default. For a deliberate CPU-only
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

## Destination-server restoration (2026-10-06)

The transferred pretrained cache is complete at `/mnt/hdd/pang/hf-cache/hub`.
The default lookup path, `/home/pang/.cache/huggingface/hub`, now points there
through a symlink. Existing commands can load models by their Hub IDs without
setting `HF_HOME`. No model downloads or package changes were needed.

| Model | Verified cached revision |
| --- | --- |
| `Exscientia/IgBert` | `81a4abb8a4dac5cdcbefc28e5b331a82224c485d` |
| `Rostlab/prot_bert` | `7a894481acdc12202f0a415dd567f6cfdb698908` |
| `facebook/esm2_t12_35M_UR50D` | `6fbf070e65b0b7291e7bbcd451118c216cff79d8` |

Cache symlinks and weight blob SHA-256 hashes passed verification. All five
processed datasets also match their preprocessing metadata hashes. Both
environments passed `pip check`, imports and GPU forward/backward on physical
GPU 0 (RTX PRO 6000 Blackwell Max-Q); `cdrprj` also passed headless plotting.
GPU 1 was occupied and was excluded from these checks.

Offline model checks passed through the real project loaders:

- ESM-2 and ProtBert preserve canonical residue tokenization and produce finite
  masked-LM logits on GPU.
- The saved scratch S2S checkpoint generated six valid candidates from three
  OAS4 validation frameworks, with finite IgBert scores and full judge coverage.
  Its input contract matched without `--force`.
- The packaged IgLM checkpoint generated two valid natural-EOS candidates from
  one validation framework through the benchmark adapter, with no failures.
- The experiment launcher activated `cdrprj` and planned all six `compare`
  trials in dry-run mode without launching training.

These are execution checks, not comparative scientific results. Evidence,
model/data hashes and logs are saved in
[`results/misc/restoration_20261006/report.json`](results/misc/restoration_20261006/report.json).
Historical run configs, checkpoints and evaluation outputs were preserved.

To repeat the environment check using the same GPU selection:

```bash
PATH=/home/pang/miniconda3/bin:$PATH CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  bash scripts/verify_env.sh all
```

The saved-checkpoint evaluation used:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  /home/pang/miniconda3/envs/cdrprj/bin/python evaluate.py \
  --config configs/s2s/h3_s2s_scratch.json \
  --model_dir results/runs/20260908_173143_compare/h3_OAS4_s2s_scratch \
  --num_examples 3 --num_samples 2 --temperature 1 --top_k 0 --top_p 0.9 \
  --seed 42 --judge Exscientia/IgBert \
  --partner_context without --judge_partner_context with
```

Other historical configs may still contain absolute `/raid/...` dataset paths.
Use relocated execution copies with verified dataset hashes; retain the original
records and record any compatibility override needed for a deliberate relocation.

### Training-logging correction (2026-10-06)

The first no-partner AR trials failed during Trainer initialization because the
base configs request `report_to: tensorboard`, but the migration requirements
omitted TensorBoard. Neither AR trial reached training. The earlier restoration
checks imported training modules and evaluated saved models, so they did not
exercise this callback setup.

Installed TensorBoard 2.20.0 and its missing dependencies in `cdrprj`, retaining
the existing PyTorch, Transformers, Accelerate and Safetensors versions. The
requirements now pin these additions, and the verifier includes the real Trainer
callback and scalar-event check above. The updated CPU verifier and `pip check`
passed after installation. The failed batch can be retried using its
original `EXPERIMENT_DIR`: no executable project source or experiment config was
changed by this correction. The updated requirements and verification script do
not affect the runner's source fingerprint.
