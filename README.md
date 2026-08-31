<!-- Note to AI: Edits allowed OUTSIDE of MANUAL PROTECTION ZONES and INSIDE of AI ACCESS ZONES -->
# CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** 2026-08-31
**Status:** Matched length-free CDRSeqGen–IgLM validation scored — **IgLM leads on every quality axis** (Findings §11). Paired confidence intervals and a clean-tree rescore pending. The core-weighted loss screen has since concluded: **no re-weighting beats plain cross-entropy** (Findings §13), closing the last sequence-side lever before Phase 2

## Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences given the surrounding framework context. 

Current implemented features include: 
- **Data Preprocessing**: standardize data into Bio-Object JSON format (IMGT), current datasets: OAS (21106 paired sequences) and SAbDab2 (3984 paired + antigen structures)
- **Addressing Data Leakage**: clustered OAS splitting with optional train/validation/test partitions that keep clonal near-duplicates together
- **Task-Agnostic Masking**: masking training method for all 6 CDR loops individually by target selection
- **Pluggable Backbone PLMs**: supports various PLMs, current list: custom transformer (scratch), ESM-2, IgBert, ProtBert
- **Partner Chain Context**: optional inclusion of non-target chain sequence context
- **Antigen Context**: optional inclusion of antigen sequence context for SAbDab2 dataset
- **Context Dropout**: `context_dropout` trains one checkpoint that works with *and* without its optional context (partner chain / antigen drawn independently), instead of a model locked to the regime it was trained in
- **Generative Sampling**: independent / iterative (Mask-Predict) decoding with temperature (`0` = greedy), top-k, top-p — samples a *library* of candidate loops instead of one argmax guess
- **Explicit Length Policies**: fixed, bounded, and inferred-length MLM generation; inferred mode classifies one gap sentinel and expands it into residue masks without reading the native loop length
- **Evaluation Metrics**: *recovery-side* — Sequence Recovery Rate (Per-Residue Accuracy), anchor-excluded core accuracy, anchor accuracy. *Design-side* — diversity, novelty, per-position entropy, pseudo-perplexity, plus judge-scored perplexity for cross-model comparison
- **Matched External Benchmark**: versioned manifests, model-specific CDRSeqGen/IgLM adapters, shared contract validation and scoring, and optional independent-judge perplexity
- **Logging & Visualization**: per-run `train.log` (the whole run's record, including the generative-eval table and any uncaught traceback), TensorBoard logging, csv metric extraction, multi-line graph plotter
- **Extensible Architecture**: method packages isolate masked-LM assumptions so future autoregressive or encoder–decoder implementations can reuse shared data, metrics, artifacts, and benchmark contracts

---

## 🏗️ Repository Structure

```text
CDRSeqGen/
├── configs/                    # 48 run configs (flat; run.sh references them by path)
│   ├── h{1,2,3}_scratch[.json|_lc|_ag|_ag_lconly] + h{1,2,3}_esm2[...]
│   ├── h{1,2,3}_igbert[.json|_lc] + h{1,2,3}_protbert[.json|_lc]
│   └── l{1,2,3}_scratch[.json|_hc] + l{1,2,3}_esm2[.json|_hc]
├── cdrseqgen/                  # Library package — grouped by concern (entrypoints stay at root)
│   ├── artifacts.py            # Shared names for persisted run artifacts and legacy aliases
│   ├── data/                   # ALL data code (the data itself lives in dataset/data/)
│   │   ├── dataset.py          # Runtime: processed JSON → tensors (tokenizers, Dataset, collator)
│   │   └── preprocess/         # Offline: raw → processed JSON (run once per dataset)
│   │       ├── oas_paired.py       # OAS paired repertoires + the leakage-safe clustered split
│   │       ├── sabdab2.py          # SAbDab2 structures (adds antigen; split ships with the data)
│   │       ├── common.py           # Shared by both: validation, chain assembly, funnel, IO
│   │       ├── download_oas_paired.sh  # Fetch OAS paired repertoires → dataset/data/raw/
│   │       └── download_sabdab2.sh     # Fetch SAbDab2 splits → dataset/data/raw/ (CWD-independent)
│   ├── methods/                # Complete generation methods; no cross-method mask assumptions
│   │   └── mlm/                # Masked-LM training, generation, and native evaluation
│   │       ├── arguments.py        # MLM model/data/generation arguments + run setup
│   │       ├── backbones.py        # Masked-LM registry — resolve(model_type) -> Backbone
│   │       ├── losses.py           # Shared loss registry + Trainer bridge for every MLM backbone
│   │       ├── training.py         # Datasets, Trainer lifecycle, post-training evaluation
│   │       ├── recovery.py         # Position recovery, anchor/core split, core coverage
│   │       ├── sampling.py         # Independent/iterative decoding + pseudo-perplexity
│   │       ├── evaluation.py       # Fixed-native generative evaluation
│   │       ├── variable_length_evaluation.py # Bounded/inferred held-out evaluation
│   │       ├── generation.py       # Length selection, mask expansion, decoder dispatch
│   │       ├── case_generation.py  # One framework -> candidates and metrics
│   │       ├── length.py           # Policies, constraints, evaluation-mode validation
│   │       ├── length_predictor.py # Gap-sentinel classifier + checkpoint artifacts
│   │       └── models/
│   │           └── transformer.py  # From-scratch masked encoder — model_type "scratch"
│   ├── train/                  # Method-neutral run logging and artifact reporting
│   │   ├── logging.py          # Logging, run dirs/latest, complete run_config.json
│   │   └── record_results.py   # epoch_metrics.csv + training-curve PNGs
│   ├── eval/
│   │   └── metrics.py          # Architecture-neutral sequence/design metrics
│   └── visualization/          # Run artifacts -> charts and tables. No GPU, no model loading.
│       ├── plot_style.py           # Shared palette, axes furniture, empty panels, saving
│       ├── linegraph_plotter.py    # Chart type: lines with optional spread bands (+ CLI)
│       ├── bargraph_plotter.py     # Chart type: grouped bars with error bars
│       ├── scatterplot_plotter.py  # Chart type: operating-point scatter
│       ├── sweep_plots.py          # Figure: sweep settings x seeds, four panels
│       ├── batch_plots.py          # Figure: the run matrix, one dashboard per dataset
│       ├── confidence_plot.py      # Figure: does per-candidate confidence rank by recovery?
│       ├── batch_tables.py         # Generates validated Markdown comparison tables from a batch
│       └── sweep_tables.py         # Ranks a sweep's settings across seeds -> summary.tsv
├── benchmark/                  # Frozen-manifest comparisons against external baselines
│   ├── adapters/               # CDRSeqGen and IgLM generation under one output contract
│   ├── manifests/              # Versioned validation/test inputs plus SHA-256 sidecars
│   ├── _common.py              # Hashing, seeding, atomic writes, git state, resume checks
│   ├── export_manifest.py      # Processed OAS split -> model-independent manifest
│   ├── scorer.py               # Shared validation, metrics, accounting, and judge scoring
│   ├── plot_comparison.py      # Matched recovery, length, position, and diversity dashboard
│   └── README.md               # End-to-end benchmark commands and protocol
├── dataset/data/               # THE DATA — gitignored, ~4.4GB (raw + processed OAS/SAbDab2).
│                               #   Deliberately outside the package; configs' data_path points here.
├── results/                    # Ignored in git. Three tiers, by cost to regenerate:
│   ├── runs/                   #   hours of GPU — <timestamp>[_name]/<run>/, latest -> newest
│   │                           #   A round reports itself when it finishes: records at the batch
│   │                           #   root (batch_tables.md, or runs.tsv/summary.tsv for a sweep),
│   │                           #   charts under <batch>/figures/, per-trial stdout under logs/.
│   ├── benchmark/              #   GPU minutes — <benchmark_id>/{generations,scores}/
│   └── figures/                #   seconds, disposable — hand-picked charts for the README and
│                               #   ad-hoc cross-batch comparisons, which belong to no one batch
├── test/                       # Unit tests plus curated qualitative JSON cases for generate.py
│   ├── test_*.py
│   └── test_{h1,h3,h3_lc,h3_ag}.json
├── misc/
│   └── CDRSeqGen260814.drawio.png  # Project workflow diagram
├── train.py                    # Entrypoint — masked-LM training
├── generate.py                 # Entrypoint — candidate generation on supplied frameworks
├── evaluate.py                 # Entrypoint — held-out generative evaluation (+ --judge)
├── fit_mlm_length.py           # Add MLM length artifacts to an older checkpoint
├── scripts/
│   ├── run.sh                  # Unified runner: ./scripts/run.sh <gpu> [oas|antigen|lcdr|igbert|protbert|plm|all ...]
│   ├── sweep.sh                # Unified sweep: h3 | pilot | arch | loss | quick presets
│   └── test.sh                 # Unit test runner (sets PYTHONPATH; no GPU or data needed)
├── env.sh
└── README.md
```

---

## ⚙️ Environment Setup
This project uses Conda for strict CUDA handling and pip for Hugging Face tools.

1. Run the setup script:
   ```bash
   chmod +x env.sh
   ./env.sh
   ```
2. Activate the environment:
   ```bash
   conda activate cdrprj
   ```
3. Run the tests to check the environment:
   ```bash
   ./scripts/test.sh                      # whole suite, a few seconds
   ./scripts/test.sh test_length_policy   # one module
   ```
   The suite is `unittest`-based and stubs every model, so it needs no GPU, no downloaded weights,
   and no processed dataset. `PYTHONPATH=.` is required (there is no `test/__init__.py`), which is
   why the runner exists rather than a bare `python -m unittest`.

---

## 🚀 Workflow & Execution

### Architecture Overview

![CDRSeqGen data, training, evaluation, analysis, and benchmark workflow](misc/CDRSeqGen260814.drawio.png)

> The diagram is the 2026-08-14 snapshot. It predates the `cdrseqgen/methods/mlm/` consolidation and
> the `generate.py` / `evaluate.py` / `fit_mlm_length.py` renames; a refreshed diagram is pending.

### Step 1: Preprocessing Data
Convert datasets into a custom "Bio-Object" JSON format, extracting the regions and preserving `ANARCI` numbering.

**OAS**: currently only including 21106 paired sequences from 1287169_1_Paired_All.csv, pending future scale-ups
```bash
python -m cdrseqgen.data.preprocess.oas_paired \
    --input_file dataset/data/raw/1287169_1_Paired_All.csv \
    --output_file dataset/data/processed/processed_oas_data.json
```

For hyperparameter tuning, create a separate three-way file. Test clusters are allocated first using
the historical shuffle, so the established test membership is preserved; validation is then taken
from the former training partition:
```bash
python -m cdrseqgen.data.preprocess.oas_paired \
    --input_file dataset/data/raw/1287169_1_Paired_All.csv \
    --output_file dataset/data/processed/processed_oas_tuning.json \
    --split cluster --test_size 0.1 --validation_size 0.1 --seed 42
```

**SAbDab2**: currently including 3984 structures - paired with resolved antigen, remaining >10k structures without antigen pending future inclusion
```bash
python -m cdrseqgen.data.preprocess.sabdab2 \
    --input_file dataset/data/raw/sabdab2/splits_final/abag_split.csv \
    --output_file dataset/data/processed/processed_sabdab2_data.json
```

Processed records preserve provenance needed by external baselines and future multi-source datasets:
`species`, both original IDs in `chain_sequence_ids`, and a stable `source` block (OAS run or
SAbDab/PDB identifiers). OAS species is read from its raw metadata row; the current SAbDab2 source
has an empty species column, so those records deliberately store `"species": null` rather than
assuming human. Each preprocessing run also writes a hash-verified `*.metadata.json` sidecar with
source details, filters, numbering convention, and split parameters.

### Step 2: Training & Batch Experimentation
We use `.json` configuration files to manage hyperparameters cleanly. The trainer calculates **Sequence Recovery Rate** (Per-Residue Accuracy), logs to TensorBoard, and auto-generates metric graphs at the end of every run.

> **⚠️ Every config must declare `eval_split`.** The split named here drives early stopping,
> `load_best_model_at_end`, and every reported `eval_*` number, so leaving it implicit means
> selecting checkpoints on whichever partition happens to be non-train — which, on a two-way file,
> is the final test set. `get_datasets` now **refuses** to start when `eval_split` is unset and the
> dataset carries a `test` split. The 36 OAS configs declare
> `processed_oas_tuning.json` + `"eval_split": "validation"`; the 12 SAbDab2 configs declare
> `"eval_split": "test"` because that dataset ships no validation partition (an open problem, not
> an endorsement). Use `test` only after the settings are frozen.

**Run a Single Experiment:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py configs/h3_scratch.json
```

**Run an Automated Experiment Suite:** [`scripts/run.sh`](scripts/run.sh) takes a GPU id and one or more suite names, skips to the next config if one fails, and writes the round's tables and dashboards into the batch folder when it finishes (Step 3).
```bash
# One runner, suites chosen (and combined) by arg:  ./scripts/run.sh <gpu> [suite ...]
./scripts/run.sh 4                # default "all": OAS heavy + SAbDab2 antigen + OAS light (36)
./scripts/run.sh 4 oas            # OAS heavy matrix, scratch+ESM-2 (h1/h2/h3)          (12)
./scripts/run.sh 4 antigen        # SAbDab2 heavy antigen matrix (_ag / _ag_lconly)     (12)
./scripts/run.sh 4 lcdr           # OAS light matrix (l1/l2/l3, heavy-chain context)    (12)
./scripts/run.sh 4 oas lcdr       # combine suites — deduped, run in order              (24)

# Antibody/protein-BERT PLM backbones on OAS heavy (first run downloads a ~1.6GB model each)
./scripts/run.sh 4 igbert         # IgBert (antibody-specific)                           (6)
./scripts/run.sh 4 protbert       # ProtBert (general-protein baseline)                  (6)
./scripts/run.sh 4 plm            # both (igbert + protbert)                            (12)
```

**Every run lives at `results/runs/<batch>/<run>/` — always two levels under `runs/`.** A `run.sh` round shares one batch
folder; a standalone `train.py` gets a batch folder of its own, so `results/` never collects loose run
dirs. **`results/runs/latest`** is a symlink to the most recent *batch*, so `results/runs/latest/<config>/`
addresses any run in that round:

```text
results/runs/
├── 20260720_191138/                    ← one ./scripts/run.sh round
│   ├── h1_OAS_scratch/
│   ├── h3_OAS_LC_scratch/
│   ├── batch_tables.md                 ← the round's own report, written when it finishes
│   └── figures/                        ← batch_oas.png, batch_sd2.png
├── 20260720_193002/                    ← a single train.py run
│   └── h3_OAS_scratch/
├── 20260811_181946_sweep_h3/           ← validation-safe sweep; results/runs/latest unchanged
│   ├── h3_OAS_scratch_lr1e-4_wu0.0_wd0.0_s42/
│   ├── h3_OAS_scratch_lr1e-3_wu0.06_wd0.01_s43/
│   ├── runs.tsv  summary.tsv  grid.tsv ← the records you read
│   ├── figures/                        ← objective_comparison.png
│   └── logs/                           ← one stdout/stderr capture per trial
└── latest -> 20260720_193002/          ← newest batch; generate.py uses latest/<config>
```

Everything a round produces about *itself* sits at its own root: records as files, charts under
`figures/`, per-trial stdout under `logs/`. The logs are kept rather than dropped because a trial
that dies before logging attaches leaves no `train.log` — sometimes no run directory at all — so
that capture is the only evidence of the failure; they just no longer outnumber the records.

A named group is `<timestamp>_<name>` — the timestamp leads so directories sort chronologically
whether or not they carry a name (`20260730_160028_seedcheck`, `20260811_181946_sweep_h3`).

The run folder is named from the **resolved arguments**, not the config filename — `h3_scratch.json
--target_region l3` produces an `l3_OAS_HC_scratch` folder, so an override can't be misfiled. The
date/time lives on the batch folder and isn't repeated inside.

**Tag a run** with `CDRSEQGEN_RUN_SUFFIX` to distinguish runs that resolve to the same name:
```bash
CDRSEQGEN_RUN_SUFFIX=ablation_noLC python train.py configs/h3_scratch.json
#  -> results/<batch>/h3_OAS_scratch_ablation_noLC/
```
`sweep.sh` tags every trial with its LR, warm-up, weight decay, epochs and seed, so folders are
self-describing. It groups trials with `CDRSEQGEN_RESULTS_ROOT`, writes `runs.tsv`/`summary.tsv`, and
sets `CDRSEQGEN_SKIP_LATEST=1` so exploratory trials do not replace the main usable batch. If two runs
still collide, the second becomes `..._2` with a logged warning; a run never silently overwrites another.

A standalone `python train.py …` sets neither environment variable and retains the normal behavior.
Export `CDRSEQGEN_RESULTS_ROOT` yourself to group manual runs:
```bash
CDRSEQGEN_RESULTS_ROOT=results/runs/my_experiment python train.py configs/h3_scratch.json
```

**Override a config from the CLI:** trailing flags override the JSON (they do **not** replace it), so any
`TrainingArguments` field is tunable without writing a new config:
```bash
python train.py configs/h3_scratch.json --learning_rate 3e-4 --num_train_epochs 20 --warmup_ratio 0.06
```

The original `masking_policy: "full_span"` remains the default: every native CDR residue is replaced
by its own mask token. Opt into mixed-mask training when training a checkpoint for iterative
decoding:

```bash
python train.py configs/h3_scratch.json \
    --masking_policy mixed --mixed_full_span_probability 0.5
```

Mixed mode keeps the requested fraction of training examples fully masked. For every other example,
it reveals at least one native loop residue and masks at least one, with loss computed only over the
masked subset. Validation, post-training generation, and standalone evaluation always use a separate
`full_span` dataset view, so partially revealed native residues cannot leak into reported recovery.
Both settings are persisted in `run_config.json`.

**Choose a training loss:** ordinary token-averaged cross-entropy remains the default and therefore
reproduces old configs. The shared loss registry in
[`cdrseqgen/methods/mlm/losses.py`](cdrseqgen/methods/mlm/losses.py) is applied after model forward,
so scratch, ESM-2, IgBert, and ProtBert use the same implementation. The first alternative raises
the contribution of the hypervariable core relative to the easier loop anchors:

```bash
python train.py configs/h3_scratch.json \
    --loss_type weighted_cross_entropy \
    --core_loss_weight 2.0 --anchor_loss_weight 1.0 --anchor_trim 2
```

or, equivalently, from a config: [`configs/h3_scratch_weighted.json`](configs/h3_scratch_weighted.json)
is `h3_scratch.json` plus the three loss keys and nothing else, so the two differ only in objective.

The weighted objective is a weighted mean over supervised tokens, so only the core-to-anchor weight
ratio matters. `anchor_trim: 2` means the first and last two residues are anchors; loops of length
four or less contain no core and are treated entirely as anchors — on H3 that is 1 of 21,106 loops,
but on L2 it is 98.8% of them, so the weighted objective is an H3/H2/H1 tool and degenerates to plain
cross-entropy on the short light-chain loops. Setting `anchor_loss_weight 0` supervises the core only
(both weights zero is refused, as is a negative weight); a batch whose supervised tokens are all
anchors under that setting raises rather than emitting NaN. The same trim is passed to the
reported `core_accuracy`/`anchor_accuracy`, and original loop offsets keep the classification correct
under mixed masking. Loss settings and any `label_smoothing_factor` are persisted in
`run_config.json`.

**Tune on validation:** [`scripts/sweep.sh`](scripts/sweep.sh) runs a validation-only,
multi-seed screen. It takes a GPU id and a preset, the same way `run.sh` takes suite names:

| preset | grid | trials |
|---|---|---:|
| `h3` (default) | 4 LR × 2 warm-up × 2 weight decay × 2 seeds, fixed 30-epoch horizon | 32 |
| `pilot` | 3 LR × 2 epoch budgets — the 2026-07 "is the budget enough?" grid | 6 |
| `arch` | 3 widths × 2 depths, FFN tracking width | 6 |
| `loss` | 5 objective settings × 3 seeds, generative eval on, ranked by judge perplexity | 15 |
| `quick` | 2 trials at 2 epochs — checks the script runs, not for results |  2 |

The `loss` grid is `cross_entropy` plus four weighted settings (core/anchor `1.0/1.0`, `1.0/0.0`,
`3.0/1.0`, `3.0/0.0`). Weight combinations are deduplicated against `cross_entropy`, which ignores
them, so the nominal 24 grid points collapse to the 15 trials the preset runs.

> **⚠️ Only 9 of those 15 are distinct objectives** — the dedup catches weights under
> `cross_entropy` but not weights that are equivalent to each other. The loss normalises by total
> weight, so only the core-to-anchor *ratio* survives: `weighted 1.0/1.0` is bit-identical to
> `cross_entropy`, and `3.0/0.0` to `1.0/0.0`. Six of the fifteen trials therefore retrain a model
> the grid has already trained. **Keeping one such pair is worth it deliberately** — it measures
> run-to-run nondeterminism, which turns out to exceed the seed spread (Findings §13) — but the
> other three are waste. Canonicalising on the ratio is a contained change to the grid materialiser.

```bash
./scripts/sweep.sh 4                      # h3 preset
./scripts/sweep.sh 4 pilot
SEEDS="42 43 44" LRS="3e-4 1e-3" ./scripts/sweep.sh 4
SWEEP_DIR=results/runs/my_sweep ./scripts/sweep.sh 4   # resume into the same directory
./scripts/sweep.sh 4 loss
DRY_RUN=1 ./scripts/sweep.sh 4 loss       # print the grid, train nothing
```

Every dimension (`LRS`, `WARMUPS`, `WEIGHT_DECAYS`, `EPOCH_LIST`, `SEEDS`, `HIDDEN_SIZES`,
`NUM_LAYERS`, `MAX_EPOCHS`, `PATIENCE`, `LOSS_TYPES`, `CORE_WEIGHTS`, `ANCHOR_WEIGHTS`) is
env-overridable, so a preset only supplies defaults. Three knobs control what each trial is scored on:
`RANK_METRIC` (`core` | `whole` | `judge_ppl` | `diversity`), `GEN_EVAL` (`true`/`false`), and
`GEN_EXAMPLES`/`GEN_SAMPLES`, which shrink the generative eval — it runs once per trial, so the
150×20 default would dominate a 15-trial sweep. The `loss` preset uses 100×10.

Rank by `judge_ppl` rather than `core` when comparing objectives: a loss that reweights core tokens
changes the very quantity `core` measures, so ranking on it partly rewards the objective for
optimising its own metric. The independent judge does not share that bias.

The script refuses a dataset without explicit `train`/`validation`/`test` labels, forces
`--eval_split validation`, ranks settings by mean best-epoch validation core recovery across seeds,
writes `runs.tsv`, `summary.tsv` and `objective_comparison.png`, never selects the test partition, skips already-completed grid
points on resume, and sets `CDRSEQGEN_SKIP_LATEST=1` so exploratory trials do not replace the main
usable batch. It also flags a winner whose margin falls below the ~0.006 seed-noise floor as a tie.

**On the `pilot` preset.** It sweeps `num_train_epochs`, which also reshapes the linear scheduler —
so a "20 epochs is better" result from it confounds budget with schedule. That is why `h3` fixes the
horizon and does not sweep the budget. `pilot` answers a different question (is the budget itself
enough?) and should not be used to select hyperparameters.

**Completed screen (2026-08-11).** All 32 trials completed on the three-way clustered OAS split
(16 settings × seeds 42/43). The leading region was shallow rather than decisive:

| LR | warm-up | weight decay | mean validation core recovery | seed SD | mean best epoch |
|---:|---:|---:|---:|---:|---:|
| **1e-3** | **0.06** | **0.01** | **0.32305** | 0.00138 | 16.5 |
| 3e-4 | 0.06 | 0.01 | 0.32269 | 0.00070 | 18.5 |
| 3e-4 | 0.00 | 0.00 | 0.32238 | 0.00032 | 13.5 |

The top-two gap is only 0.00036 and changes sign between the two seeds, so this screen identifies a
region, not a conclusive winner. Confirm the top two with additional shared seeds before freezing a
training config; evaluate the final choice only once on `test`. Full local records are in
`results/runs/20260811_181946_sweep_h3/{runs,summary}.tsv`.

### Step 3: Result Visualization & Comparison
All three recovery metrics are recorded **every epoch** (`compute_metrics` → `epoch_metrics.csv`:
`eval_accuracy`, `eval_core_accuracy`, `eval_anchor_accuracy`), and each run auto-generates four PNGs:

| file | shows |
|---|---|
| `eval_loss_curve.png` | loss |
| `eval_accuracy_curve.png` | whole-span recovery (anchor-inflated) |
| `eval_core_accuracy_curve.png` | **core** recovery — the one to report |
| `recovery_breakdown.png` | all three together — the anchor gap at a glance (H3: ~0.46 / ~0.32 / ~0.90) |

Use the **breakdown** chart to see the anchor gap, and the **individual** charts to read a trend: the
breakdown's y-axis spans ~0.3–0.9, which compresses within-line movement (ESM-2's core decay from
0.321 → 0.306 is legible on `eval_core_accuracy_curve.png`, barely visible on the breakdown).

#### A finished round reports itself
What each run does for itself, a batch now does for the whole round — the report lands in the **batch
folder**, beside the runs it describes, so nothing has to be regenerated by hand to read a result.
Records sit at the batch root and charts under `figures/`, the same split `results/` uses one level up:

| written by | at the end of | into |
|---|---|---|
| `batch_tables.py` | `run.sh` | `<batch>/batch_tables.md` — every README comparison table |
| `batch_plots.py` | `run.sh` | `<batch>/figures/batch_<ds>.png` — one dashboard per dataset |
| `sweep_tables.py` | `sweep.sh` | `<sweep>/summary.tsv` — settings ranked across seeds |
| `sweep_plots.py` | `sweep.sh` | `<sweep>/figures/objective_comparison.png` — settings × seeds |

Both steps are guarded in the shell: a batch that already trained cannot be spoiled by a reporting
failure, and any of the four can be re-run against a finished directory at any time. That matters for
`sweep_tables` in particular — it used to be a heredoc at the end of `sweep.sh`, so an interrupted
sweep produced no summary at all, even though every finished trial was already on disk.

`results/figures/` stays for what does **not** belong to one batch: hand-picked charts for this
README, and ad-hoc comparisons spanning several batches (the two CLIs below still default there).

One function draws both comparison shapes, because every line is the same triple `(run_dir, column, label)`.
Each `--run` takes `LABEL FOLDER [METRIC]`; omit `METRIC` and it falls back to the global `--metric`
(default `eval_core_accuracy`):

```bash
# same metric across runs — metric named once
python -m cdrseqgen.visualization.linegraph_plotter \
  --metric eval_core_accuracy \
  --run "H3 Scratch" "results/20260729_202635/h3_OAS_scratch" \
  --run "H3 ESM-2"   "results/20260729_202635/h3_OAS_esm2_t12_35M_UR50D" \
  --title "H3 Core Recovery: Scratch vs ESM-2"

# breakdown of ONE run — per-run metric, several columns on one chart
python -m cdrseqgen.visualization.linegraph_plotter \
  --run whole  "results/runs/latest/h3_OAS_scratch" eval_accuracy \
  --run core   "results/runs/latest/h3_OAS_scratch" eval_core_accuracy \
  --run anchor "results/runs/latest/h3_OAS_scratch" eval_anchor_accuracy \
  --ylabel Recovery --title "H3 whole vs core vs anchor"
```

A missing CSV or column is reported and skipped rather than aborting the chart, so one bad path in a
multi-run comparison doesn't lose the other lines. Charts default to `results/figures/training/`.

<!-- > **⚠️ Never mix datasets on one chart.** OAS (`_OAS`) and SAbDab2 (`_SD2`) runs are evaluated on *different test sets*, so their accuracies are not comparable on shared axes — plot one graph per dataset (e.g. `cdr3_oas` with the 4 OAS runs, `cdr3_sd2` with the 4 SAbDab2 runs). The `results/comparisons/*_oas_*.png` / `*_sd2_*.png` pairs follow this convention. -->

### Step 4: Qualitative Sampling on Curated Frameworks (`generate.py`)
Give it a framework with its CDR loop cut out (`pt1` / gap / `pt2`) and it fills the gap: samples a
library of candidate loops (default **iterative**) and prints them, each with recovery-vs-native (when
you provide the native) and pseudo-perplexity, plus per-framework diversity/novelty. It also reports
which CDR the checkpoint was trained for (read from `run_config.json`). Samples + metrics are written to
`<model_path>/generation_results.json`.

> **This is a showcase, not an evaluation.** `test/*.json` holds only a handful of curated frameworks
> (mostly without natives). For numbers you can report, use **Step 5**, which draws real held-out
> frameworks straight from the eval split.

> **⚠️ Match the checkpoint's context, or the model silently underperforms.** A checkpoint trained with
> partner-chain or antigen context does not *fail* on a bare framework — the input is merely
> out-of-distribution, so it quietly produces bad loops. Measured on `h1_OAS_LC_scratch`, 60 held-out
> frameworks, greedy: **0.9858** recovery with the light chain supplied as trained, **0.2599** without.
> In the latest batch **24 of 36 checkpoints** need context.
>
> Pass it with `--partner_chain` / `--antigen` (or per-case `partner_chain` / `antigen` keys in a
> `--test_file`), and set `--max_len` to the training config's value (512 for the SAbDab2 antigen runs).
> `generate.py` reads `run_config.json`, prints what the checkpoint expects, and **refuses** a context
> mismatch in either direction unless `--force` — supplying context a model never saw is as damaging
> as withholding context it needs (0.308 → **0.000** on `h3_OAS_scratch` with a partner chain
> prepended), because for an absolute-position backbone it shifts every learned position. Injection
> goes through the same `inject_context` / `assemble_masked_input` that build the training examples,
> so which side the antigen lands on — and which side gets truncated — follows the backbone's
> `position_type` automatically instead of being hand-encoded in the test file.
>
> The **loop** is a weaker signal and is treated accordingly. Nothing in the tensor names a loop or a
> chain, so which CDR you are designing is fixed by which checkpoint you load; `--target_loop` only
> *declares* your intent and cannot change the computation, so a mismatch **warns** rather than
> refuses — the input is still well-formed, just answered by weights fitted for another loop, which
> is a legitimate thing to probe. Because that check is opt-in, one more runs unconditionally: the
> length of `pt1` says where along the chain the gap sits, and across all 50,180 frameworks in both
> processed datasets the three CDR numbers separate cleanly (CDR1 8–26, CDR2 33–57, CDR3 78–106).
> That catches an h3 checkpoint fed an H1 framework with nothing declared, which was previously
> silent. Only the **number** is inferable — H1 (23–25) and L1 (24–26) overlap outright — so a
> heavy/light mix-up still cannot be detected this way.

```bash
# design coherent loops for a batch of curated frameworks (default: iterative sampling)
# results/runs/latest is the newest batch — name the config under it (here: h3_OAS_scratch)
python generate.py --model_path ./results/runs/latest/h3_OAS_scratch --test_file test/test_h3.json --use_native_length

# more candidates + nucleus sampling
python generate.py --model_path ./results/runs/latest/h3_OAS_scratch --test_file test/test_h3.json \
    --use_native_length --decoding iterative --num_samples 50 --top_p 0.9

# a single framework straight from the CLI (equal bounds mean an exact length)
python generate.py --model_path ./results/runs/latest/h3_OAS_scratch --pt1 EVQL...YYCAR --pt2 WGQGT... \
    --min_length 17 --max_length 17

# constrain each sampled length to an inclusive range (one-sided bounds also work)
python generate.py --model_path ./results/runs/latest/h3_OAS_scratch --pt1 EVQL...YYCAR --pt2 WGQGT... \
    --min_length 11 --max_length 15 --num_samples 50

# infer length from the framework: omit target_length/min_length/max_length
python generate.py --model_path ./results/runs/latest/h3_OAS_scratch --pt1 EVQL...YYCAR --pt2 WGQGT... \
    --num_samples 50
```

Length selection passes through one explicit policy layer. Equal bounds request an exact fixed
length; unequal or one-sided bounds truncate and renormalize the checkpoint's learned length
distribution; omitted bounds sample that distribution without consulting the native target. A
`target_truth` is scoring-only unless `--use_native_length` is passed explicitly, preventing silent
length leakage. `--target_length` remains as a deprecated fixed-length alias. Every result stores the
policy, bounds, constrained probabilities, selected lengths, and prompt truncation under
`length_policy`.

The sequence backbone is still an encoder-only MLM: one **gap sentinel** is classified into a length,
then expanded into that many residue masks before the existing decoder runs. New training runs fit
and save `length_config.json` plus `length_head.safetensors` beside the best sequence checkpoint by
default; the sequence weights remain frozen during this fit. For a checkpoint trained before these
artifacts existed, run:

```bash
python fit_mlm_length.py --model_dir results/path/to/old_run
```

This discovers the run's saved `run_config.json`, keeping the original processed dataset and split.
Use `--config path/to/config.json` only when the checkpoint has no saved run configuration.
Inspect the resulting `length_config.json` before generation: its held-out `eval_accuracy` and
`eval_length_mae` should improve on `eval_mode_baseline_accuracy` and
`eval_mode_baseline_length_mae`; otherwise the head is mostly reproducing the training-set mode.

> **First result (2026-08-17): the mechanism works, but this length head does not yet beat the naive
> baseline.** On the full 2,111-framework validation split, the tuned scratch checkpoint's head
> reached 9.24% exact-length accuracy and 3.64 MAE, versus 10.52% and 3.28 for always predicting the
> training mode (length 15). See Finding §10; this is an open modeling result, not a software failure.

`--decoding`: `independent` (per-position sampling, 1 forward pass) · `iterative` (confidence-ordered
Mask-Predict — conditions each residue on those already committed). **`--temperature 0` makes either
deterministic (greedy)** and returns a single loop. Warp with `--temperature` / `--top_k` / `--top_p`;
`--skip_perplexity` drops the compute-heavy scoring.

> **⚠️ The two decoders are currently equivalent — use `independent` for sweeps.** Measured 2026-07-28
> on `h3_OAS_esm2`, 150 stratified frameworks, 20 samples, `top_p 0.9`, seed 42, IgBert judge:
>
> | metric | iterative | independent | Δ |
> |---|---|---|---|
> | recovery | 0.4057 | 0.4072 | +0.0015 |
> | diversity | 0.5731 | 0.5703 | −0.0028 |
> | entropy (bits) | 1.8414 | 1.8242 | −0.0171 |
> | judge-ppl | 7.1434 | 7.2259 | +0.0825 |
>
> Every difference is within noise, while iterative costs **~15×** the forward passes (one per masked
> position; mean H3 length 15.1). Mask-Predict only helps when the loop joint differs from the product
> of per-position marginals — getting identical numbers says this masked-LM has learned near-independent
> marginals for H3. That is a limit of the **training objective**, not of the sampler, and it is direct
> evidence for the "marginal-based, not learned-joint" open problem below.

### Step 5: Generative Evaluation on the Held-Out Split (`evaluate.py`)
The reportable version of Step 4. Reconstructs the **exact training-time eval split** from the config
(same clustered leakage-safe holdout, same partner/antigen injection, a real native on *every*
framework — 2,111 held-out antibodies for OAS), samples `--num_samples` candidates each, and aggregates
the generative metrics — overall **and bucketed by H3 length**, since difficulty scales with it.

```bash
python evaluate.py \
    --config configs/h3_esm2.json \
    --model_dir results/20260710/20260710_184547_h3_OAS_esm2_t12_35M_UR50D \
    --num_examples 150 --stratify --num_samples 20 \
    --decoding iterative --top_p 0.9 \
    --judge Exscientia/IgBert
```

The command above is the original `fixed_native` track. Use the checkpoint length head for a bounded
or fully inferred sweep:

```bash
# inclusive range; either min or max may be omitted
python evaluate.py --config configs/h3_esm2.json --model_dir results/path/to/run \
    --length_mode bounded --min_length 11 --max_length 15 --num_samples 20

# no native mask count is supplied to generation
python evaluate.py --config configs/h3_esm2.json --model_dir results/path/to/run \
    --length_mode inferred --num_samples 20
```

Positional AAR/core/anchor recovery is calculated only for candidates whose generated length matches
the native loop. All valid candidates still contribute to length-match rate, length MAE,
alignment-aware edit similarity, diversity, validity, runtime, and judge perplexity.

> **⚠️ Self-perplexity is NOT comparable across models.** Pseudo-perplexity is scored under each model's
> *own* distribution, so it measures self-consistency — a model confident in its own outputs looks good
> regardless of quality. To compare models, pass `--judge <model_type>`: every candidate is then also
> scored by one **independent** reference LM (IgBert is a good pick — antibody-specific, and independent
> of both scratch and ESM-2). Diversity/entropy carry no such caveat; they're purely sequence-based.

`--num_examples K` subsamples the split (`-1` = all; default 200); `--stratify` balances the draw across
short/medium/long H3 buckets so rare long loops aren't washed out. Results land in
`<model_dir>/generation_eval.json`.

#### Generative evaluation runs automatically after every training run
Once all epochs finish, `train.py` fits the frozen-backbone length head, samples designs from the
finished best model, and writes `generation_eval.json` beside `epoch_metrics.csv`. Both the length
fit and fixed-native generative evaluation are **on by default** — no config change is needed.

It samples from **`trainer.model`** (with `load_best_model_at_end` that's the peak checkpoint, already
on the GPU, so nothing is reloaded from disk), runs last, and is exception-guarded: a generation
failure logs a warning and **cannot** damage a training run that already finished and saved.

Every candidate is also scored by an **independent judge** (`gen_judge`, default
`Exscientia/IgBert`), which is what makes perplexity comparable *across* runs — self-perplexity is
not. Comparability requires the same judge in every run, **not** the same process, so per-run
judging is valid. Tune or disable via config:

```json
"run_generative_eval": true,
"gen_num_examples": 150,
"gen_num_samples": 20,
"gen_decoding": "iterative",
"gen_top_p": 0.9,
"gen_length_mode": "fixed_native",
"gen_min_length": null,
"gen_max_length": null,
"gen_length_temperature": 1.0
```

Set `run_generative_eval: false` (or `gen_judge: ""` to keep the metrics but skip the judge) to opt out;
trim `gen_num_examples` for cheap runs. Cost is ~80s of sampling + ~15s to load the judge per run.

Both evaluation paths share the MLM evaluation implementation
([`cdrseqgen/methods/mlm/evaluation.py`](cdrseqgen/methods/mlm/evaluation.py)); they differ only in
where the model comes from. **Both length tracks also share one loop**: `run_eval_loop` owns example
selection, scoring, judging, reporting, and writing, while a *strategy* owns the one thing that must
not be shared — turning an eval index into candidates. `FixedNativeStrategy` reads the dataset's own
masked view (so the mask count is the native one and the context is exactly what training used);
`VariableLengthStrategy` ([`variable_length_evaluation.py`](cdrseqgen/methods/mlm/variable_length_evaluation.py))
rebuilds the framework and lets the checkpoint's length head decide. Before this, the two tracks were
separate 130-line functions that had already drifted apart in four ways (progress bar, write guard,
summary statistic, empty-mask guard) — differences that make two tracks quietly incomparable.

Length policy, checkpoint-specific classification, mask expansion, and independent-versus-iterative
decoder dispatch live in [`cdrseqgen/methods/mlm/`](cdrseqgen/methods/mlm/). Future autoregressive
and encoder-decoder methods can be added beside `mlm/` without inheriting its mask-token,
pseudo-perplexity, or length-expansion assumptions. The MLM primitives are shared by `generate.py`,
both held-out evaluation paths, and the CDRSeqGen benchmark adapter — all of which now reach the
decoder through the single `sample_length_conditioned_spans` entry point, so a fixed length is
simply the case where the length is known before the prompt is built rather than predicted from it.
`evaluate.py` stays the tool of choice for
scoring checkpoints you already trained, or re-sweeping temperature / top-p / judge without
retraining.

### Matched External Benchmark

The frozen-manifest CDRSeqGen–IgLM validation and final-test workflow is documented in
[`benchmark/README.md`](benchmark/README.md). Generation remains model-specific in the two adapters;
[`benchmark/scorer.py`](benchmark/scorer.py) validates the matched contract and applies one shared
metric implementation. Validation-smoke results are development checks, not final benchmark claims.

Current length-free status (2026-08-17): the tuned scratch adapter completed all 25 validation
frameworks × 20 candidates with zero failures and a 6.0% sampled native-length match rate. IgLM's
matched natural-EOS output, shared scoring, and the comparison plot are still pending; therefore no
length-free CDRSeqGen-versus-IgLM claim exists yet.

---

## 📊 Findings

Phase 1 (the sequence sweep) is a **leakage-safe benchmark** for CDR infilling. What it establishes:

*Numbers below are from the complete 36-run batch `results/20260729_202635` (6 loops × {scratch, ESM-2} ×
context × {OAS, SAbDab2}), read at **best epoch** under the coverage-aware recovery policy below.
Every generated table states the selected metric and minimum core coverage. Budgets are per family —
20 epochs for scratch, 10 for pretrained, both with `early_stopping_patience: 3`.*

> **⚠️ These numbers supersede the 07-24 batch, which trained every family for exactly 10 epochs.**
> That budget under-trained the scratch model, and it inflated the magnitude of **four** findings at
> once — §1, §2, §4 and §7 — all in the same direction, against the smaller model. See §9.

> **Every table here is generated, not hand-written** — regenerate them all after any new batch with:
> ```bash
> python -m cdrseqgen.visualization.batch_tables --batch results/20260729_202635
> python -m cdrseqgen.visualization.batch_tables --only regimes,loops   # or a subset
> ```
> It reads only what a finished run already writes (`epoch_metrics.csv`, `generation_eval.json`,
> `run_config.json`) plus the processed dataset JSONs, and prints markdown ready to paste. The one
> exception is the decoding table (Step 4), which needs two `evaluate.py` runs differing only in
> `--decoding`, passed via `--decoding-json` (the pair used above is kept alongside that run as
> `generation_eval_decoding_{iterative,independent}.json`).

**1. "Model choice doesn't matter" is only true in two of three regimes.** On OAS H3 a 128-dim from-scratch transformer matches ESM-2 (0.3148 vs 0.3209 core). But that generalises badly — whether pretraining pays depends entirely on how much of the test set is *memorisable* from train (see §6):

| case | unseen in train | metric | min core coverage | scratch | ESM-2 | Δ |
|---|---|---|---:|---:|---:|---:|
| OAS H1 | 1.6% | core | 100.0% | 0.9930 | 0.9935 | **+0.000** |
| SD2 H1 | 57.5% | core | 100.0% | 0.5473 | 0.6664 | **+0.119** |
| SD2 H2 | 79.6% | core | 100.0% | 0.3672 | 0.6083 | **+0.241** |
| OAS H3 | 100.0% | core | 100.0% | 0.3185 | 0.3209 | **+0.002** |

Pretraining buys ~nothing when the answer can be looked up, **+0.12–0.24** when the task needs genuine generalisation, and ~nothing again at the H3 ceiling. The original claim was drawn from regimes 1 and 3 only.

**2. CDR-H3 is the only hard loop — but the other five are easier than they look.** Best-epoch recovery on OAS under the recorded core-coverage policy:

| CDR | reported metric | min core coverage | scratch | +ctx | ESM-2 | +ctx |
|---|---|---:|---:|---:|---:|---:|
| H1 | core | 100.0% | 0.9930 | 0.9893 | 0.9935 | 0.9933 |
| H2 | core | 100.0% | 0.9897 | 0.9802 | 0.9899 | 0.9903 |
| **H3** | core | 100.0% | **0.3185** | 0.3139 | **0.3209** | **0.3243** |
| L1 | core | 100.0% | 0.9925 | 0.9841 | 0.9938 | 0.9937 |
| L2 | whole | 1.4% | 0.9927 | 0.9819 | 0.9932 | 0.9927 |
| L3 | core | 99.9% | 0.9261 | 0.9019 | 0.9274 | 0.9262 |

The biology is right — heavy CDR3 gets D-gene + junctional diversity, the rest is germline-templated — but the ~0.99s are **not** evidence the model generalises; under the current split they are largely retrieval (§6). Only H3 is measured on entirely unseen loops.

**3. The standard metric misleads (anchor inflation).** Whole-span `eval_accuracy` (~0.46 on H3) is dominated by the 2-3 V/J-gene-determined anchor residues at each loop end (~0.90 recovery, any model); the hypervariable middle actually being "designed" is ~0.32. The gap is **H3-specific** (H1/H2/L core ≈ whole-span). Measured via `core_accuracy`/`anchor_accuracy` ([`compute_metrics`](cdrseqgen/methods/mlm/recovery.py), passed to the Trainer by `train.py`); the position-resolved breakdown below is from the standalone re-scorer `eval_recovery.py` (archived in the gitignored `old/`, so it is not in a fresh clone). Illustrative (`h3_esm2`):

| Position from start | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 100% | 89% | 35% | 21% | 20% | 14% |

**4. Sequence context — partner chain and antigen — gives no measurable benefit.** The valid same-data antigen ablation (`_ag` vs `_ag_lconly`, SAbDab2, all 3 CDRs) shows no signal, and it *survives the core metric*:

| CDR | metric | min core coverage | scratch lc-only → +ag | ESM-2 lc-only → +ag |
|---|---|---:|---|---|
| H1 | core | 100.0% | 0.5473 → 0.4834 | 0.6664 → 0.6629 |
| H2 | core | 100.0% | 0.3672 → 0.3486 | 0.6083 → 0.5894 |
| H3 | core | 99.7% | 0.1308 → 0.1352 | 0.2214 → 0.2128 |

|Δ| ≤ 0.064, sign inconsistent, 5 of 6 comparisons negative. Not a truncation artifact (84% of complexes keep the full antigen at `max_len=512`).

**Partner-chain context: the earlier "it hurts the small model" claim was mostly an under-training artifact.** At a 10-epoch budget it looked like a solid effect; at 20 epochs two thirds of it disappears:

| loop | Δ at 10 epochs | Δ at 20 epochs |
|---|---|---|
| H1 | −0.0103 | −0.0038 |
| H2 | −0.0279 | −0.0095 |
| H3 | −0.0086 | −0.0046 |
| L1 | −0.0277 | −0.0084 |
| L2 | −0.0331 | −0.0108 |
| L3 | **−0.0786** | −0.0242 |
| **mean** | **−0.0310** | **−0.0102** |

Adding context lengthens the input and slows convergence: every `+ctx` scratch run peaked at **epoch 18–20**, against 4–14 without it, and five were still improving when the budget ran out — so −0.0102 is an *upper* bound. ESM-2 is unmoved either way (−0.001 to +0.003).

**With an error bar (finally).** Three seeds of `h3_scratch` ± partner chain, 20-epoch budget:

| seed | no ctx | +LC | Δ |
|---|---|---|---|
| 1 | 0.3196 | 0.3154 | −0.0042 |
| 2 | 0.3199 | 0.3123 | −0.0076 |
| 3 | 0.3187 | 0.3089 | −0.0098 |
| | sd **0.0006** | sd **0.0033** | mean −0.0072, sd **0.0028** |

The effect is real but small (2.5σ, all three seeds negative), and the noise floor on a per-loop Δ is **σ ≈ 0.0028**, i.e. ~0.006 at 2σ. Applying that to the table above, only four of the six loops survive:

| loop | Δ | |Δ|/σ | |
|---|---|---|---|
| H2 / L1 / L2 / L3 | −0.0084 … −0.0242 | 3.0 – 8.6 | real |
| **H1 (−0.0038), H3 (−0.0046)** | | **1.4 / 1.6** | **not distinguishable from noise** |

So on the loop that actually matters — H3 — partner-chain context has **no measurable effect in either direction**, and the earlier −0.0086 was noise plus under-training. A second observation from the same three seeds: the `+ctx` runs are **5× more variable** than the base runs (sd 0.0033 vs 0.0006) and their best epoch swings between 10 and 20, so context destabilises training as well as slowing it.

**Interpretation is unchanged:** sequence context — partner chain or antigen — buys nothing measurable. Binding information is geometric, not linear-sequence, so the next lever is structure / cross-attention. What changed is that the apparent *harm* was largely our own budget.

**5. The homology leak was real but immaterial *for H3*.** OAS's original random split leaked clonal near-duplicates (76 of 2,111 test entries, 3.6%, share a CDR3 cluster with train). The clustered-split fix removed it (→0%), and re-running confirmed the H3 core moved by **≤0.004**. (Now the default; see [`preprocess/oas_paired.py`](cdrseqgen/data/preprocess/oas_paired.py).) The important qualifier is *for H3* — the split clusters on heavy CDR3 only, so it controls nothing for the other five loops (§6).

**6. The clustered split controls H3 and nothing else — so the ~0.99s are retrieval.** `cluster_by_cdr3` groups on heavy CDR3, and whole clusters are held out. Two antibodies with identical H1/H2 land on opposite sides freely as long as their CDR3s differ:

| | unique H1 strings | test entries with an **unseen** H1 | test entries with an unseen H2 | recovery |
|---|---|---|---|---|
| OAS | 439 / 21,106 (2.1%) | 33 / 2,111 (**1.6%**) | 42 / 2,111 (2.0%) | ~0.99 |
| SAbDab2 | 895 / 3,984 (22.5%) | 431 / 750 (**57.5%**) | 597 / 750 (79.6%) | 0.33–0.43 |

98% of OAS test H1s appear verbatim in training, so H1/H2/L1/L2/L3 under this split are near-lookup tasks — which is why every model scores ~0.99 and why context and pretraining make no difference there. **SAbDab2 is not underperforming; it is the only place the non-H3 loops are honestly measured.** H3 is unaffected (100% of test H3s are unseen by construction). Fixing this needs a per-loop split — cluster on whichever CDR is the target — which is a contained change to `add_split_field`.

**7. Recovery and design quality disagree — high AAR can mean a perfect copier.** From the post-training generative eval (150 held-out frameworks, 20 samples each, IgBert judge):

| | recovery | diversity | entropy (bits) | judge-ppl |
|---|---|---|---|---|
| H1 scratch | 0.991 | 0.005 | 0.010 | 1.06 |
| H1 ESM-2 | 0.991 | **0.002** | **0.004** | 1.05 |
| H3 scratch | 0.405 | 0.569 | 1.836 | 7.22 |
| H3 ESM-2 | 0.407 | 0.570 | 1.824 | 7.23 |

H1 scores 0.99 AAR while emitting essentially **one** sequence with total certainty (perplexity ≈ 1.0) — a perfect copier and a useless designer. And on H3 the two metrics rank the models differently: recovery calls scratch and ESM-2 near-equal, the independent judge puts ESM-2 **1.7× ahead**. Recovery alone is insufficient to choose a model, which is the practical argument for a design-aligned objective.

**8. A model is locked to the context regime it was trained in — and `context_dropout` unlocks it.** Feeding a checkpoint the *wrong* amount of context is as damaging as feeding it none, in both directions, because for an absolute-position backbone a prepended partner chain shifts every learned position:

| H1 / OAS / scratch | eval WITH ctx | eval WITHOUT |
|---|---|---|
| trained WITHOUT ctx | **0.1588** | 0.9969 |
| trained WITH ctx | 0.9837 | **0.2612** |
| `context_dropout=0.5` | 0.9791 | 0.9931 |

Training a fraction of examples context-free (each context kind drawn independently, deterministic per example) buys one checkpoint that holds up in every regime. Worst-case recovery, 80 held-out frameworks, greedy:

| | specialist, own regime | specialist, wrong regime | `context_dropout=0.5`, worst |
|---|---|---|---|
| OAS H1 | 0.9969 | 0.1588 | **0.9791** |
| OAS H3 | 0.4353 | 0.0535 | **0.4244** |
| SAbDab2 H3 | 0.2271 | 0.1217 | **0.1766** |

**On OAS it is nearly free** (−0.011 to −0.018 against the best specialist, for a ~6× better worst case). **On SAbDab2 it is not** (−0.051): with 3,234 training entries, splitting across four context combinations leaves ~800 per regime, and the model ends up preferring the *no-context* input (0.2384) over the one it was configured for (0.1832). Read that as a data-size limit, not a refutation — dropout costs what the dataset can't afford to split.

Two side benefits on the datasets where it is cheap: the partner-chain and antigen ablations become *within-model* comparisons (same weights, two inputs) instead of comparisons between separately-trained runs with different init and SGD paths; and it removes the silent-degradation failure `generate.py` has to warn about.

**9. One un-audited hyperparameter inflated four findings at once.** Every config trained for exactly `num_train_epochs: 10`, which suited the pretrained models (ESM-2 median best epoch **4**) and starved the scratch model (median **9**, with 8/18 still improving at the cut-off). Re-running with per-family budgets — 20 epochs for scratch, 10 for pretrained, both with `early_stopping_patience: 3` — moved four separate results, all in the same direction:

| finding | at 10 epochs | at per-family budgets |
|---|---|---|
| §1 pretraining lead, SD2 H1 | +0.231 | **+0.119** |
| §4 partner-chain penalty (mean) | −0.0310 | **−0.0102** |
| §7 judge-ppl gap, H3 scratch vs ESM-2 | 1.74× | **1.00×** |
| §2 scratch `+ctx` recovery | — | +0.007 to +0.055 |

The §7 change is the starkest: scratch's judge-perplexity went **12.42 → 7.22** against ESM-2's 7.23. Recovery and the judge no longer disagree; they were only disagreeing because one model was under-trained. (Not the decoder switch — iterative and independent were measured equivalent, and ESM-2's number barely moved.)

Early stopping cost nothing on the pretrained side — mean Δ over 12 OAS runs was **−0.0000**, and zero runs were cut with their best at the final epoch — while shortening the batch from 295 to 226 minutes.

**The lesson is procedural:** a single budget shared across model families is not a neutral choice, and every distortion it caused ran against the smaller model. Anything compared across families should have its budget justified per family, or the comparison measures the budget.

**10. Removing native H3 length exposes a separate unsolved problem.** The first length-free run used
the validation-selected scratch checkpoint (`1e-3` learning rate, 0.06 warm-up, 0.01 weight decay,
seed 42). Its frozen sequence backbone supplied the representation for a gap-sentinel length
classifier; neither fitting nor generation received the native length.

| length result | learned head | mode baseline (always 15) |
|---|---:|---:|
| exact accuracy, 2,111 validation frameworks | **9.24%** | 10.52% |
| mean absolute error | **3.64** | 3.28 |

The downstream adapter then produced 500/500 syntactically valid candidates on the frozen
25-framework length-free validation manifest, with zero failures and **30/500 (6.0%)** native-length
matches. That 6.0% is a sampled-candidate rate on a small manifest, whereas the table reports one
argmax decision per example over the full validation split, so they should not be equated.

This is a useful negative result: the current representation/head does not extract more length
information from the framework than a trivial global guess. Fixed-native and length-free results
must remain separate tracks; §11 scores the length-free track against IgLM.

**11. Matched length-free comparison: IgLM leads on every quality axis.** Both models ran the frozen
`oas_h3_validation_length_free_v1` manifest — 25 validation frameworks × 20 candidates, one scorer,
one IgBert judge. CDRSeqGen predicts a length from its gap sentinel; IgLM terminates on its own EOS.
Neither receives the native length.

| metric | CDRSeqGen (tuned scratch) | IgLM | better |
|---|---:|---:|---|
| whole-span AAR | 0.3324 | **0.3532** | IgLM |
| core AAR | 0.1576 | **0.1838** | IgLM |
| anchor AAR | **0.8611** | 0.8260 | CDRSeqGen |
| length match rate | 6.0% | **8.4%** | IgLM |
| length MAE | 4.51 | **3.69** | IgLM |
| edit similarity | 0.3596 | **0.3890** | IgLM |
| judge perplexity | 9.74 | **4.07** | IgLM (2.4×) |
| diversity (Hamming) | **0.8047** | 0.7784 | CDRSeqGen |
| valid generation rate | 1.00 | 1.00 | tie |
| runtime / retained sample | **0.078 s** | 0.164 s | CDRSeqGen |

**This inverts the fixed-length result — and the fixed-length track turns out to be unfair to
IgLM.** On the fixed-native smoke manifest CDRSeqGen wins recovery outright (0.3318 vs 0.2155
whole-span, 0.1491 vs 0.1080 core), but that comparison is an artifact of the `constrain` policy.

`constrain` sets `min_length = max_length = native_length` on IgLM's `generate` call, and Hugging
Face's `min_length` suppresses the end-of-span logit until that length is reached — so IgLM is not
merely told the length, it is **forbidden from ending its span** and keeps generating into the
downstream framework. The corruption is directly measurable in the outputs:

| IgLM fixed-length H3 loops | ends in the J-gene motif (`…DV`) | matches the native's last two residues |
|---|---:|---:|
| `constrain` (fixed-length track) | **2%** | **14%** |
| natural EOS (length-free track) | 20% | 76% |
| *natives in the manifest* | *20%* | *—* |

The most common four-residue endings under `constrain` are `YWGQ`, `DYWG`, `GQGT`, `WGQG` — these
are **FR4**, the framework that follows CDR-H3, being counted as loop residues. Naturally IgLM ends
`YMDV`, `YFDY`, `WFDP`, `AFDI`. This is what drives its anchor AAR down to **0.552** against its own
natural-EOS 0.826, while CDRSeqGen's holds at ~0.9 in both tracks.

The tell was visible in the aggregate before the mechanism was found: **IgLM scores worse when handed
the native length (0.2155) than when choosing freely (0.3532)**, and extra information should never
hurt.

**Re-run under the fair protocol, the fixed-length result reverses (2026-08-21).** The adapter's
`filter` mode samples IgLM freely and retains only naturally length-matched draws. On the 20 of 25
frameworks where both models produce native-length loops (matched population, IgBert judge):

| fixed-length, matched population | CDRSeqGen | IgLM `filter` |
|---|---:|---:|
| whole-span AAR | 0.3353 | **0.3569** |
| core AAR | 0.1495 | **0.1868** |
| anchor AAR | **0.8850** | 0.8565 |
| diversity | 0.6097 | 0.6106 |
| per-position entropy (bits) | **1.3008** | 1.0858 |
| judge perplexity ↓ | 9.8244 | **4.7593** |

IgLM's anchors recover from 0.552 to 0.857 — near its natural-EOS 0.826 — confirming that `constrain`,
not the model, produced the collapse. The clearest view is the position curve under the two policies
(`results/figures/training/20260821_benchmark/constrain_vs_filter_position.png`): under `constrain` IgLM's
C-terminal recovery flatlines near 22% while CDRSeqGen climbs to 86%; under `filter` the two curves
converge across the whole loop. **So IgLM leads on both tracks.** The previously reported
fixed-length win was an artifact of the generation policy, and no CDRSeqGen-versus-IgLM recovery
advantage survives a fair protocol.

Two honest counterweights. First, `filter` has its own bias: it drops the frameworks IgLM cannot
match. Five of 25 retained nothing in 50 attempts and they skew long (native lengths 12, 20, 22, 24,
24), so the matched population averages 16.4 residues against 17.2 for the full manifest — both
models are being scored on a slightly easier subset. Neither policy is unbiased: `constrain` corrupts
the output, `filter` shrinks the population. Second, the masked-LM formulation does hold one
structural advantage the recovery numbers hide: **CDRSeqGen produces a native-length loop by
construction, every time**, while IgLM needed 1,081 attempts for 76 usable candidates (7.0%
acceptance) and failed outright on 20% of frameworks within its budget.

*Caveats, all of which must be resolved before this number is quoted outside the repo:* n = 25
examples (AAR is conditioned on length matches, so it rests on 18 CDRSeqGen / 17 IgLM examples);
no confidence intervals yet — the paired bootstrap is the next roadmap item; and the scores were
produced from a **dirty working tree at the previous commit** (`repository_commit: 0e7e499`,
`repository_dirty: true` in `validation_length_free_scores_v1.json`), so a clean-tree rescore is
required. IgLM was also pretrained on OAS, so this holdout is not a leakage-free estimate for it
(roadmap: overlap audit).

**12. The model's own confidence is nearly useless for selection — an independent judge's is not.**
Every candidate now carries its own pseudo-perplexity and judge perplexity, so the pairing with its
own recovery survives into `generation_eval.json` instead of being averaged away. That makes a
practical question answerable: given 20 candidates for one framework, does picking the most
confident one beat picking at random? (150 held-out frameworks × 20 candidates, `h3_OAS_scratch`,
IgBert judge; per-framework Spearman, then averaged.)

| ranking signal | Spearman ρ | top-1 recovery | lift over a random draw |
|---|---:|---:|---:|
| random draw (baseline) | — | 0.4066 | — |
| **self**-perplexity | 0.130 ± 0.047 | 0.4176 | **+0.011 ± 0.012 — not significant** |
| **judge** perplexity | 0.254 ± 0.046 | 0.4591 | **+0.053 ± 0.013 (8.0σ)** |

The model's own uncertainty carries a statistically detectable but practically useless signal: ρ is
5.4σ from zero, yet selecting on it does not reliably beat a random draw (95% CI crosses zero), and
its top pick beats the framework mean only 85/150 times — barely better than a coin flip. An
independent model ranks roughly twice as well (Δρ = +0.124, 4.4σ) and its top pick beats the
framework mean 103/150 times, yielding **+12.9% relative recovery for free — no retraining, just
reranking**.

Panel B of `results/figures/training/20260821_confidence/confidence_selection.png` carries the
qualifier that the means hide: the per-framework ρ distributions are wide and both straddle zero.
Self-perplexity ranks *backwards* on **46 of 150** frameworks, the judge on **29 of 150**. So even
the judge is a population-level effect, not a rule you can trust on any single framework — worth
stating before anyone reads "+12.9%" as a guarantee.

> **⚠️ The ordering did not hold on a second checkpoint.** Scored on the 08-25 screen's
> cross-entropy run, self-perplexity *outranked* the judge: top-1 lift **+0.0276 against +0.0186**,
> the reverse of the table above. That run is not a replication — 100 frameworks × 10 candidates
> against 150 × 20, on the tuned config rather than `h3_OAS_scratch` — and halving the candidate
> pool makes top-1 selection substantially noisier. But it is enough that the *ranking* of the two
> signals should be treated as unsettled until one `evaluate.py` run at 150 × 20 on the tuned
> checkpoint says otherwise. The claim that survives either way is the practical one: an
> independent judge is a usable reranker, and reranking beats a better sampler.

**The figure is written automatically.** Any evaluation that produces per-candidate scores also
drops `<output>_confidence.png` beside its JSON — the same convention as the training curves — so
every `train.py` run and every `evaluate.py` call gets one without a second command. It is skipped
silently when there is nothing to plot (`--skip_perplexity` with no judge) and is exception-guarded,
so a plotting problem cannot cost a finished evaluation. To redraw, or to overlay two runs:
```bash
python -m cdrseqgen.visualization.confidence_plot --eval <run>/generation_eval.json
python -m cdrseqgen.visualization.confidence_plot --eval a.json --label scratch --eval b.json --label ESM-2
```

This sits directly alongside the decoder finding: iterative decoding buys nothing because the
learned marginals are near-independent, and now the same model's confidence barely tracks its own
correctness. Both say the masked-LM objective produces poorly *structured* uncertainty, not merely
inaccurate predictions. The practical corollary is that a reranking stage is worth more than a
better sampler — and that the reranker must be an independent model.

**13. Re-weighting the loss across loop positions changes nothing — the anchors are not wasted
capacity.** The 08-25 screen trained the tuned scratch config (`1e-3`, 0.06 warm-up, 0.01 weight
decay, 30 epochs) under four objective settings × 3 seeds; 10 of 15 trials completed before it was
interrupted. Ranked three ways, the winner changes with the metric:

| ranked by | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| judge perplexity ↓ | **cross-entropy** (6.94) | weighted 1/1 (7.04) | weighted 3/1 (7.09) | core-only (31.60) |
| free-running core@L | **weighted 1/1** (0.2294) | cross-entropy (0.2286) | core-only (0.2266) | weighted 3/1 (0.2258) |
| teacher-forced core | **weighted 1/1** (0.3226) | weighted 3/1 (0.3224) | cross-entropy (0.3212) | core-only (0.3207) |

**The two settings trading first place are the same loss function.** `weighted_cross_entropy` at
core/anchor `1.0/1.0` is bit-identical to `cross_entropy` — with equal weights the weighted mean
reduces to the plain token mean (verified to 1e-9). The grid therefore contained only **three**
distinct objectives, not five: `3.0/0.0` is likewise identical to `1.0/0.0`, because the loss
normalises by total weight and only the core-to-anchor *ratio* survives.

That accident supplies the number that settles the comparison. Two runs of an identical objective at
identical seeds differ by **0.096 judge perplexity, 0.0014 teacher-forced core, 0.0009
free-running core** — and every real between-objective gap is at or below that floor. Weighted 3:1,
the only genuinely distinct non-degenerate setting, sits inside it on all three metrics.

**Core-only is the one decisive result, and it is a clean negative.** Setting
`anchor_loss_weight 0` collapses anchor recovery from **0.873 to 0.042**, which is what drives judge
perplexity to 4.5×. It did *not* buy anything for the region it targets: core@L 0.2266 against
cross-entropy's 0.2286. So down-weighting the anchors does not reallocate capacity to the
hypervariable middle — the core stays exactly where it was and the anchors break. The position curve
in `results/runs/20260825_165553_sweep_loss/figures/objective_comparison.png` shows it directly:
core-only tracks every other objective across relative positions 0.2–0.7 and falls away only at the
two ends. Its higher diversity (0.716 against 0.560) is an artifact of emitting implausible anchors,
not better design.

*Caveats: weighted 3:1 has n=1 — the sweep was interrupted before its other two seeds ran, so it is
bounded by the noise floor rather than properly measured. H3/OAS/scratch only, one hyperparameter
setting, generative eval at 100×10 rather than the 150×20 used in §12. One thing strengthens the
conclusion: checkpoint selection ran on `core_accuracy`, the metric a core-weighted objective
directly optimises, and the weighted settings still did not win.*

This is the third independent result pointing one way. Iterative decoding buys nothing because the
marginals are near-independent (Step 4); the model's confidence barely tracks its own correctness
(§12); and now re-weighting tokens within the objective moves neither recovery nor plausibility. The
problem is the objective *class*, not the weighting inside it.

> **Bottom line:** fixed-native sequence-only CDR-H3 infilling saturates at ~0.32 core / ~0.46
> whole-span regardless of model, context, dataset, decoder, **or loss weighting**; length inference
> is not solved by the first gap-sentinel head, and against a published baseline that picks its own
> length the current method loses on recovery, length, edit distance, and judged plausibility alike;
> and the non-H3 loops are not yet measured on a split that tests generalisation. Together these
> motivate the eventual shift to a **structure objective** (see Roadmap).

---

## 🧠 Architectural Feature Details

### 1. Task-Agnostic Masking (The "LEGO" Principle)
The `AntibodyInfillingDataset` separates sequences into `pt1` (left context), `mask` (target gap), and `pt2` (right context). The `target_region` string alone chooses the loop — `h1/h2/h3` mask a heavy CDR, `l1/l2/l3` mask the *light* chain's CDR (legacy `cdr*`/`lcdr*` spellings still parse). Switching target requires changing zero core tokenization logic. Labels are strictly length-enforced to prevent cross-entropy dimension mismatches.

### 2. Pluggable Backbones (the extensibility core)
The masked-LM backbone registry [`backbones.py`](cdrseqgen/methods/mlm/backbones.py) holds one `Backbone` entry per compatible model family; the entrypoints and `dataset.py` never branch on model type — they go through `resolve(model_type)`, which returns the matching backbone and a uniform `TokenizerAdapter`. The same MLM pipeline trains the from-scratch transformer, fine-tunes ESM-2, or fine-tunes any Hugging Face masked-LM just by changing the `"model_type"` string.

Each backbone declares two conventions **explicitly** instead of letting them be guessed from the tokenizer (the previous code overloaded a single `is_esm` boolean to mean three different things, which silently mishandled non-ESM models):
- **`position_type`** (`"absolute"` | `"relative"`) — controls where antigen context is injected (see §4). Absolute-position models (the scratch transformer, BERT-style protein LMs) need it appended after the target; relative/rotary models (ESM-2) can take it prepended.
- **`uses_special_tokens`** — whether inputs are wrapped as `[cls] … [end]`. The `TokenizerAdapter` also resolves the ESM-vs-BERT separator difference generically (`<eos>` when the tokenizer has one, else `<sep>`), so BERT-style tokenizers with `eos_token_id = None` work without special-casing.

**Adding a new masked PLM is usually one `register(Backbone(...))` entry** — occasionally *zero*, when a model is already covered by the generic `hf_encoder` fallback (any `AutoModelForMaskedLM` + `AutoTokenizer`, absolute positions, per-character vocab — just pass its hub id as `model_type`). In practice most protein LMs need an explicit entry: e.g. BERT-style ones (ProtBert, IgBert) expect *whitespace-separated* residues and ship tokenizer/loader quirks (a fast-tokenizer that needs `sentencepiece`, or a `config.json` with no `model_type`), all handled once in the `protein_bert` backbone. Either way, no edits to `train.py`/`generate.py`/`dataset.py` are needed. A new autoregressive or encoder-decoder architecture is instead a sibling method package, not a `Backbone` entry here.

### 3. Partner Chain Context Injection
Biologically, Heavy and Light chains physically interact, so the framework can prepend the paired **partner chain** (the non-target chain) to `pt1` (separated by `<eos>`) when `"include_partner_chain": true` is passed, enriching structural context. It's symmetric: for a heavy (`h*`) target the partner is the light chain (run tag `_LC`); for a light (`l*`) target the partner is the heavy chain (run tag `_HC`). A single flag covers both directions — the target region determines which chain gets added.

### 4. Antigen Context Injection (SAbDab2)
Same "LEGO" pattern as partner-chain injection (§3), but **the injection side is keyed on the backbone's `position_type`**. For **relative**-position models (ESM-2), the antigen is prepended *ahead of* the light chain into `pt1`: `antigen <sep> light <sep> heavy_pt1 <mask> pt2`. For **absolute**-position models (the scratch transformer, and BERT-style PLMs), the antigen is instead *appended* into `pt2`, with the light chain still prepended to `pt1`: `light <sep> heavy_pt1 <mask> fwr4 <sep> antigen`. Multi-chain antigens are joined with `X` (the standard "unknown residue" convention) as a soft chain-break marker, since the amino-acid tokenizer has no token for a literal chain separator.

Antigen sequences (up to ~2,300 residues for large multi-chain complexes) made the pre-existing "no truncation" gap in `AntibodyInfillingDataset` actually bite, so this feature also added a real length cap, applied on whichever side carries the antigen: truncated from the left for relative-position `pt1`, from the right for absolute-position `pt2` — always dropping antigen context first and keeping the target-chain residues immediately adjacent to the masked loop.

### 5. Automated Tracking & Evaluation
*   **TensorBoard:** Live tracking via `"report_to": "tensorboard"`.
*   **Metric Extraction:** At the end of training, `train.py` parses the Hugging Face logs, dumps `epoch_metrics.csv` (loss + recovery, whole-span **and** the anchor-excluded `core`/`anchor` accuracies), and generates standalone `.png` curves.

### 6. Extensibility Principles
The recurring rule: **a new capability is added in one place, and the rest of the pipeline stays oblivious.** Four "add-one-line" seams:
- **New masked-LM backbone** → one `Backbone` entry in [`backbones.py`](cdrseqgen/methods/mlm/backbones.py) (§2) — the entrypoints and `dataset.py` never learn about it.
- **New target loop** → change the `target_region` string (§1); masking logic is untouched.
- **New context** → one boolean flag (`include_partner_chain`, `include_antigen`), resolved symmetrically (§3–4).
- **New dataset source** → one `(substring, tag)` pair in the `DATASET_TAGS` registry ([`train/logging.py`](cdrseqgen/train/logging.py)) (`oas → OAS`, `sabdab2 → SD2`), so every run's output-dir name states its source explicitly instead of leaving it silently implied; an unrecognised path is tagged `DS` **with a logged warning** rather than left blank.

The guiding constraint: **conventions are declared explicitly, never inferred.** The predecessor overloaded a single `is_esm` boolean to mean three different things and silently mishandled non-ESM models; each backbone now states its `position_type` and `uses_special_tokens` up front. This isn't free — protein-BERT models (IgBert/ProtBert) still needed their own entry for residue-spacing and tokenizer/loader quirks (§2) — but the cost is paid once, in the registry, not scattered across the pipeline.

---

## ⚠️ Current Issues 
*(current as of 2026-08-18; update per commit)*

### Open problems
- **The split only controls the loop it clusters on.** `add_split_field` clusters heavy CDR3, so the holdout is leakage-safe for H3 and **not** for H1/H2/L1/L2/L3 — 98% of OAS test H1s appear verbatim in train (Findings §6). Every non-H3 number in this README is therefore retrieval-bounded and should not be read as generalisation. Fix: cluster on the *target* CDR (per-loop split), a contained change to `add_split_field`; then re-run the OAS matrix.
- **The first learned H3 length head is below its naive baseline.** It reaches 9.24% exact accuracy / 3.64 MAE versus 10.52% / 3.28 for always predicting length 15 (Finding §10). Before treating inferred MLM generation as a competitive method, evaluate probability calibration and top-k length coverage, then test a representation or objective that actually learns the framework-to-length signal.
- **Training still optimizes recovery, not design.** The masked-LM objective rewards reproducing the native loop; the generative metrics only *measure* design quality after the fact — nothing trains for it. Findings §7 shows the two can disagree outright (H1: 0.99 recovery, 0.002 diversity). A design-aligned objective stays open (Phase 2). Re-weighting tokens *within* the masked-LM objective has now been tested and does not help (§13), so this needs a different objective class rather than a better weighting.
- **Run-to-run nondeterminism exceeds the seed spread, and only seeds are ever reported.** Two bit-identical objectives at identical seeds — `cross_entropy` and `weighted 1.0/1.0` — ended 0.0014 apart in teacher-forced core, all three seeds in the same direction, against a within-objective seed sd of 0.0003–0.0007 (§13). The two differ only in floating-point reduction order. So a seed sd *understates* reproducibility by 2–4×, which bears directly on the ±0.006 tie threshold in `sweep.sh` and the σ ≈ 0.0028 noise floor in §4. Every "error bar" in this README is a seed spread and should be read as a lower bound. Cheap fix: keep one duplicated setting per sweep as a deliberate determinism control.
- **Checkpoint selection is circular for a loss sweep.** `sweep.sh` argues — correctly — that *ranking* a loss sweep on core recovery lets a core-weighted objective grade its own homework, and defaults `RANK_METRIC` to `judge_ppl`. It then passes `--metric_for_best_model core_accuracy` to every trial, so each checkpoint is still *selected* on the metric the objective optimises. It is also blind to the failure §13 found: the core-only run has anchor recovery 0.042 and would be selected happily, because selection never looks at anchors even though `eval_anchor_accuracy` is recorded every epoch.
- **`evaluate.py --decoding` does not default off the masking policy.** `resolve_decoding` falls back to the checkpoint's recorded `gen_decoding`, which defaults to `"iterative"` in `GenerationArguments` — so a checkpoint that never set it resolves back to the mode that halves core recovery on a `full_span` run. All 49 current configs happen to set `independent`, which is the only reason this is not live. The guard that would catch it is also dead for older checkpoints: it tests `masking_policy == "full_span"`, and `masking_policy` is absent from every `run_config.json` written before 08-25. Drive the default off `masking_policy`, treating a missing value as `full_span`.
- **`anchor_trim` does not reach the generative eval.** `run_generative_eval` never forwards it to `run_eval_loop`, so `core@L`/`anchor@L` always use the module default of 2 even when training ran with `--anchor_trim 3`, while the training metrics use the requested value. Not live today (everything is trim 2), but it is the silent metric-definition mismatch the shared `core_anchor_indices` exists to prevent.
- **The judge shares the eval distribution.** IgBert was trained on OAS paired repertoires — the same distribution as the OAS eval split it judges. It's applied identically to every model, so the *comparison* is fair, but absolute perplexities are OAS-flavoured; an off-distribution judge would be a stronger check.
- **Generative sweep is H3/OAS only.** The matrix covered CDR-H3 on OAS (150 held-out frameworks × 4 models). L-chain loops, SAbDab2, and the antigen configs have not been swept generatively.
- **Sampler is marginal-based, not learned-joint — now measured.** The `iterative` (Mask-Predict) decoder approximates the loop joint by conditioning on committed residues, but the model is still a masked-LM trained on independent-position recovery. As of 2026-07-28 this is no longer a suspicion: iterative and `independent` are **within noise on every metric** on H3/OAS/ESM-2 (recovery 0.4057 vs 0.4072; judge-ppl 7.14 vs 7.23) at ~15× the compute. The conditioning buys nothing because the learned marginals are already near-independent — true joint/generative *training* (autoregressive or diffusion) is Phase-2 territory.
- **Five runs hit their epoch ceiling, all of them context runs.** `h1_SD2_LC`, `h2_OAS_LC`, `h3_OAS_LC`, `l1_OAS_HC`, `l3_OAS_HC` peaked at epoch 19–20 of 20 in the 07-29 batch. Their numbers are lower bounds, so the §4 partner-chain penalty (−0.0102) is an upper bound on the true effect. A scratch-only sweep at 30–40 epochs would settle it (~1 hour; scratch is 60 of the batch's 226 minutes).
- **Only one comparison has an error bar.** Three seeds of `h3_scratch` ± partner chain put the seed sd of a per-loop Δ at **0.0028** (§4), which already retires two of the six per-loop context numbers as noise. Every *other* number in this README is still single-seed, and any claimed difference below ~0.006 should be treated as unmeasured until it has the same treatment.
- **Every batch through 2026-08-17 selected checkpoints on the test split.** Not just the 07-29 runs: `eval_split` defaulted to empty, which fell back to "every non-train entry", and `processed_oas_data.json` has only `train`/`test` — so early stopping, `load_best_model_at_end`, the reported `eval_*` metrics, and `generation_eval.json` all read the final holdout. The 08-17 post-refactor batch (`results/20260817_200051`) is affected too; its logs read `2111 eval (split: all non-train)`. **Fixed 2026-08-18**: `get_datasets` now refuses an unset `eval_split` whenever a `test` split exists, and the 36 OAS configs declare `processed_oas_tuning.json` + `eval_split: validation`. Every pre-08-18 number in this README is an exploratory result, not an untouched-test estimate, and the OAS matrix needs re-running before any of it is reported.
- **Neither fixed-length policy is unbiased — report both.** *(Re-run 2026-08-21; see Findings §11.)* `constrain` corrupts IgLM's loop endings; `filter` is fair per-example but drops the frameworks IgLM cannot length-match (5 of 25, skewing long), so it scores an easier subset. Current practice: report `filter` on a matched population and state the dropout. A policy that is unbiased in both respects — e.g. raising the attempt budget until every framework is covered, or reporting per-length-bucket — is still open. Historical note on `constrain`: IgLM's `constrain` policy sets `min_length = max_length = native_length`, which suppresses its end-of-span logit until that length is reached — so it writes past its loop into FR4. Only 2% of constrained loops carry the J-gene motif against 20% naturally (Findings §11), and every fixed-length CDRSeqGen-versus-IgLM number is inflated in CDRSeqGen's favour. Re-run with the adapter's `filter` mode (retain only naturally length-matched draws; ~12 attempts per accepted candidate at IgLM's 8.4% match rate) before making any fixed-length claim. The length-free track is unaffected — IgLM uses natural EOS there.
- **SAbDab2 has no validation partition.** Its split comes from the source `ab_ag_split` column, which is `train`/`test` only, so its 12 configs now declare `eval_split: test` — honest, but still selection on the holdout. A three-way SAbDab2 split (or a fixed validation carve-out of its train partition) is needed before any SAbDab2 comparison is reportable.
- **Hyperparameter selection is not yet frozen.** The validation-safe 32-trial scratch-H3 screen completed on 2026-08-11 and placed `1e-3` / 0.06 warm-up / 0.01 weight decay first at 0.32305 mean validation core recovery. Its margin over the runner-up was only 0.00036 across two seeds and changed sign by seed, so additional shared confirmation seeds are still required before touching the final test split.
- **Checkpoint-selection policy is still config-declared.** The current configs select core recovery except the four OAS-L2 configs, which select whole-span because those runs have only 1.4% core coverage. Artifact reporting no longer hard-codes that loop name: it derives the metric from recorded coverage. Training should eventually validate or derive `metric_for_best_model` from the dataset before a run, so a future dataset cannot inherit an obsolete L2 exception.
- ~~**Scratch architecture isn't tunable.**~~ **Fixed 2026-07-27.** `num_attention_heads` and `intermediate_size` are now `ModelArguments` fields forwarded by `build_model` ([`transformer.py`](cdrseqgen/methods/mlm/models/transformer.py)). `intermediate_size` defaults to `None` = **4 × hidden_size**, so the FFN ratio tracks width instead of silently drifting (at the old fixed 512, widening to 256 quietly halved the ratio and would have confounded any architecture sweep). Defaults reproduce the old geometry exactly (128 → 512), and a `hidden_size % num_attention_heads` check now fails loudly instead of deep inside `nn.MultiheadAttention`.
- **Single-seed runs.** Small differences (e.g. light chain hurts scratch ~0.01, neutral for ESM-2) need multiple seeds + error bars before they can be claimed.
- **The tests stub every model.** All 105 run in ~3s with no GPU, weights, or processed data, which is why they are cheap to run on every change — but nothing in the suite would catch a shape, dtype, or device error against a real backbone. A mutation audit (2026-08-18) confirmed they do catch the science-critical defects: native-length leakage, a non-full-span eval view, core/anchor off-by-one, attempt-accounting drift, flank-mangling in the IgLM adapter, and resume-seed collapse. `arguments.py` and the `train.py`/`evaluate.py` entrypoints remain untested.
- **Structure not yet available.** SAbDab2 preprocessing parses *sequences* only; the Phase-2 structure objective needs a coordinate-parsing step (or predicted structures via IgFold/ABlooper).

### How to read the numbers 
- **Report core only when it represents the dataset.** Whole-span recovery is anchor-inflated on H3, but a core score is also misleading when most loops have no residues left after trimming. The shared 50% policy uses `eval_core_accuracy` only when the run's recorded `eval_core_coverage` is at least 50%; otherwise core is `N/A` and the explicitly labeled whole-span score is reported.
- **OAS L2 is an observed case, not a coded exception.** With `ANCHOR_TRIM=2`, 98.8% of its loops are three residues, so only ~30/2,111 evaluation examples (1.4%) contain a core. The reporter reads that 1.4% from `epoch_metrics.csv` and selects whole-span. A future L2 dataset with adequate coverage would use core, while any other short-loop dataset would correctly fall back to whole-span.
- **Every recovery number in Findings is from the fixed-native track — the model is told the loop length.** `dataset.py` inserts exactly `len(native)` mask tokens for that protocol, so the length is given away by the input's shape. This is standard fixed-length redesign and is required for position-wise AAR to be well-defined, but it is not free information; how much it is worth varies enormously by loop:

  | loop | distinct lengths | mode freq | free info from knowing L |
  |------|------------------|-----------|--------------------------|
  | H3   | 34               | 9.9%      | **4.05 bits**            |
  | L3   | 14               | 43.4%     | 2.13 bits                |
  | L1   | 6                | 54.3%     | 2.01 bits                |
  | H2   | 6                | 62.0%     | 1.24 bits                |
  | H1   | 4                | 83.8%     | 0.73 bits                |
  | L2   | 3                | 98.8%     | **0.10 bits**            |

  On H3 that's ~4 bits handed over per example — roughly 9% of the loop's total information content (a 15-mer at the measured 2.96 bits/position ≈ 44 bits; position-conditioned entropy over the 2,098 length-15 H3 loops). On L2 it's nothing: guess 3 and you're right 98.8% of the time. So **H3 recovery is the number most flattered by this**, and cross-loop comparisons of AAR partly compare how much length information each loop gave away. Length-free results are therefore a separate track with length/edit metrics, not a replacement value in these tables.
- **Teacher-forced accuracy is not generation accuracy, and the core is where they diverge.** `eval_accuracy`/`eval_core_accuracy` are argmax over the true context; the generative eval samples. On `h3_OAS_scratch` (300 frameworks x 10 candidates, fixed-native, T=1.0, independent decoding, all 3,000 candidates length-matched):

  | metric | teacher-forced (argmax) | free-running (sampled) |
  |---|---|---|
  | whole-loop | 0.4604 | 0.3695 |
  | **core** | 0.3180 | **0.2042** |
  | anchor | — | 0.8397 |

  The whole-loop number is carried by the anchors (0.84): they are largely V/J-determined, so a model that had learned nothing about the hypervariable middle would still post a respectable whole-loop AAR. Judge the core column. Against baselines fitted on train and evaluated on validation over the same 26,205 core positions — matched to the sampling protocol, since an argmax baseline is not comparable to a sampled model — sampling from H3's marginal core amino-acid frequencies expects **0.084**, and sampling from a per-`(length, position)` frequency table expects **0.146**. At 0.204 the model beats the lookup table by ~40% relative, which is a real but modest margin, and the honest headline for free-running H3 core design.
- **The decoder must match the checkpoint's masking policy.** `iterative` decoding re-masks and resamples positions one at a time, which assumes the model saw partially revealed loops — true only under `masking_policy: mixed`. On the same `full_span` checkpoint and protocol, iterative decoding costs whole-loop 0.370 -> 0.253 and core **0.204 -> 0.077**, dropping it below even the 0.084 marginal-sampling baseline. `evaluate.py --decoding` therefore defaults to whatever `run_config.json` recorded rather than to a fixed mode, and warns when `iterative` is forced onto a `full_span` run.
- **Read best-epoch, not last-epoch** — ESM-2 overfits on SAbDab2/H3; `load_best_model_at_end` saves the peak, but the last CSV row can sit well below it.
- **Never plot OAS and SAbDab2 on shared axes** — different test sets; the `_OAS`/`_SD2` run tags enforce it, and per-dataset charts live under `results/figures/training/<batch>/*.png`.
- **`epoch_metrics.csv` and `*_results.json` measure different models — they're *supposed* to disagree.** The CSV is the per-epoch **trajectory** (the model as it stood at each epoch); `eval_results.json`/`all_results.json` come from the post-training `trainer.evaluate()`, which with `load_best_model_at_end` runs on the **best** checkpoint while still being stamped with the final epoch number. On `h3_esm2` the CSV's epoch-10 row reads core **0.3063** (the actual last model) while `eval_results.json` reads **0.3209** (the epoch-4 best model). `save_training_csv` de-duplicates on epoch and **keeps the first** occurrence precisely so the curve keeps the real last-epoch point — keeping the last would silently redraw the best model at x=10 and erase the overfitting decay entirely.

<!-- MANUAL PROTECTION ZONE BEGINS -->
### Changelog
<!-- AI ACCESS ZONE BEGINS -->
<!-- AI ACCESS ZONE ENDS -->
- **2026-08-31** -- Auto figure plotting after hyperparameter & loss sweeps, in sweep dir instead of separate `figures/` folder
- **2026-08-31** -- Data visualization rework, run/sweep type -> table/graph plotters
- **2026-08-25** -- Enabled different loss types besides cross entropy, currently added: weighted cross entropy (core vs anchor)
- **2026-08-21** -- Added confidence scores per candidate and confidence vs AAR correlation metrics
- **2026-08-21** -- `scripts/sweep.sh` [new]: merged `sweep_h3.sh`, `sweep_scratch.sh` into unified hyperparameter sweep script
- **2026-08-21** -- `results/` storage reorganized
- **2026-08-21** -- IgLM benchmark fix: fixed-length for IgLM now filters samples that match the native length than forced EOS (is this approach ok??)
- **2026-08-18** -- [RECLAIM] `dataset.py`, `preprocess/common.py`
- **2026-08-18** -- project wide claude fix
- **2026-08-17** -- Produced IgLM benchmark results
- **2026-08-17** -- Numerous `test/` scripts
- **2026-08-17** -- [RECLAIM] Code restructure, progress reset
- **2026-08-17** -- Enabled variable-length generation
- **2026-08-14** -- Flowchart updated in `misc/`
- **2026-08-14** -- `artifacts.py`[new]: stores current and legacy filenames as global constants
- **2026-08-14** -- `batch_tables.py`: reworked - auto detect core AAR availability, merged separate table functions into unified function
- **2026-08-12** -- IgLM benchmark - benchmark manifest (`species` field added); local (cdrseqgen) and IgLM adapters; smoke test run
- **2026-08-11** -- `scripts/sweep_h3.sh` [new]: hyperparameter sweep script. Result: `1e-3` LR / 0.06 warm-up / 0.01 weight decay
- **2026-08-11** -- Changed dataset split to train:val:test = 8:1:1 (old: train:test = 9:1)
- **2026-08-11** -- Update roadplan for baseline comparison: IgLM, IgCraft
- **2026-07-29** -- `dataset.py`: added `context_dropout` variable to train a fraction of examples without paired chain/antigen context
- **2026-07-29** -- Split `train.py` into `cdrseqgen/train/{arguments,logging,pipeline,record_results}.py`
- **2026-07-29** -- `train.py`: fixed `train.log`, records per epoch (used to only have "Starting Training.../Starting Evaluation...") (pending verification)
- **2026-07-27** -- Fixed print/logging issues in `models.py`, `generation.py`
- **2026-07-27** -- `data/preprocess/common.py`[new]: common scaffolding for csv -> json data preprocessing scripts
- **2026-07-27** -- moved and renamed preprocessing scripts to `data/preprocess/`, removed separate dataset README file
- **2026-07-24** -- Generate `run_config.json` for each training run
- **2026-07-24** -- fixed `results/runs/latest/` symlink to latest batch (to verify further)
- **2026-07-24** -- `linegraph_plotter.py`: moved to new `utils/` folder
- **2026-07-24** -- `sampling.py`: handled greedy algo for both independent and iterative decoders

---

## ⏭️ Roadmap

<!-- AI ACCESS ZONE BEGINS -->
> **Objective-first:** the target is **CDR-H3 design**. **Phase 1** = a leakage-controlled *sequence benchmark* (the current models plateau near 0.32 core recovery under the tested settings; see Findings). The active milestone is to position that result against a published sequence-infilling baseline under one matched evaluation protocol. **Phase 2** = *structure-conditioned* design (RMSD-to-native; exact target to lock with advisor). Everything below is a tool toward that objective.

### Active milestone — matched CDRSeqGen vs IgLM benchmark

**Primary baseline:** [IgLM](https://github.com/Graylab/IgLM), because it is the closest published and reproducible match to CDRSeqGen's sequence-infilling task. IgCraft is the intended second sequence baseline after the comparison interface is stable; structure/antigen models such as IgGM belong to Phase 2 and should not be mixed into the sequence-only table.

The comparison must run both methods on the **same frozen examples**, with the same number of candidates and the same scorer. It has two distinct tracks and they must not be collapsed into one number:

1. **Fixed-length redesign (first deliverable).** Supply the native H3 length to both models. This matches CDRSeqGen's current mask construction and permits position-wise recovery, including the existing anchor/core split.
2. **Length-free infilling.** Supply only the flanking framework. IgLM terminates with EOS; CDRSeqGen predicts a checkpoint-specific length from one gap sentinel and then expands that choice into residue masks. Report validity, length error, edit similarity, and length-matched AAR rather than collapsing this into the fixed-length result.

**Implementation plan**

- [x] Freeze versioned H3 benchmark manifests from the clustered OAS evaluation split. Each record carries a stable ID, heavy-chain `pt1`/`pt2`, native H3, native length, and split provenance; both adapters consume the exact same file.
- [x] Add a model-independent shared scorer that accepts both adapters' `{example_id, native, samples}` records, validates the paired experiment contract, and keeps model-specific sampling outside scoring.
- [x] Add a manifest-driven CDRSeqGen benchmark adapter for existing checkpoints. It validates run/task/context provenance, recreates training-time tokenization and truncation, preserves duplicate draws, hashes checkpoint artifacts, and writes resumable records in the same schema as IgLM.
- [x] Add an attempt-bounded IgLM adapter around the official pretrained checkpoint. It records model/checkpoint provenance, conditioning tokens, sampling parameters, per-example seeds, failures, raw generated lengths, runtime, and resumable progress; IgLM remains outside the masked-LM backbone registry.
- [x] **Re-run the fixed-length track in `filter` mode.** Done 2026-08-21: IgLM leads on both tracks once the constraint is removed (§11). Artifact: `results/benchmark/validation_fixed_length_filter_scores_v1.json`
- [ ] Cover the 5 frameworks `filter` drops — raise the attempt budget or report per-length-bucket, so the fixed-length population isn't silently the easier subset
- [x] Define the fixed-length IgLM policy explicitly. Default `constrain` forces IgLM's end-of-span at the supplied native length through its underlying generation API; audit mode `filter` preserves every variable-length draw and reports match/acceptance counts under a finite attempt budget.
- [x] Validate the shared evaluator on both 25-framework validation outputs: whole-span/core/anchor AAR, diversity, native-distance novelty, unique fraction, per-position entropy, length buckets, validity/length rates, generation runtime per candidate, and a revision-pinned IgBert judge. Final test scoring remains gated on freezing the sampling policy.
- [x] Add a checkpoint-specific CDRSeqGen length head, bounded/inferred generation, and migration tooling for old sequence checkpoints without changing the backbone architecture.
- [x] Add a separate `length_free` manifest/adapter/scorer contract: retain all valid lengths, condition positional AAR on length matches, and score every candidate with length error, edit similarity, diversity, and the shared judge.
- [x] Fit the tuned scratch checkpoint's first length head and run its 25-framework × 20-candidate length-free validation output; preserve the below-mode-baseline result rather than hiding it.
- [x] Run the matched IgLM natural-EOS validation output, score both adapters together, and render the length-free comparison dashboard. **Result: IgLM leads on recovery, length error, edit similarity, and judge perplexity (Findings §11).** Rescore on a clean tree before quoting — the stored scores carry `repository_dirty: true` at commit `0e7e499`.
- [ ] Add paired uncertainty: bootstrap test examples and report the CDRSeqGen-minus-IgLM difference with a 95% confidence interval. The example, not each generated candidate, is the statistical unit.
- [ ] Produce side-by-side evaluation curves: recovery by H3-length bucket, accuracy by relative H3 position, and the recovery--diversity frontier across matched sampling settings. Do not compare CDRSeqGen training curves with a frozen pretrained IgLM checkpoint; training curves become meaningful only if IgLM is retrained on the same corpus and should then use tokens/examples seen rather than epoch number.
- [x] Reproduce the official IgLM infill example and validate the adapter on 25 frozen validation frameworks (125/125 constrained draws valid and length-matched; zero adapter failures). Reproducing a reported paper-scale benchmark remains pending.
- [ ] Audit training-data overlap. A frozen IgLM model was pretrained on OAS, so performance on the local OAS holdout is not automatically a leakage-free generalization estimate. Report the frozen-model comparison with this limitation, and separately investigate either retraining IgLM on the exact local training split or constructing a post-training/independent test set.
- [ ] Write one comparison artifact containing the manifest hash, model provenance, complete parameters, per-example records, aggregate table, confidence intervals, and plot paths so the result is reproducible from a single command.

**General — next steps**
- [x] Sampler used by `generate.py` (temperature / top-k / top-p, independent + iterative Mask-Predict) + generative metrics ([`sampling.py`](cdrseqgen/methods/mlm/sampling.py), [`metrics.py`](cdrseqgen/eval/metrics.py))
- [x] Sweep the generative metrics across the H3 model/context matrix (150 held-out frameworks × 4 models, IgBert judge)
- [x] **Reporting hygiene** — recovery tables derive core-versus-whole validity from recorded `eval_core_coverage`, label both metric and coverage, and never branch on the loop name. Checkpoint-selection configs remain explicit and must be validated against future datasets (see Open problems).
- [x] Plumb `num_attention_heads` / `intermediate_size` through `build_model` — `intermediate_size` now defaults to 4 × `hidden_size` (prerequisite for any architecture sweep)
- [ ] **Per-loop split** — cluster on the *target* CDR, not always heavy CDR3, then re-run the OAS matrix. Everything non-H3 in Findings is retrieval-bounded until this lands (§6)
- [x] Implement three-way clustered OAS splitting, explicit validation/test selection, full run provenance, and the validation-safe H3 sweep
- [x] Run the 32-trial validation-safe scratch H3 screen ([`sweep.sh`](scripts/sweep.sh)); top region: `1e-3` LR / 0.06 warm-up / 0.01 weight decay at 0.32305 mean validation core recovery (two seeds)
- [ ] Confirm the top two sweep settings with additional shared seeds, freeze the selected config, then evaluate it once on the untouched test split
- [ ] Multi-seed the key H3 configs (error bars on the ±0.01 model/context deltas)
- [x] **Run the core/anchor loss screen** ([`sweep.sh`](scripts/sweep.sh) `loss` preset, 10/15 trials). Result: no re-weighting beats plain cross-entropy, and core-only collapses the anchors without helping the core (§13). Closes the "is the loss weighting a lever?" question in the negative
- [ ] Finish the loss screen's two missing `weighted 3.0/1.0` seeds (43, 44) so the only distinct untested setting is measured rather than noise-bounded; skip `3.0/0.0` entirely — it is the same objective as `1.0/0.0`
- [ ] Canonicalise the `loss` grid on the core-to-anchor ratio, keeping exactly one duplicate pair as a deliberate determinism control (6 of 15 trials are currently redundant)
- [ ] Settle the §12 self-vs-judge ordering with one 150 × 20 `evaluate.py` run on the tuned checkpoint — the 08-25 screen inverted it at 100 × 10
- [ ] Fix the four reporting/selection defects in Open problems: circular `metric_for_best_model` on loss sweeps, the `gen_decoding` fallback and its dead `masking_policy` guard, and `anchor_trim` not reaching the generative eval
- [ ] Re-run the 07-24 batch under core-based checkpoint selection (those 36 runs predate the fix; their saved weights were picked on whole-span)
- [ ] Lock the Phase-2 objective + primary metric with advisor
<!-- AI ACCESS ZONE ENDS -->

- [ ] Project flowchart (0814, outdated)

- **Operation Reclamation** -- Manual code cleaning: Comprehend and rewrite the project content in my own terms - previous work handled by AI resulted in the project almost becoming a "black box" for the developer himself 
  - [ ] `data/` scripts 
    - [x] `dataset.py`, `common.py` (0818)
    - [ ] `preprocess/` scripts
  - [ ] `cdrseqgen/methods/mlm/` 
    - [ ] arguments, backbones
    - [ ] training
    - [ ] sampling
    - [ ] generation, case_generation
    - [ ] evaluation, recovery
    - [ ] length, length_predictor, variable_length_evaluation
  - [ ] `eval/metrics.py`  and `train/` artifact/logging scripts
  - [ ] `train.py` + `train/` (logging)
  - [ ] visualization scripts
  - [ ] `evaluate.py`
  - [ ] `generate.py`
  - [ ] `fit_mlm_length.py`
  - [ ] bash scripts (`run.sh`, `sweep.sh`, `test.sh`)
  - [ ] `benchmark/` scripts (`adapters/`, `export_manifest.py`, `scorer.py`)
  - [ ] `test/` scripts
  - [ ] README

**Datasets**
- OAS
  - [x] 21,106 paired sequences (`1287169_1_Paired_All.csv`)
  - [ ] Remaining paired sequences (→ 1M+ scale)
  - [ ] Unpaired sequences
- SAbDab2
  - [x] 3,984 paired Ab with resolved antigen
  - [ ] Remaining ~10k+ structures without antigen
- Other sources
  - [ ] SAbDab / Thera-SAbDab

**Preprocessing**
- [x] Bio-Object JSON schema (14 IMGT regions)
- [x] Partner-chain context (light ↔ heavy)
- [x] Antigen context (SAbDab2)
- [x] Clustered train/test split (OAS) -- H3 homology-leak-safe
- [x] Optional clustered train/validation/test split for tuning -- preserves the historical test set
- [ ] Structure / coordinate parsing 
- [ ] Unpaired-OAS preprocessor (`preprocess_oas_unpaired.py`)

**Training / methods**
- Sequence-only
  - [x] Masked-LM infilling
    - [x] Fixed-native generation
    - [x] Bounded/inferred generation plumbing and checkpoint artifacts
    - [ ] Learned H3 length predictor that beats the naive mode baseline
    - Models
      - [x] From-scratch transformer (baseline)
      - [x] ESM-2 (general PLM)
      - [x] IgBert (antibody PLM)
      - [x] ProtBert (general PLM)
      - [ ] AbLang2  (antibody PLM)
  - [ ] Autoregressive method integration
    - Models
      - [ ] Custom
      - [x] IgLM external benchmark adapter
  - [ ] Encoder-decoder
    - Models
      - [ ] Custom
- Structure-aware
  - [ ] Invariant Point Attention (IPA)
  - [ ] E(n)-equivariant GNNs
  - [ ] Diffusion (dyMEAN / DiffAb-informed)
  - [ ] Antigen cross-attention / epitope-contact signal  

**Evaluation**
- [x] Sequence recovery (AAR) -- whole-span
- [x] Sequence recovery (AAR) -- core, anchor-excluded
- [x] Perplexity -- self, judge
- [x] Diversity -- pairwise Hamming, unique fraction, per-position entropy
- [x] Novelty -- uniqueness vs train set
- [x] Length-free metrics -- validity, length match/MAE, edit similarity, length-matched AAR
- [x] Matched fixed-length CDRSeqGen/IgLM validation smoke test
- [ ] Matched length-free CDRSeqGen/IgLM score and plot
- [ ] Paired bootstrap confidence intervals
- [ ] Structure metrics - RMSD, TM-score

**Long-term objectives / deliverables**
- Leakage-safe **benchmark** for CDR-infilling methods (the Phase-1 result)
- Structure-conditioned CDR-H3 design
- Integrated sequence + structure design framework
- Benchmark paper and/or a novel antibody-design method
<!-- MANUAL PROTECTION ZONE ENDS -->
