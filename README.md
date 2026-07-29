<!-- MANUAL PROTECTION ZONE BEGINS -->
# CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** 2026-07-29 
**Status:** Operation Reclamation - reorganized code structure for training phase

## Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences given the surrounding framework context. 

Current implemented features include: 
<!-- MANUAL PROTECTION ZONE ENDS -->
- **Data Preprocessing**: standardize data into Bio-Object JSON format (IMGT), current datasets: OAS (21106 paired sequences) and SAbDab2 (3984 paired + antigen structures)
- **Addressing Data Leakage**: clustered train/test split for OAS to prevent clonal near-duplicate leakage
- **Task-Agnostic Masking**: masking training method for all 6 CDR loops individually by target selection
- **Pluggable Backbone PLMs**: supports various PLMs, current list: custom transformer (scratch), ESM-2, IgBert, ProtBert
- **Partner Chain Context**: optional inclusion of non-target chain sequence context
- **Antigen Context**: optional inclusion of antigen sequence context for SAbDab2 dataset
- **Generative Sampling**: independent / iterative (Mask-Predict) decoding with temperature (`0` = greedy), top-k, top-p — samples a *library* of candidate loops instead of one argmax guess
- **Evaluation Metrics**: *recovery-side* — Sequence Recovery Rate (Per-Residue Accuracy), anchor-excluded core accuracy, anchor accuracy. *Design-side* — diversity, novelty, per-position entropy, pseudo-perplexity, plus judge-scored perplexity for cross-model comparison
- **Logging & Visualization**: per-run `train.log` (the whole run's record, including the generative-eval table and any uncaught traceback), TensorBoard logging, csv metric extraction, multi-line graph plotter
- **Extensible Architecture**: modular design for easy addition of new PLMs, datasets, and evaluation metrics

---

## 🏗️ Repository Structure

```text
CDRSeqGen/
├── configs/                    # 48 run configs (flat; run.sh references them by path)
│   ├── h{1,2,3}_scratch[.json|_lc|_ag|_ag_lconly] + h{1,2,3}_esm2[...]
│   ├── h{1,2,3}_igbert[.json|_lc] + h{1,2,3}_protbert[.json|_lc]
│   └── l{1,2,3}_scratch[.json|_hc] + l{1,2,3}_esm2[.json|_hc]
├── cdrseqgen/                  # Library package — grouped by concern (entrypoints stay at root)
│   ├── data/                   # ALL data code (the data itself lives in dataset/data/)
│   │   ├── dataset.py          # Runtime: processed JSON → tensors (tokenizers, Dataset, collator)
│   │   └── preprocess/         # Offline: raw → processed JSON (run once per dataset)
│   │       ├── oas_paired.py       # OAS paired repertoires + the leakage-safe clustered split
│   │       ├── sabdab2.py          # SAbDab2 structures (adds antigen; split ships with the data)
│   │       ├── common.py           # Shared by both: validation, chain assembly, funnel, IO
│   │       ├── download_oas_paired.sh  # Fetch OAS paired repertoires → dataset/data/raw/
│   │       └── download_sabdab2.sh     # Fetch SAbDab2 splits → dataset/data/raw/ (CWD-independent)
│   ├── models/
│   │   ├── registry.py         # Backbone registry — resolve(model_type) -> Backbone
│   │   └── custom/             # In-house architectures (vs. loaded PLMs)
│   │       └── transformer.py  # From-scratch encoder — model_type "scratch"
│   ├── train/                  # What train.py is assembled from (train.py itself is just main())
│   │   ├── arguments.py        # Model/Data/Generation dataclasses + setup_args
│   │   ├── workspace.py        # Logging, train.log, run dirs, results/latest, run_meta.json
│   │   ├── pipeline.py         # get_datasets, train/eval driver, post-training generative eval
│   │   └── reporting.py        # epoch_metrics.csv + the four curve PNGs
│   ├── eval/
│   │   ├── sampling.py         # Decoders (independent / iterative Mask-Predict; temperature 0 = greedy) + pseudo-perplexity
│   │   ├── metrics.py          # Design-side stats: diversity, novelty, per-position entropy
│   │   ├── recovery.py         # Recovery-side: compute_metrics, anchor/core split, core coverage
│   │   └── generation.py       # Shared generative-eval core (used by BOTH train.py and gen_eval.py)
│   └── utils/
│       ├── linegraph_plotter.py  # Training-curve plotting utility
│       └── report_findings.py    # Regenerates the README Findings tables from a batch
├── dataset/data/               # THE DATA — gitignored, ~4.4GB (raw + processed OAS/SAbDab2).
│                               #   Deliberately outside the package; configs' data_path points here.
├── results/                    # Ignored in git. Always results/<batch>/<run>/ (two levels);
│                               #   results/latest -> newest batch dir.
├── test/                       # Small curated frameworks — qualitative showcase for predict.py ONLY
│                               #   (real evaluation uses the held-out split via gen_eval.py)
│   ├── test_h1.json
│   ├── test_h3.json
│   ├── test_h3_lc.json
│   └── test_h3_ag.json
├── train.py                    # Entrypoint — training; main() only, parts in cdrseqgen/train/
├── predict.py                  # Entrypoint — sampling on curated frameworks (qualitative showcase)
├── gen_eval.py                 # Entrypoint — GENERATIVE metrics on a finished run (+ --judge)
├── scripts/
│   ├── run.sh                  # Unified runner: ./scripts/run.sh <gpu> [oas|antigen|lcdr|igbert|protbert|plm|all ...]
│   └── sweep_scratch.sh        # HP sweep for the scratch model (LR × epochs × warmup)
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

---

## 🚀 Workflow & Execution

### Step 1: Preprocessing Data
Convert datasets into a custom "Bio-Object" JSON format, extracting the regions and preserving `ANARCI` numbering.

**OAS**: currently only including 21106 paired sequences from 1287169_1_Paired_All.csv, pending future scale-ups
```bash
python -m cdrseqgen.data.preprocess.oas_paired \
    --input_file dataset/data/raw/1287169_1_Paired_All.csv \
    --output_file dataset/data/processed/processed_oas_data.json
```

**SAbDab2**: currently including 3984 structures - paired with resolved antigen, remaining >10k structures without antigen pending future inclusion
```bash
python -m cdrseqgen.data.preprocess.sabdab2 \
    --input_file dataset/data/raw/sabdab2/splits_final/abag_split.csv \
    --output_file dataset/data/processed/processed_sabdab2_data.json
```

### Step 2: Training & Batch Experimentation
We use `.json` configuration files to manage hyperparameters cleanly. The trainer calculates **Sequence Recovery Rate** (Per-Residue Accuracy), logs to TensorBoard, and auto-generates metric graphs at the end of every run.

**Run a Single Experiment:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py configs/h3_scratch.json
```

**Run an Automated Experiment Suite:** [`scripts/run.sh`](scripts/run.sh) takes a GPU id and one or more suite names, skips to the next config if one fails.
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

**Every run lives at `results/<batch>/<run>/` — always two levels.** A `run.sh` round shares one batch
folder; a standalone `train.py` gets a batch folder of its own, so `results/` never collects loose run
dirs. **`results/latest`** is a symlink to the most recent *batch*, so `results/latest/<config>/`
addresses any run in that round:

```text
results/
├── 20260720_191138/                    ← one ./scripts/run.sh round
│   ├── h1_OAS_scratch/
│   └── h3_OAS_LC_scratch/
├── 20260720_193002/                    ← a single train.py run
│   └── h3_OAS_scratch/
├── sweeps/scratch_20260720_193030/     ← a sweep: grid point in the name
│   ├── h3_OAS_scratch_lr1e-4_ep1_wu0.0/
│   └── h3_OAS_scratch_lr1e-3_ep1_wu0.0/
└── latest -> 20260720_193002/          ← newest batch; predict.py uses latest/<config>
```

The run folder is named from the **resolved arguments**, not the config filename — `h3_scratch.json
--target_region l3` produces an `l3_OAS_HC_scratch` folder, so an override can't be misfiled. The
date/time lives on the batch folder and isn't repeated inside.

**Tag a run** with `CDRSEQGEN_RUN_SUFFIX` to distinguish runs that resolve to the same name:
```bash
CDRSEQGEN_RUN_SUFFIX=ablation_noLC python train.py configs/h3_scratch.json
#  -> results/<batch>/h3_OAS_scratch_ablation_noLC/
```
`sweep_scratch.sh` sets it automatically to the grid point (`lr1e-3_ep20_wu0.06`), so sweep folders are
self-describing. If two runs still collide, the second becomes `..._2` **with a logged warning** — a run
never silently overwrites another. `results/latest` is left untouched if it exists as a real directory.

`sweep_scratch.sh` does the same, putting its runs beside its own `runs.tsv` under
`results/sweeps/scratch_<stamp>/`. Mechanism: the runner exports `CDRSEQGEN_RESULTS_ROOT` and
`train.py` creates its run dir underneath it (see `results_root()`); per-run folder names are
unchanged. A standalone `python train.py …` sets no env var and writes to `./results/<run>/` as before,
so nothing about single runs changes. Export it yourself to group manual runs:
```bash
CDRSEQGEN_RESULTS_ROOT=results/my_experiment python train.py configs/h3_scratch.json
```

**Override a config from the CLI:** trailing flags override the JSON (they do **not** replace it), so any
`TrainingArguments` field is tunable without writing a new config:
```bash
python train.py configs/h3_scratch.json --learning_rate 3e-4 --num_train_epochs 20 --warmup_ratio 0.06
```

**Sweep hyperparameters (scratch):** [`scripts/sweep_scratch.sh`](scripts/sweep_scratch.sh) grids LR × epochs ×
warmup over a base config (~50s per scratch run) and prints a summary sorted by **best-epoch** core accuracy,
flagging runs that overfit after their peak or are still improving at the last epoch.
```bash
./scripts/sweep_scratch.sh <gpu>                                     # default: 3 LRs × 2 epoch budgets
LRS="1e-4 3e-4 1e-3" EPOCHS="10 20" WARMUPS="0.0 0.06" ./scripts/sweep_scratch.sh 4
```
> **Architecture is now sweepable** *(since 2026-07-27)*. `num_attention_heads` and `intermediate_size` are
> `ModelArguments` fields forwarded by `build_model` ([`custom/transformer.py`](cdrseqgen/models/custom/transformer.py)).
> Leave `intermediate_size` unset and it resolves to **4 × `hidden_size`**, so the FFN ratio follows width instead of
> silently drifting (at the old hardcoded 512, widening to 256 halved the ratio and would have confounded width with
> FFN ratio in any architecture sweep). `hidden_size` must stay divisible by `num_attention_heads` — now a loud
> `ValueError` rather than a failure deep inside `nn.MultiheadAttention`.
>
> Also note HF's default scheduler is **linear decay over the total budget**, so `num_train_epochs` reshapes the whole
> LR trajectory — `10` vs `20` compares two schedules, not one schedule at two lengths.
>
> **Early stopping** is available but off by default: set `"early_stopping_patience": N` (and optionally
> `"early_stopping_threshold"`) to stop once `metric_for_best_model` plateaus. Useful because the two families
> misfit `num_train_epochs: 10` in opposite directions — see Open problems.

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

One function draws both comparison shapes, because every line is the same triple `(run_dir, column, label)`.
Each `--run` takes `LABEL FOLDER [METRIC]`; omit `METRIC` and it falls back to the global `--metric`
(default `eval_core_accuracy`):

```bash
# same metric across runs — metric named once
python -m cdrseqgen.utils.linegraph_plotter \
  --metric eval_core_accuracy \
  --run "H3 Scratch" "results/20260724_211745/h3_OAS_scratch" \
  --run "H3 ESM-2"   "results/20260724_211745/h3_OAS_esm2_t12_35M_UR50D" \
  --title "H3 Core Recovery: Scratch vs ESM-2"

# breakdown of ONE run — per-run metric, several columns on one chart
python -m cdrseqgen.utils.linegraph_plotter \
  --run whole  "results/latest/h3_OAS_scratch" eval_accuracy \
  --run core   "results/latest/h3_OAS_scratch" eval_core_accuracy \
  --run anchor "results/latest/h3_OAS_scratch" eval_anchor_accuracy \
  --ylabel Recovery --title "H3 whole vs core vs anchor"
```

A missing CSV or column is reported and skipped rather than aborting the chart, so one bad path in a
multi-run comparison doesn't lose the other lines. Charts default to `results/pngs/`.

<!-- > **⚠️ Never mix datasets on one chart.** OAS (`_OAS`) and SAbDab2 (`_SD2`) runs are evaluated on *different test sets*, so their accuracies are not comparable on shared axes — plot one graph per dataset (e.g. `cdr3_oas` with the 4 OAS runs, `cdr3_sd2` with the 4 SAbDab2 runs). The `results/comparisons/*_oas_*.png` / `*_sd2_*.png` pairs follow this convention. -->

### Step 4: Qualitative Sampling on Curated Frameworks (`predict.py`)
Give it a framework with its CDR loop cut out (`pt1` / gap / `pt2`) and it fills the gap: samples a
library of candidate loops (default **iterative**) and prints them, each with recovery-vs-native (when
you provide the native) and pseudo-perplexity, plus per-framework diversity/novelty. It also reports
which CDR the checkpoint was trained for (read from `run_meta.json`). Samples + metrics are written to
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
> `predict.py` reads `run_meta.json`, prints what the checkpoint expects, and **warns** when a case
> supplies none. Injection goes through the same `inject_context` / `assemble_masked_input` that build
> the training examples, so which side the antigen lands on — and which side gets truncated — follows
> the backbone's `position_type` automatically instead of being hand-encoded in the test file.
>
> There is **no `--target_loop` argument**: nothing in the tensor names a loop or a chain (the dataset
> uses `target_region` only to decide where to *cut*), so the loop is fixed by which checkpoint you
> load — `--model_path` is effectively that argument.

```bash
# design coherent loops for a batch of curated frameworks (default: iterative sampling)
# results/latest is the newest batch — name the config under it (here: h3_OAS_scratch)
python predict.py --model_path ./results/latest/h3_OAS_scratch --test_file test/test_h3.json

# more candidates + nucleus sampling
python predict.py --model_path ./results/latest/h3_OAS_scratch --test_file test/test_h3.json \
    --decoding iterative --num_samples 50 --top_p 0.9

# a single framework straight from the CLI
python predict.py --model_path ./results/latest/h3_OAS_scratch --pt1 EVQL...YYCAR --pt2 WGQGT... --target_length 17
```

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

### Step 5: Generative Evaluation on the Held-Out Split (`gen_eval.py`)
The reportable version of Step 4. Reconstructs the **exact training-time eval split** from the config
(same clustered leakage-safe holdout, same partner/antigen injection, a real native on *every*
framework — 2,111 held-out antibodies for OAS), samples `--num_samples` candidates each, and aggregates
the generative metrics — overall **and bucketed by H3 length**, since difficulty scales with it.

```bash
python gen_eval.py \
    --config configs/h3_esm2.json \
    --model_dir results/20260710/20260710_184547_h3_OAS_esm2_t12_35M_UR50D \
    --num_examples 150 --stratify --num_samples 20 \
    --decoding iterative --top_p 0.9 \
    --judge Exscientia/IgBert
```

> **⚠️ Self-perplexity is NOT comparable across models.** Pseudo-perplexity is scored under each model's
> *own* distribution, so it measures self-consistency — a model confident in its own outputs looks good
> regardless of quality. To compare models, pass `--judge <model_type>`: every candidate is then also
> scored by one **independent** reference LM (IgBert is a good pick — antibody-specific, and independent
> of both scratch and ESM-2). Diversity/entropy carry no such caveat; they're purely sequence-based.

`--num_examples K` subsamples the split (`-1` = all; default 200); `--stratify` balances the draw across
short/medium/long H3 buckets so rare long loops aren't washed out. Results land in
`<model_dir>/generation_eval.json`.

#### Generative evaluation runs automatically after every training run
Once all epochs finish, `train.py` samples designs from the finished best model and writes
`generation_eval.json` beside `epoch_metrics.csv`. **On by default** — no config change needed.

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
"gen_top_p": 0.9
```

Set `run_generative_eval: false` (or `gen_judge: ""` to keep the metrics but skip the judge) to opt out;
trim `gen_num_examples` for cheap runs. Cost is ~80s of sampling + ~15s to load the judge per run.

Both paths share one implementation ([`cdrseqgen/eval/generation.py`](cdrseqgen/eval/generation.py)) —
they differ only in where the model comes from. `gen_eval.py` stays the tool of choice for scoring
checkpoints you already trained, or re-sweeping temperature / top-p / judge without retraining.

---

## 📊 Findings

Phase 1 (the sequence sweep) is a **leakage-safe benchmark** for CDR infilling. What it establishes:

*Numbers below are from the complete 36-run batch `results/20260724_211745` (6 loops × {scratch, ESM-2} ×
context × {OAS, SAbDab2}, 10 epochs each), read at **best epoch**, core metric unless stated.*

> **Every table here is generated, not hand-written** — regenerate them all after any new batch with:
> ```bash
> python -m cdrseqgen.utils.report_findings --batch results/20260724_211745
> python -m cdrseqgen.utils.report_findings --only regimes,loops   # or a subset
> ```
> It reads only what a finished run already writes (`epoch_metrics.csv`, `generation_eval.json`,
> `run_meta.json`) plus the processed dataset JSONs, and prints markdown ready to paste. The one
> exception is the decoding table (Step 4), which needs two `gen_eval.py` runs differing only in
> `--decoding`, passed via `--decoding-json` (the pair used above is kept alongside that run as
> `generation_eval_decoding_{iterative,independent}.json`).

**1. "Model choice doesn't matter" is only true in two of three regimes.** On OAS H3 a 128-dim from-scratch transformer matches ESM-2 (0.3148 vs 0.3209 core). But that generalises badly — whether pretraining pays depends entirely on how much of the test set is *memorisable* from train (see §6):

| regime | example | test CDRs unseen in train | scratch → ESM-2 |
|---|---|---|---|
| memorisable | OAS H1 | 1.6% | 0.9928 → 0.9935 (**+0.001**) |
| learnable, not memorisable | SD2 H1 | 57.5% | 0.4349 → **0.6657** (+0.231) |
| " | SD2 H2 | 79.6% | 0.3277 → **0.6096** (+0.282) |
| neither (information ceiling) | OAS H3 | 100% | 0.3148 → 0.3209 (+0.006) |

Pretraining buys ~nothing when the answer can be looked up, **+0.23–0.28** when the task needs genuine generalisation, and ~nothing again at the H3 ceiling. The original claim was drawn from regimes 1 and 3 only.

**2. CDR-H3 is the only hard loop — but the other five are easier than they look.** Best-epoch recovery on OAS (core; L2 whole-span, see "How to read the numbers"):

| | scratch | +ctx | ESM-2 | +ctx |
|---|---|---|---|---|
| H1 | 0.9928 | 0.9825 | 0.9935 | 0.9934 |
| H2 | 0.9897 | 0.9617 | 0.9899 | 0.9903 |
| **H3** | **0.3148** | 0.3063 | **0.3209** | **0.3240** |
| L1 | 0.9928 | 0.9651 | 0.9938 | 0.9937 |
| L2 | 0.9921 | 0.9590 | 0.9932 | 0.9926 |
| L3 | 0.9252 | 0.8466 | 0.9274 | 0.9268 |

The biology is right — heavy CDR3 gets D-gene + junctional diversity, the rest is germline-templated — but the ~0.99s are **not** evidence the model generalises; under the current split they are largely retrieval (§6). Only H3 is measured on entirely unseen loops.

**3. The standard metric misleads (anchor inflation).** Whole-span `eval_accuracy` (~0.46 on H3) is dominated by the 2-3 V/J-gene-determined anchor residues at each loop end (~0.90 recovery, any model); the hypervariable middle actually being "designed" is ~0.32. The gap is **H3-specific** (H1/H2/L core ≈ whole-span). Measured via `core_accuracy`/`anchor_accuracy` (`compute_metrics` in `train.py`); the position-resolved breakdown below is from the standalone re-scorer `eval_recovery.py` (archived in `old/`). Illustrative (`h3_esm2`):

| Position from start | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 100% | 89% | 35% | 21% | 20% | 14% |

**4. Sequence context — partner chain and antigen — gives no measurable benefit.** The valid same-data antigen ablation (`_ag` vs `_ag_lconly`, SAbDab2, all 3 CDRs) shows no signal, and it *survives the core metric*:

| CDR | scratch lc-only → +ag | ESM-2 lc-only → +ag |
|---|---|---|
| H1 | 0.4349 → 0.4036 | 0.6657 → 0.6664 |
| H2 | 0.3277 → 0.3111 | 0.6096 → 0.5900 |
| H3 | 0.1337 → 0.1362 | 0.2214 → 0.2128 |

|Δ| ≤ 0.031, sign inconsistent, 5 of 6 comparisons negative. Not a truncation artifact (84% of complexes keep the full antigen at `max_len=512`). Partner-chain context is worse than neutral for the small model — it hurts scratch on **all six** loops (−0.009 to −0.079, largest on L3) while leaving ESM-2 unmoved (−0.001 to +0.003), which reads as capacity dilution rather than a signal. **Interpretation:** binding information is geometric, not linear-sequence — so the next lever is structure / cross-attention, not more sequence context.

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
| H1 scratch | 0.991 | **0.002** | **0.006** | 1.05 |
| H1 ESM-2 | 0.960 | 0.010 | 0.024 | 1.56 |
| H3 scratch | 0.322 | 0.407 | 1.229 | 12.42 |
| H3 ESM-2 | **0.406** | 0.573 | 1.841 | **7.14** |

H1 scores 0.99 AAR while emitting essentially **one** sequence with total certainty (perplexity ≈ 1.0) — a perfect copier and a useless designer. And on H3 the two metrics rank the models differently: recovery calls scratch and ESM-2 near-equal, the independent judge puts ESM-2 **1.7× ahead**. Recovery alone is insufficient to choose a model, which is the practical argument for a design-aligned objective.

> **Bottom line:** sequence-only CDR-H3 infilling saturates at ~0.32 core / ~0.46 whole-span regardless of model, context, dataset, or decoder — and the non-H3 loops are not yet measured on a split that tests generalisation. Together these motivate the shift to a **structure objective** (see Roadmap).

---

## 🧠 Architectural Feature Details

### 1. Task-Agnostic Masking (The "LEGO" Principle)
The `AntibodyInfillingDataset` separates sequences into `pt1` (left context), `mask` (target gap), and `pt2` (right context). The `target_region` string alone chooses the loop — `h1/h2/h3` mask a heavy CDR, `l1/l2/l3` mask the *light* chain's CDR (legacy `cdr*`/`lcdr*` spellings still parse). Switching target requires changing zero core tokenization logic. Labels are strictly length-enforced to prevent cross-entropy dimension mismatches.

### 2. Pluggable Backbones (the extensibility core)
The model registry [`registry.py`](cdrseqgen/models/registry.py) holds one `Backbone` entry per model family; `train.py`, `predict.py`, and `dataset.py` never branch on model type — they all go through `resolve(model_type)`, which returns the matching backbone and a uniform `TokenizerAdapter`. The same pipeline trains the from-scratch transformer, fine-tunes ESM-2, or fine-tunes any Hugging Face masked-LM just by changing the `"model_type"` string.

Each backbone declares two conventions **explicitly** instead of letting them be guessed from the tokenizer (the previous code overloaded a single `is_esm` boolean to mean three different things, which silently mishandled non-ESM models):
- **`position_type`** (`"absolute"` | `"relative"`) — controls where antigen context is injected (see §4). Absolute-position models (the scratch transformer, BERT-style protein LMs) need it appended after the target; relative/rotary models (ESM-2) can take it prepended.
- **`uses_special_tokens`** — whether inputs are wrapped as `[cls] … [end]`. The `TokenizerAdapter` also resolves the ESM-vs-BERT separator difference generically (`<eos>` when the tokenizer has one, else `<sep>`), so BERT-style tokenizers with `eos_token_id = None` work without special-casing.

**Adding a new PLM is usually one `register(Backbone(...))` entry** — occasionally *zero*, when a model is already covered by the generic `hf_encoder` fallback (any `AutoModelForMaskedLM` + `AutoTokenizer`, absolute positions, per-character vocab — just pass its hub id as `model_type`). In practice most protein LMs need an explicit entry: e.g. BERT-style ones (ProtBert, IgBert) expect *whitespace-separated* residues and ship tokenizer/loader quirks (a fast-tokenizer that needs `sentencepiece`, or a `config.json` with no `model_type`), all handled once in the `protein_bert` backbone. Either way, no edits to `train.py`/`predict.py`/`dataset.py` are needed.

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
- **New model** → one `Backbone` entry in [`registry.py`](cdrseqgen/models/registry.py) (§2) — `train.py`/`predict.py`/`dataset.py` never learn about it.
- **New target loop** → change the `target_region` string (§1); masking logic is untouched.
- **New context** → one boolean flag (`include_partner_chain`, `include_antigen`), resolved symmetrically (§3–4).
- **New dataset source** → one `(substring, tag)` pair in `train.py`'s `DATASET_TAGS` registry (`oas → OAS`, `sabdab2 → SD2`), so every run's output-dir name states its source explicitly instead of leaving it silently implied; an unrecognised path is tagged `DS` **with a logged warning** rather than left blank.

The guiding constraint: **conventions are declared explicitly, never inferred.** The predecessor overloaded a single `is_esm` boolean to mean three different things and silently mishandled non-ESM models; each backbone now states its `position_type` and `uses_special_tokens` up front. This isn't free — protein-BERT models (IgBert/ProtBert) still needed their own entry for residue-spacing and tokenizer/loader quirks (§2) — but the cost is paid once, in the registry, not scattered across the pipeline.

---

## ⚠️ Current Issues 
*(current as of 2026-07-28; update per commit)*

### Open problems
- **The split only controls the loop it clusters on.** `add_split_field` clusters heavy CDR3, so the holdout is leakage-safe for H3 and **not** for H1/H2/L1/L2/L3 — 98% of OAS test H1s appear verbatim in train (Findings §6). Every non-H3 number in this README is therefore retrieval-bounded and should not be read as generalisation. Fix: cluster on the *target* CDR (per-loop split), a contained change to `add_split_field`; then re-run the OAS matrix.
- **Training still optimizes recovery, not design.** The masked-LM objective rewards reproducing the native loop; the generative metrics only *measure* design quality after the fact — nothing trains for it. Findings §7 shows the two can disagree outright (H1: 0.99 recovery, 0.002 diversity). A design-aligned objective stays open (Phase 2).
- **The judge shares the eval distribution.** IgBert was trained on OAS paired repertoires — the same distribution as the OAS eval split it judges. It's applied identically to every model, so the *comparison* is fair, but absolute perplexities are OAS-flavoured; an off-distribution judge would be a stronger check.
- **Generative sweep is H3/OAS only.** The matrix covered CDR-H3 on OAS (150 held-out frameworks × 4 models). L-chain loops, SAbDab2, and the antigen configs have not been swept generatively.
- **Sampler is marginal-based, not learned-joint — now measured.** The `iterative` (Mask-Predict) decoder approximates the loop joint by conditioning on committed residues, but the model is still a masked-LM trained on independent-position recovery. As of 2026-07-28 this is no longer a suspicion: iterative and `independent` are **within noise on every metric** on H3/OAS/ESM-2 (recovery 0.4057 vs 0.4072; judge-ppl 7.14 vs 7.23) at ~15× the compute. The conditioning buys nothing because the learned marginals are already near-independent — true joint/generative *training* (autoregressive or diffusion) is Phase-2 territory.
- **Hyperparameters were never tuned.** *(Partly addressed 2026-07-27: opt-in early stopping is now available — set `"early_stopping_patience": N` in a config to stop once `metric_for_best_model` plateaus. Off by default, so nothing changes unless you ask for it. The epoch counts below are still un-right-sized.)* All 48 configs pin one LR per family (scratch `1e-4`, pretrained `5e-5`) and exactly `num_train_epochs: 10`, with no warmup / weight-decay / scheduler set (HF defaults throughout). The curves show 10 is wrong for **both**, in opposite directions: `h3_esm2` peaks at **epoch 4** (core 0.3209) then decays to 0.3063 by 10 — six wasted epochs, rescued only by `load_best_model_at_end` — while `h3_scratch` is still creeping at 10 (~0.001 loss/epoch). *Mitigating evidence for Finding #1:* the two reach ~0.32 core from **opposite directions** (ESM-2 from above in 1–4 epochs, scratch from below over 10), which argues the ceiling is real rather than a shared-settings artifact. Sweep with [`scripts/sweep_scratch.sh`](scripts/sweep_scratch.sh).
- ~~**Checkpoint selection uses the metric we say is misleading.**~~ **Fixed 2026-07-27.** 44 of 48 configs now set `"metric_for_best_model": "core_accuracy"` (HF resolves it to `eval_core_accuracy`). The four `l2_*` configs deliberately stay on whole-span `accuracy`: L2 core is computed over ~1.4% of eval rows, so selecting checkpoints on it would be selecting on noise — the same exception documented under "How to read the numbers". Runs before this date selected on whole-span and aren't checkpoint-comparable to later ones.
- ~~**Scratch architecture isn't tunable.**~~ **Fixed 2026-07-27.** `num_attention_heads` and `intermediate_size` are now `ModelArguments` fields forwarded by `build_model` ([`custom/transformer.py`](cdrseqgen/models/custom/transformer.py)). `intermediate_size` defaults to `None` = **4 × hidden_size**, so the FFN ratio tracks width instead of silently drifting (at the old fixed 512, widening to 256 quietly halved the ratio and would have confounded any architecture sweep). Defaults reproduce the old geometry exactly (128 → 512), and a `hidden_size % num_attention_heads` check now fails loudly instead of deep inside `nn.MultiheadAttention`.
- **Single-seed runs.** Small differences (e.g. light chain hurts scratch ~0.01, neutral for ESM-2) need multiple seeds + error bars before they can be claimed.
- **Structure not yet available.** SAbDab2 preprocessing parses *sequences* only; the Phase-2 structure objective needs a coordinate-parsing step (or predicted structures via IgFold/ABlooper).

### How to read the numbers 
- **Report core, not whole-span** — whole-span is anchor-inflated on H3 (Findings §3).
- **…except for L2, where core is meaningless.** `ANCHOR_TRIM=2` trims 2 residues from each end, so a loop needs length > 4 to contribute *any* core token — but **98.8% of light-chain CDR2 loops are 3 residues** (20,847/21,106). L2 core is therefore computed over ~**30 of 2,111** eval examples (1.4%), and its values are noise: the 07-17 batch shows L2 core of `1.0000` for scratch and `0.5957` for scratch+HC, while the *valid* whole-span numbers are `0.9921` vs `0.9590`. **Report whole-span for L2**; H1/H2/H3/L1/L3 are unaffected (2110-2111/2111 eval entries contribute core). This is now **self-detecting**: every run records **`eval_core_coverage`** (fraction of eval rows with any core residue) in `epoch_metrics.csv`, and `train.py` warns once when it drops below 50%. A coverage well under 1.0 means the core number is noise.
- **Every recovery number is conditioned on the native loop length — the model is *told* it, never predicts it.** `dataset.py` inserts exactly `len(native)` mask tokens, so the length is given away by the input's shape. This is the standard fixed-length-redesign protocol (RAbD, DiffAb, MEAN all do it) and is *required* for position-wise AAR to be well-defined at all — but it is not free information, and how much it's worth varies enormously by loop:

  | loop | distinct lengths | mode freq | free info from knowing L |
  |------|------------------|-----------|--------------------------|
  | H3   | 34               | 9.9%      | **4.05 bits**            |
  | L3   | 14               | 43.4%     | 2.13 bits                |
  | L1   | 6                | 54.3%     | 2.01 bits                |
  | H2   | 6                | 62.0%     | 1.24 bits                |
  | H1   | 4                | 83.8%     | 0.73 bits                |
  | L2   | 3                | 98.8%     | **0.10 bits**            |

  On H3 that's ~4 bits handed over per example — roughly 10% of the loop's total information content (a 15-mer at the measured ~2.45 bits/position ≈ 37 bits). On L2 it's nothing: guess 3 and you're right 98.8% of the time. So **H3 recovery is the number most flattered by this**, and cross-loop comparisons of AAR are partly comparing how much length information each loop gave away. It also means the models cannot currently be used for length-free *design* — see Roadmap.
- **Read best-epoch, not last-epoch** — ESM-2 overfits on SAbDab2/H3; `load_best_model_at_end` saves the peak, but the last CSV row can sit well below it.
- **Never plot OAS and SAbDab2 on shared axes** — different test sets; the `_OAS`/`_SD2` run tags enforce it, and per-dataset charts live in `results/pngs/*_{oas,sd2}_*.png`.
- **`epoch_metrics.csv` and `*_results.json` measure different models — they're *supposed* to disagree.** The CSV is the per-epoch **trajectory** (the model as it stood at each epoch); `eval_results.json`/`all_results.json` come from the post-training `trainer.evaluate()`, which with `load_best_model_at_end` runs on the **best** checkpoint while still being stamped with the final epoch number. On `h3_esm2` the CSV's epoch-10 row reads core **0.3063** (the actual last model) while `eval_results.json` reads **0.3209** (the epoch-4 best model). `save_training_csv` de-duplicates on epoch and **keeps the first** occurrence precisely so the curve keeps the real last-epoch point — keeping the last would silently redraw the best model at x=10 and erase the overfitting decay entirely.

### Changelog
<!-- Latest updates can be added here before protection zone -->
- **2026-07-29** -- `predict.py`: added `--partner_chain` / `--antigen` / `--max_len`, routed through the same `inject_context` the dataset uses; reads `run_meta.json` and warns when a checkpoint's expected context is missing 
<!-- MANUAL PROTECTION ZONE BEGINS -->
- **2026-07-29** -- Split `train.py` into `cdrseqgen/train/{arguments,workspace,pipeline,reporting}.py`
- **2026-07-29** -- `train.py`: fixed `train.log`, records per epoch (used to only have "Starting Training.../Starting Evaluation...") (pending verification)
- **2026-07-29** -- [RECLAIM] `dataset.py`: extracted `assemble_masked_input` and `inject_context` (partner/antigen injection + length capping) functions to be used in training and inference (`dataset.py` still under control)
- **2026-07-29** -- `utils/report_findings.py`[new]: regenerates every README Findings table from a batch (`python -m cdrseqgen.utils.report_findings`)
- **2026-07-27** -- [RECLAIM] Restructured `models/` directory (`custom/transformer.py`[`custom.py`]: rename; `registry.py`[`models.py`]: rename)
- **2026-07-27** -- Fixed print/logging issues in `models.py`, `generation.py`
- **2026-07-27** -- `data/preprocess/common.py`[new]: common scaffolding for csv -> json data preprocessing scripts
- **2026-07-27** -- moved and renamed preprocessing scripts to `data/preprocess/`, removed separate dataset README file
- **2026-07-27** -- [RECLAIM] Reclaimed: `dataset.py`, data preprocessing scripts
- **2026-07-24** -- `run_meta.json` provenance: `train.py` records the CDR loop (+ model/context) each run was trained for
- **2026-07-24** -- fixed `results/latest/` symlink to latest batch (to verify further)
- **2026-07-24** -- [RECLAIM] `linegraph_plotter.py`: enables multi-run + multi-metric breakdown all in one function
- **2026-07-24** -- [RECLAIM] Reclaimed: `metrics.py`, `sampling.py`, `generate.py`, `gen_eval.py`
- **2026-07-24** -- `linegraph_plotter.py`: moved to new `utils/` folder
- **2026-07-24** -- `sampling.py`: handled greedy algo for both independent and iterative decoders
- **2026-07-21** -- `eval_recovery.py`: temporarily discarded (now in `old/`)
- **2026-07-21** -- `train.py`: Gen eval after per training run
- **2026-07-20** -- Fixed issue with anchor excluded AAR for short CDRs
- **2026-07-20** -- Edited results directory naming (`[timestamp]/[configs]/` (timestamp at time of `run.sh` round start))
- **2026-07-20** -- Enabled line graph plotting for whole + core (no anchor) AARs for each run
- **2026-07-20** -- Run hyperparameter sweep on scratch model 
- **2026-07-20** -- [RECLAIM] Begin manual code cleaning plan (*Codename: Operation Reclamation (lol)*) (list - see roadmap)
- **2026-07-17** -- Data restructure, preparation for hyperparameter sweep on scratch model
- **2026-07-17** -- `eval_generative.py`[new]: generative eval on the held-out split (2,111 frameworks — real natives, leakage-safe, injection reused so nothing drifts)
- **2026-07-17** -- `cdrseqgen/eval/generation.py`[new]: generative eval metrics: diversity, novelty, pseudo-perplexity, per-position entropy
- **2026-07-17** -- `train.py`: opt-in post-training generative eval (`"run_generative_eval": true`)
- **2026-07-16** -- `sampling.py`[new]: decoders (greedy / independent / iterative Mask-Predict) + pseudo-perplexity scoring
- **2026-07-16** -- `metrics.py`[new]: diversity / novelty / per-position entropy 
- **2026-07-16** -- Repo restructure
- **2026-07-15** -- Polished README (half-manually), updated roadmap (pending manual edit)
<!-- MANUAL PROTECTION ZONE ENDS -->

---

## ⏭️ Roadmap

> **Objective-first:** the target is **CDR-H3 design**. **Phase 1** = a leakage-safe *sequence benchmark* (done — establishes the ~0.32 core ceiling; see Findings). **Phase 2** = *structure-conditioned* design (RMSD-to-native; exact target to lock with advisor). Everything below is a tool toward that objective.

**General — next steps**
- [x] Sampler in `predict.py` (temperature / top-k / top-p, independent + iterative Mask-Predict) + generative metrics ([`sampling.py`](cdrseqgen/eval/sampling.py), [`metrics.py`](cdrseqgen/eval/metrics.py))
- [x] Sweep the generative metrics across the H3 model/context matrix (150 held-out frameworks × 4 models, IgBert judge)
- [x] **HP hygiene** — `metric_for_best_model` → `core_accuracy` on 44/48 configs (the four `l2_*` stay whole-span, see "How to read the numbers"); opt-in early stopping wired in. *Epoch budgets are still un-right-sized — see Open problems.*
- [x] Plumb `num_attention_heads` / `intermediate_size` through `build_model` — `intermediate_size` now defaults to 4 × `hidden_size` (prerequisite for any architecture sweep)
- [ ] **Per-loop split** — cluster on the *target* CDR, not always heavy CDR3, then re-run the OAS matrix. Everything non-H3 in Findings is retrieval-bounded until this lands (§6)
- [ ] LR sweep on scratch ([`sweep_scratch.sh`](scripts/sweep_scratch.sh)) — cheap insurance for "model choice doesn't matter"; scratch is the model most likely under-tuned
- [ ] Multi-seed the key H3 configs (error bars on the ±0.01 model/context deltas)
- [ ] Re-run the 07-24 batch under core-based checkpoint selection (those 36 runs predate the fix; their saved weights were picked on whole-span)
- [ ] Lock the Phase-2 objective + primary metric with advisor

<!-- MANUAL PROTECTION ZONE BEGINS -->
- [ ] Project flowchart
 - [ ] Data section
 - [ ] Models section
 - [ ] Eval section
 - [ ] Train section
 - [ ] Analysis & Visualization section
- **Operation Reclamation** -- Manual code cleaning: Comprehend and rewrite the project content in my own terms - previous work handled by AI resulted in the project almost becoming a "black box" for the developer himself 
 - [x] `metrics.py`
 - [x] `sampling.py`
 - [x] `generation.py`
 - [x] `gen_eval.py`
 - [ ] `predict.py`
 - [x] `linegraph_plotter.py`
 - [ ] `report_findings.py`
 - [x] `custom/transformer.py` 
 - [x] `registry.py` 
 - [x] `dataset.py` 
 - [x] preprocessing scripts
 - [ ] `run.sh`
 - [ ] hyperparameter tuning script (`sweep_scratch.sh`)
 - [ ] `train/arguments.py` 
 - [ ] `train/pipeline.py` 
 - [ ] `train/reporting.py` 
 - [ ] `train/workspace.py` 
 - [ ] `train.py` 
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
- [x] Clustered split (OAS) -- homology-leak-safe 
- [ ] Structure / coordinate parsing 
- [ ] Unpaired-OAS preprocessor (`preprocess_oas_unpaired.py`)
<!-- MANUAL PROTECTION ZONE ENDS -->

**Training / methods**
- Sequence-only
  - [x] Masked-LM infilling objective
  - Models
    - [x] From-scratch transformer (baseline — *not* pretrained)
    - [x] ESM-2 (general PLM)
    - [x] IgBert (antibody PLM)
    - [x] ProtBert (general-protein PLM)
    - [ ] AbLang2  *(low priority on OAS, where model choice barely moves H3 — but pretraining is worth +0.23–0.28 on SAbDab2, so worth revisiting there; Findings §1)*
    - [ ] IgLM  *(autoregressive — needs its own generation/scoring path)*
- Structure-aware  *(Phase 2)*
  - [ ] Invariant Point Attention (IPA)
  - [ ] E(n)-equivariant GNNs
  - [ ] Diffusion (dyMEAN / DiffAb-informed)
  - [ ] Antigen cross-attention / epitope-contact signal  *(the flat-sequence version was inert)*

**Evaluation**
- [x] Sequence recovery (AAR) — whole-span
- [x] Anchor-excluded core recovery (`compute_metrics` in `train.py`; standalone re-scorer `eval_recovery.py` archived in `old/`)
- [x] Pseudo-perplexity — masked-LM PLL over the loop ([`sampling.py`](cdrseqgen/eval/sampling.py))
- [x] Diversity & novelty — pairwise Hamming, unique fraction, per-position entropy, novelty-vs-native ([`metrics.py`](cdrseqgen/eval/metrics.py))
- [ ] Novelty vs. *training set* (nearest-neighbour CDR3 identity — needs the train CDR3 index)
- [ ] Structure metrics — RMSD, TM-score  *(Phase 2)*

**Long-term objectives / deliverables**
- Leakage-safe **benchmark** for CDR-infilling methods (the Phase-1 result)
- Structure-conditioned CDR-H3 design
- Integrated sequence + structure design framework
- → benchmark paper and/or a novel antibody-design method