<!-- Note to AI: Edits allowed OUTSIDE of MANUAL PROTECTION ZONES and INSIDE of AI ACCESS ZONES, and DO NOT COMMIT ON MY BEHALF -->
# CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** 2026-10-06
**Status:** AR/S2S context-dropout and ESM-2 LR screens complete; matched controls and confirmation pending

## Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences given the surrounding framework context. 

Current implemented features include:

**Data**
- **Preprocessing**: IMGT Bio-Object JSON; current OAS4 corpus: 152,727 paired sequences from four donors (122,181 train / 15,273 validation / 15,273 test). The earlier one-donor corpus has 21,106 pairs. SAbDab2 has 3,984 paired antibodies with antigen (3,234 train / 750 test). Preprocessing can merge multiple donor CSVs and split the merged corpus once.
- **Leakage-controlled splits**: H3-clustered OAS train/validation/test splits and explicit validation selection. Representative clustering does not guarantee pairwise sequence separation; see [Open problems](#open-problems).

**Conditioning context**
- **Partner chain**: optional inclusion of non-target chain sequence context
- **Antigen**: optional inclusion of antigen sequence context for SAbDab2 dataset
- **Context dropout**: `context_dropout` trains one checkpoint that works with *and* without its optional context (partner chain / antigen drawn independently), instead of a model locked to the regime it was trained in

**Infilling methods**
- **Masked infilling**: masking training method for all 6 CDR loops individually by target selection
- **Autoregressive infilling**: causal span infilling with both framework flanks in the prompt, answer-only loss, rotary positions, and model-chosen loop length via `<eos>`
- **Encoder-decoder infilling**: two-stream span infilling — a *bidirectional* encoder over `pt1 <mask> pt2` and a cross-attending decoder that emits residues until `<eos>`. Unlike the causal layout the loop never shares a sequence with the framework, so no answer reserve competes with optional context for `max_len`
- **Pluggable backbones**: supports various PLMs, current list: custom transformer (scratch), ESM-2, IgBert, ProtBert

**Training & experiments**
- **Batch experiment runner**: one planner/executor behind `run.sh` drives named suites, hyperparameter sweeps, and declarative multi-arm specs; it shards trials across GPUs, freezes each batch as a `plan.json` (merged configs, dataset hashes, source identity) for exact resume, skips runs whose artifacts are already complete, and writes the round's tables and dashboards when it finishes

**Generation**
- **Sampling**: independent / iterative (Mask-Predict) decoding with temperature (`0` = greedy), top-k, top-p — samples a *library* of candidate loops instead of one argmax guess
- **Length policies**: fixed, bounded, and inferred-length MLM generation; inferred mode classifies one gap sentinel and expands it into residue masks without reading the native loop length

**Evaluation**
- **Sequence metrics**: length-matched amino-acid recovery (AAR), core/anchor recovery, positional and globally aligned BLOSUM62, edit similarity, length error, diversity, uniqueness and per-position entropy. `novelty_mean` is `1 - AAR` relative to the native loop.
- **Confidence & plausibility**: generator self-perplexity and optional masked-LM judge pseudo-perplexity, plus confidence-ranking summaries. Cross-model judge comparisons require matched weights, tokenization, context and scoring populations; generator and judge context are configured separately.
- **Matched external benchmark**: versioned manifests, model-specific CDRSeqGen/IgLM adapters, shared contract validation and scoring, and optional independent-judge perplexity

**Engineering**
- **Logging & visualization**: per-run `train.log` (the whole run's record, including the generative-eval table and any uncaught traceback), TensorBoard logging, csv metric extraction, multi-line graph plotter
- **Extensible architecture**: method packages isolate each method's assumptions so a new one reuses shared data, metrics, artifacts, and benchmark contracts. A method declares itself in `methods/registry.py` and fills a five-function CLI protocol; the dispatchers, `--decoding` choices and dashboard families all derive from that registry rather than from hard-coded lists

---

## Contents

**Start here** · [Project overview](#project-overview) · [Repository structure](#-repository-structure) · [Environment setup](#-environment-setup)

**[Workflow & Execution](WORKFLOW.md)** — how to run everything
&nbsp;&nbsp;[Architecture overview](WORKFLOW.md#architecture-overview) ·
[1 Preprocessing](WORKFLOW.md#step-1-preprocessing-data) ·
[2 Training & batches](WORKFLOW.md#step-2-training--batch-experimentation) ·
[3 Visualization](WORKFLOW.md#step-3-result-visualization--comparison) ·
[4 Sampling (`generate.py`)](WORKFLOW.md#step-4-qualitative-sampling-on-curated-frameworks-generatepy) ·
[5 Held-out eval (`evaluate.py`)](WORKFLOW.md#step-5-generative-evaluation-on-the-held-out-split-evaluatepy) ·
[Benchmark](WORKFLOW.md#matched-external-benchmark)
&nbsp;&nbsp;*also:* [a finished round reports itself](WORKFLOW.md#a-finished-round-reports-itself) ·
[auto-eval after training](WORKFLOW.md#generative-evaluation-runs-automatically-after-every-training-run)

**[📊 Findings](#-findings)** — what Phase 1 actually establishes

| | | |
|---|---|---|
| [1 Model choice pays only in some regimes](#1-model-choice-doesnt-matter-is-only-true-in-two-of-three-regimes) | [6 The split controls H3 and nothing else](#6-the-clustered-split-controls-h3-and-nothing-else--so-the-099s-are-retrieval) | [11 Matched IgLM comparison, with paired CIs](#11-matched-length-free-comparison-iglm-leads-on-plausibility-and-length-and-the-recovery-gap-does-not-resolve) |
| [2 H3 is the only hard loop](#2-cdr-h3-is-the-only-hard-loop--but-the-other-five-are-easier-than-they-look) | [7 Recovery and design quality disagree](#7-recovery-and-design-quality-disagree--high-aar-can-mean-a-perfect-copier) | [12 Model confidence is useless for selection](#12-the-models-own-confidence-is-nearly-useless-for-selection--an-independent-judges-is-not) |
| [3 Anchor inflation — the metric misleads](#3-the-standard-metric-misleads-anchor-inflation) | [8 Context-regime lock-in, and `context_dropout`](#8-a-model-is-locked-to-the-context-regime-it-was-trained-in--and-context_dropout-unlocks-it) | [13 Loss re-weighting changes nothing](#13-re-weighting-the-loss-across-loop-positions-changes-nothing--the-anchors-are-not-wasted-capacity) |
| [4 Historical MLM context comparisons](#4-sequence-context--partner-chain-and-antigen--gives-no-measurable-benefit) | [9 One hyperparameter inflated four findings](#9-one-un-audited-hyperparameter-inflated-four-findings-at-once) | [14 AR/S2S context-dropout trade-off](#14-context-dropout-improves-missing-partner-robustness-with-a-small-context-present-cost) |
| [5 The homology leak (immaterial for H3)](#5-the-homology-leak-was-real-but-immaterial-for-h3) | [10 Length-free exposes the length problem](#10-removing-native-h3-length-exposes-a-separate-unsolved-problem) | [15 ESM-2 learning-rate screen](#15-esm-2-learning-rate-tuning-improves-teacher-forced-recovery-more-clearly-than-generation) |

**[🧠 Architectural Feature Details](#-architectural-feature-details)**
&nbsp;&nbsp;[1 Task-agnostic masking](#1-task-agnostic-masking-the-lego-principle) ·
[2 Pluggable backbones](#2-pluggable-backbones-the-extensibility-core) ·
[3 Partner-chain context](#3-partner-chain-context-injection) ·
[4 Antigen context](#4-antigen-context-injection-sabdab2) ·
[5 Tracking & evaluation](#5-automated-tracking--evaluation) ·
[6 Extensibility principles](#6-extensibility-principles)

**[⚠️ Current Issues](#-current-issues)** · [Open problems](#open-problems) · [How to read the numbers](#how-to-read-the-numbers) · [Changelog](#changelog)

**[⏭️ Roadmap](#-roadmap)** · [Active milestone — CDRSeqGen vs IgLM](#active-milestone--matched-cdrseqgen-vs-iglm-benchmark)

---

## 🏗️ Repository Structure

`scripts/` holds thin command-line launchers; `cdrseqgen/` holds executable Python logic.
The launch direction is `run.sh → python -m cdrseqgen.experiments → train.py`; `sweep.sh`
forwards to `run.sh`. Single-run settings live in `configs/{mlm,ar,s2s}/`, multi-run recipes in
`configs/experiments/`, and resolved plans and outputs in `results/runs/`.

```text
CDRSeqGen/
├── configs/                    # Run configs and experiment recipes; no executable code.
│                               #   A config's "method" key selects the pipeline (default mlm)
│   ├── mlm/                    # Masked infilling: heavy/, light/, antigen/
│   ├── ar/                     # Autoregressive heavy-chain infilling
│   ├── s2s/                    # Encoder–decoder H3: scratch and pretrained encoders
│   ├── experiments/            # Dated multi-arm specs and protocols
│   └── README.md               # Config selection and naming guide
├── cdrseqgen/                  # Library package — grouped by concern (entrypoints stay at root)
│   ├── experiments.py          # Suite/sweep/spec planner and executor; called by scripts/run.sh
│   ├── artifacts.py            # Shared names for persisted run artifacts and legacy aliases
│   ├── datasets/               # Dataset/preprocessing code; data files live in data/
│   │   ├── dataset_common.py   # Method-NEUTRAL: tokenizers, framework/target split, context
│   │   │                       #   injection + capping, and select_split_indices (the shared
│   │   │                       #   eval_split guard — never duplicate it per method)
│   │   ├── dataset_mlm.py      # Masked-LM view: masking policies, mask layout, labels, collator
│   │   ├── dataset_ar.py       # Autoregressive view (2026-09-01): pt1 <mask> pt2 <sep> target
│   │   │                       #   <eos> — both flanks in the prompt so the model can condition on
│   │   │                       #   FR4; loss on the answer only; length learned via <eos>
│   │   ├── dataset_s2s.py      # Encoder-decoder view (2026-09-07): TWO streams — encoder reads
│   │   │                       #   pt1 <mask> pt2, decoder runs <bos> target -> target <eos>. The
│   │   │                       #   gap is ONE marker carrying no length, and the loop never sits in
│   │   │                       #   the encoder, so there is no LOOP_LENGTH_RESERVE to trade against
│   │   │                       #   context: at max_len 128 that is 128 context tokens against ar's 31
│   │   └── preprocess/         # Offline raw → processed JSON pipeline (run once per dataset)
│   │       ├── __main__.py         # Shared CLI, source registry, pipeline, and output writing
│   │       ├── oas_paired.py       # OAS paired CSVs: filtering, deduplication, H3 clustering
│   │       ├── sabdab2.py          # SAbDab2 split: IMGT regions, antigen filtering, validation carve-out
│   │       ├── common.py           # Shared schema, validation, chain assembly, funnel, and metadata IO
│   │       ├── download_oas_paired.sh  # Download and verify/extract OAS paired CSVs → data/raw/
│   │       └── download_sabdab2.sh     # Download/extract SAbDab2 split archive → data/raw/
│   ├── methods/                # Complete generation methods; no cross-method mask assumptions
│   │   ├── common/             # What every method needs to PRODUCE output. One implementation, so
│   │   │   │                   #   the methods cannot drift — they already had, with the same field
│   │   │   │                   #   called gen_judge_path on one side and gen_judge_weights on the other
│   │   │   ├── arguments.py    # The 28 shared args + config/CLI parsing + run-directory setup;
│   │   │   │                   #   each method subclasses and adds only its own
│   │   │   ├── sampling.py     # warp_logits / ban_tokens / logits_of — so "top_p=0.9" means the
│   │   │   │                   #   same thing on both sides of an AR-vs-MLM table
│   │   │   ├── callbacks.py    # build_callbacks: metric logging ALWAYS, early stopping on request
│   │   │   └── training.py     # shared train/eval lifecycle + standard finished-run artifacts
│   │   ├── ar/                 # Autoregressive (2026-09-01). Data lives in datasets/dataset_ar.py;
│   │   │   ├── backbones.py    #   this package owns model + training.
│   │   │   │                   # Causal registry — resolve(model_type) -> CausalBackbone.
│   │   │   │                   #   hf_decoder ADDS the layout's sentinels to a pretrained LM and
│   │   │   │                   #   resizes it: those rows start random (see its docstring)
│   │   │   ├── training.py     # get_datasets + teacher-forced core/anchor metrics + Trainer.
│   │   │   │                   #   No masking view (nothing to hide), no loss registry (§13)
│   │   │   ├── sampling.py     # Causal decode to <eos>. Replaces the masked side's sampling +
│   │   │   │                   #   generation + length + length_predictor: the model picks its
│   │   │   │                   #   own length, so §10's gap-sentinel head has no counterpart
│   │   │   ├── arguments.py    # Causal args: subclasses the shared bases, adds gen_max_new_tokens
│   │   │   ├── entrypoint.py   # main() for the causal pipeline; train.py dispatches here
│   │   │   ├── evaluation.py   # CausalStrategy for the SHARED loop in eval/generative.py —
│   │   │   │                   #   decode-to-eos draws, TRUE (not pseudo) perplexity, per-candidate
│   │   │   │                   #   judge prompts since causal lengths vary
│   │   │   └── models/
│   │   │       └── transformer.py  # Causal decoder, ROTARY positions (§8: absolute positions lock
│   │   │                       #   a checkpoint to its context regime; causal prompts vary in length)
│   │   ├── s2s/                # Encoder-decoder (2026-09-07). Data lives in datasets/dataset_s2s.py.
│   │   │   │                   #   The missing cell of the 2x2: bidirectional conditioning like
│   │   │   │                   #   mlm, <eos>-chosen length like ar. Contract-identical to ar
│   │   │   │                   #   throughout, so an s2s/ar/mlm table is one implementation.
│   │   │   ├── backbones.py    # scratch, ESM-2, and protein-BERT encoders. PLM encoders keep
│   │   │   │                   #   native tokenizers; all decoders use the same residue vocabulary
│   │   │   ├── training.py     # get_datasets + teacher-forced metrics + Trainer. The metrics do
│   │   │   │                   #   NOT re-shift: dataset_s2s already offset the decoder pair
│   │   │   ├── sampling.py     # Decode to <eos>. ONE encoder pass shared by every candidate, and
│   │   │   │                   #   no padding at all — every decoder row starts at <bos>
│   │   │   ├── evaluation.py   # Seq2SeqStrategy for the SHARED loop in eval/generative.py
│   │   │   └── models/
│   │   │       ├── plm.py      # PLM encoder + width projection + shared residue decoder;
│   │   │       │               #   fine-tuned/frozen/random controls, complete local checkpoints
│   │   │       └── transformer.py  # Bidirectional encoder + cross-attending causal decoder.
│   │   │                       #   Rotary on SELF-attention only. NOT capacity-matched to ar at the
│   │   │                       #   same num_layers (1.86M vs 800k) — build_model logs the count
│   │   └── mlm/                # Masked-LM training, generation, and native evaluation
│   │       ├── arguments.py        # MLM args: shared bases + loss_type/core/anchor weights
│   │       ├── entrypoint.py       # main() for the masked pipeline; train.py dispatches here
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
│   ├── eval/                   # Shared sequence metrics and strategy-based evaluation
│   │   ├── __init__.py         # ANCHOR_TRIM / MIN_CORE_COVERAGE and package responsibilities
│   │   ├── metrics.py          # Sequence comparisons, BLOSUM provenance, confidence summaries
│   │   └── generative.py       # Select -> draw -> score -> judge -> report/output helpers
│   │                           #   Strategies: draw, self_scores, judge_inputs, payload_length_policy
│   │                           #   Judge helpers load MLM implementations when needed
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
├── data/                       # THE DATA — gitignored, ~6.7GB (raw + processed OAS/SAbDab2).
│                               #   Deliberately outside the package; configs' data_path points here.
├── results/                    # Ignored in git. Three tiers, by cost to regenerate:
│   ├── runs/                   #   hours of GPU — <timestamp>[_name]/<run>/, latest -> newest
│   │                           #   A round reports itself when it finishes: records at the batch
│   │                           #   root (batch_tables.md, or runs.tsv/summary.tsv for a sweep),
│   │                           #   charts under <batch>/figures/, per-trial stdout under logs/.
│   ├── benchmark/              #   GPU minutes — <benchmark_id>/{generations,scores}/
│   └── figures/                #   seconds, disposable — hand-picked charts for the README and
│                               #   ad-hoc cross-batch comparisons, which belong to no one batch
├── unit_tests/                 # CPU unit/integration tests — no downloads or research data (./scripts/unit_tests.sh).
│   └── test_*.py               #   Named to keep "test" free for the train/validation/test split.
├── showcase/                   # Curated frameworks for generate.py — a hand-designed probe set,
│   └── {h1,h3,h3_lc,l3_hc,h3_ag}.json # from VALIDATION, so repeated inspection is free;
│                               #   stratified by H3 length, one case deliberately without a
│                               #   native, each carrying its provenance and intent
├── misc/
│   └── CDRSeqGen260814.drawio.png  # Project workflow diagram
├── train.py                    # Entrypoint — dispatches to mlm, ar, or s2s
│                               #   Dispatches on the config's "method" key; each method's own
│                               #   main() lives in methods/<method>/entrypoint.py
├── generate.py                 # Entrypoint — candidate generation on supplied frameworks
├── evaluate.py                 # Entrypoint — held-out generative evaluation (+ --judge)
├── fit_mlm_length.py           # Add MLM length artifacts to an older checkpoint
├── scripts/
│   ├── run.sh                  # Launcher: <gpus> [suite ...] or <gpus> sweep [preset] [config]
│                               #   Suites name their method first: mlm_heavy | mlm_light |
│                               #   mlm_antigen | mlm_igbert | mlm_protbert | mlm_plm |
│                               #   ar_heavy | s2s_heavy | s2s_{esm2,igbert,protbert,plm} | compare | all. Several GPUs shard the round
│   ├── sweep.sh                # Compatibility alias for run.sh <gpus> sweep [preset] [config]
│   └── unit_tests.sh           # Unit test runner (sets PYTHONPATH; no GPU or data needed)
├── env.sh
├── README.md                   # What the project is, what it found, what is still open
└── WORKFLOW.md                 # How to run it — preprocessing through benchmark
```

---

## ⚙️ Environment Setup

Create the separate training/scoring (`cdrprj`) and IgLM baseline (`iglm`)
environments from the pinned migration requirements:

```bash
bash env.sh                              # creates both; refuses existing names
bash scripts/verify_env.sh                # checks dependencies, GPUs, and plotting
conda run --no-capture-output -n cdrprj bash scripts/unit_tests.sh
conda activate cdrprj
```

Use `bash env.sh cdrprj` or `bash env.sh iglm` to create only one environment.
The baseline uses Python 3.11 and CUDA 12.8 PyTorch wheels. Check destination
hardware/driver compatibility before installing. For intentional CPU-only
validation, run `bash scripts/verify_env.sh all --cpu`.

See [Environment and migration notes](ENVIRONMENT.md) for captured versions,
validation limits, and the data, model caches, and results that must move outside
Git. See [the test guide](unit_tests/README.md) for the CPU unittest suite.

---

## 🧹 Data Preprocessing Quick Start

Run these commands from the `CDRSeqGen/` repository root after activating `cdrprj`. The
downloaders are optional when the raw files are already present; they resolve paths from the
repository root, so they can also be called from another working directory.

### OAS paired repertoires

Download and extract the eight current paired-repertoire CSVs:

```bash
bash cdrseqgen/datasets/preprocess/download_oas_paired.sh
```

Build one merged corpus and assign the split once across all donors. The default OAS pipeline
keeps productive paired chains, removes exact duplicate heavy/light pairs, clusters heavy CDR3s,
and writes both the processed JSON and its provenance sidecar:

```bash
python -m cdrseqgen.datasets.preprocess.oas_paired \
  --input_file data/raw/oas/*_Paired_All.csv \
  --output_file data/processed/processed_oas_multi.json \
  --split cluster --test_size 0.1 --validation_size 0.1 --seed 42
```

For a quick input or environment check, apply `--limit` after source filtering:

```bash
python -m cdrseqgen.datasets.preprocess.oas_paired \
  --input_file data/raw/oas/1287169_1_Paired_All.csv \
  --output_file data/processed/processed_oas_smoke.json \
  --limit 100 --split none
```

Use `--split cluster` for model development. `--split none` is only suitable for inspection or
legacy reproduction because training must then create a random split at load time.

### SAbDab2 antibody-antigen pairs

Download the upstream split archive and extract it under `data/raw/sabdab2/`:

```bash
bash cdrseqgen/datasets/preprocess/download_sabdab2.sh
```

Process the paired structures with resolved protein or peptide antigens:

```bash
python -m cdrseqgen.datasets.preprocess.sabdab2 \
  --input_file data/raw/sabdab2/splits_final/abag_split.csv \
  --output_file data/processed/processed_sabdab2_data.json
```

The SAbDab2 processor preserves the upstream `train`/`test` split and does not create validation
records unless requested. To carve validation records from `train` without touching `test`, use
`--validation_size 0.15 --seed 42`; whole antigen/CDR-H3 groups are moved together. Add
`--allow_missing_antigen` only when intentionally building an antibody-only corpus.

Every command above also runs as `python -m cdrseqgen.datasets.preprocess --source
oas_paired|sabdab2 [flags]` -- the same pipeline, reached through `preprocess/__main__.py`.
Both processors accept `--help` for all source-specific options and write a `*.metadata.json`
sidecar containing the input/filter/split settings and a SHA-256 hash of the processed JSON. The
full funnel, split caveats, multi-donor guidance, and output schema are documented in
[WORKFLOW.md, Step 1](WORKFLOW.md#step-1-preprocessing-data).

---

## 🚀 Workflow & Execution

**→ [WORKFLOW.md](WORKFLOW.md)** — the full operating manual, split out so the Findings below are
not 500 lines of CLI detail away from the top.

[Architecture overview](WORKFLOW.md#architecture-overview) ·
[1 Preprocessing](WORKFLOW.md#step-1-preprocessing-data) ·
[2 Training & batches](WORKFLOW.md#step-2-training--batch-experimentation) ·
[3 Visualization](WORKFLOW.md#step-3-result-visualization--comparison) ·
[4 Sampling (`generate.py`)](WORKFLOW.md#step-4-qualitative-sampling-on-curated-frameworks-generatepy) ·
[5 Held-out eval (`evaluate.py`)](WORKFLOW.md#step-5-generative-evaluation-on-the-held-out-split-evaluatepy) ·
[Matched benchmark](WORKFLOW.md#matched-external-benchmark)

### S2S pretrained encoders: results and next runs

From the repository root, the combined suite is equivalent to listing all three model suites:

```bash
DRY_RUN=1 ./scripts/run.sh 0 s2s_plm   # inspect the plan; no training or result files
./scripts/run.sh 0 s2s_plm            # same as: 0 s2s_esm2 s2s_igbert s2s_protbert
# With multiple available GPUs:
# ./scripts/run.sh 0,1,2 s2s_plm
```

This schedules **six H3/OAS4 validation runs**: ESM-2, IgBert, and ProtBert, each with and without
partner-chain context. One GPU runs them sequentially; multiple GPUs run separate trials.
Defaults fine-tune the pretrained encoder and train a new 128-wide, two-layer residue decoder:
LR 3e-5, warmup 0.06, weight decay 0.01, effective batch 32, and up to ten epochs with early stopping.
These are initial settings, not tuned optima. Each run performs generative evaluation afterward
(150 frameworks × 20 candidates, top_p 0.9, seed 42, IgBert judge).

`FREEZE_ENCODER=true` keeps the encoder fixed; `ENCODER_INIT=random` gives a matching architecture
with random encoder weights. Both controls work in suite and sweep mode. The default matrix is
complete; tune and repeat promising settings across seeds before claiming a pretraining benefit.
See [the workflow](WORKFLOW.md#step-2-training--batch-experimentation) for control and sweep commands.

**Completed:** the [first full S2S PLM matrix](results/runs/20260913_151152_463868_s2s_plm/batch_tables.md)
finished all six runs, including full ProtBert training and evaluation. All runs reached ten epochs;
each produced 3,000/3,000 nonempty, complete candidates with full judge coverage (18,000 total).
These are seed-42 **validation** results, not held-out test results:

| Encoder | Partner context | Judge perplexity ↓ | Exact length (%) ↑ | Length MAE ↓ | Generated core recovery (%) ↑ | Recovery frameworks / 150 |
|---|---|---:|---:|---:|---:|---:|
| ESM-2 | No | 3.441 | 8.93 | 3.832 | 21.95 | 117 |
| ESM-2 | Yes | 3.609 | 9.07 | 3.708 | 19.99 | 111 |
| IgBert | No | 3.499 | 8.10 | 3.915 | 20.84 | 113 |
| IgBert | Yes | 4.132 | 6.70 | 4.364 | 13.33 | 106 |
| ProtBert | No | 3.471 | 7.77 | 4.010 | 19.87 | 111 |
| ProtBert | Yes | 3.623 | 7.20 | 4.081 | 20.16 | 103 |

Generated core recovery is conditional on length-matched candidates; the contributing framework
populations differ. The [earlier scratch comparison](results/runs/20260908_173143_compare/batch_tables.md)
scored **3.215 / 3.250** judge perplexity without / with partner context. No PLM setting improves
that metric here, but capacities, optimizer settings, and epoch budgets differ, so this does not
isolate the effect of pretraining. Five PLM runs achieved their best teacher-forced core recovery at
epoch 10; ESM-2 + partner peaked at epoch 7. A ten-epoch screen does not establish convergence.

**Stability and comparative performance are separate questions.** The collapse described below
is a stability failure in one of six runs. All six PLM settings also score worse than the earlier
scratch reference on judge perplexity; the collapse alone does not explain that broader result.
These measurements describe the tested settings, but do not isolate the effect of pretraining:
PLMs had a ten-epoch budget against scratch's twenty, five PLM runs reached their best
teacher-forced core recovery at the final epoch, and tuning effort and model capacities differ.
ESM-2 t12, the smallest encoder tested, has the lowest PLM judge perplexity. Budget and optimizer
settings are possible explanations for this ordering; architecture and pretrained representations
remain possible explanations too. A comparison with matched data/evaluation protocols and comparable
tuning budgets would assess the complete systems more fairly. Pretrained-versus-random initialization
of the same encoder architecture is the more direct control for the contribution of pretraining.

**September 17 status:** all 12 context-dropout trials and all four ESM-2 LR trials are complete. The [September 15–17 review](results/reviews/0915-0917-review.md) verifies their saved models, matched generation populations and scoring coverage. Dropout improves robustness to a missing partner at a small context-present edit cost (§14); `1e-4` improves ESM-2 teacher-forced recovery and judge perplexity without a consistent edit-similarity gain over `3e-5` (§15). Generation from the three earlier IgBert diagnostic checkpoints remains pending.

**Next steps:**

1. **IgBert + partner: representation collapse confirmed** by the
   [September 14 checkpoint diagnostics](results/diagnostics/20260914_igbert/REPORT.md).
   On 256 validation examples, swapping the framework or partner changes no predicted tokens;
   final encoder representations are nearly constant. The no-partner checkpoint and original
   pretrained encoder retain input variation. Training/validation inputs fit within 234 tokens
   (cap 384), ruling out truncation for this corpus. The initiating event remains unresolved;
   the epoch-2 loss/anchor deterioration is the leading interval to investigate. Next compare the
   completed original-LR, lower-LR and frozen-encoder diagnostic checkpoints generatively.
   All three retained conditioning through epoch 3; replication is still needed to estimate
   collapse incidence. The separate AR/S2S context-dropout screen is now complete (§14).
2. Add the **matched no-partner training controls** before varying their optimizer. The
   partner-present ESM-2 LR screen is complete (§15). The older command below is an optional
   no-partner LR follow-up, not the completed batch: three LRs × two seeds, up to 20 epochs,
   with generative evaluation enabled. Checkpoint selection remains teacher-forced core recovery;
   sweep ranking uses judge perplexity. Inspect recovery, length, and coverage alongside that ranking.
3. Compare promising settings with **random initialization of the same encoder architecture** and
   **frozen pretrained encoders**, using comparable tuning budgets, then replicate finalists across
   additional seeds. Keep the test split untouched during selection.
4. **Compare tuned scratch S2S and PLM systems under a matched protocol.** Use the same corpus,
   split, evaluation examples, decoding settings, and scorer, with comparable tuning effort and
   explicit training budgets. Matching epoch budgets controls data exposure; it does not match
   compute across model sizes. Running the arms in one batch simplifies bookkeeping, but consistent
   protocol and provenance also permit comparison across batches. This comparison measures the
   complete systems; step 3's same-architecture initialization control addresses pretraining.

The immediate diagnostic [experiment spec](configs/experiments/20260914_igbert_diagnostic.json) reuses the IgBert + partner config for three arms: trainable at
LR 3e-5, trainable at 1e-5, and frozen pretrained encoder at 3e-5. Seed 42, context dropout 0,
and the original ten-epoch scheduler horizon are retained; a callback stops training after epoch 3.
All three epoch checkpoints are retained, including optimizer state. Generation is disabled for
this diagnostic run. `conditioning_probe.jsonl` records projected encoder variation and sensitivity
to swapping whole encoder memories between 16 fixed validation examples, before training, every
250 optimizer steps, and at epoch ends. These probes do not select checkpoints or alter training RNG.
The frozen arm still trains its projection and decoder. Do not set `FREEZE_ENCODER` or `ENCODER_INIT`
globally in spec mode; put seed and encoder controls in the file. The recipe is an experiment
record, separate from the permanent suite catalog. Generic diagnostics remain available to future runs.

**How to read it.** The trainable-3e-5 arm is intended to reproduce the original first three epochs
with the same seed, data seed, context dropout and scheduler horizon. If it does not collapse,
report that the failure was not reproduced under this rerun; a fixed seed alone does not establish
identical numerical execution. The lower-LR and frozen arms still provide stability and conditioning
measurements, but a non-collapsing reproduction control would prevent claiming that those changes
averted the original failure on the basis of this comparison alone.

The earlier checkpoint diagnostic used 256 validation examples and swapped raw framework flanks;
it observed ~28.67% changed argmax tokens (mean absolute logit change ~0.577) for the no-partner
checkpoint versus 0% (~6.6e-7) for the collapsed checkpoint. The running callback uses 16 fixed
validation examples and swaps entire encoder memories, including partner context. Those historical
values illustrate the failure; they do not calibrate a pass threshold for the callback. Interpret
projected encoder variation and swap sensitivity together over training. No numerical pass/fail
criterion was specified before this batch started. Any threshold chosen from these traces is
exploratory and should be fixed before a subsequent confirmation run.

With 122,181 training records, microbatch 4 and accumulation 8, this run has **3,819 optimizer
steps per epoch**, including the final partial accumulation window. Three complete epochs produce
45 periodic probes at 250-step intervals, plus one initial and three epoch-end probes: **49 records
per arm**. The original training-loss transition was logged around epochs 1.83–2.09; validation
anchor accuracy was measured at epoch boundaries, falling from 88.19% at epoch 1 to 81.35% at
epoch 2. The probes can narrow the transition interval but do not identify its exact update.

Each retained checkpoint in the earlier fully trainable IgBert run occupied approximately
**4.69 GiB**, including optimizer state. The frozen arm still saves the full encoder weights,
but its optimizer state covers only the trainable projection and decoder, so its checkpoints
should be smaller. Budget the two trainable arms and the frozen arm separately, plus final
model exports and logs; multiplying the fully trainable checkpoint size by all nine checkpoints
overestimates the expected total.

```bash
DRY_RUN=1 ./scripts/run.sh 0 spec configs/experiments/20260914_igbert_diagnostic.json
./scripts/run.sh 0 spec configs/experiments/20260914_igbert_diagnostic.json
# Or run the three arms concurrently on three available GPUs:
# ./scripts/run.sh 0,1,2 spec configs/experiments/20260914_igbert_diagnostic.json
```

After reviewing stability, the ESM-2 tuning command is:

```bash
# Inspect the six-trial plan first; remove DRY_RUN=1 to launch. Choose available GPU IDs.
DRY_RUN=1 LRS="1e-5 3e-5 1e-4" SEEDS="42 43" EPOCH_LIST="20" \
GEN_EVAL=true RANK_METRIC=judge_ppl \
./scripts/sweep.sh 0 lr configs/s2s/h3_s2s_esm2.json
```

The [scratch S2S LR screen](results/runs/20260907_204721_sweep_lr/summary.tsv) and the earlier
six-run MLM/AR/S2S comparison are also complete. The 414-test CPU suite passed on September 10;
that historical check and the completed GPU matrix establish execution coverage, not biological
quality or a pretraining advantage.

## 📊 Findings

Phase 1 reports sequence-infilling experiments under the split and selection limitations in
[Open problems](#open-problems). Historical findings below identify their source batches; the
[current S2S run status](#s2s-pretrained-encoders-results-and-next-runs) is recorded separately.

*Numbers below are from the complete 36-run batch `results/runs/20260818_192606` (6 loops ×
{scratch, ESM-2} × context × {OAS, SAbDab2}), read at **best epoch** under the coverage-aware
recovery policy below. Every generated table states the selected metric and minimum core coverage.
Budgets are per family — 20 epochs for scratch, 10 for pretrained, both with
`early_stopping_patience: 3`; generative eval 150 frameworks × 20 candidates, independent decoding,
IgBert judge.*

> **✅ These are the first validation-selected numbers in this README.** The 08-18 batch is the
> re-run of the OAS matrix on `processed_oas_tuning.json` with `eval_split: validation`, so early
> stopping, `load_best_model_at_end` and every `eval_*` figure below read the **validation**
> partition and the test split stays untouched. Every earlier batch — including 07-29 and the 08-17
> post-refactor run — selected on "all non-train", which on the two-way file *is* the final test
> set. Protocol is otherwise identical to 07-29, so the two are directly comparable; the differences
> below are the cost of that contamination, and they are small (OAS H3 scratch 0.3185 → 0.3180).
>
> **The SAbDab2 rows are not covered by this.** SAbDab2 ships no validation partition, so its 12
> configs declare `eval_split: test` — honest, but still selection on the holdout. Every SD2 number
> below remains exploratory until a three-way SAbDab2 split exists (see Open problems).

> **⚠️ These numbers supersede the 07-24 batch, which trained every family for exactly 10 epochs.**
> That budget under-trained the scratch model, and it inflated the magnitude of **four** findings at
> once — §1, §2, §4 and §7 — all in the same direction, against the smaller model. See §9.

> **Every table here is generated, not hand-written** — regenerate them all after any new batch with:
> ```bash
> python -m cdrseqgen.visualization.batch_tables --batch results/runs/20260818_192606
> python -m cdrseqgen.visualization.batch_tables --only regimes,loops   # or a subset
> ```
> It reads only what a finished run already writes (`epoch_metrics.csv`, `generation_eval.json`,
> `run_config.json`) plus the processed dataset JSONs, and prints markdown ready to paste. The one
> exception is the decoding table (Step 4), which needs two `evaluate.py` runs differing only in
> `--decoding`, passed via `--decoding-json` (the pair used above is kept alongside that run as
> `generation_eval_decoding_{iterative,independent}.json`).

#### 1. "Model choice doesn't matter" is only true in two of three regimes

On OAS H3 a 128-dim from-scratch transformer matches ESM-2 (0.3180 vs 0.3214 core). But that generalises badly — whether pretraining pays depends entirely on how much of the test set is *memorisable* from train (see §6):

| case | unseen in train | metric | min core coverage | scratch | ESM-2 | Δ |
|---|---|---|---:|---:|---:|---:|
| OAS H1 | 1.5% | core | 100.0% | 0.9936 | 0.9935 | **−0.000** |
| SD2 H1 | 57.5% | core | 100.0% | 0.5445 | 0.6654 | **+0.121** |
| SD2 H2 | 79.6% | core | 100.0% | 0.3685 | 0.6096 | **+0.241** |
| OAS H3 | 100.0% | core | 100.0% | 0.3180 | 0.3214 | **+0.003** |

Pretraining buys ~nothing when the answer can be looked up, **+0.12–0.24** when the task needs genuine generalisation, and ~nothing again at the H3 ceiling. The original claim was drawn from regimes 1 and 3 only.

#### 2. CDR-H3 is the only hard loop — but the other five are easier than they look

Best-epoch recovery on OAS under the recorded core-coverage policy:

| CDR | reported metric | min core coverage | scratch | +ctx | ESM-2 | +ctx |
|---|---|---:|---:|---:|---:|---:|
| H1 | core | 100.0% | 0.9936 | 0.9905 | 0.9935 | 0.9939 |
| H2 | core | 100.0% | 0.9913 | 0.9793 | 0.9909 | 0.9909 |
| **H3** | core | 100.0% | **0.3180** | 0.3141 | **0.3214** | 0.3206 |
| L1 | core | 100.0% | 0.9937 | 0.9857 | 0.9940 | 0.9938 |
| L2 | whole | 1.1% | 0.9913 | 0.9846 | 0.9932 | 0.9928 |
| L3 | core | 99.9% | 0.9290 | 0.8927 | 0.9304 | 0.9298 |

The biology is right — heavy CDR3 gets D-gene + junctional diversity, the rest is germline-templated — but the ~0.99s are **not** evidence the model generalises; under the current split they are largely retrieval (§6). Only H3 is measured on entirely unseen loops.

#### 3. The standard metric misleads (anchor inflation)

Whole-span `eval_accuracy` (~0.46 on H3) is dominated by the 2-3 V/J-gene-determined anchor residues at each loop end (~0.90 recovery, any model); the hypervariable middle actually being "designed" is ~0.32. The gap is **H3-specific** (H1/H2/L core ≈ whole-span). Measured via `core_accuracy`/`anchor_accuracy` ([`compute_metrics`](cdrseqgen/methods/mlm/recovery.py), passed to the Trainer by `train.py`); the position-resolved breakdown below is from the standalone re-scorer `eval_recovery.py` (archived in the gitignored `old/`, so it is not in a fresh clone). Illustrative (`h3_esm2`):

| Position from start | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 100% | 89% | 35% | 21% | 20% | 14% |

#### 4. Sequence context — partner chain and antigen — gives no measurable benefit

The valid same-data antigen ablation (`_ag` vs `_ag_lconly`, SAbDab2, all 3 CDRs) shows no signal, and it *survives the core metric*:

| CDR | metric | min core coverage | scratch lc-only → +ag | ESM-2 lc-only → +ag |
|---|---|---:|---|---|
| H1 | core | 100.0% | 0.5445 → 0.4872 | 0.6654 → 0.6629 |
| H2 | core | 100.0% | 0.3685 → 0.3483 | 0.6096 → 0.5884 |
| H3 | core | 99.7% | 0.1286 → 0.1353 | 0.2209 → 0.2128 |

|Δ| ≤ 0.057, sign inconsistent, 5 of 6 comparisons negative. Not a truncation artifact (84% of complexes keep the full antigen at `max_len=512`).

**Partner-chain context: the earlier "it hurts the small model" claim was mostly an under-training artifact.** At a 10-epoch budget it looked like a solid effect; at 20 epochs two thirds of it disappears:

| loop | Δ at 10 epochs | Δ at 20 epochs (validation-selected) |
|---|---|---|
| H1 | −0.0103 | −0.0031 |
| H2 | −0.0279 | −0.0120 |
| H3 | −0.0086 | −0.0039 |
| L1 | −0.0277 | −0.0080 |
| L2 | −0.0331 | −0.0067 |
| L3 | **−0.0786** | −0.0363 |
| **mean** | **−0.0310** | **−0.0117** |

Adding context lengthens the input and slows convergence: every `+ctx` scratch run peaked at **epoch 18–20**, against 4–14 without it, and five were still improving when the budget ran out — so −0.0117 is an *upper* bound. ESM-2 is unmoved either way (−0.001 to +0.003).

**With an error bar (finally).** Three seeds of `h3_scratch` ± partner chain, 20-epoch budget:

| seed | no ctx | +LC | Δ |
|---|---|---|---|
| 1 | 0.3196 | 0.3154 | −0.0042 |
| 2 | 0.3199 | 0.3123 | −0.0076 |
| 3 | 0.3187 | 0.3089 | −0.0098 |
| | sd **0.0006** | sd **0.0033** | mean −0.0072, sd **0.0028** |

The effect is real but small (2.5σ, all three seeds negative), and the noise floor on a per-loop Δ is **σ ≈ 0.0028**, i.e. ~0.006 at 2σ. Applying that to the table above, only four of the six loops survive:

| loop | Δ | \|Δ\|/σ | |
|---|---|---|---|
| L3 / H2 / L1 | −0.0363 / −0.0120 / −0.0080 | 13.0 / 4.3 / 2.9 | real |
| L2 | −0.0067 | 2.4 | marginal — just past 2σ |
| **H1 (−0.0031), H3 (−0.0039)** | | **1.1 / 1.4** | **not distinguishable from noise** |

So on the loop that actually matters — H3 — partner-chain context has **no measurable effect in either direction**, and the earlier −0.0086 was noise plus under-training. Note also that σ itself is a *seed* spread, and §13 shows run-to-run nondeterminism exceeds that; read the marginal row accordingly. A second observation from the same three seeds: the `+ctx` runs are **5× more variable** than the base runs (sd 0.0033 vs 0.0006) and their best epoch swings between 10 and 20, so context destabilises training as well as slowing it.

**Interpretation is unchanged:** sequence context — partner chain or antigen — buys nothing measurable. Binding information is geometric, not linear-sequence, so the next lever is structure / cross-attention. What changed is that the apparent *harm* was largely our own budget.

#### 5. The homology leak was real but immaterial *for H3*

OAS's original random split leaked clonal near-duplicates (76 of 2,111 test entries, 3.6%, share a CDR3 cluster with train). The clustered-split fix removed it (→0%), and re-running confirmed the H3 core moved by **≤0.004**. (Now the default; see [`preprocess/oas_paired.py`](cdrseqgen/datasets/preprocess/oas_paired.py).) The important qualifier is *for H3* — the split clusters on heavy CDR3 only, so it controls nothing for the other five loops (§6).

#### 6. The clustered split controls H3 and nothing else — so the ~0.99s are retrieval

`cluster_by_cdr3` groups on heavy CDR3, and whole clusters are held out. Two antibodies with identical H1/H2 land on opposite sides freely as long as their CDR3s differ:

| | unique H1 strings | held-out entries with an **unseen** H1 | with an unseen H2 | recovery |
|---|---|---|---|---|
| OAS | 439 / 21,106 (2.1%) | 65 / 4,222 (**1.5%**) | 83 / 4,222 (2.0%) | ~0.99 |
| SAbDab2 | 895 / 3,984 (22.5%) | 431 / 750 (**57.5%**) | 597 / 750 (79.6%) | 0.33–0.43 |

*OAS is now counted over the three-way file's held-out partitions — validation **and** test, 2,111
each — hence 4,222 rather than the 2,111 of the old two-way split. The fraction is unchanged.*

98% of OAS held-out H1s appear verbatim in training, so H1/H2/L1/L2/L3 under this split are near-lookup tasks — which is why every model scores ~0.99 and why context and pretraining make no difference there. **SAbDab2 is not underperforming; it is the only place the non-H3 loops are honestly measured.** H3 is unaffected (100% of test H3s are unseen by construction). Fixing this needs a per-loop split — cluster on whichever CDR is the target — which is a contained change to `add_split_field`.

#### 7. Recovery and design quality disagree — high AAR can mean a perfect copier

From the post-training generative eval (150 held-out frameworks, 20 samples each, IgBert judge):

| | recovery | diversity | entropy (bits) | judge-ppl |
|---|---|---|---|---|
| H1 scratch | 0.999 | **0.000** | **0.000** | 1.021 |
| H1 ESM-2 | 0.999 | **0.000** | **0.000** | 1.021 |
| H3 scratch | 0.398 | 0.574 | 1.835 | 7.223 |
| H3 ESM-2 | 0.395 | 0.588 | 1.880 | 7.490 |

H1 scores 0.999 AAR while emitting essentially **one** sequence with total certainty (perplexity ≈ 1.0, diversity and per-position entropy both 0.000) — a perfect copier and a useless designer. That is the finding, and the validation-selected batch makes it starker than before: the copier is now exact.

> **The H3 half of this finding no longer holds, and should not be quoted.** Earlier text here said the judge put ESM-2 "1.7× ahead" of scratch on H3 while recovery called them equal. That gap was an artifact of the shared 10-epoch budget (§9 recorded it closing to 1.00× once scratch got 20 epochs), and the prose outlived the table it described. On the validation-selected batch the two metrics **agree**: recovery 0.398 vs 0.395, judge 7.223 vs 7.490, both a hair in scratch's favour. Recovery and design quality can still disagree — H1 above is the proof — but not between these two H3 models.

#### 8. A model is locked to the context regime it was trained in — and `context_dropout` unlocks it

Feeding a checkpoint the *wrong* amount of context is as damaging as feeding it none, in both directions, because for an absolute-position backbone a prepended partner chain shifts every learned position:

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

#### 9. One un-audited hyperparameter inflated four findings at once

Every config trained for exactly `num_train_epochs: 10`, which suited the pretrained models (ESM-2 median best epoch **4**) and starved the scratch model (median **9**, with 8/18 still improving at the cut-off). Re-running with per-family budgets — 20 epochs for scratch, 10 for pretrained, both with `early_stopping_patience: 3` — moved four separate results, all in the same direction:

| finding | at 10 epochs | at per-family budgets |
|---|---|---|
| §1 pretraining lead, SD2 H1 | +0.231 | **+0.119** |
| §4 partner-chain penalty (mean) | −0.0310 | **−0.0102** |
| §7 judge-ppl gap, H3 scratch vs ESM-2 | 1.74× | **1.00×** |
| §2 scratch `+ctx` recovery | — | +0.007 to +0.055 |

The §7 change is the starkest: scratch's judge-perplexity went **12.42 → 7.22** against ESM-2's 7.23. Recovery and the judge no longer disagree; they were only disagreeing because one model was under-trained. (Not the decoder switch — iterative and independent were measured equivalent, and ESM-2's number barely moved.)

Early stopping cost nothing on the pretrained side — mean Δ over 12 OAS runs was **−0.0000**, and zero runs were cut with their best at the final epoch — while shortening the batch from 295 to 226 minutes.

**The lesson is procedural:** a single budget shared across model families is not a neutral choice, and every distortion it caused ran against the smaller model. Anything compared across families should have its budget justified per family, or the comparison measures the budget.

#### 10. Removing native H3 length exposes a separate unsolved problem

The first length-free run used
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

#### 11. Matched length-free comparison: IgLM leads on plausibility and length, and the recovery gap does not resolve

Both models ran the frozen `oas_h3_validation_length_free_v1` manifest — 25
validation frameworks × 20 candidates, one scorer, one IgBert judge. CDRSeqGen predicts a length
from its gap sentinel; IgLM terminates on its own EOS. Neither receives the native length.

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

**With a paired bootstrap over examples, only half of those margins survive (2026-08-31).** The
example is the resampling unit, not the candidate — 500 draws from 25 frameworks are not 500
observations — and one index draw is applied to both arms, so per-framework difficulty cancels in
the difference. A metric uses only examples where *both* models report it, which for positional AAR
means the 14 frameworks where both produced a native-length candidate. 10,000 resamples, seed 42:

| metric | n | CDRSeqGen | IgLM | difference | 95% CI | resolved |
|---|---:|---:|---:|---:|---|---|
| whole-span AAR | 14 | 0.3288 | 0.3340 | −0.0052 | [−0.0414, +0.0308] | no |
| core AAR | 14 | 0.1540 | 0.1686 | −0.0146 | [−0.0624, +0.0339] | no |
| anchor AAR | 14 | 0.8304 | 0.8065 | +0.0238 | [−0.0238, +0.0685] | no |
| length match rate | 25 | 0.0600 | 0.0840 | −0.0240 | [−0.0520, +0.0040] | no |
| length MAE ↓ | 25 | 4.5100 | 3.6900 | +0.8200 | [+0.3660, +1.2340] | **IgLM** |
| edit similarity | 25 | 0.3596 | 0.3890 | −0.0294 | [−0.0481, −0.0124] | **IgLM** |
| judge perplexity ↓ | 25 | 9.7364 | 4.0709 | +5.6655 | [+5.0044, +6.3255] | **IgLM** |
| diversity (Hamming) | 25 | 0.8047 | 0.7784 | +0.0263 | [+0.0147, +0.0387] | **CDRSeqGen** |

**So the headline should not be "IgLM leads on every quality axis" — that was this section's
original claim and the intervals do not support it.** What survives is narrower and more specific:
IgLM is decisively more *plausible* (judge perplexity, a 5.67 gap whose interval is nowhere near
zero) and decisively better at *length* (MAE, edit similarity). Every **recovery** comparison —
whole-span, core, and CDRSeqGen's apparent anchor advantage alike — is unresolved at n = 25, and the
point estimates sit well inside their intervals. CDRSeqGen's diversity edge is individually resolved but small.

**Correcting for the eight endpoints changes no verdict.** Eight unadjusted 95% intervals carry a
family-wise error near 34% under independence, so the multiplicity caveat is worth stating — but at
Bonferroni-adjusted 99.375% intervals all four resolved metrics still exclude zero: judge perplexity
[+4.75, +6.61], length MAE [+0.16, +1.39], edit similarity [−0.057, −0.006], diversity
[+0.011, +0.044]. Edit similarity sits nearest the boundary and still clears it.

Two readings follow. The recovery axes are not evidence of parity either: an interval spanning zero
at this sample size means *not measured*, not *equal*. And the axis that does resolve is the one
CDRSeqGen loses worst — judged plausibility — which is the same signal §7 and §12 keep returning.
Reproduce with:

```bash
python benchmark/bootstrap.py \
    --scores results/benchmark/oas_h3_validation_length_free_v1/scores/length_free.json --markdown
```

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
(`results/figures/benchmark/20260821_regenerated/constrain_vs_filter_position.png`): under `constrain` IgLM's
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
examples — AAR is conditioned on length matches, so it rests on 18 CDRSeqGen / 17 IgLM examples
individually and only **14 paired** ones, which is why the recovery intervals above are as wide as
they are. The bootstrap resamples frameworks while holding each framework's finite candidate pool
fixed, so it does not include generation-seed uncertainty; the eight per-metric 95% intervals are
also unadjusted for multiple endpoints. ~~No confidence intervals yet~~ **done 2026-08-31**
([`bootstrap.py`](benchmark/bootstrap.py));
the remaining blockers are that the scores were produced from a **dirty working tree at the previous
commit** (`repository_commit: 0e7e499`, `repository_dirty: true`), so a clean-tree rescore is still
required, and that IgLM was pretrained on OAS, so this holdout is not a leakage-free estimate for it
(roadmap: overlap audit). The clean-tree rescore is expected to move the point estimates only
slightly — but the intervals above should be regenerated from it rather than carried over.

#### 12. The model's own confidence is nearly useless for selection — an independent judge's is not
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

#### 13. Re-weighting the loss across loop positions changes nothing — the anchors are not wasted capacity

The 08-25 screen trained the tuned scratch config (`1e-3`, 0.06 warm-up, 0.01 weight
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

> **Scope of the historical findings:** the ~0.32 core / ~0.46 whole-span plateau describes the tested fixed-native MLM settings. It is not an objective-independent ceiling. The AR/S2S teacher-forced and length-free generation results below use different conditioning and must be reported separately. Non-H3 generalization still requires target-loop-specific splits.

#### 14. Context dropout improves missing-partner robustness, with a small context-present cost

The [12-run context screen](results/runs/20260915_131629_103719_context_dropout/runs.tsv) completed and was reviewed on **2026-09-17**: scratch AR, scratch S2S and ESM-2 S2S × dropout 0/0.5 × seeds 42/43. Each checkpoint generated 20 candidates for the same 150 OAS4 validation frameworks, with and without generator partner context; the IgBert judge always received partner context. Checkpoints were selected by teacher-forced core recovery with partner context present. Two-seed means:

| Model | Train dropout | Edit, partner present ↑ | Edit, partner absent ↑ | Judge PPL, present ↓ | Judge PPL, absent ↓ |
|---|---|---|---|---|---|
| AR scratch | 0.0 | 0.3910 | 0.3815 | 3.261 | 3.890 |
| AR scratch | 0.5 | 0.3900 | 0.3893 | 3.180 | 3.242 |
| S2S scratch | 0.0 | 0.3905 | 0.3829 | 3.208 | 3.405 |
| S2S scratch | 0.5 | 0.3889 | 0.3878 | 3.269 | 3.278 |
| S2S ESM-2 | 0.0 | 0.3932 | 0.3689 | 3.526 | 4.368 |
| S2S ESM-2 | 0.5 | 0.3899 | 0.3872 | 3.531 | 3.558 |

Dropout improves partner-absent edit similarity in **both seeds for all three models**: mean gains **+0.0078 / +0.0050 / +0.0183** for AR / scratch S2S / ESM-2 S2S. Partner-present edit costs are **−0.0010 / −0.0016 / −0.0032**, also consistent in direction across seeds. Partner-absent judge perplexity improves in every pair. This is a robustness trade-off; it does not show that dropout improves every metric or that training without partner context would be worse. Those matched no-partner training controls remain pending.

**Coverage and limits:** 71,999/72,000 candidates were valid. The sole exception was a complete-empty draw from ESM-2/dropout 0/seed 43 without partner context. All 150 frameworks contributed valid candidates in every regime, and judge coverage was complete over valid candidates. Edit similarity excludes that empty draw; generated core recovery is conditional on matching length and has different contributing populations. Two seeds are screening evidence, without a significance or equivalence claim. The existing automatic `batch_tables.md` omits three ESM-2 trials through family-key collisions; use the complete per-seed tables in the [September 15–17 review](results/reviews/0915-0917-review.md).

#### 15. ESM-2 learning-rate tuning improves teacher-forced recovery more clearly than generation

The [four-run LR screen](results/runs/20260915_214327_432282_sweep_lr/summary.tsv) completed at `1e-5` and `1e-4`, seeds 42/43, up to 20 epochs. These are **partner-trained, dropout-0** models evaluated with partner context present; partner-absent evaluation was not part of this sweep. The `3e-5` reference below comes from the context screen. All use the same 150 × 20 generation protocol; means across two seeds:

| ESM-2 LR | Teacher-forced core ↑ | Generated edit ↑ | Generated core@L ↑ | Judge PPL ↓ | Length match | Length MAE ↓ |
|---|---|---|---|---|---|---|
| 1e-05 | 0.3880 | 0.3917 | 0.2137 | 3.941 | 9.07% | 3.834 |
| 3e-05 | 0.3980 | 0.3932 | 0.2260 | 3.526 | 8.53% | 3.813 |
| 1e-04 | 0.4032 | 0.3928 | 0.2104 | 3.345 | 8.62% | 3.762 |

Within the new sweep, **`1e-4` beats `1e-5` in both seeds** on teacher-forced core, edit similarity and judge perplexity. Against the `3e-5` reference, `1e-4` improves mean teacher-forced core by **0.0052** and lowers judge perplexity by **0.180**, but edit similarity is slightly lower on average (**−0.0003**) with opposite seed directions. Generated core@L also falls, but its population differs (107/106 contributing frameworks at `1e-4`, versus 116/109 at `3e-5`). **Keep `1e-4` as a follow-up candidate, not an overall generation winner; no default config was promoted.** Its interaction with dropout 0.5 remains untested.

All **12,000/12,000** LR-sweep candidates were valid with full judge coverage. The final teacher-forced evaluations cover 15,271/15,273 validation examples for core accuracy. Native answer prefixes make these ~0.40 teacher-forced scores different from historical MLM masked recovery; they are not evidence of a direct improvement over its ~0.32 result. All 16 runs across the two batches record dirty-tree provenance, and the batch source hashes differ. Preserve those records and confirm selected comparisons under a fixed clean revision before final paper claims. See the [September 15–17 review](results/reviews/0915-0917-review.md) for config differences, per-seed results and input hashes.

---

## 🧠 Architectural Feature Details

### 1. Task-Agnostic Masking (The "LEGO" Principle)
The `MaskedInfillingDataset` separates sequences into `pt1` (left context), `mask` (target gap), and `pt2` (right context). The `target_region` string alone chooses the loop — `h1/h2/h3` mask a heavy CDR, `l1/l2/l3` mask the *light* chain's CDR (legacy `cdr*`/`lcdr*` spellings still parse). Switching target requires changing zero core tokenization logic. Labels are strictly length-enforced to prevent cross-entropy dimension mismatches.

### 2. Pluggable Backbones (the extensibility core)
The masked-LM backbone registry [`backbones.py`](cdrseqgen/methods/mlm/backbones.py) holds one `Backbone` entry per compatible model family; the entrypoints and the `dataset_*.py` classes never branch on model type — they go through `resolve(model_type)`, which returns the matching backbone and a uniform `TokenizerAdapter`. The same MLM pipeline trains the from-scratch transformer, fine-tunes ESM-2, or fine-tunes any Hugging Face masked-LM just by changing the `"model_type"` string.

Each backbone declares two conventions **explicitly** instead of letting them be guessed from the tokenizer (the previous code overloaded a single `is_esm` boolean to mean three different things, which silently mishandled non-ESM models):
- **`position_type`** (`"absolute"` | `"relative"`) — controls where antigen context is injected (see §4). Absolute-position models (the scratch transformer, BERT-style protein LMs) need it appended after the target; relative/rotary models (ESM-2) can take it prepended.
- **`uses_special_tokens`** — whether inputs are wrapped as `[cls] … [end]`. The `TokenizerAdapter` also resolves the ESM-vs-BERT separator difference generically (`<eos>` when the tokenizer has one, else `<sep>`), so BERT-style tokenizers with `eos_token_id = None` work without special-casing.

**Adding a new masked PLM is usually one `register(Backbone(...))` entry** — occasionally *zero*, when a model is already covered by the generic `hf_encoder` fallback (any `AutoModelForMaskedLM` + `AutoTokenizer`, absolute positions, per-character vocab — just pass its hub id as `model_type`). In practice most protein LMs need an explicit entry: e.g. BERT-style ones (ProtBert, IgBert) expect *whitespace-separated* residues and ship tokenizer/loader quirks (a fast-tokenizer that needs `sentencepiece`, or a `config.json` with no `model_type`), all handled once in the `protein_bert` backbone. Either way, no edits to `train.py`/`generate.py`/`dataset_mlm.py` are needed. A new autoregressive or encoder-decoder architecture is instead a sibling method package, not a `Backbone` entry here.

### 3. Partner Chain Context Injection
Biologically, Heavy and Light chains physically interact, so the framework can prepend the paired **partner chain** (the non-target chain) to `pt1` (separated by `<eos>`) when `"include_partner_chain": true` is passed, enriching structural context. It's symmetric: for a heavy (`h*`) target the partner is the light chain (run tag `_LC`); for a light (`l*`) target the partner is the heavy chain (run tag `_HC`). A single flag covers both directions — the target region determines which chain gets added.

### 4. Antigen Context Injection (SAbDab2)
Same "LEGO" pattern as partner-chain injection (§3), but **the injection side is keyed on the backbone's `position_type`**. For **relative**-position models (ESM-2), the antigen is prepended *ahead of* the light chain into `pt1`: `antigen <sep> light <sep> heavy_pt1 <mask> pt2`. For **absolute**-position models (the scratch transformer, and BERT-style PLMs), the antigen is instead *appended* into `pt2`, with the light chain still prepended to `pt1`: `light <sep> heavy_pt1 <mask> fwr4 <sep> antigen`. Multi-chain antigens are joined with `X` (the standard "unknown residue" convention) as a soft chain-break marker, since the amino-acid tokenizer has no token for a literal chain separator.

Antigen sequences (up to ~2,300 residues for large multi-chain complexes) made the pre-existing "no truncation" gap in `MaskedInfillingDataset` actually bite, so this feature also added a real length cap, applied on whichever side carries the antigen: truncated from the left for relative-position `pt1`, from the right for absolute-position `pt2` — always dropping antigen context first and keeping the target-chain residues immediately adjacent to the masked loop.

### 5. Automated Tracking & Evaluation
*   **TensorBoard:** Live tracking via `"report_to": "tensorboard"`.
*   **Metric Extraction:** At the end of training, `train.py` parses the Hugging Face logs, dumps `epoch_metrics.csv` (loss + recovery, whole-span **and** the anchor-excluded `core`/`anchor` accuracies), and generates standalone `.png` curves.

Generated-loop evaluation follows the shared workflow in [`generative.py`](cdrseqgen/eval/generative.py):

```text
select frameworks -> strategy.draw -> sequence metrics + self/judge scores
                  -> candidate details + confidence -> framework records
                  -> report / JSON / optional confidence figure
```

The strategy owns generation and validity accounting; the method wrapper assembles and saves the payload. [`metrics.py`](cdrseqgen/eval/metrics.py) defines the sequence comparisons. Their eligible candidates differ:

| Metric | Candidates included |
|---|---|
| AAR, core/anchor recovery, positional BLOSUM | Native-length candidates; empty regions and undefined ratios are omitted |
| Edit similarity and length error/match rate | All retained candidates |
| Aligned BLOSUM | Non-empty candidates of any length; ratios require a non-zero native self-score |
| Confidence summaries | Native-length candidates with finite recovery and a finite self/judge score |

Repeated draws count separately. `novelty_mean = 1 - aar_mean` measures difference from the native loop; it does not compare against training sequences. Reports give each contributing framework equal weight, with **population standard deviation across frameworks**, rather than a confidence interval or seed spread. The displayed `n` counts records; individual metrics can have fewer contributors.

MLM self-scores and all judge scores use masked pseudo-perplexity; AR/S2S self-scores use their causal factorizations. BLOSUM provenance records the matrix, gap penalties and alignment version; the current version-1 recurrence is unchanged by the October cleanup.

### 6. Extensibility Principles
The recurring rule: **a new capability is added in one place, and the rest of the pipeline stays oblivious.** Five seams:

- **New generation method** → a package beside [`methods/mlm/`](cdrseqgen/methods/mlm/), subclassing `AntibodyFrameworkDataset` from [`datasets/dataset_common.py`](cdrseqgen/datasets/dataset_common.py), plus an `entrypoint.py` that [`train.py`](train.py) dispatches to on the config's `"method"` key. **Exercised 2026-09-01** by [`methods/ar/`](cdrseqgen/methods/ar/), which is a third the size of the masked package — the difference is the length-policy layer, the gap-sentinel head, the second eval track and the loss registry, none of which a causal model can need.

  **Exercised again 2026-09-07** by [`methods/s2s/`](cdrseqgen/methods/s2s/), which is the real test: the second method could still have been accommodated by generalising toward it, while the third had to fit a boundary drawn without it in mind. It did — a registry entry, the five-function CLI protocol, and a strategy for the shared evaluation loop — and the parts that did *not* fit were informative. `SpanLayout` does not generalise, because it builds one concatenated stream and an encoder-decoder needs two; that is a new data view, not a parameterisation of the old one. And three *enumerations* had to be found by running the method rather than by the type system: the dashboard family map, the `--decoding` choice list duplicated into two argparse calls, and `sweep.sh`'s architecture flag. Each was a list that had to be edited in a second place, which is the shape of seam this section exists to avoid; `--decoding` and the method classification now derive from `methods/registry.py`, and a test asserts every registered method has a dashboard family.

  Adding the second method is what made the shared layers honest, and each extraction was checked against two implementations rather than guessed from one: `select_split_indices` (the eval-split guard — a second copy could recontaminate one method's results silently), `eval/generative.py` (the evaluation loop; a *strategy* supplies `draw()`, `self_scores()`, `judge_inputs()` and `payload_length_policy()`), `build_judge` (§11 rests on one judge scoring both models), and `methods/common/` (arguments, sampling, callbacks). The last one caught a live defect: the causal entrypoint had quietly dropped `LogMetricsCallback`, reintroducing the empty-`train.log` regression fixed on 2026-07-29. *Added 2026-08-31*: the framework/target split, context injection and length capping are now method-neutral, and `TokenizerAdapter.mask_id` is optional — requiring a mask token was the one thing that made a causal tokenizer impossible to construct. `dataset_mlm.py` is now just the masked-LM view of that base, and re-exports the moved names so existing imports are unchanged. The benchmark contract was already method-agnostic, as [`adapters/iglm.py`](benchmark/adapters/iglm.py) demonstrates.
- **New masked-LM backbone** → one `Backbone` entry in [`backbones.py`](cdrseqgen/methods/mlm/backbones.py) (§2) — the entrypoints and `dataset_mlm.py` never learn about it. The causal and seq2seq registries have the same shape, so the seam is one rule across all three methods.
- **New target loop** → change the `target_region` string (§1); masking logic is untouched.
- **New context** → one boolean flag (`include_partner_chain`, `include_antigen`), resolved symmetrically (§3–4).
- **New dataset source** → one `(substring, tag)` pair in the `DATASET_TAGS` registry ([`train/logging.py`](cdrseqgen/train/logging.py)) (`oas → OAS`, `sabdab2 → SD2`), so every run's output-dir name states its source explicitly instead of leaving it silently implied; an unrecognised path is tagged `DS` **with a logged warning** rather than left blank.

The guiding constraint: **conventions are declared explicitly, never inferred.** The predecessor overloaded a single `is_esm` boolean to mean three different things and silently mishandled non-ESM models; each backbone now states its `position_type` and `uses_special_tokens` up front. This isn't free — protein-BERT models (IgBert/ProtBert) still needed their own entry for residue-spacing and tokenizer/loader quirks (§2) — but the cost is paid once, in the registry, not scattered across the pipeline.

---

## ⚠️ Current Issues
*(status reviewed 2026-09-14; historical measurements retain their original batch dates)*

### Open problems
- **Generative evaluation edge cases remain open (2026-10-06 review).** `run_eval_loop` can silently truncate candidate details when self-score counts disagree with the draws. Score means, judge coverage and framework aggregation still accept NaN/Inf, although `confidence_metrics` now filters non-finite pairs and rejects unequal input lengths. Stratified selection also favors the first-observed buckets when the framework budget is smaller than the number of non-empty buckets.
- **The `OAS4` corpus confounds donor with disease, by construction.** Jaffe 2022 contains exactly four donors and no more (all 580 catalogued paired runs probed 2026-09-03), and their conditions do not overlap: Donor-1 and Donor-2 are SARS-CoV-2, Donor-3 healthy, Donor-4 CMV. A donor-held-out split is therefore also a disease-held-out split, and neither effect can be attributed separately. Mitigated but not removed by every run being **Naive-B-Cells**: naive repertoires are largely pre-antigen-selection, so disease shapes them far less than it shapes memory or plasmablast compartments. Recorded before any `OAS4` experiment was run.
- **More donors would cost B-cell-type purity.** 96 further donors exist across the other 18 paired studies, but only **one** naive paired run among them (Woodruff_2020); the rest are unsorted, memory, or plasmablast. Somatic hypermutation differs sharply between those compartments, and it lands hardest on the germline-encoded H1/H2 — the loops §6 already shows to be retrieval-bound. Any expansion past four donors must record `BType` per entry (`read_oas_metadata` does not capture it today) and stratify on it, or the heterogeneity becomes invisible in the merged corpus.
- **`OAS4` results are not comparable to any earlier OAS number.** A different donor set is a different test split. The `oas_multi -> OAS4` tag keeps the two apart in run names and batch tables mechanically, but no table should place them on one axis — the same rule the `_OAS` / `_SD2` tags exist to enforce.
- **H3 clustering does not guarantee pairwise separation.** The split clusters heavy CDR3 against representatives. The [September audit](results/reviews/0903-0904-review.md) found same-length training neighbors at ≥85% identity for 3/150 evaluated OAS4 H3 frameworks; disjoint cluster IDs do not rule out such neighbors. Other loops have a separate exact-overlap problem: 98% of the older OAS test H1s occur in train (§6). Audit pairwise H3 separation and use target-loop clustering before interpreting non-H3 results as generalisation.
- **H3 length is close to unpredictable from the framework, and every learned method is below the achievable ceiling.** A non-neural lookup — group training examples by (last 12 residues of FR3 | first 6 of FR4), a crude V/J-gene proxy, and take the modal H3 length — reaches **11.07% exact / 3.01 MAE** on the OAS4 validation split, covering 98.3% of it. Against that: the §10 length head 9.24% / 3.64, causal `<eos>` 7.23% / 4.00, and always predicting the global mode 10.12% / 3.53. So there IS signal (MAE 3.01 against a constant's 3.53, a 15% reduction) but very little, and **neither learned method captures even that much**. Three interventions moved nothing — decoding temperature, a 10x learning-rate range, and 7x more data — which is what a near-flat ceiling predicts. The same verdict holds on the older split the length head was measured on: 9.24% / 3.64 against 10.52% / 3.28 for always predicting length 15 (§10) — below its naive baseline there too. Treat length accuracy as a property to report against these baselines, not a target to optimise; the V/J lookup should replace the global mode as the reference number. Before treating inferred MLM generation as competitive, evaluate probability calibration and top-k length coverage, then test a representation or objective that actually learns the framework-to-length signal.
- **Training still optimizes recovery, not design.** The masked-LM objective rewards reproducing the native loop; the generative metrics only *measure* design quality after the fact — nothing trains for it. Findings §7 shows the two can disagree outright (H1: 0.99 recovery, 0.002 diversity). A design-aligned objective stays open (Phase 2). Re-weighting tokens *within* the masked-LM objective has now been tested and does not help (§13), so this needs a different objective class rather than a better weighting.
- **Run-to-run nondeterminism exceeds the seed spread, and only seeds are ever reported.** The strongest evidence is qualitative: on 2026-09-14 the IgBert + partner collapse **failed to reproduce** at the same seed and settings — the 09-13 run degraded through epoch 2 (loss 1.7379 → 1.7684, anchor 0.8819 → 0.8135) while the replay improved (1.745 → 1.651, 0.8757 → 0.8820). Same nominal configuration, opposite outcomes. The quantitative case is older and milder: two bit-identical objectives at identical seeds — `cross_entropy` and `weighted 1.0/1.0`, differing only in floating-point reduction order — ended 0.0014 apart in teacher-forced core, all three seeds in the same direction, against a within-objective seed sd of 0.0003–0.0007 (§13). So a seed sd *understates* reproducibility by at least 2–4× on stable runs and can miss a total training failure on unstable ones, which bears directly on the ±0.006 tie threshold in `sweep.sh` and the σ ≈ 0.0028 noise floor in §4. Every "error bar" in this README is a seed spread and should be read as a lower bound. Cheap fix: keep one duplicated setting per sweep as a deliberate determinism control.
- **Loss-sweep selection and ranking are separate.** The shared runner now defaults the `loss` preset to whole-span checkpoint selection (`SELECT_METRIC=accuracy`) and judge-perplexity ranking (`RANK_METRIC=judge_ppl`). This makes anchor collapse visible to selection, but does not train directly for design quality. Historical loss-screen results retain their original checkpoint-selection policy; new summaries label both selection and ranking metrics.
- **Scratch and pretrained S2S have completed initial experiments.** The [September 8 comparison](results/runs/20260908_173143_compare/batch_tables.md) contains six completed runs. Without partner context, judge perplexity is 7.512 / 3.154 / 3.215 for MLM / AR / S2S, with inferred length for MLM and EOS termination for AR/S2S. This is one training seed, and conditional recovery uses different length-matched subsets. It does not establish an AR-versus-S2S winner. The [first S2S PLM matrix](#s2s-pretrained-encoders-results-and-next-runs) also completed all six runs: judge perplexity is 3.441–4.132, with no improvement over the earlier scratch S2S reference. Unequal capacities/budgets, one seed, and incomplete convergence prevent attributing this to pretraining; IgBert + partner has confirmed encoder representation collapse ([diagnostics](results/diagnostics/20260914_igbert/REPORT.md)), but the 2026-09-14 replay at the same seed and settings did **not** reproduce it (`20260914_140305_226060_igbert_diagnostic`), so its trigger, incidence and remedy remain open; this rerun did not reproduce collapse within three epochs.
- **The S2S LR screen and shipped defaults use different optimizer settings.** The [six-trial scratch screen](results/runs/20260907_204721_sweep_lr/summary.tsv) tested LR 1e-4 / 3e-4 / 1e-3 at warmup 0.06 and decay 0.01, seeds 42/43; 3e-4 has the lowest mean judge perplexity in that screen. The fresh comparison and current scratch configs use LR 1e-4 with zero warmup/decay. A sweep does not automatically promote its settings into configs. New PLM configs start at LR 3e-5 with warmup 0.06 and decay 0.01 and still need tuning.
- **The `compare` suite's three arms are matched on protocol but not on capacity.** Corpus, split, evaluation frameworks, candidate count, `top_p`, `max_len`, judge and seed all match; parameter counts do not. `ar` is 800k, and `s2s` is 933k (1.17x) even after pinning its stacks to 2/2 — the closest match that keeps real depth on both sides, since the nearest-by-count alternative needed a single-layer encoder and would have gutted the one thing the method exists to test. The counts are logged in every `train.log`; **quote them in any table**, and do not read a 1.17x-larger model's win as an architecture result until the `arch` sweep separates the two.
- **The causal method's plausibility advantage is judge-specific, and a native-referenced metric contradicts it.** On OAS4 H3 the causal model reaches judge perplexity 3.15 against masked 9.35 at equal recovery — but BLOSUM62 global alignment over retained non-empty candidates puts it behind: 0.146 against 0.199. The two disagree because both are confounded by length in opposite directions: the aligned score charges gap penalties for the 93% of causal draws whose length is wrong, while the masked arm is handed the native length and never pays one; judge perplexity is length-blind and native-free, so it is indifferent to exactly what causal is bad at. Neither is a clean method comparison until both arms predict their own length (the `compare` suite). Also worth recording: **core BLOSUM is ~0 for every H3 model** (causal 0.020, masked -0.001, ESM-2 0.003) — core substitutions are on average no better than the matrix expects between unrelated residues, which exact-match core recovery of ~0.20 made look like partial credit.
- **The judge shares the eval distribution.** IgBert was trained on OAS paired repertoires — the same distribution as the OAS eval split it judges. It's applied identically to every model, so the *comparison* is fair, but absolute perplexities are OAS-flavoured; an off-distribution judge would be a stronger check.
- **Generative sweep is H3/OAS only.** The matrix covered CDR-H3 on OAS (150 held-out frameworks × 4 models). L-chain loops, SAbDab2, and the antigen configs have not been swept generatively.
- **Sampler is marginal-based, not learned-joint — now measured.** The `iterative` (Mask-Predict) decoder approximates the loop joint by conditioning on committed residues, but the model is still a masked-LM trained on independent-position recovery. As of 2026-07-28 this is no longer a suspicion: iterative and `independent` are **within noise on every metric** on H3/OAS/ESM-2 (recovery 0.4057 vs 0.4072; judge-ppl 7.14 vs 7.23) at ~15× the compute. The conditioning buys nothing because the learned marginals are already near-independent — AR and S2S now provide autoregressive training paths; their free-running evaluations must be assessed separately from this historical MLM decoding comparison.
- **Three runs hit their epoch ceiling, all of them context runs.** `h1_OAS_LC`, `h1_SD2_LC` and `l3_OAS_HC` (all scratch) peaked at their final epoch, 20 of 20, in [`20260818_192606`](results/runs/20260818_192606) — the batch the Findings tables quote. Their numbers are lower bounds, so the §4 partner-chain penalty (−0.0102) is an upper bound on the true effect. A scratch-only sweep at 30–40 epochs would settle it (~1 hour; scratch is 60 of the batch's 226 minutes). The earlier 07-29 batch had five such runs; `h2_OAS_LC`, `h3_OAS_LC` and `l1_OAS_HC` no longer hit the ceiling in the 08-18 re-run.
- **Uncertainty coverage is incomplete.** MLM context/loss studies and LR screens include multiple seeds, and the IgLM comparison has paired bootstrap intervals. The fresh three-method comparison and the S2S PLM matrix are still single-seed. Seed spread, repeated-run variation, and uncertainty across evaluation frameworks measure different things; small method/context differences — light chain hurting scratch by ~0.01 while staying neutral for ESM-2, for example — need matched replication and error bars before they can be claimed.
- **Every batch through 2026-08-17 selected checkpoints on the test split.** Not just the 07-29 runs: `eval_split` defaulted to empty, which fell back to "every non-train entry", and `processed_oas_data.json` has only `train`/`test` — so early stopping, `load_best_model_at_end`, the reported `eval_*` metrics, and `generation_eval.json` all read the final holdout. The 08-17 post-refactor batch (`results/runs/20260817_200051`) is affected too; its logs read `2111 eval (split: all non-train)`. **Fixed 2026-08-18**: `get_datasets` now refuses an unset `eval_split` whenever a `test` split exists, and the 36 OAS configs declare `processed_oas_tuning.json` + `eval_split: validation`. **Resolved for OAS 2026-08-31**: the matrix was re-run as `results/runs/20260818_192606` and every Findings table now quotes it, so the OAS numbers are validation-selected and the test partition is untouched. Two things still stand: the **SAbDab2** rows come from configs declaring `eval_split: test` and remain selection-on-holdout (see the SAbDab2 entry below), and any figure elsewhere in this README still sourced from a pre-08-18 batch — the Step 4 decoding table, the §12 confidence numbers, and the §3 position breakdown — has not been re-derived and stays exploratory.
- **Neither fixed-length policy is unbiased — report both.** *(Re-run 2026-08-21; see Findings §11.)* `constrain` corrupts IgLM's loop endings; `filter` is fair per-example but drops the frameworks IgLM cannot length-match (5 of 25, skewing long), so it scores an easier subset. Current practice: report `filter` on a matched population and state the dropout. A policy that is unbiased in both respects — e.g. raising the attempt budget until every framework is covered, or reporting per-length-bucket — is still open. Historical note on `constrain`: IgLM's `constrain` policy sets `min_length = max_length = native_length`, which suppresses its end-of-span logit until that length is reached — so it writes past its loop into FR4. Only 2% of constrained loops carry the J-gene motif against 20% naturally (Findings §11), and every fixed-length CDRSeqGen-versus-IgLM number is inflated in CDRSeqGen's favour. Re-run with the adapter's `filter` mode (retain only naturally length-matched draws; ~12 attempts per accepted candidate at IgLM's 8.4% match rate) before making any fixed-length claim. The length-free track is unaffected — IgLM uses natural EOS there.
- **SAbDab2 has no validation partition.** Its split comes from the source `ab_ag_split` column, which is `train`/`test` only, so its 12 configs now declare `eval_split: test` — honest, but still selection on the holdout. A three-way SAbDab2 split (or a fixed validation carve-out of its train partition) is needed before any SAbDab2 comparison is reportable.
- **Hyperparameter selection is not yet frozen.** The validation-safe 32-trial scratch-H3 screen completed on 2026-08-11 and placed `1e-3` / 0.06 warm-up / 0.01 weight decay first at 0.32305 mean validation core recovery. Its margin over the runner-up was only 0.00036 across two seeds and changed sign by seed, so additional shared confirmation seeds are still required before touching the final test split.
- **Checkpoint-selection policy is still config-declared.** The current configs select core recovery except the four OAS-L2 configs, which select whole-span because those runs have only 1.4% core coverage. Artifact reporting no longer hard-codes that loop name: it derives the metric from recorded coverage. Training should eventually validate or derive `metric_for_best_model` from the dataset before a run, so a future dataset cannot inherit an obsolete L2 exception.
- **Implementation checks do not establish experimental conclusions.** The suite passed 419 tests as of 2026-09-14, including real tiny ESM/BERT weights, encoder freezing, gradients, separate vocabularies, and exact checkpoint reloads. Actual ESM-2/IgBert GPU smoke runs also exercised training and generation with the real judge. The September 13 PLM batch subsequently completed full ESM-2, IgBert, and ProtBert training and generation in all six settings. Passing checks does not establish split separation, convergence, or biological quality; see [the test guide](unit_tests/README.md).
- **Structure not yet available.** SAbDab2 preprocessing parses *sequences* only; the Phase-2 structure objective needs a coordinate-parsing step (or predicted structures via IgFold/ABlooper).

### How to read the numbers
- **Report core only when it represents the dataset.** Whole-span recovery is anchor-inflated on H3, but a core score is also misleading when most loops have no residues left after trimming. The shared 50% policy uses `eval_core_accuracy` only when the run's recorded `eval_core_coverage` is at least 50%; otherwise core is `N/A` and the explicitly labeled whole-span score is reported.
- **OAS L2 is an observed case, not a coded exception.** With `ANCHOR_TRIM=2`, 98.8% of its loops are three residues, so only ~30/2,111 evaluation examples (1.4%) contain a core. The reporter reads that 1.4% from `epoch_metrics.csv` and selects whole-span. A future L2 dataset with adequate coverage would use core, while any other short-loop dataset would correctly fall back to whole-span.
- **The historical fixed-native MLM tables supply the true loop length.** Length-free results in §10–11 and the fresh three-method comparison are separate protocols. `dataset_mlm.py` inserts exactly `len(native)` mask tokens for that protocol, so the length is given away by the input's shape. This is standard fixed-length redesign and is required for position-wise AAR to be well-defined, but it is not free information; how much it is worth varies enormously by loop:

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
- **The decoder must match the checkpoint's masking policy.** `iterative` decoding re-masks and resamples positions one at a time, which assumes the model saw partially revealed loops — true only under `masking_policy: mixed`. On the same `full_span` checkpoint and protocol, iterative decoding costs whole-loop 0.370 -> 0.253 and core **0.204 -> 0.077**, dropping it below even the 0.084 marginal-sampling baseline. `evaluate.py` therefore resolves the default from `masking_policy` (missing legacy metadata means `full_span` → `independent`) and warns when iterative decoding is requested explicitly off distribution.
- **Read best-epoch, not last-epoch** — ESM-2 overfits on SAbDab2/H3; `load_best_model_at_end` saves the peak, but the last CSV row can sit well below it.
- **Never plot OAS and SAbDab2 on shared axes** — different test sets; the `_OAS`/`_SD2` run tags enforce it, and per-dataset charts live under `results/figures/training/<batch>/*.png`.
- **`epoch_metrics.csv` and `*_results.json` measure different models — they're *supposed* to disagree.** The CSV is the per-epoch **trajectory** (the model as it stood at each epoch); `eval_results.json`/`all_results.json` come from the post-training `trainer.evaluate()`, which with `load_best_model_at_end` runs on the **best** checkpoint while still being stamped with the final epoch number. On `h3_esm2` the CSV's epoch-10 row reads core **0.3063** (the actual last model) while `eval_results.json` reads **0.3209** (the epoch-4 best model). `save_training_csv` de-duplicates on epoch and **keeps the first** occurrence precisely so the curve keeps the real last-epoch point — keeping the last would silently redraw the best model at x=10 and erase the overfitting decay entirely.

<!-- MANUAL PROTECTION ZONE BEGINS -->
### Changelog
<!-- AI ACCESS ZONE BEGINS -->
<!-- AI ACCESS ZONE ENDS -->
- **2026-10-06** -- [RECLAIM] `eval/`: `metrics.py`, `generative.py`
- **2026-09-22** -- Reclassified config dirs
- **2026-09-22** -- Revised environment settings (server migration)
- **2026-09-18** -- [RECLAIM] `datasets/`: revised dataset class scripts
- **2026-09-17** -- [RECLAIM] `datasets/preprocess/`: Redesigned the data preprocessing pipeline
- **2026-09-17** -- Data storage restructure: dataset code to `cdrseqgen/datasets/`; raw/processed files to root `data/` (old: `datasets/data/`)
- **2026-09-15** -- **Experiments table added:** timeline for future framework development and experiments for conference-ready results
- **2026-09-15** -- Span layout + context dropout edits
- **2026-09-14** -- [RECLAIM] Revised reclamation list
- **2026-09-10** -- Added PLMs for S2S: ESM-2, IgBert, ProtBert
- **2026-09-10** -- `cdrseqgen/experiments.py` [new]: unified suite/sweep planner + executor for `run.sh`/`sweep.sh` (to be moved to `scripts/`?)
- **2026-09-07** -- **SEQ2SEQ UPDATE** (`methods/s2s/`)
- **2026-09-04** -- Enabled different span layouts: `SpanLayout` in [`datasets/dataset_ar.py`]
- **2026-09-04** -- Added BLOSUM62 in `eval/metrics.py`
- **2026-09-04** -- AR context fix: `max_len` 256 -> 384 (to make space for partner chain since answer reserve is long (96 toks))
- **2026-09-04** -- First OAS4 results (18 runs, mlm+ar)
- **2026-09-03** -- `run.sh`: renamed suites, takes several GPUs. To run all mlm + ar experiments: `run.sh <gpu, sep=,> <suite+>` (eg. `scripts/run.sh 0,1,2 mlm_heavy ar_heavy`)
- **2026-09-03** -- Expanded OAS corpus (8 csvs, 4 donors, total 152727 unique paired entries)
- **2026-09-03** -- Hyperparameter sweep for autoregressive (optimal: 1e-4 LR, 0.06 warm-up, 0.01 weight decay)
- **2026-09-02** -- AR prompt fix for training vs generation (native length was leaked into flank truncation)
- **2026-09-01** -- **AUTOREGRESSIVE UPDATE** (`methods/ar/`) (can be run directly with `run.sh <gpu> ar`(0903: deprecated))
- **2026-08-31** -- Data layer split (`frameworks.py`) for mlm to make way for autoregressive dataset class
- **2026-08-31** -- Former `test/` dir split into `showcase/` (jsons for inference) and `unit_tests/` (for examining train scripts)
- **2026-08-31** -- Auto figure plotting after hyperparameter & loss sweeps, in sweep dir instead of separate `figures/` folder
- **2026-08-31** -- Data visualization rework, run/sweep type -> table/graph plotters
- **2026-08-25** -- Enabled different loss types besides cross entropy, currently added: weighted cross entropy (core vs anchor)
- **2026-08-21** -- Added confidence scores per candidate and confidence vs AAR correlation metrics
- **2026-08-21** -- `scripts/sweep.sh` [new]: merged `sweep_h3.sh`, `sweep_scratch.sh` into unified hyperparameter sweep script
- **2026-08-21** -- `results/` storage reorganized
- **2026-08-21** -- IgLM benchmark fix: fixed-length for IgLM now filters samples that match the native length than forced EOS (is this approach ok??)
- **2026-08-18** -- project wide claude fix
- **2026-08-17** -- Produced IgLM benchmark results
- **2026-08-17** -- Numerous `test/` scripts
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
- **2026-07-27** -- `datasets/preprocess/common.py`[new]: common scaffolding for csv -> json data preprocessing scripts
- **2026-07-27** -- moved and renamed preprocessing scripts to `datasets/preprocess/`, removed separate dataset README file
- **2026-07-24** -- Generate `run_config.json` for each training run
- **2026-07-24** -- `sampling.py`: handled greedy algo for both independent and iterative decoders

---

## ⏭️ Roadmap

<!-- AI ACCESS ZONE BEGINS -->
> **Objective-first:** the target is **CDR-H3 design**. **Phase 1** = a leakage-controlled *sequence benchmark* (historical MLM settings plateau near 0.32 masked core recovery; AR/S2S teacher-forced and generated recovery are separate measures; see Findings), positioned against a published sequence-infilling baseline under one matched evaluation protocol. **Phase 2** = *structure-conditioned* design (RMSD-to-native; exact target to lock with advisor). Everything below is a tool toward that objective.

### Medium-term milestone — conference-ready CDR-H3 infilling study

**Research question:** how do generation method, encoder pretraining and partner-chain context affect framework-conditioned CDR-H3 generation under one evaluation protocol?

**Scope:** sequence-only H3 infilling; length-free generation is the primary track and native-length redesign a separate diagnostic. Structure-aware models and other loops are follow-ups. The [Active milestone](#active-milestone--matched-cdrseqgen-vs-iglm-benchmark) below is this study's external-baseline component, not a parallel effort.

**The negative results are the likely spine, not a byproduct.** Historical MLM context comparisons show little benefit (§4), while AR/S2S dropout improves missing-partner robustness with a context-present cost (§14); loss re-weighting changes nothing (§13); iterative decoding buys nothing over independent at ~15× the compute (Open problems); H3 length is near-unpredictable, with every learned method below a non-neural V/J lookup (§10); and the model's own confidence is nearly useless for selection where an independent judge is not (§12). Decide early whether the contribution is that controlled set of negatives or something else — the evidence each needs differs sharply.

- [ ] **Agree the contribution and target venue with the advisor.** Controlled empirical study, evaluation benchmark, or new method. State what it adds beyond existing infilling work, then derive the evidence list from that claim rather than from what has already been run
- [ ] **Freeze the evaluation protocol before any final experiment**, and gate everything below on it: inputs available to each method, length policy, example selection, candidate/attempt budgets, decoding settings, checkpoint-selection metric, primary endpoint and supporting metrics. Relabel every existing result as exploratory under the old protocol
- [ ] **Resolve the judge conflict.** `gen_judge` is `Exscientia/IgBert` in every config — including `h3_s2s_igbert[_lc]`, where IgBert is *also* the encoder, so the judge scores its own encoder's output. IgBert was additionally trained on OAS paired repertoires, the same distribution as the eval split. Either add an off-distribution judge or drop judge perplexity as a primary endpoint; a shared-identity judge will not survive review
- [ ] **Finish the leakage audit rather than restart it.** Already done: OAS4 donor/disease confounding recorded, the [September pairwise audit](results/reviews/0903-0904-review.md) (3 of 150 H3 frameworks with ≥85%-identity same-length training neighbours), and versioned frozen manifests. Still open: target-loop clustering for non-H3, nearest-neighbour similarity at scale, and overlap with *PLM pretraining* corpora — IgLM and IgBert were both trained on OAS, so neither is a leakage-free comparator on an OAS holdout. If the audit forces a revised split, retrain before comparing
- [ ] **Complete the minimum controlled comparison.** Stable MLM, AR and S2S against IgLM plus cheap retrieval/empirical baselines (the V/J length lookup is already one). Report parameter count, training exposure, tuning effort and compute per arm: the `compare` arms already differ 1.17× (`ar` 800k vs `s2s` 933k) and the PLM arms span 34M to 420M, so unstated capacity would confound every conclusion
- [ ] **Run the controls each claim needs.** Pretraining: same-architecture pretrained versus random initialization. Context: matched context settings plus dropout-trained models evaluated in both regimes. Conditioning: framework/context perturbation response, via the encoder-memory probe once its criteria are calibrated. Frozen encoders are a feature-extractor control, not a pretraining control
- [ ] **Report training stability as a result, not a footnote.** The IgBert + partner collapse and its 2026-09-14 non-reproduction at an identical seed show a run can fail catastrophically and nondeterministically. Report failure incidence per configuration alongside performance, and state the run-selection rule — silently dropping failed runs would overstate every arm that has one
- [ ] **Replicate finalists and quantify uncertainty.** Three seeds is a floor, not a target: run-to-run nondeterminism at a *fixed* seed already exceeds the seed spread by 2–4× on stable runs (§13) and can flip an unstable one to total failure, so seed spread alone understates reproducibility. Include repeated identical-setting runs as a deliberate determinism control, use paired framework-level comparisons accounting for clustering, and keep evaluation uncertainty separate from training variability
- [ ] **Test whether conclusions survive evaluation choices.** Report validity and length/alignment metrics over all attempts; label length-matched recovery with its contributing population; check dependence on judge, length bucket and decoding setting. Judge perplexity and BLOSUM alignment already disagree in direction on AR versus MLM (Open problems), so this is a known risk rather than a hypothetical. Include diversity and novelty beside recovery
- [ ] **Run the final held-out evaluation.** Freeze settings and reporting first. Test-set history is known and must be stated rather than rediscovered: every batch through 2026-08-17 selected checkpoints on the test split, OAS was re-run clean as `20260818_192606`, but **SAbDab2 still selects on its own holdout** and has no validation partition — carve one out or keep SAbDab2 out of the headline claims. Do not tune on the final results
- [ ] **State the regime honestly.** 122,181 training pairs, 0.8M–420M parameters, single-digit-hour runs, one corpus family. The conclusions are about this scale; say so rather than letting a reader generalise them
- [ ] **Produce the evidence package and release artifacts.** Main comparison table, conditioning/control figure, length and coverage breakdown, uncertainty estimates, compute and failure accounting — each figure with a defined claim and a reproducible source artifact. Pin code/environment versions, configs, manifests, model provenance and scoring commands, preserving historical runs' real provenance rather than back-dating it. Write methods and limitations alongside the experiments, then related work, results and discussion

### Experiment coverage across methods

Status snapshot: **2026-09-17**. **☑** experiment completed; **◐** partial or historical evidence; **↻** in progress; **☐** pending; **—** requires a pretrained AR adapter first. Completion means exploratory results exist, not that the paper's final confirmation is finished. Pending cells are candidates, not a requirement to fill the entire matrix; prioritize matched context comparisons, the controls needed to interpret them, and confirmation across seeds. The detailed checklists below specify the follow-up work.

The [experiment-oriented results index](RESULTS.md) links each category to its saved evidence and ranks the next runs. Run `python scripts/organize_results.py` to regenerate the local `results/by_experiment/` view from the catalog; original timestamped batches remain the provenance records. Hyperparameter sweeps stay in their original folders and are excluded from this view.

| Experiment category | MLM | AR | S2S |
|---|---|---|---|
| **[Scratch baseline](results/by_experiment/scratch_baseline/)** | ☑ | ☑ | ☑ |
| **[Pretrained models](results/by_experiment/pretrained_models/)** | ☑ ESM-2, IgBert, ProtBert | ☐ Pretrained causal adapter and experiments | ☑ ESM-2, IgBert, ProtBert **encoders** |
| **Scratch learning-rate tuning** | ☑ Broader hyperparameter screen | ☑ LR screen | ☑ LR screen; settings not yet promoted |
| **PLM learning-rate tuning** | ☐ Matched follow-up as needed | — | ☑ ESM-2 four-run partner-present screen reviewed (§15); confirmation pending |
| **[Train with versus without partner context](results/by_experiment/partner_context/)** | ◐ Existing comparisons; match final protocol | ◐ Historical runs; **2 matched no-partner runs pending** | ◐ Existing comparisons; **4 matched no-partner runs pending** |
| **[Context dropout 0 versus 0.5; evaluate both regimes](results/by_experiment/context_dropout/)** | ☑ Historical experiments; protocol differs | ☑ 4/4 complete and reviewed (§14) | ☑ Scratch 4/4 and ESM-2 4/4 complete and reviewed (§14) |
| **Same architecture: pretrained versus random initialization** | ☐ | — | ☐ ESM-2 control; code available |
| **[Frozen versus trainable pretrained weights](results/by_experiment/frozen_encoder/)** | ☐ | — | ◐ IgBert diagnostic trained; generation pending. ☐ ESM-2; code available |
| **[Conditioning/collapse diagnostics](results/by_experiment/conditioning_diagnostics/)** | ◐ Context sensitivity studied | ◐ Context-ablation results available | ◐ IgBert probes complete; generation and replication pending |
| **[Length-free generation and length analysis](results/by_experiment/length_free_generation/)** | ☑ Length-head experiments | ☑ EOS/length analysis | ☑ Generation metrics; comparative analysis ongoing |
| **[Decoding/sampling sensitivity](results/by_experiment/decoding_sampling/)** | ☑ Independent/iterative and sampling studies | ☑ Temperature/length study | ☐ Controlled sampling screen |
| **[Matched external IgLM benchmark](results/by_experiment/external_iglm_benchmark/)** | ◐ Small pilot; final comparison pending | ☐ | ☐ |
| **[Additional seeds for selected contrasts](results/by_experiment/additional_seeds/)** | ☐ | ☐ Seeds 44–46 | ☐ Seeds 44–46 |
| **[Light-chain experiments under target-loop-specific splits](results/by_experiment/light_chain/)** | ◐ Historical runs; corrected split required | ☐ After split work | ☐ After split work |
| **Final frozen protocol, uncertainty estimates and held-out evaluation** | ☐ | ☐ | ☐ |

### Active milestone — matched CDRSeqGen vs IgLM benchmark

**Primary baseline:** [IgLM](https://github.com/Graylab/IgLM) — the closest published, reproducible match to CDRSeqGen's sequence-infilling task. IgCraft is the intended second sequence baseline once the comparison interface is stable; structure/antigen models such as IgGM belong to Phase 2 and must not enter the sequence-only table.

Both methods run on the **same frozen examples**, with the same candidate count and the same scorer, across two tracks that must never be collapsed into one number:

1. **Fixed-length redesign (first deliverable).** Native H3 length supplied to both models. Matches CDRSeqGen's mask construction and permits position-wise recovery with the anchor/core split.
2. **Length-free infilling.** Only the flanking framework supplied. IgLM terminates with EOS; CDRSeqGen predicts a length from one gap sentinel, then expands it into residue masks. Report validity, length error, edit similarity and length-matched AAR separately.

**Done — harness and first comparison (through 2026-08-21).** Versioned H3 manifests from the clustered OAS split; model-independent shared scorer; manifest-driven CDRSeqGen adapter (provenance-validated, checkpoint-hashed, resumable) and attempt-bounded IgLM adapter (kept outside the MLM backbone registry); explicit fixed-length policy (`constrain` default, `filter` audit mode); checkpoint-specific length head with bounded/inferred generation and migration tooling; separate `length_free` manifest/adapter/scorer contract; validated shared evaluator (AAR, diversity, novelty, entropy, length buckets, runtime, revision-pinned IgBert judge); official infill example reproduced (125/125 constrained draws valid, zero adapter failures). **Result: IgLM leads on both tracks** (§11) — recovery, length error, edit similarity and judge perplexity. Paired bootstrap ([`bootstrap.py`](benchmark/bootstrap.py), example as the resampling unit): **only 4 of 8 metrics resolve at 95%** — IgLM's judge-perplexity, length-MAE and edit-similarity leads hold, CDRSeqGen's diversity edge holds, and every recovery comparison is unresolved at n = 25. ⚠️ **Rescore on a clean tree before quoting** — stored scores carry `repository_dirty: true` at commit `0e7e499`, and the intervals must be regenerated after that rescore.

- [ ] Cover the 5 frameworks `filter` drops — raise the attempt budget or report per-length-bucket, so the fixed-length population isn't silently the easier subset
- [ ] Produce side-by-side curves: recovery by H3-length bucket, accuracy by relative H3 position, and the recovery–diversity frontier across matched sampling settings. Do **not** compare CDRSeqGen training curves against a frozen IgLM checkpoint; that becomes meaningful only if IgLM is retrained on the same corpus, and should then use tokens/examples seen rather than epoch number
- [ ] Audit training-data overlap. Frozen IgLM was pretrained on OAS, so the local OAS holdout is not automatically a leakage-free generalization estimate. Report the limitation, and investigate either retraining IgLM on the exact local training split or building an independent test set
- [ ] Write one comparison artifact — manifest hash, model provenance, full parameters, per-example records, aggregate table, confidence intervals, plot paths — reproducible from a single command

**General**

**Done.** Sampler (temperature/top-k/top-p, independent + iterative Mask-Predict) and generative metrics ([`sampling.py`](cdrseqgen/methods/mlm/sampling.py), [`metrics.py`](cdrseqgen/eval/metrics.py)); generative sweep across the H3 model/context matrix (150 held-out frameworks × 4 models, IgBert judge); reporting hygiene (core-vs-whole validity derived from recorded `eval_core_coverage`, both labeled, never branching on loop name); `num_attention_heads`/`intermediate_size` plumbed through `build_model` (`intermediate_size` defaults to 4 × `hidden_size`); three-way clustered OAS splitting with explicit validation/test selection and full run provenance; 32-trial validation-safe H3 screen — top region `1e-3` LR / 0.06 warmup / 0.01 weight decay at **0.32305** mean two-seed core recovery; core/anchor loss screen — **no re-weighting beats plain cross-entropy**, and core-only collapses the anchors without helping the core (§13); `loss` grid canonicalised by core-to-anchor ratio (2026-09-10); policy-aware decoding with `anchor_trim` propagated into generative evaluation (2026-08-31); checkpoint selection separated from ranking (whole-span for selection, judge perplexity for ranking; historical results keep their original policy).

- [ ] **Per-loop split** — cluster on the *target* CDR, not always heavy CDR3, then re-run the OAS matrix. Everything non-H3 in Findings is retrieval-bounded until this lands (§6)
- [ ] Confirm the top two sweep settings with additional shared seeds, freeze the selected config, then evaluate it once on the untouched test split
- [ ] Multi-seed the key H3 configs (error bars on the ±0.01 model/context deltas)
- [ ] Finish the loss screen's two missing `weighted 3.0/1.0` seeds (43, 44) so the only distinct untested setting is measured rather than noise-bounded; skip `3.0/0.0` — it is the same objective as `1.0/0.0`
- [ ] Settle the §12 self-vs-judge ordering with one 150 × 20 `evaluate.py` run on the tuned checkpoint — the 08-25 screen inverted it at 100 × 10
- [ ] Re-run the 07-24 batch under core-based checkpoint selection (those 36 runs predate the fix; their saved weights were picked on whole-span)
- [ ] Lock the Phase-2 objective and primary metric with advisor

**Autoregressive method**

**Done — the length question is closed (through 2026-09-04).** All six `ar` configs retrained under the fixed `LOOP_LENGTH_RESERVE` (`20260902_164544_ar`; `valid_rate` ~1.0 confirms the `<eos>` mechanism is healthy). H1/H2 are perfect and therefore meaningless (§6 retrieval regime); on H3 the causal model reaches **8.3% exact length / 3.80 MAE**, against §10's length head at 9.24%/3.64 and a modal-length baseline at 10.52%/3.28 — `<eos>` does **not** beat the length head, and both sit below the constant baseline on the natural length distribution. Three causes excluded: decoding (temperature sweep moves MAE only 4.31 → 3.48 from T=1.0 to greedy, exact-match never leaves ~8%; mean length well calibrated at 16.11 vs 16.17 native but under-dispersed, correlating 0.396 with the target); learning rate (`20260903_155940_sweep_h3` — teacher-forced core moves 0.0005 across a 10× range against a 0.006 noise floor); data scale (`20260903_192305` — at 7.2× the data the model held at exactly 1.13× its corpus's own modal-length MAE, and a V/J-conditioned lookup puts the achievable ceiling at 11.07% / 3.01). **Length is now a reported property, not an optimisation target.** The causal `generate.py` counterpart landed 2026-09-07 (`ar/cli.py` + `ar/case_generation.py`, dispatching through the registry).

- [ ] **Decide the showcase fixture** — blocking the causal CLI (diagnosed 2026-09-07). Every `showcase/*.json` carries one "design only" case with no `target_truth` and an explicit `target_length`, because a masked LM must be told a length when there is no native to derive one from. Both length-free methods reject that key and abort the whole file; dropping it runs all three H3 cases correctly. Either add length-free probe sets, or relax the two CLIs to warn-and-ignore a length request they cannot honour
- [ ] **Sweep capacity** (`arch` preset) on the expanded corpus rather than the current one: at 0.8M parameters capacity and data scale plausibly interact, so an answer at 21k would not transfer. Also the axis that would settle whether `s2s`'s 1.17× parameter edge matters; `sweep.sh` drives both encoder and decoder depth on an s2s base config
- [ ] **Harden `SpanLayout` before registering a second layout** (review finding 2026-09-04; none of it affects the shipped `CDRSEQGEN_CAUSAL` or any current result): `build_prompt` receives the whole framework dict **including `target`**, so a layout could leak the answer into its own prompt and the training-equals-generation test would still pass; `n_layout_tokens` is unverified metadata that drives truncation; `CausalInfillingDataset` does not validate its layout, so an invalid one trains a whole run before generation rejects it; and `validate_layout` accepts a residue token as the terminator, which would stop decoding at that amino acid
- [ ] Record `span_layout_id` in `run_config.json` and compare it in the checkpoint-compatibility check — otherwise changing a backbone's layout silently reinterprets older checkpoints
- [ ] **A pretrained causal backbone** through `methods/ar/backbones.py` (registry has `scratch` and a generic `hf_decoder` fallback). `SpanLayout` is only a partial prerequisite: it cannot express a native format needing generated-output conversion or wrapper tokens, and IgLM's format additionally needs `validate_generated_seq` + `iglm_to_infilled`. Still missing: layout-aware extraction, per-config layout selection, conditioning fields on the framework dict, and a wired layout on the `hf_decoder` path. Format and checkpoint are separate questions — adopting IgLM's *layout* is clean, fine-tuning IgLM's *weights* is not, since it was pretrained on OAS and is the §11 baseline
- [ ] **Automate one GPU + real-judge smoke run.** Three defects reached live runs past a fully green CPU suite — device mismatch, prompt divergence, and the missing `pseudo_log_likelihood` import — because no test exercised the full GPU training-to-generation path with a real judge. The 2026-09-07 `s2s` landing ran this by hand and caught exactly those three; that is the argument for automating it, not a substitute

**Encoder-decoder method**

**Done.** `s2s` implemented end to end 2026-09-07 (two-stream data view, bidirectional encoder + cross-attending decoder, backbone registry, training, decode-to-`<eos>` sampling, shared-evaluation strategy, CLI, entrypoint, configs, `s2s_heavy` suite, `compare` extended to three arms; 44 new tests), with the full path verified outside the suite on GPU against the real IgBert judge (`valid_generation_rate` 0.975, `judge-cov` 1.000). Scratch S2S LR screen complete ([20260907_204721_sweep_lr](results/runs/20260907_204721_sweep_lr/summary.tsv) — 3 LRs × 2 seeds; settings not promoted into the scratch configs), as is the three-arm [`compare` suite](results/runs/20260908_173143_compare/batch_tables.md) on OAS4 validation with length-free generation for all three methods. Pretrained encoders landed 2026-09-10: ESM-2, IgBert and ProtBert feed the shared residue decoder through a learned width projection, with separate encoder tokenizers and `encoder_init`/`freeze_encoder` controls — implementation capability, not evidence of improved generation. The [full PLM matrix](results/runs/20260913_151152_463868_s2s_plm/batch_tables.md) completed 2026-09-13 (six runs, 18,000 complete candidates, full judge coverage), and the IgBert + partner [collapse was diagnosed](results/diagnostics/20260914_igbert/REPORT.md) 2026-09-14.

- [x] **Complete the three-arm IgBert diagnostic** (`20260914_140305_226060_igbert_diagnostic`). All arms finished three epochs, with 49 finite probe records and three epoch checkpoints each. The original-LR replay did not reproduce the earlier collapse within this window. Epoch-3 validation loss / core recovery: original 1.6092 / 0.3836; lower LR 1.6954 / 0.3566; frozen 1.6185 / 0.3814. All retained conditioning. This establishes neither failure incidence nor a preventive effect of freezing/lower LR. See the [completed diagnostic report](results/diagnostics/20260915_igbert_summary/REPORT.md).
- [ ] **Evaluate the three saved diagnostic checkpoints generatively.** Same validation examples and 150 × 20 generation protocol, with fixed partner context for generator and judge; no retraining. [Commands](configs/experiments/20260915_context_dropout.md#evaluate-the-completed-igbert-diagnostic-checkpoints-first).
- [x] **Run and review the AR/S2S context-dropout screen.** All 12 trials completed, with both evaluation regimes and fixed judge context. Dropout 0.5 improves missing-partner edit similarity in both seeds for every model, at a small partner-present cost (§14; [September 15–17 review](results/reviews/0915-0917-review.md)). Matched no-partner training controls and additional seeds remain pending.
- [x] **Preserve trial identity in batch tables.** Fixed 2026-10-06: new tables and dashboards retain every seed/dropout trial and show generator/judge context separately. Artifact-only regression against the completed context batch discovers all 12 runs and both evaluation regimes. Historical reports remain as originally saved.
- [ ] **Add six models trained without partner context — next training batch.** Scratch AR, scratch S2S and ESM-2 S2S × seeds 42/43, with `include_partner_chain: false` and `context_dropout: 0`. Match the current screen within each model: OAS4 H3 validation split, optimizer, batch size, 20-epoch maximum, early stopping and checkpoint-selection metric. Evaluate the same 150 × 20 examples/candidates without generator partner context and with fixed partner context for the judge. Compare against partner-trained dropout 0 and 0.5 models; removing the partner only at inference does not replace this training control. The [six-trial spec](configs/experiments/20261006_no_partner_controls.json) was launched 2026-10-06; initial AR attempts failed before training on missing TensorBoard, now corrected. See the [protocol/retry commands](configs/experiments/20261006_h3_followup.md).
- [ ] **Confirm selected contrasts with seeds 44–46.** Select follow-ups using validation results, then extend both sides of each contrast, including both dropout conditions when testing a dropout benefit. The [initial four-run ESM-2 spec](configs/experiments/20261006_esm2_dropout_confirmation.json) prepares dropout 0/0.5 at seeds 44/45 after the six controls; seed 46 remains optional pending time and results. Report individual seeds and paired differences; keep the test split untouched until settings and endpoints are frozen. For IgBert stability follow-ups, also report collapse incidence: one failure and one clean replay establish neither its rate nor run-to-run variance.
- [ ] **Separate ESM-2 pretraining from capacity and adaptation.** After reviewing the context screen, compare trainable pretrained, trainable randomly initialized, and frozen pretrained encoders of the same ESM-2 architecture, initially at seeds 42/43. Hold the decoder, corpus, context policy and evaluation protocol fixed; reuse current runs only where settings match. Use `encoder_init`/`freeze_encoder` in a JSON spec. Allow comparable validation tuning effort, since random initialization may require a different learning rate. Report generation quality, validity and contributing populations alongside teacher-forced recovery.
- [x] **Screen ESM-2 learning rates on a longer budget.** Four trials at `1e-5`/`1e-4`, seeds 42/43, up to 20 epochs, completed and reviewed against the `3e-5` context-screen reference (§15; [September 15–17 review](results/reviews/0915-0917-review.md)). All use dropout 0 and partner-present generation. `1e-4` is a candidate for confirmation, not a promoted default; its dropout interaction and partner-absent behavior remain unmeasured.
- [ ] **Compare tuned scratch and PLM systems under a matched protocol.** Match corpus, split, evaluation and comparable tuning effort; state epoch and compute budgets. The initial matrix's judge-perplexity gap (PLMs 3.441–4.132 versus scratch 3.215/3.250) is historical; the new partner-present ESM-2 `1e-4` mean is 3.345 (§15), and differing capacities and architectures still confound attribution to pretraining. Retain the same-architecture pretrained-versus-random control
- [ ] Calibrate conditioning criteria for the exact 16-example encoder-memory probe and fix them before a confirmation run. This batch began without a numerical pass/fail criterion, so thresholds developed from its own traces are exploratory; the earlier 256-example raw-framework-swap measurements are illustrative references, not calibrated callback thresholds
- [ ] **A full pretrained encoder-decoder backbone**: ProtT5 remains a future adapter. The released checkpoint used BART-like residue masking, **not** original T5 span denoising; validate its real pretraining/input/output contract before designing missing-span extraction ([model card](https://huggingface.co/Rostlab/prot_t5_xl_uniref50))
<!-- AI ACCESS ZONE ENDS -->

- [ ] Project flowchart (0814, outdated)

- **Operation Reclamation** -- Manual code cleaning: Comprehend and rewrite the project content in my own terms - previous work handled by AI resulted in the project almost becoming a "black box" for the developer himself 
  - [ ] `benchmark/`
    - [ ] `adapters/` 
    - [ ] `_common`, `bootstrap`, `export_manifest`, `scorer`
    - [ ] `plot_comparison`
    - [ ] README
  - [ ] `cdrseqgen/` 
    - [x] `datasets/`
      - [x] `preprocess/` (1006)
      - [x] `dataset_<method>` (1006)
    - [x] `eval/` (`generative`, `metrics`) (1006)
    - [ ] `methods/mlm/`
      - [ ] custom model
      - [ ] length scripts (`length`, `length_predictor`, `variable_length...`)
      - [ ] `arguments`, `backbones`, `losses`
      - [ ] `cli`, `entrypoint`, `training`
      - [ ] `evaluation`
      - [ ] `case_generation`, `sampling`
    - [ ] `methods/ar/`
      - [ ] custom model
      - [ ] `arguments`, `backbones`
      - [ ] `cli`, `entrypoint`, `training`
      - [ ] `evaluation`
      - [ ] `case_generation`, `sampling`
    - [ ] `methods/s2s/`
      - [ ] custom model
      - [ ] `arguments`, `backbones`
      - [ ] `cli`, `entrypoint`, `training`
      - [ ] `evaluation`
      - [ ] `case_generation`, `sampling`
      - [ ] `diagnostics` (??)
    - [ ] `methods/common/`
      - [ ] `arguments` 
      - [ ] `backbones`
      - [ ] `callbacks`
      - [ ] `evaluation`
      - [ ] `sampling`
      - [ ] `training`
    - [ ] `train/` (logging scripts)
    - [ ] `visualization/` 
  - [ ] `unit_tests/`
  - [ ] entrypoint scripts
  - [ ] README
  - [ ] local `results/`

**Datasets**
- OAS
  - [x] 21,106 paired sequences (`1287169_1_Paired_All.csv`; pre-2026-09-03)
  - [x] OAS4: 152,727 paired sequences (8 CSVs, 4 donors; Naive-B-Cells)
  - [ ] Expand paired data after auditing available donors, B-cell types and attainable scale
  - [ ] Unpaired sequences
- SAbDab2
  - [x] 3,984 paired antibodies with resolved antigen
  - [ ] Inventory and process remaining structures without antigen
- Other sources
  - [ ] SAbDab / Thera-SAbDab

**Preprocessing**
- [x] Bio-Object JSON schema (14 IMGT regions)
- [x] Partner-chain context (light <-> heavy)
- [x] Antigen context (SAbDab2)
- [x] Clustered split (H3) (prevents data leakage)
- [ ] Structure / coordinate parsing
- [ ] Unpaired-OAS preprocessor

**Training / methods**
- Sequence-only
  - General
    - [x] Context dropout (partner chain, antigen)
  - [x] Masked-LM infilling
    - [x] Fixed-native generation
    - [x] Sampling strategies: independent vs iterative
    - [x] Bounded/inferred generation plumbing and checkpoint artifacts
    - [x] Predicts length before infilling residues
    - Models
      - [x] Custom transformer
      - [x] ESM-2
      - [x] IgBert
      - [x] ProtBert
      - [ ] AbLang2
  - [x] Autoregressive infilling
    - Models
      - [x] Custom transformer
  - [x] Encoder-decoder infilling
    - [x] Frozen-encoder and same-architecture random-initialization controls
    - [x] Optional conditioning diagnostics and early stopping that preserves the LR schedule
    - [ ] Establish stable IgBert + partner training with diagnostic controls
    - [ ] Tune and replicate promising settings; compare pretrained and random initialization
    - [ ] Full pretrained encoder-decoder adapter
    - Models
      - [x] Custom transformer
      - [x] ESM-2
      - [x] IgBert
      - [x] ProtBert
- Structure-aware
  - [ ] Invariant Point Attention (IPA)
  - [ ] E(n)-equivariant GNNs
  - [ ] Diffusion (dyMEAN / DiffAb-informed)
  - [ ] Antigen cross-attention / epitope-contact signal

**Evaluation**
- [x] Sequence recovery (AAR) — whole-span, core and anchors
- [x] Self perplexity / pseudo-perplexity and shared judge scoring
- [x] Diversity — pairwise Hamming, unique fraction, per-position entropy
- [x] Novelty — uniqueness versus training set
- [x] Length-free metrics — validity, length match/MAE, edit similarity, conditional length-matched AAR
- [x] Coverage and contributing populations reported alongside conditional recovery
- [ ] Paired bootstrap confidence intervals for the final matched comparisons
- [ ] Structure metrics — RMSD, TM-score

**Benchmark**
- [x] IgLM external benchmark adapter
- [x] Matched fixed-length CDRSeqGen/IgLM validation smoke test
- [ ] Matched length-free CDRSeqGen/IgLM score and plot
- [ ] Final held-out evaluation after model and protocol selection

**Long-term objectives / deliverables**
- Leakage-audited benchmark for CDR-infilling methods
- Structure-conditioned CDR-H3 design
- Integrated sequence + structure design framework
- Benchmark paper and/or a novel antibody-design method

<!-- MANUAL PROTECTION ZONE ENDS -->
