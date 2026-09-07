<!-- Note to AI: Edits allowed OUTSIDE of MANUAL PROTECTION ZONES and INSIDE of AI ACCESS ZONES -->
# CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** 2026-09-07
**Status:** **Three methods now dispatch from one `train.py`** — masked (`mlm`), causal (`ar`), and encoder-decoder (`s2s`, added 2026-09-07). The length question is closed: §9/§10 put the achievable H3 ceiling at 11.07% / 3.01 MAE via a non-neural V/J lookup, barely above always guessing the modal length, so length is reported against that baseline rather than optimised. `s2s` is the missing cell of the 2x2 — bidirectional framework encoding like `mlm`, left-to-right loop generation with model-chosen length like `ar` — motivated by the 2026-09-04 matched comparison, which ties on every native-referenced metric and separates only on judged plausibility (causal 3.15 vs masked 7.55). It is **implemented and verified end to end but not yet measured**: no `s2s` experiment has been run. Next: an `s2s` learning-rate screen (its hyperparameters are inherited from the AR sweep, not its own), then the three-arm `compare` run, multi-seeded.

## Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences given the surrounding framework context. 

Current implemented features include: 
- **Data Preprocessing**: standardize data into Bio-Object JSON format (IMGT), current datasets: OAS (21106 paired sequences, one donor; a four-donor 156k corpus is downloaded and awaiting its first preprocess) and SAbDab2 (3984 paired + antigen structures). One preprocessing run can merge any number of donor CSVs into a single corpus, splitting once over the merged set
- **Addressing Data Leakage**: clustered OAS splitting with optional train/validation/test partitions that keep clonal near-duplicates together
- **Task-Agnostic Masking**: masking training method for all 6 CDR loops individually by target selection
- **Pluggable Backbone PLMs**: supports various PLMs, current list: custom transformer (scratch), ESM-2, IgBert, ProtBert
- **Partner Chain Context**: optional inclusion of non-target chain sequence context
- **Antigen Context**: optional inclusion of antigen sequence context for SAbDab2 dataset
- **Context Dropout**: `context_dropout` trains one checkpoint that works with *and* without its optional context (partner chain / antigen drawn independently), instead of a model locked to the regime it was trained in
- **Generative Sampling**: independent / iterative (Mask-Predict) decoding with temperature (`0` = greedy), top-k, top-p — samples a *library* of candidate loops instead of one argmax guess
- **Explicit Length Policies**: fixed, bounded, and inferred-length MLM generation; inferred mode classifies one gap sentinel and expands it into residue masks without reading the native loop length
- **Autoregressive Infilling**: causal span infilling with both framework flanks in the prompt, answer-only loss, rotary positions, and model-chosen loop length via `<eos>`
- **Encoder-Decoder Infilling**: two-stream span infilling — a *bidirectional* encoder over `pt1 <mask> pt2` and a cross-attending decoder that emits residues until `<eos>`. Unlike the causal layout the loop never shares a sequence with the framework, so no answer reserve competes with optional context for `max_len`
- **Evaluation Metrics**: *recovery-side* — Sequence Recovery Rate (Per-Residue Accuracy), anchor-excluded core accuracy, anchor accuracy. *Design-side* — diversity, novelty, per-position entropy, pseudo-perplexity, plus judge-scored perplexity for cross-model comparison
- **Matched External Benchmark**: versioned manifests, model-specific CDRSeqGen/IgLM adapters, shared contract validation and scoring, and optional independent-judge perplexity
- **Logging & Visualization**: per-run `train.log` (the whole run's record, including the generative-eval table and any uncaught traceback), TensorBoard logging, csv metric extraction, multi-line graph plotter
- **Extensible Architecture**: method packages isolate each method's assumptions so a new one reuses shared data, metrics, artifacts, and benchmark contracts. A method declares itself in `methods/registry.py` and fills a five-function CLI protocol; the dispatchers, `--decoding` choices and dashboard families all derive from that registry rather than from hard-coded lists

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
| [4 Sequence context buys nothing](#4-sequence-context--partner-chain-and-antigen--gives-no-measurable-benefit) | [9 One hyperparameter inflated four findings](#9-one-un-audited-hyperparameter-inflated-four-findings-at-once) | |
| [5 The homology leak (immaterial for H3)](#5-the-homology-leak-was-real-but-immaterial-for-h3) | [10 Length-free exposes the length problem](#10-removing-native-h3-length-exposes-a-separate-unsolved-problem) | |

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

```text
CDRSeqGen/
├── configs/                    # 57 run configs (flat; run.sh references them by path).
│                               #   A config's "method" key selects the pipeline (default mlm)
│   ├── h{1,2,3}_scratch[.json|_lc|_ag|_ag_lconly] + h{1,2,3}_esm2[...]
│   ├── h{1,2,3}_igbert[.json|_lc] + h{1,2,3}_protbert[.json|_lc]
│   ├── l{1,2,3}_scratch[.json|_hc] + l{1,2,3}_esm2[.json|_hc]
│   ├── h{1,2,3}_ar_scratch[.json|_lc]
│   └── h3_s2s_scratch[.json|_lc]   # pins num_encoder/decoder_layers 2/2 = 933k params, so the
│                               #   compare suite is not confounded by 2.3x capacity (see s2s below)
├── cdrseqgen/                  # Library package — grouped by concern (entrypoints stay at root)
│   ├── artifacts.py            # Shared names for persisted run artifacts and legacy aliases
│   ├── data/                   # ALL data code (the data itself lives in dataset/data/)
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
│   │   └── preprocess/         # Offline: raw → processed JSON (run once per dataset)
│   │       ├── oas_paired.py       # OAS paired repertoires + the leakage-safe clustered split
│   │       ├── sabdab2.py          # SAbDab2 structures (adds antigen; split ships with the data)
│   │       ├── common.py           # Shared by both: validation, chain assembly, funnel, IO
│   │       ├── download_oas_paired.sh  # Fetch OAS paired repertoires → dataset/data/raw/
│   │       └── download_sabdab2.sh     # Fetch SAbDab2 splits → dataset/data/raw/ (CWD-independent)
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
│   │   ├── ar/                 # Autoregressive (2026-09-01). Data lives in data/dataset_ar.py;
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
│   │   ├── s2s/                # Encoder-decoder (2026-09-07). Data lives in data/dataset_s2s.py.
│   │   │   │                   #   The missing cell of the 2x2: bidirectional conditioning like
│   │   │   │                   #   mlm, <eos>-chosen length like ar. Contract-identical to ar
│   │   │   │                   #   throughout, so an s2s/ar/mlm table is one implementation.
│   │   │   ├── backbones.py    # Seq2seq registry. Only `scratch` is registered: a generic
│   │   │   │                   #   hf_seq2seq fallback would force our <mask> onto ProtT5 and throw
│   │   │   │                   #   away the very thing it is wanted for — span infilling IS its
│   │   │   │                   #   pretraining objective. resolve() fails loudly instead
│   │   │   ├── training.py     # get_datasets + teacher-forced metrics + Trainer. The metrics do
│   │   │   │                   #   NOT re-shift: dataset_s2s already offset the decoder pair
│   │   │   ├── sampling.py     # Decode to <eos>. ONE encoder pass shared by every candidate, and
│   │   │   │                   #   no padding at all — every decoder row starts at <bos>
│   │   │   ├── evaluation.py   # Seq2SeqStrategy for the SHARED loop in eval/generative.py
│   │   │   └── models/
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
│   ├── eval/                   # Architecture-neutral evaluation. Imports from NO method package.
│   │   ├── metrics.py          # Sequence/design metrics + ANCHOR_TRIM / MIN_CORE_COVERAGE
│   │   └── generative.py       # The shared eval loop (2026-09-01): select -> draw -> score ->
│   │                           #   judge -> write. A *strategy* owns draw() and self_scores(),
│   │                           #   the only two steps a method cannot share
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
├── unit_tests/                 # CPU unit/integration tests — no downloads or research data (./scripts/unit_tests.sh).
│   └── test_*.py               #   Named to keep "test" free for the train/validation/test split.
├── showcase/                   # Curated frameworks for generate.py — a hand-designed probe set,
│   └── {h1,h3,h3_lc,l3_hc,h3_ag}.json # from VALIDATION, so repeated inspection is free;
│                               #   stratified by H3 length, one case deliberately without a
│                               #   native, each carrying its provenance and intent
├── misc/
│   └── CDRSeqGen260814.drawio.png  # Project workflow diagram
├── train.py                    # Entrypoint — dispatches to mlm or ar
│                               #   Dispatches on the config's "method" key; each method's own
│                               #   main() lives in methods/<method>/entrypoint.py
├── generate.py                 # Entrypoint — candidate generation on supplied frameworks
├── evaluate.py                 # Entrypoint — held-out generative evaluation (+ --judge)
├── fit_mlm_length.py           # Add MLM length artifacts to an older checkpoint
├── scripts/
│   ├── run.sh                  # Unified runner: ./scripts/run.sh <gpu>[,<gpu>…] [suite ...]
│                               #   Suites name their method first: mlm_heavy | mlm_light |
│                               #   mlm_antigen | mlm_igbert | mlm_protbert | mlm_plm |
│                               #   ar_heavy | compare | all. Several GPUs shard the round
│   ├── sweep.sh                # Unified sweep: h3 | pilot | arch | loss | quick presets
│   └── unit_tests.sh           # Unit test runner (sets PYTHONPATH; no GPU or data needed)
├── env.sh
├── README.md                   # What the project is, what it found, what is still open
└── WORKFLOW.md                 # How to run it — preprocessing through benchmark
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
   ./scripts/unit_tests.sh                      # whole suite, a few seconds
   ./scripts/unit_tests.sh test_length_policy   # one module
   ```
   The `unittest` suite combines synthetic fixtures, stubs for external models, real tiny CPU
   scratch models, and isolated shell-runner checks. It needs no GPU, downloaded weights, or
   research dataset. See [the test guide](unit_tests/README.md) for guarantees and limitations.
   `PYTHONPATH=.` is required (there is no `unit_tests/__init__.py`),
   which is why the runner exists rather than a bare `python -m unittest`.

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

## 📊 Findings

Phase 1 (the sequence sweep) is a **leakage-safe benchmark** for CDR infilling. What it establishes:

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

OAS's original random split leaked clonal near-duplicates (76 of 2,111 test entries, 3.6%, share a CDR3 cluster with train). The clustered-split fix removed it (→0%), and re-running confirmed the H3 core moved by **≤0.004**. (Now the default; see [`preprocess/oas_paired.py`](cdrseqgen/data/preprocess/oas_paired.py).) The important qualifier is *for H3* — the split clusters on heavy CDR3 only, so it controls nothing for the other five loops (§6).

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
The recurring rule: **a new capability is added in one place, and the rest of the pipeline stays oblivious.** Five seams:

- **New generation method** → a package beside [`methods/mlm/`](cdrseqgen/methods/mlm/), subclassing `AntibodyFrameworkDataset` from [`data/dataset_common.py`](cdrseqgen/data/dataset_common.py), plus an `entrypoint.py` that [`train.py`](train.py) dispatches to on the config's `"method"` key. **Exercised 2026-09-01** by [`methods/ar/`](cdrseqgen/methods/ar/), which is a third the size of the masked package — the difference is the length-policy layer, the gap-sentinel head, the second eval track and the loss registry, none of which a causal model can need.

  **Exercised again 2026-09-07** by [`methods/s2s/`](cdrseqgen/methods/s2s/), which is the real test: the second method could still have been accommodated by generalising toward it, while the third had to fit a boundary drawn without it in mind. It did — a registry entry, the five-function CLI protocol, and a strategy for the shared evaluation loop — and the parts that did *not* fit were informative. `SpanLayout` does not generalise, because it builds one concatenated stream and an encoder-decoder needs two; that is a new data view, not a parameterisation of the old one. And three *enumerations* had to be found by running the method rather than by the type system: the dashboard family map, the `--decoding` choice list duplicated into two argparse calls, and `sweep.sh`'s architecture flag. Each was a list that had to be edited in a second place, which is the shape of seam this section exists to avoid; `--decoding` and the method classification now derive from `methods/registry.py`, and a test asserts every registered method has a dashboard family.

  Adding the second method is what made the shared layers honest, and each extraction was checked against two implementations rather than guessed from one: `select_split_indices` (the eval-split guard — a second copy could recontaminate one method's results silently), `eval/generative.py` (the evaluation loop; a *strategy* owns only `draw()` and `self_scores()`), `build_judge` (§11 rests on one judge scoring both models), and `methods/common/` (arguments, sampling, callbacks). The last one caught a live defect: the causal entrypoint had quietly dropped `LogMetricsCallback`, reintroducing the empty-`train.log` regression fixed on 2026-07-29. *Added 2026-08-31*: the framework/target split, context injection and length capping are now method-neutral, and `TokenizerAdapter.mask_id` is optional — requiring a mask token was the one thing that made a causal tokenizer impossible to construct. `dataset_mlm.py` is now just the masked-LM view of that base, and re-exports the moved names so existing imports are unchanged. The benchmark contract was already method-agnostic, as [`adapters/iglm.py`](benchmark/adapters/iglm.py) demonstrates.
- **New masked-LM backbone** → one `Backbone` entry in [`backbones.py`](cdrseqgen/methods/mlm/backbones.py) (§2) — the entrypoints and `dataset.py` never learn about it. The causal and seq2seq registries have the same shape, so the seam is one rule across all three methods.
- **New target loop** → change the `target_region` string (§1); masking logic is untouched.
- **New context** → one boolean flag (`include_partner_chain`, `include_antigen`), resolved symmetrically (§3–4).
- **New dataset source** → one `(substring, tag)` pair in the `DATASET_TAGS` registry ([`train/logging.py`](cdrseqgen/train/logging.py)) (`oas → OAS`, `sabdab2 → SD2`), so every run's output-dir name states its source explicitly instead of leaving it silently implied; an unrecognised path is tagged `DS` **with a logged warning** rather than left blank.

The guiding constraint: **conventions are declared explicitly, never inferred.** The predecessor overloaded a single `is_esm` boolean to mean three different things and silently mishandled non-ESM models; each backbone now states its `position_type` and `uses_special_tokens` up front. This isn't free — protein-BERT models (IgBert/ProtBert) still needed their own entry for residue-spacing and tokenizer/loader quirks (§2) — but the cost is paid once, in the registry, not scattered across the pipeline.

---

## ⚠️ Current Issues
*(current as of 2026-08-31; update per commit)*

### Open problems
- **The `OAS4` corpus confounds donor with disease, by construction.** Jaffe 2022 contains exactly four donors and no more (all 580 catalogued paired runs probed 2026-09-03), and their conditions do not overlap: Donor-1 and Donor-2 are SARS-CoV-2, Donor-3 healthy, Donor-4 CMV. A donor-held-out split is therefore also a disease-held-out split, and neither effect can be attributed separately. Mitigated but not removed by every run being **Naive-B-Cells**: naive repertoires are largely pre-antigen-selection, so disease shapes them far less than it shapes memory or plasmablast compartments. Recorded before any `OAS4` experiment was run.
- **More donors would cost B-cell-type purity.** 96 further donors exist across the other 18 paired studies, but only **one** naive paired run among them (Woodruff_2020); the rest are unsorted, memory, or plasmablast. Somatic hypermutation differs sharply between those compartments, and it lands hardest on the germline-encoded H1/H2 — the loops §6 already shows to be retrieval-bound. Any expansion past four donors must record `BType` per entry (`read_oas_metadata` does not capture it today) and stratify on it, or the heterogeneity becomes invisible in the merged corpus.
- **`OAS4` results are not comparable to any earlier OAS number.** A different donor set is a different test split. The `oas_multi -> OAS4` tag keeps the two apart in run names and batch tables mechanically, but no table should place them on one axis — the same rule the `_OAS` / `_SD2` tags exist to enforce.
- **The split only controls the loop it clusters on.** `add_split_field` clusters heavy CDR3, so the holdout is leakage-safe for H3 and **not** for H1/H2/L1/L2/L3 — 98% of OAS test H1s appear verbatim in train (Findings §6). Every non-H3 number in this README is therefore retrieval-bounded and should not be read as generalisation. Fix: cluster on the *target* CDR (per-loop split), a contained change to `add_split_field`; then re-run the OAS matrix.
- **H3 length is close to unpredictable from the framework, and every learned method is below the achievable ceiling.** A non-neural lookup — group training examples by (last 12 residues of FR3 | first 6 of FR4), a crude V/J-gene proxy, and take the modal H3 length — reaches **11.07% exact / 3.01 MAE** on the OAS4 validation split, covering 98.3% of it. Against that: the §10 length head 9.24% / 3.64, causal `<eos>` 7.23% / 4.00, and always predicting the global mode 10.12% / 3.53. So there IS signal (MAE 3.01 against a constant's 3.53, a 15% reduction) but very little, and **neither learned method captures even that much**. Three interventions moved nothing — decoding temperature, a 10x learning-rate range, and 7x more data — which is what a near-flat ceiling predicts. Treat length accuracy as a property to report against this baseline, not a target to optimise; the V/J lookup should replace the global mode as the reference number.
- **The first learned H3 length head is below its naive baseline.** It reaches 9.24% exact accuracy / 3.64 MAE versus 10.52% / 3.28 for always predicting length 15 (Finding §10). Before treating inferred MLM generation as a competitive method, evaluate probability calibration and top-k length coverage, then test a representation or objective that actually learns the framework-to-length signal.
- **Training still optimizes recovery, not design.** The masked-LM objective rewards reproducing the native loop; the generative metrics only *measure* design quality after the fact — nothing trains for it. Findings §7 shows the two can disagree outright (H1: 0.99 recovery, 0.002 diversity). A design-aligned objective stays open (Phase 2). Re-weighting tokens *within* the masked-LM objective has now been tested and does not help (§13), so this needs a different objective class rather than a better weighting.
- **Run-to-run nondeterminism exceeds the seed spread, and only seeds are ever reported.** Two bit-identical objectives at identical seeds — `cross_entropy` and `weighted 1.0/1.0` — ended 0.0014 apart in teacher-forced core, all three seeds in the same direction, against a within-objective seed sd of 0.0003–0.0007 (§13). The two differ only in floating-point reduction order. So a seed sd *understates* reproducibility by 2–4×, which bears directly on the ±0.006 tie threshold in `sweep.sh` and the σ ≈ 0.0028 noise floor in §4. Every "error bar" in this README is a seed spread and should be read as a lower bound. Cheap fix: keep one duplicated setting per sweep as a deliberate determinism control.
- **Checkpoint selection is circular for a loss sweep.** `sweep.sh` argues — correctly — that *ranking* a loss sweep on core recovery lets a core-weighted objective grade its own homework, and defaults `RANK_METRIC` to `judge_ppl`. It then passes `--metric_for_best_model core_accuracy` to every trial, so each checkpoint is still *selected* on the metric the objective optimises. It is also blind to the failure §13 found: the core-only run has anchor recovery 0.042 and would be selected happily, because selection never looks at anchors even though `eval_anchor_accuracy` is recorded every epoch.
- **Nothing in this README is an `s2s` *result*.** The encoder-decoder method is implemented and verified to run end to end — GPU training, real judge, `generate.py`, `evaluate.py` — but **no experiment has been run on it**. Every claim about it here is about mechanism or cost, never performance. The one number quoted from a real corpus (a 1-epoch smoke run) exists to show `<eos>` terminates, not to say anything about quality. Do not let the method's presence in the `compare` suite imply it has been measured.
- **`s2s` hyperparameters are inherited from the causal sweep, not fitted to it.** The shipped configs carry `1e-4` / 0.06 warmup / 0.01 decay, which the 2026-09-03 screen selected for a single-stack causal decoder. An encoder-decoder has two stacks and cross-attention and may well want something else. Until the LR screen runs, an `s2s` loss in any comparison is uninterpretable — it would confound the architecture with an untuned optimiser, which is the §9 failure exactly.
- **The `compare` suite's three arms are matched on protocol but not on capacity.** Corpus, split, evaluation frameworks, candidate count, `top_p`, `max_len`, judge and seed all match; parameter counts do not. `ar` is 800k, and `s2s` is 933k (1.17x) even after pinning its stacks to 2/2 — the closest match that keeps real depth on both sides, since the nearest-by-count alternative needed a single-layer encoder and would have gutted the one thing the method exists to test. The counts are logged in every `train.log`; **quote them in any table**, and do not read a 1.17x-larger model's win as an architecture result until the `arch` sweep separates the two.
- **The causal method's plausibility advantage is judge-specific, and a native-referenced metric contradicts it.** On OAS4 H3 the causal model reaches judge perplexity 3.15 against masked 9.35 at equal recovery — but BLOSUM62 global alignment over *every* attempt puts it behind: 0.146 against 0.199. The two disagree because both are confounded by length in opposite directions: the aligned score charges gap penalties for the 93% of causal draws whose length is wrong, while the masked arm is handed the native length and never pays one; judge perplexity is length-blind and native-free, so it is indifferent to exactly what causal is bad at. Neither is a clean method comparison until both arms predict their own length (the `compare` suite). Also worth recording: **core BLOSUM is ~0 for every H3 model** (causal 0.020, masked -0.001, ESM-2 0.003) — core substitutions are on average no better than the matrix expects between unrelated residues, which exact-match core recovery of ~0.20 made look like partial credit.
- **The judge shares the eval distribution.** IgBert was trained on OAS paired repertoires — the same distribution as the OAS eval split it judges. It's applied identically to every model, so the *comparison* is fair, but absolute perplexities are OAS-flavoured; an off-distribution judge would be a stronger check.
- **Generative sweep is H3/OAS only.** The matrix covered CDR-H3 on OAS (150 held-out frameworks × 4 models). L-chain loops, SAbDab2, and the antigen configs have not been swept generatively.
- **Sampler is marginal-based, not learned-joint — now measured.** The `iterative` (Mask-Predict) decoder approximates the loop joint by conditioning on committed residues, but the model is still a masked-LM trained on independent-position recovery. As of 2026-07-28 this is no longer a suspicion: iterative and `independent` are **within noise on every metric** on H3/OAS/ESM-2 (recovery 0.4057 vs 0.4072; judge-ppl 7.14 vs 7.23) at ~15× the compute. The conditioning buys nothing because the learned marginals are already near-independent — true joint/generative *training* (autoregressive or diffusion) is Phase-2 territory.
- **Five runs hit their epoch ceiling, all of them context runs.** `h1_SD2_LC`, `h2_OAS_LC`, `h3_OAS_LC`, `l1_OAS_HC`, `l3_OAS_HC` peaked at epoch 19–20 of 20 in the 07-29 batch. Their numbers are lower bounds, so the §4 partner-chain penalty (−0.0102) is an upper bound on the true effect. A scratch-only sweep at 30–40 epochs would settle it (~1 hour; scratch is 60 of the batch's 226 minutes).
- **Only one comparison has an error bar.** Three seeds of `h3_scratch` ± partner chain put the seed sd of a per-loop Δ at **0.0028** (§4), which already retires two of the six per-loop context numbers as noise. Every *other* number in this README is still single-seed, and any claimed difference below ~0.006 should be treated as unmeasured until it has the same treatment.
- **Every batch through 2026-08-17 selected checkpoints on the test split.** Not just the 07-29 runs: `eval_split` defaulted to empty, which fell back to "every non-train entry", and `processed_oas_data.json` has only `train`/`test` — so early stopping, `load_best_model_at_end`, the reported `eval_*` metrics, and `generation_eval.json` all read the final holdout. The 08-17 post-refactor batch (`results/runs/20260817_200051`) is affected too; its logs read `2111 eval (split: all non-train)`. **Fixed 2026-08-18**: `get_datasets` now refuses an unset `eval_split` whenever a `test` split exists, and the 36 OAS configs declare `processed_oas_tuning.json` + `eval_split: validation`. **Resolved for OAS 2026-08-31**: the matrix was re-run as `results/runs/20260818_192606` and every Findings table now quotes it, so the OAS numbers are validation-selected and the test partition is untouched. Two things still stand: the **SAbDab2** rows come from configs declaring `eval_split: test` and remain selection-on-holdout (see the SAbDab2 entry below), and any figure elsewhere in this README still sourced from a pre-08-18 batch — the Step 4 decoding table, the §12 confidence numbers, and the §3 position breakdown — has not been re-derived and stays exploratory.
- **Neither fixed-length policy is unbiased — report both.** *(Re-run 2026-08-21; see Findings §11.)* `constrain` corrupts IgLM's loop endings; `filter` is fair per-example but drops the frameworks IgLM cannot length-match (5 of 25, skewing long), so it scores an easier subset. Current practice: report `filter` on a matched population and state the dropout. A policy that is unbiased in both respects — e.g. raising the attempt budget until every framework is covered, or reporting per-length-bucket — is still open. Historical note on `constrain`: IgLM's `constrain` policy sets `min_length = max_length = native_length`, which suppresses its end-of-span logit until that length is reached — so it writes past its loop into FR4. Only 2% of constrained loops carry the J-gene motif against 20% naturally (Findings §11), and every fixed-length CDRSeqGen-versus-IgLM number is inflated in CDRSeqGen's favour. Re-run with the adapter's `filter` mode (retain only naturally length-matched draws; ~12 attempts per accepted candidate at IgLM's 8.4% match rate) before making any fixed-length claim. The length-free track is unaffected — IgLM uses natural EOS there.
- **SAbDab2 has no validation partition.** Its split comes from the source `ab_ag_split` column, which is `train`/`test` only, so its 12 configs now declare `eval_split: test` — honest, but still selection on the holdout. A three-way SAbDab2 split (or a fixed validation carve-out of its train partition) is needed before any SAbDab2 comparison is reportable.
- **Hyperparameter selection is not yet frozen.** The validation-safe 32-trial scratch-H3 screen completed on 2026-08-11 and placed `1e-3` / 0.06 warm-up / 0.01 weight decay first at 0.32305 mean validation core recovery. Its margin over the runner-up was only 0.00036 across two seeds and changed sign by seed, so additional shared confirmation seeds are still required before touching the final test split.
- **Checkpoint-selection policy is still config-declared.** The current configs select core recovery except the four OAS-L2 configs, which select whole-span because those runs have only 1.4% core coverage. Artifact reporting no longer hard-codes that loop name: it derives the metric from recorded coverage. Training should eventually validate or derive `metric_for_best_model` from the dataset before a run, so a future dataset cannot inherit an obsolete L2 exception.
- ~~**Scratch architecture isn't tunable.**~~ **Fixed 2026-07-27.** `num_attention_heads` and `intermediate_size` are now `ModelArguments` fields forwarded by `build_model` ([`transformer.py`](cdrseqgen/methods/mlm/models/transformer.py)). `intermediate_size` defaults to `None` = **4 × hidden_size**, so the FFN ratio tracks width instead of silently drifting (at the old fixed 512, widening to 256 quietly halved the ratio and would have confounded any architecture sweep). Defaults reproduce the old geometry exactly (128 → 512), and a `hidden_size % num_attention_heads` check now fails loudly instead of deep inside `nn.MultiheadAttention`.
- **Single-seed runs.** Small differences (e.g. light chain hurts scratch ~0.01, neutral for ESM-2) need multiple seeds + error bars before they can be claimed.
- **CPU tests do not validate external PLMs or experimental conclusions.** The suite exercises real scratch-model attention, gradients, optimizer updates, and checkpoint reloads alongside synthetic-data and scoring tests. External pretrained models and the real judge still need a separate integration run. Passing tests does not establish cross-split sequence separation or biological quality; see [the test guide](unit_tests/README.md).
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
- **The decoder must match the checkpoint's masking policy.** `iterative` decoding re-masks and resamples positions one at a time, which assumes the model saw partially revealed loops — true only under `masking_policy: mixed`. On the same `full_span` checkpoint and protocol, iterative decoding costs whole-loop 0.370 -> 0.253 and core **0.204 -> 0.077**, dropping it below even the 0.084 marginal-sampling baseline. `evaluate.py` therefore resolves the default from `masking_policy` (missing legacy metadata means `full_span` → `independent`) and warns when iterative decoding is requested explicitly off distribution.
- **Read best-epoch, not last-epoch** — ESM-2 overfits on SAbDab2/H3; `load_best_model_at_end` saves the peak, but the last CSV row can sit well below it.
- **Never plot OAS and SAbDab2 on shared axes** — different test sets; the `_OAS`/`_SD2` run tags enforce it, and per-dataset charts live under `results/figures/training/<batch>/*.png`.
- **`epoch_metrics.csv` and `*_results.json` measure different models — they're *supposed* to disagree.** The CSV is the per-epoch **trajectory** (the model as it stood at each epoch); `eval_results.json`/`all_results.json` come from the post-training `trainer.evaluate()`, which with `load_best_model_at_end` runs on the **best** checkpoint while still being stamped with the final epoch number. On `h3_esm2` the CSV's epoch-10 row reads core **0.3063** (the actual last model) while `eval_results.json` reads **0.3209** (the epoch-4 best model). `save_training_csv` de-duplicates on epoch and **keeps the first** occurrence precisely so the curve keeps the real last-epoch point — keeping the last would silently redraw the best model at x=10 and erase the overfitting decay entirely.

<!-- MANUAL PROTECTION ZONE BEGINS -->
### Changelog
<!-- AI ACCESS ZONE BEGINS -->
- **2026-09-07** -- **ENCODER-DECODER UPDATE** (`methods/s2s/`, `data/dataset_s2s.py`). The third method, and the missing cell of the 2x2: `mlm` conditions bidirectionally but fills a fixed-length span in one pass, `ar` factorises the loop properly but encodes the framework causally, `s2s` does both. **Two streams, not one sequence, is the whole structural difference from `ar`** — and the second consequence is the useful one. The causal prompt must hold back `LOOP_LENGTH_RESERVE` (96) positions of `max_len` for a loop it has not generated, and that reserve competes with optional context for the same budget; raising it 40 -> 96 is what silently cut the `_lc` light chain from ~96% retained to ~45%. The encoder spends **one** token on the gap marker, so context gets the whole cap: at `max_len` 128 that is 128 context tokens against `ar`'s 31 (asserted in `test_dataset_s2s`). Note what this is NOT: `ar` conditions on FR4 perfectly well, since both flanks are in its prompt — what it cannot do is encode FR1 *with FR4 in view*. Run `scripts/run.sh <gpu> s2s_heavy`, or `compare` for all three arms in one batch folder
- **2026-09-07** -- **Capacity is not matched by default, deliberately.** At `num_layers: 4` an `s2s` model is 1.86M parameters against `ar`'s 800k — 4+4 blocks plus cross-attention. Silently halving the depth to fake a match would itself be the un-audited hyperparameter §9 is about, so instead: `build_model` logs the parameter count into every `train.log`, `num_encoder_layers`/`num_decoder_layers` are separate knobs, and the shipped compare configs pin 2/2 for 933k (1.17x) — the closest match that keeps real depth in both stacks. **Any table putting `s2s` beside `ar` must quote both counts**
- **2026-09-07** -- **Three enumeration bugs a third method exposed**, all found by running it rather than by the CPU suite. `batch_plots` classified any family not starting with `ar_` as "masked", which mislabelled `s2s` *and* inverted the comparability caveat — `ar` and `s2s` both predict position i from the true prefix, so the masked arm is the odd one out. `--decoding` was hard-coded in both `generate.py` and `evaluate.py`, so `--decoding seq2seq` was rejected by argparse before the s2s CLI ever saw it. And `sweep.sh`'s `arch` preset drives `--num_layers`, which an s2s config overrides with pinned depths — every arch trial would have run at identical architecture and reported the noise as a capacity result. Methods now declare their decoders in `methods/registry.py`, and the dashboards derive method from it via `batch_tables.method_of_family`
- **2026-09-07** -- Finished the method dispatch started by the 09-07 refactor: `resolve_decoding`, `resolve_case_length` and the masked `generate_case` moved from `generate.py`/`evaluate.py` into `methods/mlm/cli.py`, which had been importing *back* out of the entrypoints to reach them. The dependency now runs one way only — entrypoint -> registry -> method — and `methods/{mlm,ar,s2s}/cli.py` expose an identical five-function protocol
<!-- AI ACCESS ZONE ENDS -->
- **2026-09-04** -- Enabled different span layouts: `SpanLayout` in [`data/dataset_ar.py`]
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
- [x] **Re-run the fixed-length track in `filter` mode.** Done 2026-08-21: IgLM leads on both tracks once the constraint is removed (§11). Artifact: `results/benchmark/oas_h3_validation_smoke_v1/scores/filter_matched.json`
- [ ] Cover the 5 frameworks `filter` drops — raise the attempt budget or report per-length-bucket, so the fixed-length population isn't silently the easier subset
- [x] Define the fixed-length IgLM policy explicitly. Default `constrain` forces IgLM's end-of-span at the supplied native length through its underlying generation API; audit mode `filter` preserves every variable-length draw and reports match/acceptance counts under a finite attempt budget.
- [x] Validate the shared evaluator on both 25-framework validation outputs: whole-span/core/anchor AAR, diversity, native-distance novelty, unique fraction, per-position entropy, length buckets, validity/length rates, generation runtime per candidate, and a revision-pinned IgBert judge. Final test scoring remains gated on freezing the sampling policy.
- [x] Add a checkpoint-specific CDRSeqGen length head, bounded/inferred generation, and migration tooling for old sequence checkpoints without changing the backbone architecture.
- [x] Add a separate `length_free` manifest/adapter/scorer contract: retain all valid lengths, condition positional AAR on length matches, and score every candidate with length error, edit similarity, diversity, and the shared judge.
- [x] Fit the tuned scratch checkpoint's first length head and run its 25-framework × 20-candidate length-free validation output; preserve the below-mode-baseline result rather than hiding it.
- [x] Run the matched IgLM natural-EOS validation output, score both adapters together, and render the length-free comparison dashboard. **Result: IgLM leads on recovery, length error, edit similarity, and judge perplexity (Findings §11).** Rescore on a clean tree before quoting — the stored scores carry `repository_dirty: true` at commit `0e7e499`.
- [x] Add paired uncertainty ([`bootstrap.py`](benchmark/bootstrap.py)): the example is the resampling unit and one draw is applied to both arms. **Result: only 4 of 8 metrics resolve at 95%** — IgLM's judge-perplexity, length-MAE and edit-similarity leads hold, CDRSeqGen's diversity edge holds, and every recovery comparison is unresolved at n = 25 (§11). Regenerate the intervals after the clean-tree rescore rather than carrying these over.
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
- [x] Make decoding policy-aware and propagate `anchor_trim` into generative evaluation (2026-08-31)
- [ ] Fix circular `metric_for_best_model` checkpoint selection on loss sweeps
- [ ] Re-run the 07-24 batch under core-based checkpoint selection (those 36 runs predate the fix; their saved weights were picked on whole-span)
- [ ] Lock the Phase-2 objective + primary metric with advisor

**Autoregressive method — next steps**
- [x] **Retrain all six `ar` configs** under the fixed `LOOP_LENGTH_RESERVE`. Done 2026-09-02, batch `20260902_164544_ar`; `valid_rate` ~1.0 confirms the `<eos>` mechanism is healthy post-fix
- [x] **Answer the length question on H3, not H1.** H1/H2 are perfect (1.000 length, 1.000 core, 0.000 diversity) and meaningless — §6's retrieval regime. On H3: **8.3% exact length, 3.80 MAE**, against §10's length head at 9.24%/3.64 and a modal-length baseline at 10.52%/3.28. `<eos>` does **not** beat the length head, and both sit below the constant baseline on the natural length distribution
- [x] **Exclude decoding as the cause** (2026-09-03). A temperature sweep on the trained checkpoint cuts MAE only 4.31 -> 3.48 from T=1.0 to greedy, and exact-match never moves off ~8%. The model's mean length is well calibrated (16.11 vs 16.17 native) but under-dispersed, correlating 0.396 with the target — it hedges toward the mean rather than reading the framework
- [x] **Exclude learning rate as the cause** (2026-09-03, `20260903_155940_sweep_h3`). 3e-4/1e-3/3e-3 x 2 seeds: every metric unresolved, rankings mutually contradictory, and the shipped `1e-4` matches or beats all three. Teacher-forced core moves 0.0005 across a 10x range against a 0.006 noise floor
- [x] **Tested whether the length ceiling is data-limited** — it is not (2026-09-04, `20260903_192305`). At 7.2x the data the causal model held at exactly 1.13x its corpus's own modal-length MAE, the same ratio as on 21k. A V/J-conditioned lookup then put the achievable ceiling at 11.07% / 3.01, barely above a constant. **Length is now a reported property, not an optimisation target.**
- [ ] **Sweep capacity** (`arch` preset) on the expanded corpus rather than the current one: at 0.8M parameters capacity and data scale plausibly interact, so an answer at 21k would not transfer. Now also the axis that would settle whether `s2s`'s 1.17x parameter edge matters; `sweep.sh` drives both encoder and decoder depth on an s2s base config as of 2026-09-07
- [x] A causal counterpart to [`generate.py`](generate.py). Done 2026-09-07 (`ar/cli.py` + `ar/case_generation.py`); `generate.py` now dispatches through the registry and drives a causal checkpoint. **The remaining blocker is the fixture, not the code** (diagnosed 2026-09-07): every `showcase/*.json` carries one "design only" case with no `target_truth` and an explicit `target_length`, because a masked LM must be told a length when there is no native to derive one from. Both length-free methods reject that key and abort the whole file. Verified that dropping it runs all three H3 cases correctly. **Decision needed:** add length-free probe sets, or relax the two CLIs to warn-and-ignore a length request they cannot honour
- [ ] **Harden `SpanLayout` before registering a second layout** (found in review, 2026-09-04; none of it affects the shipped `CDRSEQGEN_CAUSAL` or any current result): `build_prompt` receives the whole framework dict **including `target`**, so a layout could leak the answer into its own prompt and the training-equals-generation test would still pass; `n_layout_tokens` is unverified metadata that drives truncation, with no assertion that the assembled example matches it; `CausalInfillingDataset` does not validate its layout, so an invalid one trains a whole run before generation rejects it; and `validate_layout` accepts a residue token as the terminator, which would stop decoding at that amino acid
- [ ] Record `span_layout_id` in `run_config.json` and compare it in the checkpoint-compatibility check — otherwise changing a backbone's layout silently reinterprets older checkpoints
- [ ] A pretrained causal backbone through `methods/ar/backbones.py` (the registry exists; only `scratch` is registered). **Partial prerequisite 2026-09-04**: `SpanLayout` lets a backbone declare its own prompt/answer arrangement. It is NOT yet sufficient for a native format that needs generated-output conversion or wrapper tokens — `validate_layout` deliberately refuses those, and IgLM's own format needs `validate_generated_seq` + `iglm_to_infilled` on top. A layout-aware extraction step is still missing, along with per-config layout selection, conditioning fields on the framework dict, and a wired layout on the `hf_decoder` path. Note the format and the checkpoint are separate questions — adopting IgLM's *layout* is clean, fine-tuning IgLM's *weights* is not, since it was pretrained on OAS and is the §11 baseline, making causal scratch-vs-pretrained the mirror of the masked scratch-vs-ESM-2 axis
- [ ] **Automate one GPU + real-judge smoke run.** Three defects reached live runs past a fully green 293-test suite — the device mismatch, the prompt divergence and the missing `pseudo_log_likelihood` import — because the suite did not exercise the full GPU training-to-generation path with a real judge. CPU scratch-model checks now cover part of that path; an opt-in end-to-end GPU test remains needed. **The 2026-09-07 `s2s` landing ran this by hand** (1 epoch on `processed_oas_tuning`, GPU, real IgBert judge, then `generate.py` and `evaluate.py` against the checkpoint) and it is exactly what caught the three enumeration bugs — none of which the 389-test CPU suite could see. That is the argument for automating it, not a substitute

**Encoder-decoder method — next steps**
- [x] **Implement `s2s` end to end** (2026-09-07): two-stream data view, bidirectional encoder + cross-attending decoder, backbone registry, training, decode-to-`<eos>` sampling, a strategy for the shared evaluation loop, CLI, entrypoint, configs, an `s2s_heavy` suite, and `compare` extended to three arms. 44 new tests
- [x] **Verify the full path outside the test suite** (2026-09-07): GPU training with the real IgBert judge (`valid_generation_rate` 0.975, `judge-cov` 1.000), plus `generate.py` and `evaluate.py` against the resulting checkpoint. After one epoch the model already emits canonical H3s — `AQRGVGAFDP`, `ARGSSGGIPFFDY` — with `AR`/`AQ` starts and `FDY`/`MDV` J-gene ends
- [ ] **Screen the learning rate before any comparison.** `s2s` currently inherits `1e-4` from the *causal* sweep (2026-09-03), which was fitted to a single-stack decoder, not to two stacks plus cross-attention. An untuned arm losing a comparison establishes nothing. `LRS="1e-4 3e-4 1e-3" WARMUPS="0.06" WEIGHT_DECAYS="0.01" SEEDS="42 43" EPOCH_LIST="20" scripts/sweep.sh <gpu> h3 configs/h3_s2s_scratch.json` — 6 trials at the shipped warmup/decay, two seeds so a rank flip is visibly noise
- [ ] **Run the three-arm `compare` suite**, all six runs in one batch folder. Do NOT splice `s2s` against the 09-04 batch instead: framework selection is deterministic in `gen_seed` so per-framework pairing is *mechanically* valid, but the tree has changed under those numbers, `batch_tables.md` and the dashboards are per-batch-folder artifacts, and the 09-04 table predates the reporting fix. Hand-splicing a comparison table is finding #2 of the last review
- [ ] Multi-seed it, like every other headline number in this README still needs
- [ ] **A pretrained encoder-decoder backbone** — the durable argument for this method family. ProtT5 is the only pretrained option across all three methods whose *native* objective is span infilling, so unlike `hf_decoder` it needs no randomly-initialised sentinels bolted on, and unlike IgLM its weights are not the §11 baseline. `methods/s2s/backbones.py` deliberately registers no generic fallback and lists what the real path needs: an encoder-stream layout using `<extra_id_0>`, an extraction step reading the decoder stream back out, and per-config layout selection
- [ ] Expect a negative result on the architecture axis, and record it as one. §4 measured no benefit from partner-chain or antigen context, and `mlm`/`ar` tie on recovery, so the prior against "better framework conditioning helps recovery" is strong. The informative outcome is judge perplexity: if the 3.15-vs-7.55 gap is really about loop factorisation, `s2s` should hold near 3.15
<!-- AI ACCESS ZONE ENDS -->

- [ ] Project flowchart (0814, outdated)

- **Operation Reclamation** -- Manual code cleaning: Comprehend and rewrite the project content in my own terms - previous work handled by AI resulted in the project almost becoming a "black box" for the developer himself 
  - [ ] `data/` scripts 
    - [ ] `dataset.py`, `frameworks.py` (0818)
    - [ ] `preprocess/` scripts
  - [ ] `cdrseqgen/methods/common/`
  - [ ] `cdrseqgen/methods/mlm/` 
    - [ ] arguments, backbones
    - [ ] training
    - [ ] sampling
    - [ ] generation, case_generation
    - [ ] evaluation, recovery
    - [ ] length, length_predictor, variable_length_evaluation
  - [ ] `cdrseqgen/methods/ar/`
  - [ ] `eval/metrics.py`  and `train/` artifact/logging scripts
  - [ ] `train.py` + `train/` (logging)
  - [ ] visualization scripts
  - [ ] `evaluate.py`
  - [ ] `generate.py`
  - [ ] `fit_mlm_length.py`
  - [ ] bash scripts (`run.sh`, `sweep.sh`, `unit_tests.sh`)
  - [ ] `benchmark/` scripts (`adapters/`, `export_manifest.py`, `scorer.py`)
  - [ ] `unit_tests/` scripts
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
