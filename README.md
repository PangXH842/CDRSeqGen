<!-- Note to AI: Edits allowed OUTSIDE of MANUAL PROTECTION ZONES and INSIDE of AI ACCESS ZONES, and DO NOT COMMIT ON MY BEHALF -->
# CDRSeqGen: Modular Antibody Infilling Framework

CDRSeqGen connects antibody data processing, model training, checkpoint loading,
CDR generation and benchmarking through a shared framework. It currently implements
masked language modeling (MLM), autoregressive generation (AR) and encoder-decoder
generation (S2S), with method-specific objectives and decoders.

**Updated:** 26 October 2026. **Status:** the iAIM abstract was submitted on 23 October.
Current work is manual code review, manuscript integration and poster preparation;
final held-out evaluation remains pending.

## Project Overview

| Approach | Generation | Length selection |
|---|---|---|
| MLM | Fill masked loop positions independently or iteratively | Native/fixed length or a fitted length head |
| AR | Generate the loop causally from a prompt containing both flanks | EOS termination |
| S2S | Encode framework context and decode the loop with cross-attention | EOS termination |

The common data schema supports heavy- and light-chain CDR targets, optional partner
and antigen context, and training-time context dropout. Method and backbone registries
connect these inputs to shared configuration, run artifacts and scoring.
Benchmark adapters support local methods, external generators and training-only references.

## Documentation

Supporting guides live in [docs/](docs/README.md).

| Read | Purpose |
|---|---|
| [Environment setup](docs/ENVIRONMENT.md) | Dependencies, server restoration, data/cache transfer and verification |
| [Workflow manual](docs/WORKFLOW.md) | Preprocessing, training, generation, evaluation and historical run recipes |
| [Experiment guide](configs/experiments/README.md) | Named batches, planning, execution, status and resume |
| [Architecture reference](docs/ARCHITECTURE.md) | Full repository map, data layout and extension points |
| [Method workflow map](cdrseqgen/methods/common/README.md) | Call graph, stage ownership and the manual review sequence |
| [Results index](docs/RESULTS.md) | Completed evidence and links to source artifacts |
| [Detailed findings](docs/FINDINGS.md) | Measurements, protocols, historical findings and limitations |
| [Project status](docs/PROJECT_STATUS.md) | Research handoff, manuscript tasks and final-evaluation decisions |
| [Roadmap](docs/ROADMAP.md) | Research milestones, experiment options and feature plans |
| [Operation Reclamation](docs/RECLAMATION.md) | Manual code-review checklist and review marks |
| [Benchmark guide](benchmark/README.md) | Adapter contracts, matched evaluation and scoring |
| [Test guide](unit_tests/README.md) | Offline CPU checks and their coverage |
| [Changelog](docs/CHANGELOG.md) | Detailed AI-assisted implementation and audit history |

<a id="current-handoff--2026-10-23"></a>

## Current handoff

1. **Continue MLM generation review:** `gen_cli.py` → shared generation lifecycle →
   `generation.py`, `case_generation.py`, `sampling.py` and the consolidated `length.py`.
   The manual checklist marks the MLM model, training block and `eval_cli`/`evaluation`
   reclaimed. AR, S2S and shared code follow later in the review sequence.
2. **Develop the manuscript and poster** from the completed validation evidence.
   The [paper outline](../iAIM/PAPER_OUTLINE.md) maps claims to evidence; the
   [canonical abstract](../iAIM/sections/0_abstract.tex) is shared by both LaTeX entry points.
   Poster presentations are scheduled for 4–5 December; acceptance is not yet recorded here.
3. **Freeze the final held-out protocol** before inference: cohort, systems, checkpoints,
   decoding, length support, primary comparisons and reporting. The audit-eligible
   test pool is not yet a frozen evaluation manifest.

Completed validation batches need no completion rerun. Research experiments remain
user-run. See [project status](docs/PROJECT_STATUS.md) for the detailed handoff and
the [reclamation checklist](docs/RECLAMATION.md) for your review marks.

## ⚙️ Environment Setup

Run from the repository root. For a fresh installation:

```bash
bash env.sh                         # creates cdrprj and iglm; refuses existing names
bash scripts/verify_env.sh           # dependency, GPU and plotting checks
conda run --no-capture-output -n cdrprj bash scripts/unit_tests.sh
conda activate cdrprj
```

Use an existing verified environment directly when returning to the project.
[ENVIRONMENT.md](docs/ENVIRONMENT.md) records hardware/version requirements, CPU-only
verification, restoration limits and the data/model caches that are transferred outside Git.

## 🧹 Data Preprocessing Quick Start

| Corpus | Current records | Scope |
|---|---:|---|
| OAS4 | 152,727 paired sequences | Four donors; 122,181 train / 15,273 validation / 15,273 test |
| Earlier OAS pilot | 21,106 paired sequences | Historical one-donor corpus |
| SAbDab2 | 3,984 antibody-antigen pairs | Upstream 3,234 train / 750 test; validation must be requested |

Data lives in `data/raw/` and `data/processed/`. Preprocessors write split/filter settings
and processed-file hashes to metadata sidecars. Existing processed data can be reused.
See [source-specific commands](docs/WORKFLOW.md#preprocessing-quick-reference) and
[the full preprocessing guide](docs/WORKFLOW.md#step-1-preprocessing-data).

## 🚀 Workflow & Execution

Discover and inspect batches before selecting a run:

```bash
./scripts/run.sh list
./scripts/run.sh plan compare --gpus 0
./scripts/run.sh status results/runs/BATCH
```

`list` and `plan` do not launch training. `BATCH` is the saved batch directory to inspect.
The [experiment guide](configs/experiments/README.md) documents `run`, `resume`, recipes
and the distinction between restarting an unfinished trial and continuing an optimizer checkpoint.
The [workflow manual](docs/WORKFLOW.md) covers the individual commands and output artifacts.

| Root command | Method entrypoint | Main responsibility |
|---|---|---|
| `train.py` | `train_cli.main()` | Configure a training run and its optional post-training evaluation |
| `evaluate.py` | `eval_cli.main(args, config, run_config)` | Connect checkpoint evaluation setup and scoring to the shared lifecycle |
| `generate.py` | `gen_cli.main(args, config)` | Generate candidates for supplied framework cases |

For MLM, the current paths are:

```text
train.py    → train_cli.main() → training.run_training()
generate.py → gen_cli.main()   → common.generation.run_generation_pipeline()
                             → generation.generate_case()
evaluate.py → eval_cli.main()  → common.evaluation.run_evaluation_pipeline()
                             → evaluation.run_generative_eval()
```

MLM generation and evaluation share `sampling.sample_masked_span()`. An explicit
decoder request takes priority over a compatible saved setting; the fallback is
**independent**. Recorded iterative decoding is used automatically only with mixed
masking. Length policies and residue decoding are separate choices.

Root commands configure console logging. Training additionally attaches `train.log`,
which includes post-training evaluation logs; standalone evaluation does not attach
a file handler. Shared scoring helpers log reports and output messages, while
decoder-selection `print()` messages bypass logging handlers.

## 🏗️ Repository Structure

```text
docs/                       Project guides, research notes and manual review checklist
configs/                    Single-run settings and named experiment recipes
cdrseqgen/datasets/         Source preprocessing and method-specific dataset views
cdrseqgen/methods/          Registry, common workflows, and mlm/ar/s2s implementations
cdrseqgen/eval/             Shared sequence metrics, scoring loops and reports
cdrseqgen/visualization/    Tables, plots and experiment dashboards
scripts/                   Experiment and maintenance command interfaces
benchmark/                 External/local adapters, frozen protocols and shared scoring
data/                      Raw inputs and processed corpora
results/                   Saved plans, checkpoints, candidates, scores and audits
unit_tests/                Offline CPU regression checks
```

The [full architecture reference](docs/ARCHITECTURE.md) describes individual modules,
context placement, backbone conventions and extension points. The
[method workflow map](cdrseqgen/methods/common/README.md) identifies stage ownership.

## 📊 Findings

The current matched benchmark uses **150 OAS4 validation frameworks × 20 attempts**
per arm, with heavy-chain flanks and no native H3 length or partner-chain context.
The primary endpoint is normalized edit similarity to the native loop.

| System/reference | Edit similarity |
|---|---:|
| MLM / AR / S2S, seven-seed means | 0.3918 / 0.3868 / 0.3882 |
| IgLM / IgCraft | 0.3880 / 0.3769 |
| Primary flank retrieval | 0.3589 |
| Unconditional training-H3 sampling | 0.2817 |

The learned systems exceed primary retrieval in the adjusted comparisons; adjusted
intervals do not resolve differences among the three scratch method families.
Same-seed repeats, larger-capacity runs, context/dropout studies and ten no-partner
controls are also complete. These studies have separate input/judge settings and
overlapping source runs. See the [evidence inventory](docs/RESULTS.md#completed-validation-evidence)
and [expanded analysis](docs/FINDINGS.md#17-expanded-validation-evidence-seeds-retrieval-and-capacity).

## ⚠️ Current Issues

### Open problems

- Current headline comparisons use validation data; final held-out evaluation remains pending.
- Representative H3 clustering leaves residual cross-split similarity. The local audit
  flags 3/150 benchmark frameworks; 14,818/15,273 test records meet its exclusion rule.
  Different-length/gapped similarity and external pretraining exposure remain unresolved.
- Confidence intervals account for their declared framework/seed populations; repeated
  candidate-generation variability is unmeasured. Historical studies retain their own limits.
- Sequence similarity and judge scores do not establish structure or binding quality.
- Detailed manual code review continues. Shared scoring edge cases remain documented in
  [the research notes](docs/FINDINGS.md#open-problems).

### How to read the numbers

| Quantity | Population/meaning |
|---|---|
| Edit similarity and length errors | Retained generated candidates, including different lengths |
| AAR, core/anchor recovery and positional BLOSUM | Candidates matching the native length; inspect coverage |
| Judge pseudo-perplexity | Successfully judged candidates; compare with the same judge/input contract |
| Generative report mean ± SD | Equal-weight contributing frameworks and population SD; not a confidence interval |

Teacher-forced validation recovery and generated-candidate recovery are separate stages.
`novelty_mean = 1 - AAR` measures difference from the native loop. Historical findings,
metric populations and protocol caveats are preserved in [FINDINGS.md](docs/FINDINGS.md).

<details>
<summary>Earlier README section links</summary>

These sections now live in the linked guides; their previous README anchors are retained.

- <a id="contents"></a>[Contents](#documentation)

- <a id="oas-paired-repertoires"></a>[OAS paired repertoires](docs/WORKFLOW.md#oas-paired-repertoires)

- <a id="sabdab2-antibody-antigen-pairs"></a>[SAbDab2 antibody-antigen pairs](docs/WORKFLOW.md#sabdab2-antibody-antigen-pairs)

- <a id="s2s-pretrained-encoders-results-and-next-runs"></a>[S2S pretrained encoders: results and next runs](docs/WORKFLOW.md#s2s-pretrained-encoders-results-and-next-runs)

- <a id="1-model-choice-doesnt-matter-is-only-true-in-two-of-three-regimes"></a>[1. "Model choice doesn't matter" is only true in two of three regimes](docs/FINDINGS.md#1-model-choice-doesnt-matter-is-only-true-in-two-of-three-regimes)

- <a id="2-cdr-h3-is-the-only-hard-loop--but-the-other-five-are-easier-than-they-look"></a>[2. CDR-H3 is the only hard loop — but the other five are easier than they look](docs/FINDINGS.md#2-cdr-h3-is-the-only-hard-loop--but-the-other-five-are-easier-than-they-look)

- <a id="3-the-standard-metric-misleads-anchor-inflation"></a>[3. The standard metric misleads (anchor inflation)](docs/FINDINGS.md#3-the-standard-metric-misleads-anchor-inflation)

- <a id="4-sequence-context--partner-chain-and-antigen--gives-no-measurable-benefit"></a>[4. Sequence context — partner chain and antigen — gives no measurable benefit](docs/FINDINGS.md#4-sequence-context--partner-chain-and-antigen--gives-no-measurable-benefit)

- <a id="5-the-homology-leak-was-real-but-immaterial-for-h3"></a>[5. The homology leak was real but immaterial *for H3*](docs/FINDINGS.md#5-the-homology-leak-was-real-but-immaterial-for-h3)

- <a id="6-the-clustered-split-controls-h3-and-nothing-else--so-the-099s-are-retrieval"></a>[6. The clustered split controls H3 and nothing else — so the ~0.99s are retrieval](docs/FINDINGS.md#6-the-clustered-split-controls-h3-and-nothing-else--so-the-099s-are-retrieval)

- <a id="7-recovery-and-design-quality-disagree--high-aar-can-mean-a-perfect-copier"></a>[7. Recovery and design quality disagree — high AAR can mean a perfect copier](docs/FINDINGS.md#7-recovery-and-design-quality-disagree--high-aar-can-mean-a-perfect-copier)

- <a id="8-a-model-is-locked-to-the-context-regime-it-was-trained-in--and-context_dropout-unlocks-it"></a>[8. A model is locked to the context regime it was trained in — and `context_dropout` unlocks it](docs/FINDINGS.md#8-a-model-is-locked-to-the-context-regime-it-was-trained-in--and-context_dropout-unlocks-it)

- <a id="9-one-un-audited-hyperparameter-inflated-four-findings-at-once"></a>[9. One un-audited hyperparameter inflated four findings at once](docs/FINDINGS.md#9-one-un-audited-hyperparameter-inflated-four-findings-at-once)

- <a id="10-removing-native-h3-length-exposes-a-separate-unsolved-problem"></a>[10. Removing native H3 length exposes a separate unsolved problem](docs/FINDINGS.md#10-removing-native-h3-length-exposes-a-separate-unsolved-problem)

- <a id="11-matched-length-free-comparison-iglm-leads-on-plausibility-and-length-and-the-recovery-gap-does-not-resolve"></a>[11. Matched length-free comparison: IgLM leads on plausibility and length, and the recovery gap does not resolve](docs/FINDINGS.md#11-matched-length-free-comparison-iglm-leads-on-plausibility-and-length-and-the-recovery-gap-does-not-resolve)

- <a id="12-the-models-own-confidence-is-nearly-useless-for-selection--an-independent-judges-is-not"></a>[12. The model's own confidence is nearly useless for selection — an independent judge's is not](docs/FINDINGS.md#12-the-models-own-confidence-is-nearly-useless-for-selection--an-independent-judges-is-not)

- <a id="13-re-weighting-the-loss-across-loop-positions-changes-nothing--the-anchors-are-not-wasted-capacity"></a>[13. Re-weighting the loss across loop positions changes nothing — the anchors are not wasted capacity](docs/FINDINGS.md#13-re-weighting-the-loss-across-loop-positions-changes-nothing--the-anchors-are-not-wasted-capacity)

- <a id="14-context-dropout-improves-missing-partner-robustness-across-mlm-ar-and-s2s"></a>[14. Context dropout improves missing-partner robustness across MLM, AR and S2S](docs/FINDINGS.md#14-context-dropout-improves-missing-partner-robustness-across-mlm-ar-and-s2s)

- <a id="15-esm-2-learning-rate-tuning-improves-teacher-forced-recovery-more-clearly-than-generation"></a>[15. ESM-2 learning-rate tuning improves teacher-forced recovery more clearly than generation](docs/FINDINGS.md#15-esm-2-learning-rate-tuning-improves-teacher-forced-recovery-more-clearly-than-generation)

- <a id="16-matched-oas4-length-free-validation-benchmark"></a>[16. Matched OAS4 length-free validation benchmark](docs/FINDINGS.md#16-matched-oas4-length-free-validation-benchmark)

- <a id="17-expanded-validation-evidence-seeds-retrieval-and-capacity"></a>[17. Expanded validation evidence: seeds, retrieval and capacity](docs/FINDINGS.md#17-expanded-validation-evidence-seeds-retrieval-and-capacity)

- <a id="-architectural-feature-details"></a>[🧠 Architectural Feature Details](docs/ARCHITECTURE.md#-architectural-feature-details)

- <a id="1-task-agnostic-masking-the-lego-principle"></a>[1. Task-Agnostic Masking (The "LEGO" Principle)](docs/ARCHITECTURE.md#1-task-agnostic-masking-the-lego-principle)

- <a id="2-pluggable-backbones-the-extensibility-core"></a>[2. Pluggable Backbones (the extensibility core)](docs/ARCHITECTURE.md#2-pluggable-backbones-the-extensibility-core)

- <a id="3-partner-chain-context-injection"></a>[3. Partner Chain Context Injection](docs/ARCHITECTURE.md#3-partner-chain-context-injection)

- <a id="4-antigen-context-injection-sabdab2"></a>[4. Antigen Context Injection (SAbDab2)](docs/ARCHITECTURE.md#4-antigen-context-injection-sabdab2)

- <a id="5-automated-tracking--evaluation"></a>[5. Automated Tracking & Evaluation](docs/ARCHITECTURE.md#5-automated-tracking--evaluation)

- <a id="6-extensibility-principles"></a>[6. Extensibility Principles](docs/ARCHITECTURE.md#6-extensibility-principles)

- <a id="-roadmap"></a>[⏭️ Roadmap](docs/ROADMAP.md#-roadmap)
- <a id="medium-term-milestone--framework-paper-and-final-evaluation"></a>[Medium-term milestone — framework paper and final evaluation](docs/ROADMAP.md#medium-term-milestone--framework-paper-and-final-evaluation)
- <a id="experiment-coverage-across-methods"></a>[Experiment coverage across methods](docs/ROADMAP.md#experiment-coverage-across-methods)
- <a id="active-milestone--code-review-and-paper-draft"></a>[Active milestone — code review and paper draft](docs/ROADMAP.md#active-milestone--code-review-and-paper-draft)
- <a id="operation-reclamation"></a>[Operation Reclamation](docs/RECLAMATION.md)

</details>

## Development history and manual review

The dated development history is retained below. Planning lives in
[ROADMAP.md](docs/ROADMAP.md), manual review marks in [RECLAMATION.md](docs/RECLAMATION.md),
and the detailed AI-assisted changelog in [CHANGELOG.md](docs/CHANGELOG.md).

<details>
<summary>Expand the protected development history</summary>

<!-- MANUAL PROTECTION ZONE BEGINS -->
### Changelog
<!-- AI ACCESS ZONE BEGINS -->
<!-- AI ACCESS ZONE ENDS -->
- **2026-10-26** -- Merged MLM fixed/variable length generation mechanics into `methods/mlm/length.py`
- **2026-10-23** -- [RECLAIM] `methods/mlm/`: `eval_cli`, `evaluation` (partial) -- updated generative decoder defaults
- **2026-10-23** -- **Abstract submission to iAIM 2026**
- **2026-10-22** -- [RECLAIM] `methods/mlm/` training block (`train_cli.py` and all relevant scripts)
- **2026-10-22** -- [EXPERIMENTS] 2 MLM ESM-2 no-partner trials 
- **2026-10-12** -- [BENCHMARK] Completed IgCraft, retrieval, 7 seed, repeat and capacity studies; added seed-aware analysis and independently verified saved evidence
- **2026-10-08** -- [BENCHMARK] Audited matched OAS4 validation and the empirical follow-up; added paired comparisons and manuscript tables/figures
- **2026-10-07** -- [EXPERIMENTS] 6 AR/S2S no-partner controls, 4 ESM-2 dropout confirmations, 2 MLM no-partner controls, 4 MLM dropout trials
- **2026-10-07** -- [RECLAIM] `methods/mlm/`: `arguments`, `train_cli`
- **2026-10-07** -- [RECLAIM] `methods/mlm/`: Redesigned root and method entrypoint scripts -- `entrypoint.py` -> `train_cli.py`; `cli.py` -> `eval_cli.py`, `gen_cli.py`
- **2026-10-07** -- Experiment scripts rework
- **2026-10-06** -- [RECLAIM] `eval/`: `metrics.py`, `generative.py`
- **2026-09-22** -- Reclassified config dirs
- **2026-09-22** -- Revised environment settings (server migration)
- **2026-09-18** -- [RECLAIM] `datasets/`: revised dataset class scripts
- **2026-09-17** -- [RECLAIM] `datasets/preprocess/`: Redesigned the data preprocessing pipeline
- **2026-09-17** -- Data storage restructure: dataset code to `cdrseqgen/datasets/`; raw/processed files to root `data/` (old: `datasets/data/`)
- **2026-09-15** -- [EXPERIMENTS] table and timeline created 
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

<!-- MANUAL PROTECTION ZONE ENDS -->

</details>
