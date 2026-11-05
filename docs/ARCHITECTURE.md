# Architecture and repository reference

[Back to README](../README.md) · [Method workflow map](../cdrseqgen/methods/common/README.md)

The full repository map and feature notes moved here from the README on **26 October 2026**.
The method workflow map and source-module docstrings describe the current call graph;
dated implementation examples below retain their historical context. Detailed personal
code-review records are kept locally; reviewed source remains open to further revision.

## 🏗️ Repository Structure

`scripts/` holds command-line interfaces; `cdrseqgen/` holds planning, training and evaluation logic.
Named commands follow `run.sh → scripts/experiments.py → cdrseqgen.experiments → train.py`;
legacy GPU-first commands call the same planner/executor directly. `sweep.sh`
forwards to `run.sh`. Single-run settings live in `configs/{mlm,ar,s2s}/`, multi-run recipes in
`configs/experiments/`, and resolved plans and outputs in `results/runs/`.

Benchmark commands follow `run.sh benchmark → benchmark/run.py → adapters → scorer → plotter`.
Their recipes live in `benchmark/recipes/`, frozen inputs in `benchmark/manifests/`, and generated
candidates, scores and plots in `results/benchmark/`. See the [current recipe](../benchmark/recipes/oas4_h3_length_free.md).
The [follow-up](../benchmark/recipes/oas4_h3_followup.md) uses `run.sh benchmark followup →
benchmark/followup.py → empirical → scorer → analysis/capacity` and reuses the completed score rows.

For the methods reclamation review, the [shared workflow map](../cdrseqgen/methods/common/README.md)
shows the run sequence, stage ownership and method extension points, with a reference
training-command skeleton alongside the three executable method training commands.

The root-to-method command paths use the same names for MLM, AR and S2S:

| Root command | Method command module | Responsibility |
|---|---|---|
| `train.py` | `methods/<method>/train_cli.main()` | Configure and conduct a training run, including optional post-training evaluation. |
| `evaluate.py` | `methods/<method>/eval_cli.main()` | Validate evaluation options, prepare method-specific state and call the evaluator. |
| `generate.py` | `methods/<method>/gen_cli.main()` | Validate supplied cases, prepare method-specific state and generate each case. |

For all three methods, the CLI functions connect method callbacks to shared workflows.
Each method's `train_cli.main()`, `eval_cli.main()` and `gen_cli.main()` configures
console logging; root scripts parse/read routing inputs and select the method.
`train_cli.main()` parses arguments and passes `training.prepare_training()` and
`training.finish_training()` to `common.training.run_training_pipeline()`.
`eval_cli.main()` supplies evaluation setup and `evaluation.run_generative_eval()`;
`gen_cli.main()` supplies generation setup, optional state and `generation.generate_case()`.
Shared standalone loading, compatibility and output handling live in
`methods/common/evaluation.py` and `methods/common/generation.py`.

```text
CDRSeqGen/
├── docs/                       # Project guides, research notes and implementation history
├── configs/                    # Run configs and experiment recipes; no executable code.
│                               #   A config's "method" key selects the pipeline (default mlm)
│   ├── mlm/                    # Masked infilling: heavy/, light/, antigen/
│   ├── ar/                     # Autoregressive heavy-chain infilling
│   ├── s2s/                    # Encoder–decoder H3: scratch and pretrained encoders
│   ├── experiments/            # Named multi-arm specs, protocols and shared bases/
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
│   │   │   ├── arguments.py    # The shared args + config/CLI parsing + run-directory setup;
│   │   │   │                   #   each method subclasses and adds only its own
│   │   │   ├── backbones.py    # Shared Backbone/BackboneRegistry, checkpoint and tokenizer helpers
│   │   │   ├── evaluation.py   # Standalone loading/provenance and post-training evaluation setup
│   │   │   ├── generation.py   # Standalone checks, model loading, case execution and output
│   │   │   ├── logging.py      # Console/file logging and uncaught exceptions
│   │   │   ├── run_setup.py    # Run directories/latest, dataset tags and config/provenance
│   │   │   ├── losses.py       # Aligned token cross-entropy shared by method objectives
│   │   │   ├── transformer.py  # Custom-model validation, rotary self-attention, FFNs and initialization
│   │   │   ├── sampling.py     # warp_logits / ban_tokens / logits_of — so "top_p=0.9" means the
│   │   │   │                   #   same thing on both sides of an AR-vs-MLM table
│   │   │   ├── callbacks.py    # LogMetricsCallback + build_callbacks; optional early stopping
│   │   │   ├── eos_decoding.py # AR/S2S only: recovery metrics, dataset split, evaluation strategy and
│   │   │   │                   #   generation record; they differ in context, sampler and alignment
│   │   │   └── training.py     # Preparation/finish workflow, Trainer lifecycle and loss bridge
│   │   ├── ar/                 # Autoregressive (2026-09-01). Data lives in datasets/dataset_ar.py;
│   │   │   ├── backbones.py    #   this package owns model + training.
│   │   │   │                   # Causal registry — resolve(model_type) -> CausalBackbone.
│   │   │   │                   #   iglm keeps IgLM's own vocabulary, layout and sampler.
│   │   │   │                   #   hf_decoder ADDS the layout's sentinels to a pretrained LM and
│   │   │   │                   #   resizes it: those rows start random (see its docstring)
│   │   │   ├── training.py     # Preparation/finish hooks, datasets, recovery metrics and Trainer.
│   │   │   │                   #   No masking view (nothing to hide), no loss registry (§13)
│   │   │   ├── sampling.py     # Causal decode to <eos>. Replaces the masked side's sampling +
│   │   │   │                   #   generation + length mechanics: the model picks its
│   │   │   │                   #   own length, so §10's gap-sentinel head has no counterpart
│   │   │   ├── arguments.py    # Causal args: subclasses the shared bases, adds gen_max_new_tokens
│   │   │   ├── train_cli.py    # Logging/config setup -> common training workflow + AR hooks
│   │   │   ├── eval_cli.py     # main(): saved-checkpoint evaluation entrypoint
│   │   │   ├── gen_cli.py      # main(): supplied-framework generation entrypoint
│   │   │   ├── evaluation.py   # CausalStrategy for the SHARED loop in eval/generative.py —
│   │   │   │                   #   decode-to-eos draws, TRUE (not pseudo) perplexity, per-candidate
│   │   │   │                   #   judge prompts since causal lengths vary
│   │   │   ├── generation.py   # generate.py callbacks: EOS-only length checks, causal case record
│   │   │   └── models/
│   │   │       ├── transformer.py  # Causal decoder, ROTARY positions (§8: absolute positions lock
│   │   │       │               #   a checkpoint to its context regime; causal prompts vary in length)
│   │   │       └── iglm.py         # Released IgLM / IgLM-S as an AR backbone: native 33-token
│   │   │                       #   vocabulary, [chain][species] layout, conditioning, sampler
│   │   ├── s2s/                # Encoder-decoder (2026-09-07). Data lives in datasets/dataset_s2s.py.
│   │   │   │                   #   The missing cell of the 2x2: bidirectional conditioning like
│   │   │   │                   #   mlm, <eos>-chosen length like ar. Contract-identical to ar
│   │   │   │                   #   throughout, so an s2s/ar/mlm table is one implementation.
│   │   │   ├── train_cli.py    # Logging/config setup -> common training workflow + S2S hooks
│   │   │   ├── eval_cli.py     # main(): saved-checkpoint evaluation entrypoint
│   │   │   ├── gen_cli.py      # main(): supplied-framework generation entrypoint
│   │   │   ├── backbones.py    # scratch, ESM-2, and protein-BERT encoders. PLM encoders keep
│   │   │   │                   #   native tokenizers; all decoders use the same residue vocabulary
│   │   │   ├── training.py     # Preparation/finish hooks, datasets, metrics and Trainer. Metrics do
│   │   │   │                   #   NOT re-shift: dataset_s2s already offset the decoder pair
│   │   │   ├── sampling.py     # Decode to <eos>. ONE encoder pass shared by every candidate, and
│   │   │   │                   #   no padding at all — every decoder row starts at <bos>
│   │   │   ├── evaluation.py   # Seq2SeqStrategy for the SHARED loop in eval/generative.py
│   │   │   ├── generation.py   # generate.py callbacks: EOS-only length checks, seq2seq case record
│   │   │   └── models/
│   │   │       ├── plm.py      # PLM encoder + width projection + shared residue decoder;
│   │   │       │               #   fine-tuned/frozen/random controls, complete local checkpoints
│   │   │       └── transformer.py  # Bidirectional encoder + cross-attending causal decoder.
│   │   │                       #   Rotary on SELF-attention only. NOT capacity-matched to ar at the
│   │   │                       #   same num_layers (1.86M vs 800k) — build_model logs the count
│   │   └── mlm/                # Masked-LM training, generation, and native evaluation
│   │       ├── arguments.py        # MLM args: shared bases + loss_type/core/anchor weights
│   │       ├── train_cli.py        # Logging/config/CLI setup -> common workflow + MLM hooks
│   │       ├── eval_cli.py         # main(): connect MLM setup/scoring to the common lifecycle
│   │       ├── gen_cli.py          # main(): supplied-framework generation entrypoint
│   │       ├── backbones.py        # Masked-LM registry — resolve(model_type) -> Backbone
│   │       ├── losses.py           # Shared loss registry + Trainer bridge for every MLM backbone
│   │       ├── training.py         # MLM preparation/finish hooks, datasets and evaluation helpers
│   │       ├── recovery.py         # Position recovery, anchor/core split, core coverage
│   │       ├── sampling.py         # Masked-span dispatch, independent/iterative decoders, pseudo-ppl
│   │       ├── evaluation.py       # MLM setup, strategy selection, scoring and reporting
│   │       ├── generation.py       # MLM setup, saved-head loading, case generation and scoring
│   │       ├── length.py           # Policies, prompts, fitted head, span sampling and eval strategies
│   │       └── models/
│   │           └── transformer.py  # From-scratch masked encoder — model_type "scratch"
│   ├── eval/                   # Shared sequence metrics and strategy-based evaluation
│   │   ├── __init__.py         # ANCHOR_TRIM / MIN_CORE_COVERAGE and package responsibilities
│   │   ├── metrics.py          # Sequence comparisons, BLOSUM provenance, confidence summaries
│   │   └── generative.py       # Select -> draw -> score -> judge -> report/output helpers
│   │                           #   Strategies: draw, self_scores, judge_inputs, payload_length_policy
│   │                           #   Judge helpers load MLM implementations when needed
│   └── visualization/          # Run artifacts -> charts and tables. No GPU, no model loading.
│       ├── training_results.py # Trainer history -> epoch_metrics.csv and training-curve PNGs
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
│   ├── run.py                  # plan / smoke / run / status / resume; freezes inputs and source
│   ├── followup.py             # Reuse completed scores; plan/run/status/resume the follow-up
│   ├── empirical.py            # Training-only H3 sampling; original cluster/split verification
│   ├── analysis.py             # Paired group intervals and population reporting across arms
│   ├── capacity.py             # Learned tensor counts, actual epochs/steps and recorded runtime
│   ├── recipes/                # Matched protocols, checkpoint selections and user-run commands
│   ├── adapters/               # CDRSeqGen and IgLM generation under one output contract
│   ├── manifests/              # Versioned validation/test inputs plus SHA-256 sidecars
│   ├── _common.py              # Hashing, seeding, atomic writes, git state, resume checks
│   ├── export_manifest.py      # Processed OAS split -> model-independent manifest
│   ├── scorer.py               # Shared validation, metrics, accounting, and judge scoring
│   ├── bootstrap.py            # Historical two-arm framework bootstrap; not cluster-aware
│   ├── plot_comparison.py      # Matched recovery, length, position, and diversity dashboard
│   └── README.md               # End-to-end benchmark commands and protocol
├── data/                       # THE DATA — gitignored, ~6.7GB (raw + processed OAS/SAbDab2).
│                               #   Deliberately outside the package; configs' data_path points here.
├── results/                    # Ignored in git. Three tiers, by cost to regenerate:
│   ├── runs/                   #   hours of GPU — <timestamp>[_name]/<run>/, latest -> newest
│   │                           #   A round reports itself when it finishes: records at the batch
│   │                           #   root (batch_tables.md, or runs.tsv/summary.tsv for a sweep),
│   │                           #   charts under <batch>/figures/, per-trial stdout under logs/.
│   ├── benchmark/              #   inference + judge scoring — <benchmark_id>/{generations,scores}/
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
│                               #   main() lives in methods/<method>/train_cli.py
├── generate.py                 # Entrypoint — candidate generation on supplied frameworks
├── evaluate.py                 # Entrypoint — held-out generative evaluation (+ --judge)
├── fit_mlm_length.py           # Add MLM length artifacts to an older checkpoint
├── scripts/
│   ├── run.sh                  # Training commands + benchmark subcommands; legacy GPU-first route
│   ├── experiments.py          # Named CLI; discovers recipes and uses the shared planner/executor
│   ├── make_iglm_randinit.py   # Random-init control with the released IgLM architecture (recipe iglm_controls)
│   ├── iglm_replay.py          # Replay a stored external IgLM batch draw for draw with the iglm backbone
│   ├── sweep.sh                # Compatibility alias for run.sh <gpus> sweep [preset] [config]
│   └── unit_tests.sh           # Unit test runner (sets PYTHONPATH; no GPU or data needed)
├── env.sh
├── README.md                   # What the project is, what it found, what is still open
└── WORKFLOW.md                 # How to run it — preprocessing through benchmark
```

---

## 🧠 Architectural Feature Details

### 1. Task-Agnostic Masking (The "LEGO" Principle)
The `MaskedInfillingDataset` separates sequences into `pt1` (left context), `mask` (target gap), and `pt2` (right context). The `target_region` string alone chooses the loop — `h1/h2/h3` mask a heavy CDR, `l1/l2/l3` mask the *light* chain's CDR (legacy `cdr*`/`lcdr*` spellings still parse). Switching target requires changing zero core tokenization logic. Labels are strictly length-enforced to prevent cross-entropy dimension mismatches.

### 2. Pluggable Backbones (the extensibility core)

All three methods use the descriptor and ordered registry implementation in
[common/backbones.py](../cdrseqgen/methods/common/backbones.py). Each method owns a separate
registry instance and its model/tokenizer registration blocks. MLM uses the shared
`Backbone` directly; S2S exports it as `Seq2SeqBackbone`. AR's `CausalBackbone` extends
`build_tokenizer()` with span-layout validation and persistence. Besides `scratch` and the
generic `hf_decoder`, it registers `iglm`, a pretrained model with its own span layout,
conditioning fields and sampler (see [MODELS.md](MODELS.md#pretrained-ar-backbone-iglm)); an
adapter carries these on its tokenizer, so the shared training and generation code do not
branch on the model name. Matching is first-wins;
the registry rejects duplicate IDs and checks fallback placement after registration.
MLM and AR require one generic fallback, while S2S permits having none.

IgLM's source layout and any recorded sampling policy are checked before both fine-tuning
and inference. Its loader rejects missing learned weights and preserves native token IDs;
its sampler isolates generation defaults and records malformed/unterminated attempts.
Original checkpoints without run metadata can use `generate.py --method ar`; saved runs
retain their declared method. Pretrained benchmark arms must carry their own saved tokenizer
assets, which are hashed with model/configuration inputs.

Backbone registrations may also supply `validate_model_args(model_args, data_args)`.
Argument setup invokes it through `backbone.validate_training_args()` before run
creation. Custom and PLM families therefore own their architecture checks; adding
another custom model does not require a special-case name test in the parser.

The current custom models retain their config/model classes, `build_model()`
factories and saved config identifiers. The factories construct `Config → Model`.
Shared validation, rotary self-attention, FFNs and initialization live in
[common/transformer.py](../cdrseqgen/methods/common/transformer.py).
Attention layout and label alignment remain method-specific. An additional
architecture belongs in its method's `models/` directory with its own backbone entry;
width/depth variants of the current model remain configuration choices.

The [model specifications](MODELS.md) record the architecture comparison, implicit
layer defaults, forward/loss contracts and parameter counts. Builder defaults and
the saved main-benchmark configurations are documented separately for paper writing.

The MLM registration module [backbones.py](../cdrseqgen/methods/mlm/backbones.py) holds one
descriptor per compatible model family. Callers use `resolve(model_type)` to select the
descriptor, then `backbone.build_tokenizer()` to obtain its tokenizer adapter. Model creation
and checkpoint restoration use the descriptor's two model loaders. The same MLM pipeline
can train the scratch transformer or fine-tune a compatible pretrained masked LM by
changing `model_type`.

Each backbone declares two conventions **explicitly** instead of letting them be guessed from the tokenizer (the previous code overloaded a single `is_esm` boolean to mean three different things, which silently mishandled non-ESM models):
- **`position_type`** (`"absolute"` | `"relative"`) — controls where antigen context is injected (see §4). Absolute-position models (the scratch transformer, BERT-style protein LMs) need it appended after the target; relative/rotary models (ESM-2) can take it prepended.
- **`uses_special_tokens`** — whether inputs are wrapped as `[cls] … [end]`. The `TokenizerAdapter` also resolves the ESM-vs-BERT separator difference generically (`<eos>` when the tokenizer has one, else `<sep>`), so BERT-style tokenizers with `eos_token_id = None` work without special-casing.

**Adding a new masked PLM is usually one `register(Backbone(...))` entry** — occasionally *zero*, when a model is already covered by the generic `hf_encoder` fallback (any `AutoModelForMaskedLM` + `AutoTokenizer`, absolute positions, per-character vocab — just pass its hub id as `model_type`). In practice most protein LMs need an explicit entry: e.g. BERT-style ones (ProtBert, IgBert) expect *whitespace-separated* residues and ship tokenizer/loader quirks (a fast-tokenizer that needs `sentencepiece`, or a `config.json` with no `model_type`), all handled once in the `protein_bert` backbone. Either way, no edits to `train.py`/`generate.py`/`dataset_mlm.py` are needed. A compatible autoregressive or encoder-decoder architecture belongs in the existing AR or S2S model module and backbone registry. Add a sibling method package when the objective or task contract changes.

### 3. Partner Chain Context Injection
Biologically, Heavy and Light chains physically interact, so the framework can prepend the paired **partner chain** (the non-target chain) to `pt1` (separated by `<eos>`) when `"include_partner_chain": true` is passed, enriching structural context. It's symmetric: for a heavy (`h*`) target the partner is the light chain (run tag `_LC`); for a light (`l*`) target the partner is the heavy chain (run tag `_HC`). A single flag covers both directions — the target region determines which chain gets added.

### 4. Antigen Context Injection (SAbDab2)
Same "LEGO" pattern as partner-chain injection (§3), but **the injection side is keyed on the backbone's `position_type`**. For **relative**-position models (ESM-2), the antigen is prepended *ahead of* the light chain into `pt1`: `antigen <sep> light <sep> heavy_pt1 <mask> pt2`. For **absolute**-position models (the scratch transformer, and BERT-style PLMs), the antigen is instead *appended* into `pt2`, with the light chain still prepended to `pt1`: `light <sep> heavy_pt1 <mask> fwr4 <sep> antigen`. Multi-chain antigens are joined with `X` (the standard "unknown residue" convention) as a soft chain-break marker, since the amino-acid tokenizer has no token for a literal chain separator.

Antigen sequences (up to ~2,300 residues for large multi-chain complexes) made the pre-existing "no truncation" gap in `MaskedInfillingDataset` actually bite, so this feature also added a real length cap, applied on whichever side carries the antigen: truncated from the left for relative-position `pt1`, from the right for absolute-position `pt2` — always dropping antigen context first and keeping the target-chain residues immediately adjacent to the masked loop.

### 5. Automated Tracking & Evaluation

Method `training.py` modules use matching preparation, dataset, Trainer-construction
and finishing stages. `prepare_training()` builds resources through `get_datasets()`
and `build_trainer()`; the shared lifecycle then trains/evaluates/saves before
`finish_training()`. That hook calls `common.evaluation.post_training_generation()`
directly with the method name and its decoding/length options. The shared helper
dispatches through the registry's `Method.evaluate()` to `eval_cli.evaluate()` and
then `evaluation.run_generative_eval()`, using the loaded model. Objective-specific
metrics, MLM length fitting and S2S diagnostics stay in their method modules.
See the [workflow map](../cdrseqgen/methods/common/README.md).

*   **TensorBoard:** Live tracking via `"report_to": "tensorboard"`.
*   **Metric Extraction:** `common.training.train_and_save()` calls [`visualization/training_results.py`](../cdrseqgen/visualization/training_results.py) to export Trainer history as `epoch_metrics.csv` (loss + recovery, whole-span **and** the anchor-excluded `core`/`anchor` accuracies) and standalone `.png` curves.

Generated-loop evaluation follows the shared workflow in [`generative.py`](../cdrseqgen/eval/generative.py):

```text
select frameworks -> strategy.draw -> sequence metrics + self/judge scores
                  -> candidate details + confidence -> framework records
                  -> report / JSON / optional confidence figure
```

The strategy owns generation and validity accounting; the method's `run_generative_eval()` assembles and saves the payload. [`metrics.py`](../cdrseqgen/eval/metrics.py) defines the sequence comparisons. Their eligible candidates differ:

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

- **New generation method** → a package beside [`methods/mlm/`](../cdrseqgen/methods/mlm), subclassing `AntibodyFrameworkDataset` from [`datasets/dataset_common.py`](../cdrseqgen/datasets/dataset_common.py), plus `train_cli.py`, `eval_cli.py` and `gen_cli.py` for the three root commands. [`train.py`](../train.py) dispatches to `train_cli.main()` on the config's `"method"` key. **Exercised 2026-09-01** by [`methods/ar/`](../cdrseqgen/methods/ar), which is a third the size of the masked package — the difference is the length-policy layer, the gap-sentinel head, the second eval track and the loss registry, none of which a causal model can need.

  **Exercised again 2026-09-07** by [`methods/s2s/`](../cdrseqgen/methods/s2s), which is the real test: the second method could still have been accommodated by generalising toward it, while the third had to fit a boundary drawn without it in mind. It did — a registry entry, the five-function CLI protocol, and a strategy for the shared evaluation loop — and the parts that did *not* fit were informative. `SpanLayout` does not generalise, because it builds one concatenated stream and an encoder-decoder needs two; that is a new data view, not a parameterisation of the old one. And three *enumerations* had to be found by running the method rather than by the type system: the dashboard family map, the `--decoding` choice list duplicated into two argparse calls, and `sweep.sh`'s architecture flag. Each was a list that had to be edited in a second place, which is the shape of seam this section exists to avoid; `--decoding` and the method classification now derive from `methods/registry.py`, and a test asserts every registered method has a dashboard family.

  Adding the second method is what made the shared layers honest, and each extraction was checked against two implementations rather than guessed from one: `select_split_indices` (the eval-split guard — a second copy could recontaminate one method's results silently), `eval/generative.py` (the evaluation loop; a *strategy* supplies `draw()`, `self_scores()`, `judge_inputs()` and `payload_length_policy()`), `build_judge` (§11 rests on one judge scoring both models), and `methods/common/` (arguments, sampling, callbacks). The last one caught a live defect: the causal entrypoint had quietly dropped `LogMetricsCallback`, reintroducing the empty-`train.log` regression fixed on 2026-07-29. *Added 2026-08-31*: the framework/target split, context injection and length capping are now method-neutral, and `TokenizerAdapter.mask_id` is optional — requiring a mask token was the one thing that made a causal tokenizer impossible to construct. `dataset_mlm.py` is now just the masked-LM view of that base, and re-exports the moved names so existing imports are unchanged. The benchmark contract was already method-agnostic, as [`adapters/iglm.py`](../benchmark/adapters/iglm.py) demonstrates.
- **New masked-LM backbone** → one `Backbone` entry in [`backbones.py`](../cdrseqgen/methods/mlm/backbones.py) (§2) — the entrypoints and `dataset_mlm.py` never learn about it. The causal and seq2seq registries have the same shape, so the seam is one rule across all three methods.
- **New target loop** → change the `target_region` string (§1); masking logic is untouched.
- **New context** → one boolean flag (`include_partner_chain`, `include_antigen`), resolved symmetrically (§3–4).
- **New dataset source** → one `(substring, tag)` pair in the `DATASET_TAGS` registry ([`methods/common/run_setup.py`](../cdrseqgen/methods/common/run_setup.py)) (`oas → OAS`, `sabdab2 → SD2`), so every run's output-dir name states its source explicitly instead of leaving it silently implied; an unrecognised path is tagged `DS` **with a logged warning** rather than left blank.

The guiding constraint: **conventions are declared explicitly, never inferred.** The predecessor overloaded a single `is_esm` boolean to mean three different things and silently mishandled non-ESM models; each backbone now states its `position_type` and `uses_special_tokens` up front. This isn't free — protein-BERT models (IgBert/ProtBert) still needed their own entry for residue-spacing and tokenizer/loader quirks (§2) — but the cost is paid once, in the registry, not scattered across the pipeline.

---
