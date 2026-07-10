# 🧬 CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** July 8, 2026  
**Status:** Custom Transformer + ESM-2 + antibody/protein BERT (IgBert) backbones · heavy (h1-h3) & light (l1-l3) CDR infilling · partner-chain + SAbDab2 antigen context · anchor-excluded metric. Full heavy 3-CDR × 2-dataset matrix (24 runs) trained; antigen ablation complete (no measurable antigen benefit yet — see Known Issues).

## 📌 Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences given the surrounding framework context. 

Unlike standard text-based string models, this framework treats antibodies as **Structured Biological Objects**. By cleanly separating the 14 IMGT-defined regions (Frameworks 1-4 and CDRs 1-3 for both Heavy and Light chains), the codebase scales seamlessly from simple CDR-H3 infilling to multi-loop predictions, light-chain aware generation, antigen-aware generation, and eventually 3D structural embeddings.

---

## 🏗️ Repository Structure

```text
CDRSeqGen/
├── configs/                    # target (h1/h2/h3 heavy, l1/l2/l3 light) x model (scratch/esm2/igbert/protbert) x context
│   ├── h{1,2,3}_scratch[.json|_lc|_ag|_ag_lconly] + h{1,2,3}_esm2[...]   # heavy-CDR suite (target = h*)
│   ├── h{1,2,3}_igbert[.json|_lc] + h{1,2,3}_protbert[.json|_lc]         # protein-BERT PLMs on OAS (IgBert/ProtBert)
│   └── l{1,2,3}_scratch[.json|_hc] + l{1,2,3}_esm2[.json|_hc]            # light-CDR suite (target = l*)
│       # TARGET: h1/h2/h3 mask a heavy CDR; l1/l2/l3 mask a light CDR (target_region in the config).
│       # PARTNER CONTEXT (include_partner_chain): prepends the *other* chain — light for an h*
│       #   target (run tag _LC), heavy for an l* target (run tag _HC). One flag, symmetric.
│       # DATASET MATTERS: base/_lc/_hc train on OAS; _ag AND _ag_lconly train on SAbDab2 (different data).
│       # _lc = OAS heavy + light context. _hc = OAS light + heavy context. _ag = SAbDab2 + light + antigen.
│       # _ag_lconly = SAbDab2 + light chain, NO antigen -- the matched ablation baseline for _ag
│       #              (the ONLY valid same-test-set comparison for isolating the antigen effect).
│       # Run names carry an explicit dataset tag (_OAS / _SD2) resolved from an extensible
│       # DATASET_TAGS registry in train.py -- an unrecognised source is tagged _DS with a warning,
│       # never left blank. (Runs before 2026-07-04 used the older implicit scheme: no OAS tag, _SD;
│       # and target_region used the older cdr*/lcdr* spelling, still accepted for back-compat.)
├── dataset/
│   ├── data/                   # Ignored in git (Raw and Processed OAS + SAbDab2 data)
│   │   └── download_sabdab2.sh # Downloads + extracts SAbDab2's ML-grade splits.tar.gz from Zenodo
│   ├── preprocess_paired.py    # OAS: data cleaning and IMGT formatting
│   └── preprocess_sabdab2.py   # SAbDab2: paired antibody-antigen structures -> Bio-Object JSON
├── results/                    # Auto-generated timestamped model checkpoints
├── test/                       # JSON files containing curated sequences for benchmarking
│   ├── test_h1.json
│   ├── test_h3.json
│   ├── test_h3_lc.json
│   └── test_h3_ag.json
├── backbones.py                # Model registry — add a new PLM (AbLang, ProtBert, ...) in ONE entry here
├── dataset.py                  # Tokenizers, TokenizerAdapter, Dataset logic, and Collator
├── env.sh                      # One-click environment setup script
├── model.py                    # The from-scratch custom transformer definition
├── predict.py                  # Inference script for generating novel loops
├── train.py                    # HF Trainer pipeline with automated CSV/PNG metric extraction
├── rescore.py                  # Re-evaluate a finished run with the anchor-aware metric (no retraining)
├── scripts/
│   └── run.sh                  # Unified runner: ./scripts/run.sh <gpu> [oas|antigen|lcdr|igbert|protbert|plm|all ...]
├── linegraph_plotter.py        # CLI tool for multi-line graph comparisons
└── README.md                   # This file
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
Convert a raw OAS dataset into our custom "Bio-Object" JSON format. This filters for productive sequences, extracts the 14 regions, and preserves `ANARCI` numbering.
```bash
python dataset/preprocess_paired.py \
    --input_file dataset/data/raw/1287169_1_Paired_All.csv \
    --output_file dataset/data/processed/processed_paired_data.json
```

**Antigen-conditioned data (SAbDab2):** for the `_ag` configs, download SAbDab2's ML-grade structural splits (~876MB, one-time) and preprocess them into the same Bio-Object schema plus an `antigen` field. Unlike OAS, SAbDab2's own `abag_split.csv` already carries IMGT-numbered CDR columns and a leakage-safe train/test split, so this step is pure CSV parsing — no structure/CIF parsing or re-numbering needed.
```bash
bash dataset/data/download_sabdab2.sh
python dataset/preprocess_sabdab2.py \
    --input_file dataset/data/raw/sabdab2/splits_final/abag_split.csv \
    --output_file dataset/data/processed/processed_sabdab2_data.json
```

### Step 2: Training & Batch Experimentation
We use `.json` configuration files to manage hyperparameters cleanly. The trainer calculates **Sequence Recovery Rate** (Per-Residue Accuracy), logs to TensorBoard, and auto-generates metric graphs at the end of every run.

**Run a Single Experiment:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py configs/h3_scratch.json
```

**Run an Automated Experiment Suite:** [`scripts/run.sh`](scripts/run.sh) takes a GPU id and one or more suite names, fails fast if the processed data a suite needs isn't there, runs scratch first (cheap — a broken config surfaces before burning time on ESM-2/PLMs), and skips to the next config if one fails.
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

### Step 3: Result Visualization & Comparison
Compare multiple runs on a single multi-line graph using the automated `epoch_metrics.csv` files generated by `train.py`.

```bash
python linegraph_plotter.py \
  --run "H3 Scratch" "./results/20260608_121522_cdr3_scratch" \
  --run "H3 ESM-2" "./results/20260608_143000_cdr3_esm2_t12_35M_UR50D" \
  --metric eval_accuracy \
  --title "Sequence Recovery Comparison: Scratch vs ESM-2"
```

> **⚠️ Never mix datasets on one chart.** OAS (`_OAS`) and SAbDab2 (`_SD2`) runs are evaluated on *different test sets*, so their accuracies are not comparable on shared axes — plot one graph per dataset (e.g. `cdr3_oas` with the 4 OAS runs, `cdr3_sd2` with the 4 SAbDab2 runs). The `results/comparisons/*_oas_*.png` / `*_sd2_*.png` pairs follow this convention.

### Step 4: Batch Inference & Evaluation
Evaluate your trained model on a library of specific antibody frameworks using JSON test files. The model acts as a "Design Tool," filling the gap and comparing it to the ground truth.

```bash
python predict.py \
    --model_type facebook/esm2_t12_35M_UR50D \
    --model_path ./results/latest \
    --test_file test/test_h3.json
```

---

## 🧠 Key Architectural Features

### 1. Task-Agnostic Masking (The "LEGO" Principle)
The `AntibodyInfillingDataset` separates sequences into `pt1` (left context), `mask` (target gap), and `pt2` (right context). The `target_region` string alone chooses the loop — `h1/h2/h3` mask a heavy CDR, `l1/l2/l3` mask the *light* chain's CDR (the mirror case; the older `cdr*`/`lcdr*` spellings are still accepted). Switching target requires changing zero core tokenization logic. Labels are strictly length-enforced to prevent cross-entropy dimension mismatches.

### 2. Pluggable Backbones (the extensibility core)
Model choice is a **registry**, not a chain of `if "esm2" in name` checks. [`backbones.py`](backbones.py) holds one `Backbone` entry per model family; `train.py`, `predict.py`, and `dataset.py` never branch on model type — they all go through `resolve(model_type)`, which returns the matching backbone and a uniform `TokenizerAdapter`. The same pipeline trains the from-scratch transformer, fine-tunes ESM-2, or fine-tunes any Hugging Face masked-LM just by changing the `"model_type"` string.

Each backbone declares two conventions **explicitly** instead of letting them be guessed from the tokenizer (the previous code overloaded a single `is_esm` boolean to mean three different things, which silently mishandled non-ESM models):
- **`position_type`** (`"absolute"` | `"relative"`) — controls where antigen context is injected (see §4). Absolute-position models (the scratch transformer, BERT-style protein LMs) need it appended after the target; relative/rotary models (ESM-2) can take it prepended.
- **`uses_special_tokens`** — whether inputs are wrapped as `[cls] … [end]`. The `TokenizerAdapter` also resolves the ESM-vs-BERT separator difference generically (`<eos>` when the tokenizer has one, else `<sep>`), so BERT-style tokenizers with `eos_token_id = None` work without special-casing.

**Adding a new PLM (e.g. AbLang, ProtBert, ESM-C) is one registry entry** — often *zero*, because any `AutoModelForMaskedLM` + `AutoTokenizer` with absolute positions is already covered by the generic `hf_encoder` fallback (just pass its hub id as `model_type`). Add an explicit `register(Backbone(...))` above the fallback only when a model needs different conventions (e.g. relative positions, or a bespoke tokenizer/model class). No edits to `train.py`/`predict.py`/`dataset.py` are needed either way.

### 3. Partner Chain Context Injection
Biologically, Heavy and Light chains physically interact, so the framework can prepend the paired **partner chain** (the non-target chain) to `pt1` (separated by `<eos>`) when `"include_partner_chain": true` is passed, enriching structural context. It's symmetric: for a heavy (`h*`) target the partner is the light chain (run tag `_LC`); for a light (`l*`) target the partner is the heavy chain (run tag `_HC`). A single flag covers both directions — the target region determines which chain gets added.

### 4. Antigen Context Injection (SAbDab2)
Same "LEGO" pattern as light-chain injection, but **the injection side is keyed on the backbone's `position_type`** (§2) — this isn't cosmetic, it fixes a real bug (see Known Issues below). For **relative**-position models (ESM-2), the antigen is prepended *ahead of* the light chain into `pt1`: `antigen <sep> light <sep> heavy_pt1 <mask> pt2`. For **absolute**-position models (the scratch transformer, and BERT-style PLMs), the antigen is instead *appended* into `pt2`, with the light chain still prepended to `pt1`: `light <sep> heavy_pt1 <mask> fwr4 <sep> antigen`. Multi-chain antigens are joined with `X` (the standard "unknown residue" convention) as a soft chain-break marker, since the amino-acid tokenizer has no token for a literal chain separator.

Antigen sequences (up to ~2,300 residues for large multi-chain complexes) made the pre-existing "no truncation" gap in `AntibodyInfillingDataset` actually bite, so this feature also added a real length cap, applied on whichever side carries the antigen: truncated from the left for relative-position `pt1`, from the right for absolute-position `pt2` — always dropping antigen context first and keeping the heavy-chain residues immediately adjacent to the masked loop.

This data comes from `dataset/preprocess_sabdab2.py`, which is a *different, much smaller* dataset than the OAS-derived one — 3,984 paired-chain (heavy+light) structures with a resolved protein/peptide antigen (3,234 train / 750 test), versus OAS's repertoire-scale sequencing data. `train.py` detects the `"split"` field this preprocessing script attaches to every entry and uses SAbDab2's own leakage-safe split instead of drawing a fresh random one (OAS-derived data has no such field and is unaffected). Both preprocessors report a per-stage filter breakdown (how many rows were dropped for non-productive/wrong-type/malformed-numbering/missing-regions) so the final count is auditable rather than a black box.

**Explicit dataset tagging (extensible).** Every run's output-dir name states its data source so runs are never silently confused on a chart (the old scheme left OAS *untagged*, so an `_SD`-less name ambiguously read as "OAS" or "self-distillation" or "std-dev"). `train.py` resolves the tag from a module-level `DATASET_TAGS` registry — an ordered `(substring, tag)` list (`sabdab2 → SD2`, `paired → OAS`), first match wins — via `dataset_tag(data_path)`. Adding a new source is one line; an unrecognised path is tagged `DS` **with a logged warning** rather than left blank. Runs produced before 2026-07-04 predate this and carry the old implicit tags (`_SD`, no OAS tag).

### 5. Automated Tracking & Evaluation
*   **TensorBoard:** Live tracking via `"report_to": "tensorboard"`.
*   **Metric Extraction:** At the end of training, `train.py` automatically parses Hugging Face logs, dumps `epoch_metrics.csv`, and generates standalone `.png` curves for Loss and Sequence Recovery Rate.

---

## ⚠️ Known Issues & Evaluation Caveats (as of July 8, 2026)

### Live caveats — keep these in mind when reading any result

**`eval_accuracy` is anchor-inflated (worst for H3).** The whole-span average is dominated by the 2-3 V/J-gene-determined anchor residues at each loop end (predicted ~80-100% by any model); the hypervariable middle that's actually being "designed" is far lower. On H3 the gap is large — whole-span 0.361 vs anchor-excluded **core 0.221** (`h3_esm2`, SAbDab2). This is now measurable: `compute_metrics` logs `core_accuracy`/`anchor_accuracy`, and [`rescore.py`](rescore.py) re-scores any finished run. Report core (or position-stratified) recovery, not the raw average — especially for the high H1/H2 numbers (0.97-0.996 on OAS). Position-resolved (`h3_esm2`):

| Position from start | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 100% | 89% | 35% | 21% | 20% | 14% |

**ESM-2's benefit is dataset-size-specific.** On OAS (21k) ESM-2 ≈ scratch — H3 0.455 vs 0.458, H1/H2 both ~0.99 — the from-scratch transformer already saturates the task, so pretraining adds nothing (and light-chain context slightly *hurts* H3). On the ~5× smaller SAbDab2 pretraining wins big: H1 0.49→0.72, H2 0.37→0.60, H3 0.21→0.32 (scratch→ESM-2, lc-only). "ESM-2 doesn't help" holds *only* in the large-data regime.

**ESM-2 overfits on SAbDab2.** Curves peak then decline over 10 epochs (H3 esm2+lc 0.361→0.320). `load_best_model_at_end` saves the peak, so read best-epoch not final-epoch numbers there, and consider early stopping.

**Antigen conditioning gives no measurable benefit.** The valid same-data ablation (`_ag` vs `_ag_lconly`, SAbDab2, all 3 CDRs) shows no signal — and it *survives the anchor-excluded core metric* too, so it isn't a measurement artifact:

| CDR | scratch lc-only → +ag | ESM-2 lc-only → +ag |
|---|---|---|
| H1 | 0.485 → 0.456 | 0.716 → 0.718 |
| H2 | 0.370 → 0.365 | 0.599 → 0.595 |
| H3 | 0.205 → 0.212 | 0.320 → 0.324 |

All |Δ| ≤ 0.03, sign inconsistent. Not a truncation artifact (60% of examples keep the full antigen at `max_len=512`). Likely the data budget (~3.2k) and/or the flat-context mechanism — cross-attention or an explicit epitope/contact signal is the promising next lever. **Never plot OAS and SAbDab2 on shared axes** (different test sets); the `_OAS`/`_SD2` run tags enforce the distinction, and per-dataset charts live in `results/pngs/*_{oas,sd2}_*.png`.

**OAS and SAbDab2 use different leakage standards.** SAbDab2 has a sequence-identity-cluster split; OAS uses a plain random `train_test_split` ([train.py](train.py)) that leaks clonal near-duplicates (same V/J, CDR3s a few mutations apart) into eval, making **OAS numbers optimistic relative to SAbDab2**. CD-HIT / clustered-split dedup is a *prerequisite* for the 1M+ scale-up, not a nice-to-have.

### Fixed bugs (changelog)
- **2026-07-08** — `CustomAntibodyModel.from_pretrained` silently returned *randomly initialized* weights (never loaded the checkpoint, no warning) — had also been breaking `predict.py` for scratch models. Now loads the state dict explicitly (strict).
- **2026-07-03** — `collate_fn` padded with a hardcoded `0`, which is ESM-2's `<cls>` id, zeroing the real `<cls>` out of every `attention_mask`. Now threads the correct `pad_token_id` per tokenizer.
- **2026-07-03** — `predict.py` built ESM-2 inputs without the `<cls>`/`<eos>` wrap that training uses (train/inference mismatch). Now mirrors the training-time construction.
- **2026-07-03** — scratch model couldn't learn from antigen context (flat ~0.145): absolute position embeddings + a variable-length *prepended* antigen shifted the target unpredictably. Fixed by appending antigen after the target for absolute-position models (see §4); climbs to ~0.21.
- **2026-07-03** — `CustomAntibodyModel.forward()` ignored `attention_mask` (attended over padding); now passed as `src_key_padding_mask`.
- **2026-07-03** — config `hidden_size`/`num_layers`/`max_len` were dropped by the model builder (defaults always used) and `max_position_embeddings` was hardcoded to 512; now threaded through and sized `max(max_len, 512)`.
- **2026-07-03** — preprocessing: narrowed a bare `except: continue`, added null/empty-region guards and a per-stage filter breakdown (dropped 1 degenerate SAbDab2 entry, 3,985→3,984).

---

## ⏭️ Roadmap

**Done** (findings live in Known Issues; bug fixes in its changelog):
- Full 3-CDR × 2-dataset **heavy matrix (24 runs)** trained; the valid antigen ablation (`_ag` vs `_ag_lconly`) is complete → **no measurable antigen benefit**, and it survives the anchor-excluded core metric. [07-03/04/08]
- **Anchor-excluded metric** (`core_accuracy`/`anchor_accuracy` + [`rescore.py`](rescore.py)); explicit extensible **dataset tagging** (`DATASET_TAGS`, `_OAS`/`_SD2`); re-ran the full post-fix ESM-2 matrix. [07-03/04/08]
- **Light-CDR (`l1/l2/l3`) infilling**: dataset generalized to target either chain, single `include_partner_chain` flag, unified `h*/l*` target vocabulary, 12 configs (run via the `lcdr` suite in [`scripts/run.sh`](scripts/run.sh)). [07-08]
- **Antibody/protein BERT backbones** (IgBert, ProtBert) via the registry + a residue-spacing tokenizer shim; 6 IgBert OAS configs. [07-08]

**Open — near-term:**
- **Make antigen conditioning actually work** (the core research question): cross-attention to the antigen or an explicit epitope/contact signal instead of flat context, and/or more antigen data (~3.2k is thin). This supersedes fiddling with the current injection.
- **Run the IgBert and light-CDR suites** (both wired, not yet swept). IgBert vs ESM-2 on OAS tests whether the OAS ceiling is model-bound — the cheap way to check before any data scale-up.
- **Fine-tune `scratch_ag` from an OAS-pretrained checkpoint** rather than from scratch — pretraining is what carries the small-data SAbDab2 regime (`h3_scratch_ag` ~0.21 vs `h3_esm2_ag` ~0.32).
- Early stopping / best-epoch reporting for the overfitting SAbDab2 runs.
- **Clustered/dedup OAS split** (CD-HIT) — fixes the optimistic-eval leakage and is a prerequisite for any scale-up.
- **Generative eval beyond recovery** (diversity / novelty / perplexity) — recovery rewards copying the native, the opposite of design.

**Open — mid/long-term:**
- **AbLang2** proper — ships a non-HF tokenizer, so it needs a small `load_tokenizer` shim (unlike the HF-native IgBert/ProtBert already supported).
- **IgLM** baseline — the field-standard comparison, but GPT-2/autoregressive, so it needs its own generation/scoring path rather than a drop-in `model_type`.
- **Data expansion:** OAS → 1M+ (paired with the clustered split above).
- **Structure:** IgFold/ABlooper-derived CDR coordinates as conditioning; antigen-aware structure-conditioned generation (informed by dyMEAN, DiffAb rather than building diffusion machinery from scratch).