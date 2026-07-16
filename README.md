# CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** 2026-07-16  
**Status:** Adjusting roadmap for specified objective

## Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences given the surrounding framework context. 

Current implemented features include: 
- **Data Preprocessing**: standardize data into Bio-Object JSON format (IMGT), current datasets: OAS (21106 paired sequences) and SAbDab2 (3984 paired + antigen structures)
- **Addressing Data Leakage**: clustered train/test split for OAS to prevent clonal near-duplicate leakage
- **Task-Agnostic Masking**: masking training method for all 6 CDR loops individually by target selection
- **Pluggable Backbone PLMs**: supports various PLMs, current list: custom transformer (scratch), ESM-2, IgBert, ProtBert
- **Partner Chain Context**: optional inclusion of non-target chain sequence context
- **Antigen Context**: optional inclusion of antigen sequence context for SAbDab2 dataset
- **Evaluation Metrics**: currently implemented: Sequence Recovery Rate (Per-Residue Accuracy), anchor-excluded core accuracy, and anchor accuracy
- **Logging & Visualization**: TensorBoard logging, csv metric extraction, multi-line graph plotter 
- **Extensible Architecture**: modular design for easy addition of new PLMs, datasets, and evaluation metrics

---

## 🏗️ Repository Structure

```text
CDRSeqGen/
├── configs/                    
│   ├── h{1,2,3}_scratch[.json|_lc|_ag|_ag_lconly] + h{1,2,3}_esm2[...]   
│   ├── h{1,2,3}_igbert[.json|_lc] + h{1,2,3}_protbert[.json|_lc]         
│   └── l{1,2,3}_scratch[.json|_hc] + l{1,2,3}_esm2[.json|_hc]            
├── dataset/
│   ├── data/                   # Ignored in git (Raw and Processed OAS + SAbDab2 data)
│   ├── preprocess_oas_paired.py    
│   └── preprocess_sabdab2.py   
├── results/                    # Ignored in git (Auto-generated timestamped model checkpoints)
├── test/                       # JSON files containing curated sequences for benchmarking
│   ├── test_h1.json
│   ├── test_h3.json
│   ├── test_h3_lc.json
│   └── test_h3_ag.json
├── backbones.py                # PLM model registry 
├── dataset.py                  # Tokenizers, TokenizerAdapter, Dataset logic, and Collator
├── env.sh                      
├── model.py                    # Custom transformer (scratch) definition
├── predict.py                  # Inference script for CDR loop generation
├── train.py                    # Main training script
├── rescore.py                  # Re-evaluate finished runs with anchor-aware metric (no retraining)
├── scripts/
│   └── run.sh                  # Unified runner: ./scripts/run.sh <gpu> [oas|antigen|lcdr|igbert|protbert|plm|all ...]
├── linegraph_plotter.py        # Utility for data visualization
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
python dataset/preprocess_oas_paired.py \
    --input_file dataset/data/raw/1287169_1_Paired_All.csv \
    --output_file dataset/data/processed/processed_oas_data.json
```

**SAbDab2**: currently including 3984 structures - paired with resolved antigen, remaining >10k structures without antigen pending future inclusion
```bash
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

### Step 3: Result Visualization & Comparison
Compare multiple runs on a single multi-line graph using the automated `epoch_metrics.csv` files generated by `train.py`.

```bash
python linegraph_plotter.py \
  --run "H3 Scratch" "./results/20260608_121522_cdr3_scratch" \
  --run "H3 ESM-2" "./results/20260608_143000_cdr3_esm2_t12_35M_UR50D" \
  --metric eval_accuracy \
  --title "Sequence Recovery Comparison: Scratch vs ESM-2"
```

<!-- > **⚠️ Never mix datasets on one chart.** OAS (`_OAS`) and SAbDab2 (`_SD2`) runs are evaluated on *different test sets*, so their accuracies are not comparable on shared axes — plot one graph per dataset (e.g. `cdr3_oas` with the 4 OAS runs, `cdr3_sd2` with the 4 SAbDab2 runs). The `results/comparisons/*_oas_*.png` / `*_sd2_*.png` pairs follow this convention. -->

### Step 4: Batch Inference & Evaluation
Evaluate the trained model on a library of specific antibody frameworks using JSON test files. 

```bash
python predict.py \
    --model_type scratch \
    --model_path ./results/latest \
    --test_file test/test_h3.json
```

---

## 📊 Findings

Phase 1 (the sequence sweep) is a **leakage-safe benchmark** for CDR infilling. What it establishes:

**1. Model choice doesn't matter on OAS.** A 128-dim from-scratch transformer matches ESM-2 *and* antibody-specific IgBert on the H3 anchor-excluded core (0.31–0.32, clustered split); every model clusters within ~0.02. → the OAS ceiling is **data/task-bound, not model-bound.** Pretraining only pays off in the *small-data* regime — on the ~5× smaller SAbDab2, scratch→ESM-2 lifts H3 0.21→0.32 (though ESM-2 overfits there — read best-epoch).

**2. CDR-H3 is the only hard loop.** Anchor-excluded core recovery: H1/H2/L1/L2/L3 all ~0.90+, **H3 ~0.32**. Matches the biology — heavy CDR3 gets D-gene + junctional diversity; the rest is comparatively templated. The entire design difficulty concentrates in one loop.

**3. The standard metric misleads (anchor inflation).** Whole-span `eval_accuracy` (~0.46 on H3) is dominated by the 2-3 V/J-gene-determined anchor residues at each loop end (~0.90 recovery, any model); the hypervariable middle actually being "designed" is ~0.32. The gap is **H3-specific** (H1/H2/L core ≈ whole-span). Measured via `core_accuracy`/`anchor_accuracy` + [`rescore.py`](rescore.py). Position-resolved (illustrative, `h3_esm2`):

| Position from start | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 100% | 89% | 35% | 21% | 20% | 14% |

**4. Sequence context — partner chain and antigen — gives no measurable benefit.** The valid same-data antigen ablation (`_ag` vs `_ag_lconly`, SAbDab2, all 3 CDRs) shows no signal, and it *survives the core metric*:

| CDR | scratch lc-only → +ag | ESM-2 lc-only → +ag |
|---|---|---|
| H1 | 0.485 → 0.456 | 0.716 → 0.718 |
| H2 | 0.370 → 0.365 | 0.599 → 0.595 |
| H3 | 0.205 → 0.212 | 0.320 → 0.324 |

|Δ| ≤ 0.03, sign inconsistent. Not a truncation artifact (60% of examples keep the full antigen at `max_len=512`). Partner-chain context is likewise net-neutral-to-negative (overhead for the small model). **Interpretation:** binding information is geometric, not linear-sequence — so the next lever is structure / cross-attention, not more sequence context.

**5. The homology leak was real but immaterial.** OAS's original random split leaked clonal near-duplicates (3.6% of the test set share a CDR3 cluster with train). The clustered-split fix removed it (→0%), and re-running confirmed the H3 core moved by **≤0.004** — the conclusions above are robust to it. (Now the default; see [`preprocess_oas_paired.py`](dataset/preprocess_oas_paired.py).)

> **Bottom line:** sequence-only CDR infilling saturates at ~0.32 H3 core regardless of model, context, or dataset quality — which is exactly what motivates the shift to a **structure objective** (see Roadmap).

---

## 🧠 Architectural Feature Details

### 1. Task-Agnostic Masking (The "LEGO" Principle)
The `AntibodyInfillingDataset` separates sequences into `pt1` (left context), `mask` (target gap), and `pt2` (right context). The `target_region` string alone chooses the loop — `h1/h2/h3` mask a heavy CDR, `l1/l2/l3` mask the *light* chain's CDR (legacy `cdr*`/`lcdr*` spellings still parse). Switching target requires changing zero core tokenization logic. Labels are strictly length-enforced to prevent cross-entropy dimension mismatches.

### 2. Pluggable Backbones (the extensibility core)
The model registry [`backbones.py`](backbones.py) holds one `Backbone` entry per model family; `train.py`, `predict.py`, and `dataset.py` never branch on model type — they all go through `resolve(model_type)`, which returns the matching backbone and a uniform `TokenizerAdapter`. The same pipeline trains the from-scratch transformer, fine-tunes ESM-2, or fine-tunes any Hugging Face masked-LM just by changing the `"model_type"` string.

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
- **New model** → one `Backbone` entry in [`backbones.py`](backbones.py) (§2) — `train.py`/`predict.py`/`dataset.py` never learn about it.
- **New target loop** → change the `target_region` string (§1); masking logic is untouched.
- **New context** → one boolean flag (`include_partner_chain`, `include_antigen`), resolved symmetrically (§3–4).
- **New dataset source** → one `(substring, tag)` pair in `train.py`'s `DATASET_TAGS` registry (`oas → OAS`, `sabdab2 → SD2`), so every run's output-dir name states its source explicitly instead of leaving it silently implied; an unrecognised path is tagged `DS` **with a logged warning** rather than left blank.

The guiding constraint: **conventions are declared explicitly, never inferred.** The predecessor overloaded a single `is_esm` boolean to mean three different things and silently mishandled non-ESM models; each backbone now states its `position_type` and `uses_special_tokens` up front. This isn't free — protein-BERT models (IgBert/ProtBert) still needed their own entry for residue-spacing and tokenizer/loader quirks (§2) — but the cost is paid once, in the registry, not scattered across the pipeline.

---

## ⚠️ Current Issues 
*(current as of 2026-07-15; update per commit)*

### Open problems
- **Evaluation is narrow.** Only accuracy + loss (+ the anchor-excluded core). No diversity / novelty / perplexity — and recovery *rewards copying the native*, the opposite of design. This is the main blocker on judging the design objective.
- **Decoder is argmax-only.** [`predict.py`](predict.py) decodes greedily per masked position — no temperature/sampling, no joint decoding — so no generative evaluation is possible yet (gates the metrics above).
- **Single-seed runs.** Small differences (e.g. light chain hurts scratch ~0.01, neutral for ESM-2) need multiple seeds + error bars before they can be claimed.
- **Structure not yet available.** SAbDab2 preprocessing parses *sequences* only; the Phase-2 structure objective needs a coordinate-parsing step (or predicted structures via IgFold/ABlooper).

### How to read the numbers 
- **Report core, not whole-span** — whole-span is anchor-inflated on H3 (Findings §3).
- **Read best-epoch, not last-epoch** — ESM-2 overfits on SAbDab2/H3; `load_best_model_at_end` saves the peak, but the last CSV row can sit well below it.
- **Never plot OAS and SAbDab2 on shared axes** — different test sets; the `_OAS`/`_SD2` run tags enforce it, and per-dataset charts live in `results/pngs/*_{oas,sd2}_*.png`.

### Changelog
- **2026-07-16** Updated roadmap (checklists)
- **2026-07-15** -- Polished README (half-manually), updated roadmap (pending manual edit)

---

## ⏭️ Roadmap

> **Objective-first:** the target is **CDR-H3 design**. **Phase 1** = a leakage-safe *sequence benchmark* (done — establishes the ~0.32 core ceiling; see Findings). **Phase 2** = *structure-conditioned* design (RMSD-to-native; exact target to lock with advisor). Everything below is a tool toward that objective.

**General — next steps**
- [ ] Sampler in `predict.py` (temperature / top-k) + generative metrics — **gating item**: recovery can't judge design
- [ ] Multi-seed the key H3 configs (error bars on the ±0.01 model/context deltas)
- [ ] Lock the Phase-2 objective + primary metric with advisor

**Datasets**
- OAS
  - [x] 21,106 paired sequences (`1287169_1_Paired_All.csv`)
  - [ ] Remaining paired sequences (→ 1M+ scale)
  - [ ] Unpaired sequences
- SAbDab2
  - [x] 3,984 paired Ab with resolved antigen
  - [ ] Remaining ~10k+ structures without antigen
- Other sources
  - [ ] SAbDab (full) / Thera-SAbDab

**Preprocessing**
- [x] Bio-Object JSON schema (14 IMGT regions)
- [x] Partner-chain context (light ↔ heavy)
- [x] Antigen context (SAbDab2)
- [x] Clustered split — homology-leak-safe (OAS)
- [ ] Structure / coordinate parsing (for Phase 2)
- [ ] Unpaired-OAS preprocessor (`preprocess_oas_unpaired.py`)

**Training / methods**
- Sequence-only
  - [x] Masked-LM infilling objective
  - Models
    - [x] From-scratch transformer (baseline — *not* pretrained)
    - [x] ESM-2 (general PLM)
    - [x] IgBert (antibody PLM)
    - [x] ProtBert (general-protein PLM)
    - [ ] AbLang2  *(low priority — model choice barely moves OAS; see Findings)*
    - [ ] IgLM  *(autoregressive — needs its own generation/scoring path)*
- Structure-aware  *(Phase 2)*
  - [ ] Invariant Point Attention (IPA)
  - [ ] E(n)-equivariant GNNs
  - [ ] Diffusion (dyMEAN / DiffAb-informed)
  - [ ] Antigen cross-attention / epitope-contact signal  *(the flat-sequence version was inert)*

**Evaluation**
- [x] Sequence recovery (AAR) — whole-span
- [x] Anchor-excluded **core** recovery (+ [`rescore.py`](rescore.py))
- [ ] Perplexity / pseudo-perplexity
- [ ] Diversity & novelty  *(needs the sampler above)*
- [ ] Structure metrics — RMSD, TM-score  *(Phase 2)*

**Long-term objectives / deliverables**
- Leakage-safe **benchmark** for CDR-infilling methods (the Phase-1 result)
- Structure-conditioned CDR-H3 design
- Integrated sequence + structure design framework
- → benchmark paper and/or a novel antibody-design method