# 🧬 CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** July 3, 2026  
**Status:** Custom Transformer + ESM-2 + CDR H1-H3 Infilling + Light Chain Context + SAbDab2 Antigen Context

## 📌 Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences given the surrounding framework context. 

Unlike standard text-based string models, this framework treats antibodies as **Structured Biological Objects**. By cleanly separating the 14 IMGT-defined regions (Frameworks 1-4 and CDRs 1-3 for both Heavy and Light chains), the codebase scales seamlessly from simple CDR-H3 infilling to multi-loop predictions, light-chain aware generation, antigen-aware generation, and eventually 3D structural embeddings.

---

## 🏗️ Repository Structure

```text
CDRSeqGen/
├── configs/                    # Full 3x2x3 matrix: loop (h1/h2/h3) x model (scratch/esm2) x context ([none]/_lc/_ag)
│   ├── h1_scratch.json / h1_scratch_lc.json / h1_scratch_ag.json / h1_esm2.json / h1_esm2_lc.json / h1_esm2_ag.json
│   ├── h2_scratch.json / h2_scratch_lc.json / h2_scratch_ag.json / h2_esm2.json / h2_esm2_lc.json / h2_esm2_ag.json
│   └── h3_scratch.json / h3_scratch_lc.json / h3_scratch_ag.json / h3_esm2.json / h3_esm2_lc.json / h3_esm2_ag.json
│       # _lc = paired light chain injected as context. _ag = SAbDab2 antigen + light chain injected as context.
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
├── dataset.py                  # Tokenizers, Dataset logic, and Collator
├── env.sh                      # One-click environment setup script
├── model.py                    # Unified Factory (Scratch Transformer & ESM-2)
├── predict.py                  # Inference script for generating novel loops
├── train.py                    # HF Trainer pipeline with automated CSV/PNG metric extraction
├── run_all.sh                  # Runs the OAS and/or SAbDab2-antigen experiment suites sequentially on one GPU
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

**Run an Automated Experiment Suite:**
`run_all.sh` takes a GPU id and an optional suite name (`oas` | `antigen` | `all`, default `all`). It fails fast with a clear message if the processed data a suite needs isn't there yet, rather than crashing mid-matrix. Scratch runs go first within each suite since they're cheap, so a broken config fails fast before burning time on ESM-2. If one run fails, it safely skips to the next one.
```bash
./run_all.sh 4            # both suites (12 OAS + 6 antigen configs)
./run_all.sh 4 oas        # just the original 3x2x2 OAS matrix
./run_all.sh 4 antigen    # just the 3x2 SAbDab2 antigen-conditioned matrix
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
The `AntibodyInfillingDataset` separates sequences into `pt1` (left context), `mask` (target gap), and `pt2` (right context). Transitioning from predicting `CDR-H3` to `CDR-H1` requires changing zero core tokenization logic. Labels are strictly length-enforced to prevent cross-entropy dimension mismatches.

### 2. Universal Model Interface
The `model.py` factory outputs Hugging Face `PreTrainedModel` objects. The exact same pipeline trains custom Transformers from scratch OR fine-tunes Meta's `ESM-2` models just by changing the `"model_type"` string. Special tokens (`<cls>`, `<eos>`) are dynamically managed based on the active tokenizer.

### 3. Light Chain Context Injection
Biologically, Heavy and Light chains physically interact. The framework dynamically prepends the paired Light Chain sequence to `pt1` (separated by `<eos>`) when `"include_light_chain": true` is passed, enriching structural context.

### 4. Antigen Context Injection (SAbDab2)
Same "LEGO" pattern as light-chain injection, one layer further out: when `"include_antigen": true`, the bound antigen's resolved sequence is prepended *ahead of* the light chain, giving the stacking order `antigen <sep> light <sep> heavy_pt1`. Multi-chain antigens are joined with `X` (the standard "unknown residue" convention) as a soft chain-break marker, since the amino-acid tokenizer has no token for a literal chain separator.

Antigen sequences (up to ~2,300 residues for large multi-chain complexes) made the pre-existing "no truncation" gap in `AntibodyInfillingDataset` actually bite, so this feature also added a real length cap: `pt1` is truncated from the *left* to fit `max_len`, dropping the most distal context first (antigen, then light chain) and always preserving the heavy-chain residues immediately before the masked loop.

This data comes from `dataset/preprocess_sabdab2.py`, which is a *different, much smaller* dataset than the OAS-derived one — 3,985 paired-chain (heavy+light) structures with a resolved protein/peptide antigen (3,235 train / 750 test), versus OAS's repertoire-scale sequencing data. `train.py` detects the `"split"` field this preprocessing script attaches to every entry and uses SAbDab2's own leakage-safe split instead of drawing a fresh random one (OAS-derived data has no such field and is unaffected).

### 5. Automated Tracking & Evaluation
*   **TensorBoard:** Live tracking via `"report_to": "tensorboard"`.
*   **Metric Extraction:** At the end of training, `train.py` automatically parses Hugging Face logs, dumps `epoch_metrics.csv`, and generates standalone `.png` curves for Loss and Sequence Recovery Rate.

---

## ⚠️ Known Issues & Evaluation Caveats (as of July 3, 2026)

**`eval_accuracy` is inflated for CDR3 by anchor residues.** The metric averages per-residue accuracy over the *entire* masked span, but the 2-3 residues at each end of CDR-H3 (adjacent to the FWR3/FWR4 junction) are largely determined by V/J-gene identity and are predicted with 78-100% accuracy regardless of model. The truly hypervariable middle of the loop — the part actually being "designed" — sits at only 13-35% accuracy. Position-resolved analysis on the `h3_esm2` run:

| Position from start | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 100% | 89% | 35% | 21% | 20% | 14% |

Verified full-matrix final-epoch numbers: **H1/H2 are easy for every model/config combo (97-99.6%)**; **H3 is hard for every combo (~44-46%), with ESM-2 giving essentially zero benefit over the scratch model**, and light-chain context slightly *hurting* H3 rather than helping. Before drawing conclusions from any CDR3 run, report position-stratified or anchor-excluded recovery, not the raw average.

> ⚠️ **The above ESM-2 numbers predate the `collate_fn` fix below and should be treated as unreliable.** Every ESM-2 training/eval batch (train and eval both go through the same `data_collator=collate_fn`) had its `<cls>` token wrongly zeroed out of `attention_mask` for every example. It's plausible this — not an inherent ESM-2/scratch parity — is why "ESM-2 gives essentially zero benefit" was observed. The full ESM-2 matrix should be re-run and these conclusions re-checked before citing them further.

**Two latent bugs found — fixed 2026-07-03:**
- `predict.py`'s `generate_sequence()` built ESM-2 inputs as `pt1 + mask + pt2`, omitting the `<cls>`/`<eos>` tokens that `dataset.py`'s training-time construction (`[cls] + pt1 + mask + pt2 + [eos]`) includes, causing a train/inference mismatch. Fixed by mirroring the exact training-time construction (and `start_mask_idx` offset) for the ESM-2 branch.
- `collate_fn` padded `input_ids` with a hardcoded `0`. For ESM-2's tokenizer, `0` is also the `<cls>` token id, so the legitimate leading `<cls>` token was zeroed out of `attention_mask` for every example. Fixed by adding a `pad_token_id` parameter (default `0`, correct for the scratch tokenizer); `train.py` now resolves and passes the real pad id for whichever tokenizer is active (`tokenizer.pad_token_id` for ESM-2, `tokenizer.pad_id` for the scratch tokenizer) via `functools.partial`.

**SAbDab2 antigen-conditioned runs are unvalidated beyond a smoke test.** The `_ag` configs have only been confirmed to train and infer without crashing (correct split sizes, no length-overflow) — no real recovery-rate numbers have been generated yet, and the H3 anchor-residue inflation caveat above almost certainly applies here too. The left-truncation policy for long antigens (drop the antigen first, keep the heavy-chain-proximal context) is a first-pass heuristic, not something validated against alternatives (e.g. keeping only the epitope-proximal antigen residues). Also note every `_ag` config bundles `include_light_chain` and `include_antigen` together — there's no isolated "antigen-only" config yet, so any effect measured against the `_lc` baseline currently can't distinguish "antigen helped" from "light chain helped."

---

## ⏭️ Roadmap

**Immediate next steps (antigen-conditioning follow-up):**
- [ ] Run the full antigen matrix — `./run_all.sh <gpu> antigen` — and compare `eval_accuracy`/loss curves against the matched `_lc` baselines with `linegraph_plotter.py`.
- [ ] Apply the position-stratified / anchor-excluded accuracy metric (below) to the antigen H3 runs specifically before trusting any raw recovery number — the anchor-inflation issue found for OAS H3 almost certainly recurs here.
- [ ] Add antigen-only configs (`include_antigen: true`, `include_light_chain: false`) so antigen and light-chain effects can be isolated instead of always measured bundled together.
- [ ] Given SAbDab2 is ~2 orders of magnitude smaller than OAS (3,985 vs. the OAS matrix's scale), try fine-tuning from an OAS-pretrained H3 checkpoint rather than training the antigen configs from scratch — plain from-scratch training on ~3.2k examples is likely to underfit.
- [ ] Revisit the naive left-truncation policy for long/multi-chain antigens (currently drops the antigen first, keeps heavy-chain-proximal context) once there's a recovery-rate baseline to test alternatives against.

**Near-term:**
- [ ] Add a position-stratified / anchor-excluded accuracy metric alongside the current average.
- [x] Fix the two latent bugs above. Fixed 2026-07-03.
- [ ] **Re-run the full ESM-2 matrix** (`./run_all.sh <gpu> oas`) now that the attention-mask bug is fixed, and re-check whether "ESM-2 gives essentially zero benefit over scratch" still holds.
- [ ] Add **AbLang** as a `model_type` backbone (BERT-style, drops into the existing factory like ESM-2 does).

**Mid-term:**
- [x] Antigen sequence-context conditioning (same pattern as light-chain injection — SAbDab2 antigen sequences). Landed 2026-07-03; see "Immediate next steps" above for validation still outstanding.
- [ ] Add **IgLM** as a baseline — it's the field-standard comparison for this exact framework-conditioned CDR-infilling task, but it's GPT-2/autoregressive, so it needs its own generation/scoring path rather than a drop-in `model_type`.

**Long-term:**
- [ ] **Data Expansion:** Scale OAS processing to 1M+ sequences with aggressive CD-HIT deduplication.
- [ ] Structural features via IgFold/ABlooper-derived CDR coordinates as additional conditioning input.
- [ ] **Antigen-Aware structure-conditioned generation**, informed by reference implementations (dyMEAN, DiffAb) rather than building equivariant/diffusion machinery from scratch.