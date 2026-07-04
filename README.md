# 🧬 CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** July 4, 2026  
**Status:** Custom Transformer + ESM-2 + CDR H1-H3 Infilling + Light Chain Context + SAbDab2 Antigen Context. Full 3-CDR × 2-dataset matrix (24 runs) trained end-to-end; antigen ablation complete (result: no measurable antigen benefit yet — see Known Issues).

## 📌 Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences given the surrounding framework context. 

Unlike standard text-based string models, this framework treats antibodies as **Structured Biological Objects**. By cleanly separating the 14 IMGT-defined regions (Frameworks 1-4 and CDRs 1-3 for both Heavy and Light chains), the codebase scales seamlessly from simple CDR-H3 infilling to multi-loop predictions, light-chain aware generation, antigen-aware generation, and eventually 3D structural embeddings.

---

## 🏗️ Repository Structure

```text
CDRSeqGen/
├── configs/                    # loop (h1/h2/h3) x model (scratch/esm2) x context ([none]/_lc/_ag/_ag_lconly)
│   ├── h1_scratch.json / h1_scratch_lc.json / h1_scratch_ag.json / h1_esm2.json / h1_esm2_lc.json / h1_esm2_ag.json
│   ├── h2_scratch.json / h2_scratch_lc.json / h2_scratch_ag.json / h2_esm2.json / h2_esm2_lc.json / h2_esm2_ag.json
│   ├── h3_scratch.json / h3_scratch_lc.json / h3_scratch_ag.json / h3_esm2.json / h3_esm2_lc.json / h3_esm2_ag.json
│   └── h3_scratch_ag_lconly.json / h3_esm2_ag_lconly.json  # SAbDab2 + light chain, antigen OFF
│       # DATASET MATTERS: base/_lc train on OAS; _ag AND _ag_lconly train on SAbDab2 (different data).
│       # _lc = OAS + light chain. _ag = SAbDab2 + light chain + antigen.
│       # _ag_lconly = SAbDab2 + light chain, NO antigen -- the matched ablation baseline for _ag
│       #              (the ONLY valid same-test-set comparison for isolating the antigen effect).
│       # Run names carry an explicit dataset tag (_OAS / _SD2) resolved from an extensible
│       # DATASET_TAGS registry in train.py -- an unrecognised source is tagged _DS with a warning,
│       # never left blank. (Runs before 2026-07-04 used the older implicit scheme: no OAS tag, _SD.)
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
The `AntibodyInfillingDataset` separates sequences into `pt1` (left context), `mask` (target gap), and `pt2` (right context). Transitioning from predicting `CDR-H3` to `CDR-H1` requires changing zero core tokenization logic. Labels are strictly length-enforced to prevent cross-entropy dimension mismatches.

### 2. Pluggable Backbones (the extensibility core)
Model choice is a **registry**, not a chain of `if "esm2" in name` checks. [`backbones.py`](backbones.py) holds one `Backbone` entry per model family; `train.py`, `predict.py`, and `dataset.py` never branch on model type — they all go through `resolve(model_type)`, which returns the matching backbone and a uniform `TokenizerAdapter`. The same pipeline trains the from-scratch transformer, fine-tunes ESM-2, or fine-tunes any Hugging Face masked-LM just by changing the `"model_type"` string.

Each backbone declares two conventions **explicitly** instead of letting them be guessed from the tokenizer (the previous code overloaded a single `is_esm` boolean to mean three different things, which silently mishandled non-ESM models):
- **`position_type`** (`"absolute"` | `"relative"`) — controls where antigen context is injected (see §4). Absolute-position models (the scratch transformer, BERT-style protein LMs) need it appended after the target; relative/rotary models (ESM-2) can take it prepended.
- **`uses_special_tokens`** — whether inputs are wrapped as `[cls] … [end]`. The `TokenizerAdapter` also resolves the ESM-vs-BERT separator difference generically (`<eos>` when the tokenizer has one, else `<sep>`), so BERT-style tokenizers with `eos_token_id = None` work without special-casing.

**Adding a new PLM (e.g. AbLang, ProtBert, ESM-C) is one registry entry** — often *zero*, because any `AutoModelForMaskedLM` + `AutoTokenizer` with absolute positions is already covered by the generic `hf_encoder` fallback (just pass its hub id as `model_type`). Add an explicit `register(Backbone(...))` above the fallback only when a model needs different conventions (e.g. relative positions, or a bespoke tokenizer/model class). No edits to `train.py`/`predict.py`/`dataset.py` are needed either way.

### 3. Light Chain Context Injection
Biologically, Heavy and Light chains physically interact. The framework dynamically prepends the paired Light Chain sequence to `pt1` (separated by `<eos>`) when `"include_light_chain": true` is passed, enriching structural context.

### 4. Antigen Context Injection (SAbDab2)
Same "LEGO" pattern as light-chain injection, but **the injection side is keyed on the backbone's `position_type`** (§2) — this isn't cosmetic, it fixes a real bug (see Known Issues below). For **relative**-position models (ESM-2), the antigen is prepended *ahead of* the light chain into `pt1`: `antigen <sep> light <sep> heavy_pt1 <mask> pt2`. For **absolute**-position models (the scratch transformer, and BERT-style PLMs), the antigen is instead *appended* into `pt2`, with the light chain still prepended to `pt1`: `light <sep> heavy_pt1 <mask> fwr4 <sep> antigen`. Multi-chain antigens are joined with `X` (the standard "unknown residue" convention) as a soft chain-break marker, since the amino-acid tokenizer has no token for a literal chain separator.

Antigen sequences (up to ~2,300 residues for large multi-chain complexes) made the pre-existing "no truncation" gap in `AntibodyInfillingDataset` actually bite, so this feature also added a real length cap, applied on whichever side carries the antigen: truncated from the left for relative-position `pt1`, from the right for absolute-position `pt2` — always dropping antigen context first and keeping the heavy-chain residues immediately adjacent to the masked loop.

This data comes from `dataset/preprocess_sabdab2.py`, which is a *different, much smaller* dataset than the OAS-derived one — 3,984 paired-chain (heavy+light) structures with a resolved protein/peptide antigen (3,234 train / 750 test), versus OAS's repertoire-scale sequencing data. `train.py` detects the `"split"` field this preprocessing script attaches to every entry and uses SAbDab2's own leakage-safe split instead of drawing a fresh random one (OAS-derived data has no such field and is unaffected). Both preprocessors report a per-stage filter breakdown (how many rows were dropped for non-productive/wrong-type/malformed-numbering/missing-regions) so the final count is auditable rather than a black box.

**Explicit dataset tagging (extensible).** Every run's output-dir name states its data source so runs are never silently confused on a chart (the old scheme left OAS *untagged*, so an `_SD`-less name ambiguously read as "OAS" or "self-distillation" or "std-dev"). `train.py` resolves the tag from a module-level `DATASET_TAGS` registry — an ordered `(substring, tag)` list (`sabdab2 → SD2`, `paired → OAS`), first match wins — via `dataset_tag(data_path)`. Adding a new source is one line; an unrecognised path is tagged `DS` **with a logged warning** rather than left blank. Runs produced before 2026-07-04 predate this and carry the old implicit tags (`_SD`, no OAS tag).

### 5. Automated Tracking & Evaluation
*   **TensorBoard:** Live tracking via `"report_to": "tensorboard"`.
*   **Metric Extraction:** At the end of training, `train.py` automatically parses Hugging Face logs, dumps `epoch_metrics.csv`, and generates standalone `.png` curves for Loss and Sequence Recovery Rate.

---

## ⚠️ Known Issues & Evaluation Caveats (as of July 4, 2026)

**`eval_accuracy` is inflated for CDR3 by anchor residues.** The metric averages per-residue accuracy over the *entire* masked span, but the 2-3 residues at each end of CDR-H3 (adjacent to the FWR3/FWR4 junction) are largely determined by V/J-gene identity and are predicted with 78-100% accuracy regardless of model. The truly hypervariable middle of the loop — the part actually being "designed" — sits at only 13-35% accuracy. Position-resolved analysis on the `h3_esm2` run:

| Position from start | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 100% | 89% | 35% | 21% | 20% | 14% |

Verified full-matrix final-epoch numbers: **H1/H2 are easy for every model/config combo (97-99.6%)**; **H3 is hard for every combo (~44-46%), with ESM-2 giving essentially zero benefit over the scratch model**, and light-chain context slightly *hurting* H3 rather than helping. Before drawing conclusions from any CDR3 run, report position-stratified or anchor-excluded recovery, not the raw average.

> ⚠️ **The above ESM-2 numbers predated the `collate_fn` fix below.** Every ESM-2 batch had its `<cls>` token wrongly zeroed out of `attention_mask`. **The full post-fix matrix was re-run 2026-07-03 (all 3 CDRs × 2 datasets, 24 runs), and the result stands: on OAS, ESM-2 still gives essentially zero benefit over the scratch model** — H3 scratch 0.458 vs ESM-2 0.455/best 0.465; H1/H2 both saturate at 0.98-0.996 for either. So OAS is large enough that the from-scratch transformer already learns the task and pretraining adds nothing. **On the much smaller SAbDab2 data the story flips hard — pretraining helps a lot:** H1 0.49→0.72, H2 0.37→0.60, H3 0.21→0.32 (scratch→ESM-2, lc-only). The "ESM-2 doesn't help" conclusion is therefore *dataset-size-specific*, not general.

> ⚠️ **ESM-2 overfits on SAbDab2.** The ESM-2 SAbDab2 curves peak then decline over 10 epochs (H3 esm2+lc best 0.361 → final 0.320; H2 0.613 → 0.599), symptomatic of a ~3.2k-example train set. `load_best_model_at_end` means the *saved* checkpoint is the peak, but read best-epoch, not final-epoch, numbers for these runs, and consider early stopping.

**Two latent bugs found — fixed 2026-07-03:**
- `predict.py`'s `generate_sequence()` built ESM-2 inputs as `pt1 + mask + pt2`, omitting the `<cls>`/`<eos>` tokens that `dataset.py`'s training-time construction (`[cls] + pt1 + mask + pt2 + [eos]`) includes, causing a train/inference mismatch. Fixed by mirroring the exact training-time construction (and `start_mask_idx` offset) for the ESM-2 branch.
- `collate_fn` padded `input_ids` with a hardcoded `0`. For ESM-2's tokenizer, `0` is also the `<cls>` token id, so the legitimate leading `<cls>` token was zeroed out of `attention_mask` for every example. Fixed by adding a `pad_token_id` parameter (default `0`, correct for the scratch tokenizer); `train.py` now resolves and passes the real pad id for whichever tokenizer is active (`tokenizer.pad_token_id` for ESM-2, `tokenizer.pad_id` for the scratch tokenizer) via `functools.partial`.

**Found and fixed 2026-07-03: the scratch model couldn't learn from antigen context at all.** First real (non-smoke-test) runs showed `h3_scratch_ag`'s `eval_accuracy` completely flat at ~0.14-0.15 for all 10 epochs — not slow convergence, no movement whatsoever, versus `scratch`/`scratch+lc` reaching ~0.44. Meanwhile `h3_esm2_ag` trained fine (settling around ~0.34-0.38, a reasonable amount below `esm2+lc`'s ~0.45), which ruled out "the antigen task is just too hard/data-starved" as the explanation, since both configs train on the same data.

Root cause: `CustomAntibodyModel` uses learned **absolute** position embeddings (`model.py`'s `pos_embedding`), while ESM-2 uses a relative/rotary scheme. Light-chain injection prepends a length-stable segment (~105-120 residues) ahead of the heavy chain, so the masked CDR3 target lands at a roughly consistent absolute position across examples — learnable. Antigens vary enormously in length (mean ~254, std ~224, up to ~2,300 residues), so prepending one shifts the target to a wildly different absolute position on almost every example, a signal too unstable for ~3,235 examples over 10 epochs to learn. ESM-2's relative positional scheme doesn't have this failure mode.

Fix: for the scratch model only, the antigen is now appended *after* the target (into `pt2`) instead of prepended before everything (into `pt1`) — see "Antigen Context Injection" above. Since this is a non-causal, fully bidirectional encoder, moving the antigen doesn't hide any information, it just keeps the target's position anchored the same low-variance way `scratch+lc` already was. Result: `h3_scratch_ag`'s `eval_accuracy` now climbs monotonically to ~0.21 (0.144 → 0.173 → 0.187 → ... → 0.213) instead of sitting flat at ~0.145 — a real, if still partial, fix (still below `esm2+lc+ag`'s ~0.34-0.38, which is expected given the scratch model has no pretraining and less capacity).

Separately, `model.py`'s `CustomAntibodyModel.forward()` also previously ignored `attention_mask` entirely (attending over padding as if it were real content) — now fixed by passing it through as `src_key_padding_mask`. This didn't turn out to be the cause of the above flat-lining (confirmed empirically: applying it alone changed nothing), but it's a correct fix worth keeping regardless.

> 🚨 **The 6-line `h3_lc_ag_acc.png` comparison chart is not a valid comparison, and no "antigen helps / hurts" conclusion can be drawn from it.** The `_ag` runs train and evaluate on **SAbDab2** (3,985 crystallized structures, ~750-example test split); the base and `_lc` runs train and evaluate on **OAS** (21,106 repertoire sequences, random 10% split). These are different train *and* test distributions, so plotting them on shared axes and reading the gap as an antigen effect is confounded by the entire dataset change — SAbDab2's held-out set is simply a harder, structurally-distinct distribution. The `esm2` (0.45, OAS) → `esm2+lc+ag` (0.34, SAbDab2) drop is dominated by that swap, not by antigen conditioning. **The only clean way to isolate the antigen effect is same-model, same-SAbDab2-data, antigen toggled on vs off** — i.e. `_ag` vs the `_ag_lconly` baseline configs. **This ablation is now complete (2026-07-03/04) for all three CDRs, and the answer is that antigen conditioning as currently implemented provides no measurable benefit** (final-epoch `eval_accuracy` on the SAbDab2 test split, lc-only vs lc+antigen):
>
> | CDR | scratch lc-only → +ag | ESM-2 lc-only → +ag |
> |---|---|---|
> | H1 | 0.485 → 0.456 (**−0.029**) | 0.716 → 0.718 (+0.002) |
> | H2 | 0.370 → 0.365 (−0.005) | 0.599 → 0.595 (−0.004) |
> | H3 | 0.205 → 0.212 (+0.007) | 0.320 → 0.324 (+0.004) |
>
> Every delta is within ±0.03 and the sign is inconsistent — no signal. **This is not a truncation artifact:** at `max_len=512` the antigen budget is ~281 tokens vs a median antigen of 198 residues, and **60% of examples retain the full antigen** (median retention 100%), so the model is genuinely seeing the antigen and not using it. The likely causes are the data budget (~3.2k train examples is thin for learning antigen-conditioned generation) and/or the conditioning *mechanism* — prepending/appending the antigen as flat context and hoping self-attention discovers the epitope is weak; cross-attention to the antigen or an explicit contact/epitope signal is the more promising next lever. Runs now carry an explicit `_SD2`/`_OAS` dataset tag (see §4) so SAbDab2 and OAS runs can never be mistaken for each other on a chart.

**Still open:** the `_ag` configs still bundle light chain + antigen, so `_ag` vs `_ag_lconly` (not `_ag` vs the OAS `_lc`) is the comparison to plot. The H3 anchor-residue inflation caveat above almost certainly applies to antigen runs too and **still hasn't been checked position-stratified** — and note H1/H2's very high raw recovery (0.99 on OAS, 0.49-0.72 on SAbDab2) is exactly the regime where anchor inflation is most misleading, so those numbers especially need anchor-excluded re-analysis before they mean much. All three CDRs have now been trained end-to-end on both datasets (the 24-run matrix); H1/H2 antigen configs are no longer untrained.

**Config parameters were silently ignored — fixed 2026-07-03.** `get_model()` only received `model_type`/`vocab_size`, so `hidden_size` and `num_layers` set in any config were dropped and the `CustomAntibodyConfig` defaults (128 / 4) were always used — harmless only because every config happened to match those defaults. Additionally, the scratch model's `max_position_embeddings` was hardcoded to 512 and decoupled from `max_len`, so any config with `max_len > 512` would have crashed the scratch model on a full-length input (shape mismatch on the position-embedding add). Fixed: `train.py` now threads `hidden_size`, `num_layers`, and `max_len` into the model builder, and `max_position_embeddings` is sized as `max(max_len, 512)`.

**OAS and SAbDab2 splits are held to different leakage standards.** SAbDab2 uses its own sequence-identity-cluster split (`ab_ag_split`), which keeps clonally-related antibodies out of the test set. OAS uses a plain random `train_test_split` ([train.py](train.py)). Exact duplication in the current single-file OAS set is low (~0.9%), so this isn't catastrophic today — but a random split still leaks *clonal near-duplicates* (same V/J genes, CDR3s differing by a few somatic mutations) into both train and eval, which SAbDab2's clustering prevents. Net effect: **OAS eval numbers are optimistic relative to SAbDab2 eval**, independent of (and compounding) the different-test-set caveat above. This becomes a real correctness problem, not a mild one, once OAS scales to the multi-file 1M+ roadmap item — CD-HIT/identity-cluster dedup should be treated as a prerequisite for that scale-up, not a nice-to-have, and ideally OAS should move to a clustered split to match SAbDab2's rigor.

**Preprocessing hardening — 2026-07-03.** `preprocess_paired.py`'s bare `except: continue` (which would silently swallow *any* error, e.g. a renamed column, as "malformed numbering") was narrowed to the actual parse exceptions; both preprocessors now guard against null/empty region fields (a `NaN` region cell would otherwise become JSON `null` and crash the string concatenation in `dataset.py`) and print a per-stage filter breakdown. This dropped 1 degenerate SAbDab2 entry with an empty target CDR (3,985 → 3,984); OAS is unchanged.

---

## ⏭️ Roadmap

**Immediate next steps (antigen-conditioning follow-up):**
- [x] H3 scratch/ESM-2 antigen configs trained end-to-end and the scratch flat-lining bug (see Known Issues) diagnosed and fixed 2026-07-03.
- [x] Added the matched `_ag_lconly` ablation baselines (SAbDab2 + light chain, antigen off) needed for a valid same-test-set antigen comparison, 2026-07-03.
- [x] **Ran the valid antigen ablation (`_ag` vs `_ag_lconly`) to 10 epochs for all three CDRs and plotted per-dataset** (`results/comparisons/cdr{1,2,3}_sd2_*.png`), 2026-07-03/04. **Result: no measurable antigen benefit** (|Δ| ≤ 0.03, sign inconsistent — see Known Issues table). Confirmed *not* a truncation artifact (60% of examples keep the full antigen).
- [x] Ran H1/H2 antigen configs — the full 3-CDR × 2-dataset matrix (24 runs) is trained.
- [x] Made dataset tagging explicit + extensible (`DATASET_TAGS` registry, `_OAS`/`_SD2` tags), 2026-07-04.
- [ ] **Make antigen conditioning actually work — the open research question now.** The null result points at mechanism/data, not plumbing: (a) try **cross-attention to the antigen** (or an explicit epitope/contact-residue signal) instead of flat prepend/append context; (b) get more antigen-conditioned data — ~3.2k train examples is thin. This supersedes fiddling with the current injection.
- [ ] Apply the position-stratified / anchor-excluded accuracy metric (below) to the SAbDab2 runs before trusting any raw recovery number — the anchor-inflation issue found for OAS H3 almost certainly recurs, and matters most for the high-scoring H1/H2 numbers.
- [ ] `h3_scratch_ag` (~0.21) still trails `h3_esm2_ag` (~0.32) — the SAbDab2 results confirm pretraining is what carries this small-data regime, so **fine-tune the scratch antigen model from an OAS-pretrained `h3_scratch`/`h3_scratch_lc` checkpoint** rather than training from scratch.
- [ ] ~~Revisit the antigen truncation policy~~ — **deprioritized**: measured retention shows 60% of examples keep the full antigen (median 100%), so truncation is not what's limiting the antigen signal.

**Near-term:**
- [ ] Add a position-stratified / anchor-excluded accuracy metric alongside the current average.
- [x] Fix the two latent bugs above. Fixed 2026-07-03.
- [x] **Re-ran the full ESM-2 matrix** (2026-07-03, post attention-mask fix). Re-checked conclusion: "ESM-2 ≈ scratch" **holds on OAS** (large data), but **ESM-2 clearly beats scratch on SAbDab2** (small data) — the benefit is dataset-size-dependent, not absent (see Known Issues).
- [ ] Add early stopping / best-epoch reporting for the SAbDab2 runs (ESM-2 overfits past its peak there).
- [ ] Add **AbLang / AbLang2** as a backbone. Now a `backbones.py` registry entry (or possibly zero edits via the generic `hf_encoder` fallback) rather than surgery across `train.py`/`predict.py`/`dataset.py` — see §2. Verify its tokenizer is HF-compatible; if AbLang ships its own non-HF tokenizer, the entry just needs a small `load_tokenizer` shim. Being BERT-style (absolute positions), it correctly gets the append-to-`pt2` antigen treatment automatically.

**Mid-term:**
- [x] Antigen sequence-context conditioning (same pattern as light-chain injection — SAbDab2 antigen sequences). Landed 2026-07-03 and validated via the completed `_ag` vs `_ag_lconly` ablation (2026-07-04). Verdict: **implemented and correct, but not yet beneficial** — flat sequence-context conditioning gives no measurable lift; making it help is now an open item under "Immediate next steps".
- [ ] Add **IgLM** as a baseline — it's the field-standard comparison for this exact framework-conditioned CDR-infilling task, but it's GPT-2/autoregressive, so it needs its own generation/scoring path rather than a drop-in `model_type`.

**Long-term:**
- [ ] **Data Expansion:** Scale OAS processing to 1M+ sequences with aggressive CD-HIT deduplication.
- [ ] Structural features via IgFold/ABlooper-derived CDR coordinates as additional conditioning input.
- [ ] **Antigen-Aware structure-conditioned generation**, informed by reference implementations (dyMEAN, DiffAb) rather than building equivariant/diffusion machinery from scratch.