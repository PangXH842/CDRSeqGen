# 🧬 CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** June 2, 2026  
**Status:** Phase 2 (Multi-Loop & Paired Chain Expansion)

## 📌 Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences given the surrounding framework context. 

Unlike standard text-based string models, this framework treats antibodies as **Structured Biological Objects**. By cleanly separating the 14 IMGT-defined regions (Frameworks 1-4 and CDRs 1-3 for both Heavy and Light chains), the codebase scales seamlessly from simple CDR-H3 infilling to multi-loop predictions, light-chain aware generation, and eventually 3D structural embeddings.

---

## 🏗️ Repository Structure

```text
CDRSeqGen/
├── configs/                    # JSON config files for training runs
│   ├── h1_scratch.json
│   ├── h3_esm2.json
│   ├── h3_esm2_lc.json 
│   └── h3_scratch.json
├── dataset/                    
│   ├── data/                   # Ignored in git
│   │   ├── raw/                # Raw .csv.gz from OAS
│   │   └── processed/          # Processed .json biological objects
│   ├── preprocess_paired.py    # Data cleaning and IMGT formatting
│   └── README.md               # Dataset-specific documentation
├── results/                    # Auto-generated timestamped model checkpoints
├── test/                       # JSON files containing curated sequences for benchmarking
│   └── test_h3.json
├── dataset.py                  # Tokenizers, Dataset logic, and Collator
├── env.sh                      # One-click environment setup script
├── model.py                    # Unified Factory (Scratch Transformer & ESM-2)
├── predict.py                  # Inference script for generating novel loops
├── train.py                    # HF Trainer pipeline with Dataclass args
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

### Step 2: Training
We use `.json` configuration files to manage hyperparameters cleanly. The trainer automatically calculates **Sequence Recovery Rate** (Per-Residue Accuracy) on the masked loop at the end of every epoch.

You can toggle exactly which loop to target and whether to include the paired Light Chain context natively in the JSON config:
```json
{
    "model_type": "facebook/esm2_t12_35M_UR50D",
    "data_path": "dataset/data/processed/processed_paired_data.json",
    "target_region": "cdr3",
    "include_light_chain": true,
    "do_train": true
}
```
*(⚠️ **Note:** Always use `facebook/esm2...` checkpoints. Avoid `nvidia/esm2...` checkpoints as their state dictionary keys are incompatible with standard HuggingFace models).*

**Run the Trainer:**
```bash
python train.py configs/h3_esm2.json
```
*(Tip: Use `CUDA_VISIBLE_DEVICES=0` before the command to specify a target GPU).*

### Step 3: Batch Inference & Evaluation
Evaluate your trained model on a library of specific antibody frameworks using JSON test files. The model acts as a "Design Tool," filling the gap and comparing it to the ground truth (if provided).

**Run Batch Inference:**
```bash
python predict.py \
    --model_type scratch \
    --model_path ./results/latest \
    --test_file test_cases/sample_h3.json
```

**Example Output:**
```text
=================================================================
🧪 Test Case:             Jaffe_2022_Donor4 (Easy)
🔹 Context (Left / pt1):  EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAP...
🔹 Context (Right / pt2): WGQGTTVTVSS
🔹 Target Gap Length:     17
-----------------------------------------------------------------
✨ Predicted Infill:      E R W G V H S S S Y Y R H G M D V
🍀 Ground Truth:          E R W G I H S S S W Y R H G M D V
🎯 Alignment Map:         | | | | . | | | | . | | | | | | |
📈 Sequence Recovery:     88.2%
=================================================================
```

---

## 🧠 Key Architectural Features

### 1. Task-Agnostic Masking (The "LEGO" Principle)
The `AntibodyInfillingDataset` separates sequences into `pt1` (left context), `mask` (target gap), and `pt2` (right context). 
*   **Why?** This ensures perfect mathematical alignment of the `-100` CrossEntropy labels. Transitioning from predicting `CDR-H3` to `CDR-H1` leaves the core tokenization engine untouched.

### 2. Universal Model Interface
The `model.py` factory outputs Hugging Face `PreTrainedModel` objects. 
*   **Why?** The exact same `train.py` loop can train a custom Transformer from scratch OR fine-tune Meta's massive `ESM-2` models just by changing the `"model_type"` string. Special tokens (`<cls>`, `<eos>`) are strictly managed based on the active tokenizer.

### 3. Light Chain Context Injection
Biologically, the Heavy and Light chains physically interact. The framework dynamically prepends the paired Light Chain sequence to `pt1` (separated by `<eos>` or `<sep>`) when `include_light_chain: true` is passed, enriching structural context without altering downstream label alignments.

### 4. Length-Conditioned Constraint Decoding
During inference, the model is told the gap size (`[MASK] * target_len`). The inference script dynamically applies logit constraints (`-inf`) to special tokens to force biologically valid amino acid generation, preventing early-training hallucination.

---

## ⏭️ Future Roadmap (Phase 3 & 4)
- [ ] **Data Expansion:** Scale OAS processing to 1M+ sequences with aggressive CD-HIT deduplication.
- [ ] **Phase 3:** SAbDab Integration for **Antigen-Aware** conditional generation via cross-attention.
- [ ] **Phase 4:** 3D coordinate injection utilizing the preserved IMGT numbering dictionaries.
