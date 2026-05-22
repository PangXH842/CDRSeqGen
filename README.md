# 🧬 CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** May 22, 2026  

## 📌 Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences (starting with **CDR-H3**) given the surrounding framework context. 

Unlike standard text-based string models, this framework treats antibodies as **Structured Biological Objects**. By cleanly separating the 14 IMGT-defined regions (Frameworks 1-4 and CDRs 1-3 for both Heavy and Light chains), the codebase is built to scale seamlessly to multi-loop predictions, antigen-awareness, and 3D structural embeddings.

---

## 🏗️ Repository Structure

```text
CDRSeqGen/
├── configs/                    # JSON config files for training runs
│   ├── h3_esm2.json
│   └── h3_scratch.json
├── dataset/                    
│   ├── data/                   # Ignored in git
│   │   ├── raw/                # Raw .csv.gz from OAS
│   │   └── processed/          # Processed .json biological objects
│   ├── preprocess_paired.py    # Data cleaning and IMGT formatting
│   └── README.md               # Dataset-specific documentation
├── results/                    # Auto-generated timestamped model checkpoints
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
python preprocess_paired.py \
    --input_file data/raw/1287169_1_Paired_All.csv \
    --output_file data/processed/processed_paired_data.json
```

### Step 2: Training
We use `.json` configuration files to manage hyperparameters cleanly. The trainer automatically calculates **Sequence Recovery Rate** (Per-Residue Accuracy) on the masked loop at the end of every epoch.

**Run the Trainer:**
```bash
python train.py configs/h3_scratch.json
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
The `AntibodyInfillingDataset` separates sequences into `pt1` (left anchor), `mask` (target gap), and `pt2` (right anchor). 
*   **Why?** This ensures perfect mathematical alignment of the `-100` CrossEntropy labels. Transitioning from predicting `CDR-H3` to `CDR-H1` requires changing only 3 lines of variable assignment, leaving the core tokenization engine untouched.

### 2. Universal Model Interface
The `model.py` factory outputs Hugging Face `PreTrainedModel` objects. 
*   **Why?** The exact same `train.py` loop can train a custom Transformer from scratch OR fine-tune Meta's 650M parameter `ESM-2` just by changing the `"model_type"` string in the config file. Special tokens (`<cls>`, `<eos>`) are strictly managed based on the active tokenizer.

### 3. Length-Conditioned Constraint Decoding
During inference, the model is told the gap size (`[MASK] * target_len`). The inference script dynamically applies logit constraints (`-inf`) to special tokens to force biologically valid amino acid generation, preventing early-training hallucination.

---
