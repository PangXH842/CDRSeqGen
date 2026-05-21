# CDRSeqGen: Modular Antibody Infilling Framework

**Date Updated:** 20260514 

## 📌 Project Overview
CDRSeqGen is a computational immunology framework designed to predict antibody loop sequences (starting with **CDR-H3**) given the surrounding framework context. 

Unlike standard text-based string models, this framework treats antibodies as **Structured Biological Objects**. By cleanly separating the 14 IMGT-defined regions (Frameworks 1-4 and CDRs 1-3 for both Heavy and Light chains), the codebase is built to scale seamlessly to multi-loop predictions, antigen-awareness, and 3D structural embeddings.

---

## 🏗️ Repository Structure

```text
CDRSeqGen/
├── configs/                    # JSON config files for training runs
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
*(Note: If you encounter the `iJIT_NotifyEvent` error on Linux, run `conda install "mkl<2024.1" -c conda-forge -y` to fix the PyTorch CPU backend bug).*

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
We use `.json` configuration files to manage hyperparameters cleanly, avoiding terminal command clutter. 

**Create a config file (e.g., `configs/h3_scratch.json`):**
```json
{
    "model_type": "scratch",
    "data_path": "data/processed/processed_paired_data.json",
    "output_dir": "./results",
    "do_train": true,
    "do_eval": true,
    "per_device_train_batch_size": 32,
    "learning_rate": 1e-4,
    "num_train_epochs": 10
}
```
**Run the Trainer:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py configs/h3_scratch.json
```
*Results, logs, and config snapshots are automatically saved to a timestamped folder like `./results/scratch_20260514_153022/`.*

### Step 3: Inference (Prediction)
Generate a loop sequence for a given framework. The model acts as a "Design Tool" where you specify the desired length of the gap to fill.

```bash
python predict.py \
    --model_type scratch \
    --model_path ./results/scratch_20260514_153022/checkpoint-500 \
    --target_length 15 \
    --pt1 "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR" \
    --pt2 "WGQGTTVTVSS"
```

---

## 🧠 Key Architectural Features

### 1. Task-Agnostic Masking (The "LEGO" Principle)
The `AntibodyInfillingDataset` separates sequences into `pt1` (left anchor), `mask` (target gap), and `pt2` (right anchor). 
*   **Why?** This ensures 100% mathematical alignment of the `-100` CrossEntropy labels. It also means transitioning from predicting `CDR-H3` to `CDR-H1` only requires changing 3 lines of variable assignment, leaving the core tokenization engine untouched.

### 2. Unified HF Model Interface
The `model.py` factory outputs Hugging Face `PreTrainedModel` objects. 
*   **Why?** The exact same `train.py` loop can train a 4-layer custom Transformer from scratch OR fine-tune Meta's 650-parameter `ESM-2` just by changing the `"model_type"` string in the config file.

### 3. Length-Conditioned Generation
During Phase 1, the model does not guess the length. We pass `[MASK] * target_len`. 
*   **Why?** In therapeutic design, loop length is often a known prerequisite constraint based on the epitope pocket. Passing the length provides a rigid spatial anchor connecting FW3 and FW4.

---

## ⏭️ Future Roadmap
- [ ] **Phase 1.5:** Length-Discovery mode (using `<EOS>` tokens to let the model guess optimal loop length).
- [ ] **Phase 2:** Inject Light Chain (`pt1` expanded to include V_L) for paired structural context.
- [ ] **Phase 3:** SAbDab Integration for **Antigen-Aware** conditional generation via cross-attention.
- [ ] **Phase 4:** 3D coordinate injection utilizing the preserved IMGT numbering dictionaries.