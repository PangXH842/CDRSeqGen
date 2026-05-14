# 🧬 Antibody CDR-H3 Infiller (Phase 1)
Updated: 2026-05-13

This project is a computational immunology framework designed to predict the **CDR-H3** amino acid sequence of an antibody given its surrounding framework regions. It is built to be modular, supporting both custom architectures and large-scale Pre-trained Language Models (PLMs).

## 🗺️ Framework Overview

Phase 1 focuses on **Conditional Infilling**. 
*   **The Task:** Given an antibody heavy chain where the CDR-H3 is masked, the model must predict the correct amino acids for that loop.
*   **The Data:** Paired Human IgG sequences from the **Observed Antibody Space (OAS)**.

### Core Philosophy: "Bio-Object" over "Strings"
Instead of treating antibodies as simple text strings, we parse them into 14 distinct regions (FWs and CDRs). This allows the model to eventually handle Light-Chain context, Antigen context, and Structural embeddings without re-processing the raw data.

---

## 📂 Project Structure

| File | Purpose |
| :--- | :--- |
| `preprocess_paired.py` | Cleans raw OAS `.csv.gz` files into structured `.json` objects. |
| `dataset.py` | Contains the `AntibodyTokenizer`, `Dataset` logic, and `collate_fn` for padding. |
| `model.py` | A factory that switches between a **Custom Transformer** and **ESM-2**. |
| `train.py` | The main training script using the Hugging Face `Trainer` API. |
| `data/README.md` | (Internal) Detailed logic on OAS column extraction and filtering. |

---

## 🚀 How to Run

### 1. Preprocess the Data
Take a raw OAS file (e.g., `1287169_1_Paired_All.csv.gz`) and run:
```bash
python preprocess_paired.py --input_file data/raw/your_file.csv.gz --output_file data/processed_paired.json
```
*   **Filtering:** Only "Productive" and "Paired" antibodies are kept.
*   **Output:** A JSON list containing 14 regional sequences and IMGT numbering for every record.

### 2. Train the Model
You have two options for training. Both use the same `Dataset` class but different Tokenizers.

#### Option A: Custom Transformer (Scratch)
Best for rapid prototyping and understanding loop-framework dependencies.
```bash
python train.py --model_type scratch --epochs 10 --batch_size 32 --lr 1e-4
```

#### Option B: ESM-2 (Pre-trained)
Best for high-accuracy biological modeling using Meta's protein language model.
```bash
python train.py --model_type facebook/esm2_t6_8M_UR50D --epochs 5 --batch_size 8 --lr 5e-5
```

---

## 🧠 Technical Highlights

### Fixed-Length Masking
In Phase 1, we use a **length-aware mask**. If a CDR-H3 is 12 amino acids long, we provide the model with 12 `<mask>` tokens. This forces the model to focus on sequence patterns given a known loop size.

### The `-100` Label Trick
To ensure the model doesn't waste capacity learning the Framework regions (which it can already see), we set the labels for all framework positions to `-100`. PyTorch’s `CrossEntropyLoss` ignores these indices, calculating gradients **only** for the predicted CDR-H3 tokens.

### Unified Interface
Both the "Scratch" model and the "ESM-2" model follow the Hugging Face `PreTrainedModel` interface. They both return a dictionary containing `{"loss": ..., "logits": ...}`, making them perfectly interchangeable in the training script.

---

## ⏭️ Next Steps (Roadmap)

- [ ] **Inference Script:** Create a `predict.py` to generate and visualize CDR-H3 suggestions for a given framework.
- [ ] **Phase 2 (Light Chain):** Modify the `input_ids` to include the paired Light Chain sequence as additional context.
- [ ] **Phase 3 (Antigen-Aware):** Integrate SAbDab data to provide the model with the target antigen sequence.
- [ ] **Phase 4 (Structural):** Use the stored `imgt_numbering` to transition from 1D sequence prediction to 3D coordinate prediction.
