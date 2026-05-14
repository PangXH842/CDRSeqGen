# Data Processing: OAS Antibody Infilling Framework (Phase 1)
Updated: 2026-05-13

## 📌 Project Goal
To develop a modular machine learning framework for predicting antibody sequences, specifically starting with **Conditional CDR-H3 Infilling**, with the ability to scale to **Antigen-Aware** and **Structure-Aware** models.

## 🛠 The Data Source
*   **Database:** Observed Antibody Space (OAS).
*   **Subset:** Human, Paired (Heavy + Light), Productive.
*   **Format:** `.csv.gz` (Gzipped CSVs with a single-line metadata header).

## 🏗 Data Preprocessing Logic
To ensure the model can eventually handle structural data and multiple tasks, we do not treat the antibody as a simple string. Instead, we treat it as a **Structured Biological Object**.

### 1. Quality Control (Filtering)
*   **Productive Only:** We filter for `productive_heavy == 'T'` and `productive_light == 'T'`. This removes sequences with stop codons or frameshifts.
*   **Paired Requirement:** Although Phase 1 focuses on CDR-H3 (Heavy), we use paired data from the start. The Light chain is a critical context for CDR-H3 folding and binding in later phases.

### 2. Regional Segmentation (The "14-Region Split")
The antibody is parsed into its 14 IMGT-defined regions:
*   **Heavy:** FWRH1, CDRH1, FWRH2, CDRH2, FWRH3, **CDRH3**, FWRH4.
*   **Light:** FWRL1, CDRL1, FWRL2, CDRL2, FWRL3, CDRL3, FWRL4.

**Why?** This allows the `DataCollator` to dynamically mask *any* part of the antibody during training (e.g., masking CDR-H3 today, but masking CDR-L3 or all CDRs tomorrow) without re-processing the raw data.

### 3. Structural Preservation (IMGT Numbering)
*   We extract and store the `ANARCI_numbering` dictionaries.
*   **Why?** This maps every amino acid to a standardized 3D coordinate position. When we move to **Structural Phase**, we can use these keys to align sequences with PDB structures or inject 3D positional embeddings.

---

## 📂 Output Schema
The processed data is saved as a List of Dictionaries (JSON or Parquet). Each entry follows this schema:

```json
{
  "sequence_id": "...",
  "isotype": "IgG",
  "heavy": {
    "fwr1": "...", "cdr1": "...", "fwr2": "...", "cdr2": "...", 
    "fwr3": "...", "cdr3": "...", "fwr4": "...",
    "full_aa": "...",
    "imgt_numbering": {"1": "E", "2": "V", ...}
  },
  "light": {
    "fwr1": "...", "cdr1": "...", "fwr2": "...", "cdr2": "...", 
    "fwr3": "...", "cdr3": "...", "fwr4": "...",
    "full_aa": "...",
    "imgt_numbering": {"1": "D", "2": "I", ...}
  },
  "antigen": null  // Placeholder for Phase 2
}
```

---

## 🚀 Phase 1 Roadmap: Conditional Infilling
For the first training task, we prepare the data as follows:

1.  **Input (Context):** `FWRH1 + CDRH1 + FWRH2 + CDRH2 + FWRH3 + [MASK] + FWRH4`
2.  **Target:** `CDRH3`
3.  **Task:** Predict the amino acid sequence of the masked H3 loop given the surrounding Framework 1-4 and CDR 1-2.

## 📝 Usage Notes
*   **File Decompression:** Use `pd.read_csv(file, compression='gzip', skiprows=1)`.
*   **Deduplication:** CDR-H3 sequences should be clustered or deduplicated before final training to prevent the model from memorizing common clones.
*   **Small Subset:** Start with `1287169_1_Paired_All.csv.gz` containing 21106 sequences for debugging before scaling to the full 3M+ dataset.

---

### 🔜 Future Phases
- [ ] **Phase 2:** Include Light Chain as additional context.
- [ ] **Phase 3:** Antigen-Awareness (Injecting antigen sequences into the Transformer cross-attention).
- [ ] **Phase 4:** Structural prediction using the saved IMGT numbering keys.