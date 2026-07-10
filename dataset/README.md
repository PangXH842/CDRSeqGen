# Data Processing: Antibody Infilling Framework
Updated: 2026-07-04

## 📌 Goal
Prepare data for **conditional CDR infilling** — predicting a masked CDR loop (H1/H2/H3) from its surrounding framework context — with the ability to layer in **light-chain context**, **antigen context**, and eventually **structure**. Antibodies are never treated as flat strings; each is a **Structured Biological Object** (the 14 IMGT regions kept separate) so any region can be masked at train time without re-processing raw data.

## 🛠 Data Sources
Two independent datasets feed the framework. **They are different train *and* test distributions — keep them on separate charts** (see the top-level README's Known Issues).

| | **OAS** (`processed_oas_data.json`) | **SAbDab2** (`processed_sabdab2_data.json`) |
|---|---|---|
| Origin | Observed Antibody Space — repertoire sequencing | SAbDab2 ML-grade structural splits (Zenodo) |
| Content | Human, paired (H+L), productive | Paired antibody + resolved protein/peptide **antigen** |
| Size | 21,106 sequences | 3,984 complexes (**3,234 train / 750 test**) |
| Antigen | none | yes (`antigen` field populated) |
| Split | CDR3-clustered holdout via `preprocess_paired.py --split cluster` (default), written to a `split` field; `--split random` for the legacy leaky split | dataset's **own leakage-safe** cluster split (carried in a `split` field) |
| Preprocessor | `preprocess_paired.py` | `preprocess_sabdab2.py` |
| Download | (raw `.csv.gz` provided) | `data/download_sabdab2.sh` (~876MB, one-time) |

**Why two?** OAS gives repertoire-scale sequence data for the base infilling task; SAbDab2 is the only source with bound antigens, needed for antigen-conditioned generation — but it is ~5× smaller, so numbers from the two are not comparable.

## 🏗 Preprocessing Logic
Both preprocessors emit the same Bio-Object schema and **print a per-stage filter breakdown** (rows dropped for non-productive / wrong-type / malformed-numbering / missing-regions) so the final count is auditable, not a black box.

### 1. Quality Control (OAS)
*   **Productive Only:** `productive_heavy == 'T'` and `productive_light == 'T'` (drops stop codons / frameshifts).
*   **Paired Requirement:** paired data from the start — the light chain is critical context for H3 folding/binding.

### 2. Regional Segmentation (the "14-Region Split")
Each antibody is parsed into its 14 IMGT-defined regions:
*   **Heavy:** FWRH1, CDRH1, FWRH2, CDRH2, FWRH3, **CDRH3**, FWRH4.
*   **Light:** FWRL1, CDRL1, FWRL2, CDRL2, FWRL3, CDRL3, FWRL4.

This lets the dataset dynamically mask *any* region at train time — heavy `h1/h2/h3` or light `l1/l2/l3`, selected by `target_region` alone — with zero raw re-processing.

### 3. Structural Preservation (IMGT Numbering)
`imgt_numbering` dictionaries are stored per chain, mapping each residue to a standardized position — the hook for aligning to PDB structures / injecting 3D positional embeddings in the structural phase.

### 4. SAbDab2 specifics
*   SAbDab2's `abag_split.csv` already carries IMGT-numbered CDR columns **and** a leakage-safe train/test split, so preprocessing is pure CSV parsing — no CIF/structure parsing or re-numbering.
*   Each entry gains an `antigen` object (`full_aa`, `chains`, `type`) and a `split` field. `train.py` honors that `split` instead of a random one (OAS has no such field and is unaffected).
*   Multi-chain antigens are joined with `X` (unknown-residue) as a soft chain-break marker, since the amino-acid tokenizer has no literal chain-separator token.
*   Hardening (2026-07-03) dropped 1 degenerate entry with an empty target CDR (3,985 → 3,984).

---

## 📂 Output Schema
A List of Dictionaries (JSON). OAS entries have `antigen: null`; SAbDab2 entries populate it and add `split`.

```json
{
  "sequence_id": "...",
  "isotype": "IgG",                         // OAS only
  "heavy": {
    "fwr1": "...", "cdr1": "...", "fwr2": "...", "cdr2": "...",
    "fwr3": "...", "cdr3": "...", "fwr4": "...",
    "full_aa": "...",
    "imgt_numbering": {"1": "E", "2": "V", "...": "..."}
  },
  "light": { "...": "... same shape as heavy ..." },
  "antigen": {                              // null for OAS; populated for SAbDab2
    "full_aa": "YTTSTRGDLAHVT",
    "chains": ["P"],
    "type": "PEPTIDE"                        // e.g. PEPTIDE | PROTEIN
  },
  "split": "train"                          // SAbDab2 only: "train" | "test"
}
```

---

## 📝 Usage Notes
*   **Decompress OAS:** `pd.read_csv(file, compression='gzip', skiprows=1)`.
*   **Debug subset:** `1287169_1_Paired_All.csv.gz` (21,106 seqs) before scaling to the full 3M+ dataset.
*   **OAS clustered split (homology-leakage fix).** A plain random split leaks *clonal near-duplicates* (CDR3s a few somatic mutations apart) into both train and eval, making OAS eval optimistic. `preprocess_paired.py --split cluster` (now the default) fixes this: it greedily clusters heavy CDR3s at ≥85% identity and holds out **whole clusters**, writing a leakage-safe `split` field that `train.py` honors automatically. Measured on the current 21,106-seq file: the random split leaks **3.6%** of the test set (test entries with a clonal relative in train); the clustered split leaks **0%**. Runs preprocessed before this used the leaky random split and should be re-run for honest numbers. Note the greedy clustering is same-length/identity-based (captures somatic hypermutation, not indels) and O(n²)-ish within length groups — fine at 21k (~10s); for the 1M+ scale-up, swap in MMseqs2/CD-HIT.

---

## 🚀 Phase Status
1. **[x] Conditional CDR infilling** — `FWR1..FWR3 + [MASK] + FWR4` → predict the masked loop. Heavy `h1/h2/h3` all trained on both datasets; light `l1/l2/l3` now supported too (path verified, full suite pending). The `target_region` string alone selects the loop *and* the chain (`h*` heavy, `l*` light; legacy `cdr*`/`lcdr*` still accepted).
2. **[x] Partner-chain context** — the paired non-target chain is prepended as additional context via one symmetric flag, `include_partner_chain`: light chain for a heavy (`h*`) target, heavy chain for a light (`l*`) target.
3. **[~] Antigen-awareness** — **sequence-context injection landed** (antigen prepended/appended per the backbone's position type; `include_antigen`), but the completed `_ag` vs `_ag_lconly` ablation shows **no measurable benefit yet**, and this survives the anchor-excluded (core) metric (see top-level README). **True antigen cross-attention** — the originally-envisioned mechanism, and the promising next lever — is *not yet built*.
4. **[ ] Structure** — use the saved IMGT numbering keys for structural conditioning (IgFold/ABlooper-derived CDR coordinates, etc.).
