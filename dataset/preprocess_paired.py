import pandas as pd
import ast
import json
import argparse
from tqdm import tqdm

REGIONS = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"]


def valid_regions(region_dict):
    """Guard against malformed rows before they reach dataset.py.

    Every region must be a real string (empty CSV cells arrive as float NaN, which would
    later crash the string concatenation in AntibodyInfillingDataset), and every CDR must be
    non-empty since any of the three can be the masked target loop.
    """
    if not all(isinstance(region_dict.get(r), str) for r in REGIONS):
        return False
    return all(region_dict[cdr] for cdr in ("cdr1", "cdr2", "cdr3"))


def build_regions(row, suffix):
    return {r: row[f"{r}_aa_{suffix}"] for r in REGIONS}


def preprocess_oas_paired(file_path):
    # 1. Load data, skipping the first row of metadata
    # We use engine='python' because OAS files sometimes have complex quoting
    df = pd.read_csv(file_path, skiprows=1, low_memory=False)

    # 2. Filter for quality
    # We only want productive antibodies (no stop codons or frameshifts)
    n_raw = len(df)
    df = df[(df['productive_heavy'] == 'T') & (df['productive_light'] == 'T')]
    n_productive = len(df)

    processed_data = []
    skipped = {"numbering": 0, "regions": 0}

    for _, row in tqdm(df.iterrows(), total=n_productive):
        # 3. Clean and parse ANARCI numbering (a stringified dict in OAS).
        # Narrow except: only swallow parse failures (incl. NaN passed to literal_eval),
        # NOT e.g. a KeyError from a renamed column -- that should surface loudly.
        try:
            heavy_imgt = ast.literal_eval(row['ANARCI_numbering_heavy'])
            light_imgt = ast.literal_eval(row['ANARCI_numbering_light'])
        except (ValueError, SyntaxError, TypeError):
            skipped["numbering"] += 1
            continue

        heavy_regions = build_regions(row, "heavy")
        light_regions = build_regions(row, "light")
        if not (valid_regions(heavy_regions) and valid_regions(light_regions)):
            skipped["regions"] += 1
            continue

        # 4. Construct our "Structured Biological Object"
        entry = {
            "sequence_id": row['sequence_id_heavy'],
            "isotype": row['Isotype_heavy'],
            "heavy": {**heavy_regions, "full_aa": row['sequence_alignment_aa_heavy'], "imgt_numbering": heavy_imgt},
            "light": {**light_regions, "full_aa": row['sequence_alignment_aa_light'], "imgt_numbering": light_imgt},
        }
        processed_data.append(entry)

    print(f"Rows: {n_raw} raw -> {n_productive} productive -> {len(processed_data)} kept "
          f"(skipped {skipped['numbering']} malformed-numbering, {skipped['regions']} missing/empty regions).")
    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OAS paired antibody data.")
    parser.add_argument("--input_file", help="Path to the input CSV file", default="./raw/1287169_1_Paired_All.csv")
    parser.add_argument("--output_file", help="Path to the output JSON file", default="./processed/processed_paired_data.json")
    args = parser.parse_args()

    data = preprocess_oas_paired(args.input_file)

    with open(args.output_file, 'w') as f:
        json.dump(data, f)

    print(f"Wrote {len(data)} entries to {args.output_file}")