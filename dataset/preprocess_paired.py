import pandas as pd
import ast
import json
import argparse
from tqdm import tqdm

def preprocess_oas_paired(file_path):
    # 1. Load data, skipping the first row of metadata
    # We use engine='python' because OAS files sometimes have complex quoting
    df = pd.read_csv(file_path, skiprows=1, low_memory=False)

    # 2. Filter for quality
    # We only want productive antibodies (no stop codons or frameshifts)
    df = df[(df['productive_heavy'] == 'T') & (df['productive_light'] == 'T')]
    
    processed_data = []

    for _, row in tqdm(df.iterrows()):
        # 3. Clean and parse ANARCI numbering 
        # (This is stored as a stringified dictionary in OAS)
        try:
            # ast.literal_eval is safer than eval() for string-to-dict conversion
            heavy_imgt = ast.literal_eval(row['ANARCI_numbering_heavy'])
            light_imgt = ast.literal_eval(row['ANARCI_numbering_light'])
        except:
            continue # Skip if numbering is malformed

        # 4. Construct our "Structured Biological Object"
        entry = {
            "sequence_id": row['sequence_id_heavy'],
            "isotype": row['Isotype_heavy'],
            "heavy": {
                "fwr1": row['fwr1_aa_heavy'],
                "cdr1": row['cdr1_aa_heavy'],
                "fwr2": row['fwr2_aa_heavy'],
                "cdr2": row['cdr2_aa_heavy'],
                "fwr3": row['fwr3_aa_heavy'],
                "cdr3": row['cdr3_aa_heavy'],
                "fwr4": row['fwr4_aa_heavy'],
                "full_aa": row['sequence_alignment_aa_heavy'],
                "imgt_numbering": heavy_imgt
            },
            "light": {
                "fwr1": row['fwr1_aa_light'],
                "cdr1": row['cdr1_aa_light'],
                "fwr2": row['fwr2_aa_light'],
                "cdr2": row['cdr2_aa_light'],
                "fwr3": row['fwr3_aa_light'],
                "cdr3": row['cdr3_aa_light'],
                "fwr4": row['fwr4_aa_light'],
                "full_aa": row['sequence_alignment_aa_light'],
                "imgt_numbering": light_imgt
            }
        }
        processed_data.append(entry)

    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OAS paired antibody data.")
    parser.add_argument("--input_file", help="Path to the input CSV file", default="./raw/1287169_1_Paired_All.csv")
    parser.add_argument("--output_file", help="Path to the output JSON file", default="./processed/processed_paired_data.json")
    args = parser.parse_args()

    data = preprocess_oas_paired(args.input_file)

    with open(args.output_file, 'w') as f:
        json.dump(data, f)

    print(f"Successfully processed {len(data)} productive paired sequences.")