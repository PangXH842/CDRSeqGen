import pandas as pd
import os

# --- CONFIGURATION ---
# All filtering criteria are defined in this dictionary.
# Modify these values to change the dataset selection criteria.
FILTER_CRITERIA = {
    # Method used to determine the structure. 'X-RAY DIFFRACTION' is the highest quality.
    "METHOD": "X-RAY DIFFRACTION",
    
    # Maximum resolution in Angstroms (Å). Lower is better. 3.0 is a common cutoff.
    "MAX_RESOLUTION": 3.0,
    
    # The type of antigen molecule. We are interested in protein/peptide targets.
    "ANTIGEN_TYPES": ["protein", "peptide"],
    
    # (Optional) Specify species. Set to None to include all species.
    # Examples: 'homo sapiens', 'mus musculus' (mouse)
    "HEAVY_CHAIN_SPECIES": None, 
    "LIGHT_CHAIN_SPECIES": None,
    "ANTIGEN_SPECIES": None,
    
    # Set to True to only keep entries that have a reported affinity value.
    # Note: This will result in a MUCH smaller dataset.
    "REQUIRE_AFFINITY": False,
}

# --- INPUT AND OUTPUT FILES ---
INPUT_SUMMARY_FILE = "20250909_0409239_summary.tsv" 
OUTPUT_FILTERED_FILE = "sabdab_summary_filtered.csv" 

# -----------------------------------------------------------------------------
# --- MAIN SCRIPT ---
# -----------------------------------------------------------------------------

def filter_sabdab_summary(summary_file_path: str, output_path: str, criteria: dict):
    """
    Filters the SAbDab summary file based on a set of criteria to select
    high-quality, relevant antibody-antigen complexes.
    
    Args:
        summary_file_path (str): Path to the input SAbDab summary TSV file.
        output_path (str): Path to save the filtered CSV file.
        criteria (dict): A dictionary containing all the filtering parameters.
    """
    if not os.path.exists(summary_file_path):
        print(f"Error: Input file not found at '{summary_file_path}'")
        print("Please download the summary file from SAbDab and place it in the same directory.")
        return

    print("Loading SAbDab summary file...")
    # SAbDab summary is a Tab-Separated Value (TSV) file
    df = pd.read_csv(summary_file_path, sep='\t')
    print(f"Initial number of entries: {len(df)}")

    # --- Apply Filters Sequentially ---

    # 1. Filter by experimental method
    df = df[df['method'] == criteria["METHOD"]]
    print(f"After method filter ('{criteria['METHOD']}'): {len(df)} entries")

    # 2. Filter by resolution (lower is better)
    df = df[df['resolution'] <= str(criteria["MAX_RESOLUTION"])]
    print(f"After resolution filter (<= {criteria['MAX_RESOLUTION']} Å): {len(df)} entries")
    
    # 3. Ensure required chains are present (H, L, and antigen)
    df = df.dropna(subset=['Hchain', 'Lchain', 'antigen_chain'])
    print(f"After requiring H, L, and antigen chains: {len(df)} entries")

    # 4. Filter by antigen type
    df = df[df['antigen_type'].isin(criteria["ANTIGEN_TYPES"])]
    print(f"After antigen type filter ({criteria['ANTIGEN_TYPES']}): {len(df)} entries")

    # 5. (Optional) Filter by species
    if criteria["HEAVY_CHAIN_SPECIES"]:
        df = df[df['heavy_species'] == criteria["HEAVY_CHAIN_SPECIES"]]
        print(f"After heavy chain species filter ('{criteria['HEAVY_CHAIN_SPECIES']}'): {len(df)} entries")
    if criteria["LIGHT_CHAIN_SPECIES"]:
        df = df[df['light_species'] == criteria["LIGHT_CHAIN_SPECIES"]]
        print(f"After light chain species filter ('{criteria['LIGHT_CHAIN_SPECIES']}'): {len(df)} entries")
    if criteria["ANTIGEN_SPECIES"]:
        df = df[df['antigen_species'] == criteria["ANTIGEN_SPECIES"]]
        print(f"After antigen species filter ('{criteria['ANTIGEN_SPECIES']}'): {len(df)} entries")

    # 6. (Optional) Filter for entries with known affinity
    if criteria["REQUIRE_AFFINITY"]:
        df = df.dropna(subset=['affinity'])
        print(f"After requiring affinity data: {len(df)} entries")

    # Select only the most important columns for the final output
    final_columns = [
        'pdb', 'Hchain', 'Lchain', 'antigen_chain', 
        'antigen_type', 'heavy_species', 'light_species', 
        'antigen_species', 'resolution', 'affinity'
    ]
    df_final = df[final_columns].copy()

    # Save the filtered DataFrame to a CSV file
    df_final.to_csv(output_path, index=False)
    print("-" * 30)
    print(f"Successfully filtered the dataset.")
    print(f"Final number of selected PDBs: {len(df_final)}")
    print(f"Filtered summary saved to: '{output_path}'")
    print("\nNext step: Use the 'pdb' column of this file to download the PDB files.")


if __name__ == "__main__":
    filter_sabdab_summary(
        summary_file_path=INPUT_SUMMARY_FILE,
        output_path=OUTPUT_FILTERED_FILE,
        criteria=FILTER_CRITERIA
    )