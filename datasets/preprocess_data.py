import pandas as pd
import torch
import os
import subprocess
import tempfile
from Bio.PDB import PDBParser, Polypeptide
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILTERED_CSV = "sabdab_summary_filtered.csv"
INPUT_PDB_DIR = "data/raw/sabdab/"
OUTPUT_PROCESSED_DIR = "data/processed/"
OUTPUT_FILE_NAME = "sabdab_processed_H_chain_infill.pt" # New name for the focused dataset

ANARCI_EXECUTABLE = "ANARCI"

# --- MAIN SCRIPT ---
def run_anarci(pdb_file: str, chain_id: str) -> dict:
    """
    Runs the ANARCI command-line tool on a single chain from a PDB file
    and parses its output to find the CDR boundaries.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("temp", pdb_file)
        chain = structure[0][chain_id]
        
        ppb = Polypeptide.Polypeptide(chain)
        # Convert the sequence object to a simple string
        sequence = str(ppb.get_sequence())
    except Exception as e:
        # This will catch errors if the PDB file is weird or the chain doesn't exist
        print(f"DEBUG: BioPython failed to parse {pdb_file} chain {chain_id}. Error: {e}")
        return None

    # ANARCI expects a simple FASTA file as input
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".fasta") as tmp:
        # Use a simple header, the content of the header doesn't matter much
        tmp.write(f">temp_seq\n{sequence}\n")
        fasta_path = tmp.name

    try:
        # Run ANARCI. It does not take a chain argument because the FASTA file only has one chain.
        # We specify the species as 'human' to help ANARCI find the best HMM match.
        # You could make this dynamic based on the 'heavy_species' column if needed.
        result = subprocess.run(
            [ANARCI_EXECUTABLE, "-i", fasta_path, "--scheme", "imgt", "--use_species", "human"],
            capture_output=True, text=True, check=True, timeout=30
        )
        
        # --- DEBUGGING: See what ANARCI is actually outputting ---
        # if not result.stdout.strip(): # Check if the output is empty
        #     print(f"DEBUG: ANARCI ran for {pdb_file} chain {chain_id} but produced NO stdout.")
        #     print(f"DEBUG: Stderr was: {result.stderr}")
        # -----------------------------------------------------------
        
        # Parse the output to find CDRs
        cdr_indices = {"H1": [], "H2": [], "H3": []} # Use lists to append residues
        
        lines = result.stdout.splitlines()
        # The actual numbered output starts after the header lines marked with '#'
        data_lines = [line for line in lines if not line.startswith('#')]

        if not data_lines: # If there's no data, ANARCI failed to number it
            print(f"DEBUG: ANARCI produced no data lines for {pdb_file} chain {chain_id}.")
            return None

        # The ANARCI output format is: chain, position, insertion_code, residue
        # We only care about position and residue
        for line in data_lines:
            parts = line.split()
            if len(parts) < 4 or parts[1] == '-': # Skip lines without a valid position
                continue
            
            try:
                position, residue = int(parts[1]), parts[3]
            except (ValueError, IndexError):
                continue # Skip malformed lines

            # IMGT numbering definitions for CDRs
            if 27 <= position <= 38:
                cdr_indices["H1"].append(residue)
            elif 56 <= position <= 65:
                cdr_indices["H2"].append(residue)
            elif 105 <= position <= 117:
                cdr_indices["H3"].append(residue)

        # Join the lists of residues into strings
        for cdr, residues in cdr_indices.items():
            if residues: # If the list is not empty
                cdr_indices[cdr] = "".join(residues)
            else: # If a CDR was not found, set to None
                cdr_indices[cdr] = None

        return cdr_indices

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"DEBUG: ANARCI subprocess failed for {pdb_file} chain {chain_id}.")
        if hasattr(e, 'stderr'):
            print(f"DEBUG: Stderr: {e.stderr}")
        return None
    finally:
        os.remove(fasta_path)


def process_pdb_for_infilling(pdb_id: str, row: pd.Series, pdb_dir: str) -> dict:
    """
    Processes a single PDB to extract the heavy chain and identify the CDR-H3 for an infilling task.
    """
    pdb_file = os.path.join(pdb_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(pdb_file):
        # print(f"DEBUG: PDB file {pdb_file} does not exist.")
        return None

    try:
        # 1. Get the full heavy chain sequence from the PDB file
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)
        heavy_chain_seq = str(Polypeptide.Polypeptide(structure[0][row['Hchain']]).get_sequence())

        # 2. Use ANARCI to reliably find the CDR-H3 sequence
        cdr_data = run_anarci(pdb_file, row['Hchain'])
        if cdr_data is None or cdr_data['H3'] is None:
            # print(f"DEBUG: Skipping {pdb_id} chain {row['Hchain']} due to ANARCI failure or missing CDR-H3.")
            return None
        cdr_h3_seq = cdr_data['H3']

        # 3. Create the masked sequence for the model's input
        # This will fail if the ANARCI-identified CDR is not a perfect substring,
        # which can happen with numbering quirks. We just skip these cases for our baseline.
        if heavy_chain_seq.find(cdr_h3_seq) == -1:
            # print(f"DEBUG: CDR-H3 sequence '{cdr_h3_seq}' not found in heavy chain of {pdb_id}. Skipping.")
            return None
        masked_heavy_chain_seq = heavy_chain_seq.replace(cdr_h3_seq, "X" * len(cdr_h3_seq))

        return {
            "pdb_id": pdb_id,
            "masked_heavy_chain": masked_heavy_chain_seq,
            "full_heavy_chain": heavy_chain_seq, # The ground truth label
        }
    except Exception as e:
        # print(f"DEBUG: Exception processing {pdb_id} chain {row['Hchain']}: {e}")
        return None


def main_workflow():
    df_filtered = pd.read_csv(INPUT_FILTERED_CSV)
    # partial dataset for testing
    # df_partial = df_filtered.sample(n=20, random_state=42).reset_index(drop=True)
    df_partial = df_filtered
    
    all_data = []
    for index, row in tqdm(df_partial.iterrows(), total=len(df_partial), desc="Processing PDBs for H-Chain Infilling"):
    # for index, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing PDBs for H-Chain Infilling"):
        processed = process_pdb_for_infilling(row['pdb'], row, INPUT_PDB_DIR)
        if processed:
            all_data.append(processed)

    print(f"\nSuccessfully processed {len(all_data)} out of {len(df_partial)} PDB files.")
    
    # ... (train_test_split and saving logic remains the same) ...
    train_data, temp_data = train_test_split(all_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_PROCESSED_DIR, OUTPUT_FILE_NAME)
    
    torch.save({'train': train_data, 'validation': val_data, 'test': test_data}, output_path)
    print(f"\nProcessed data saved to '{output_path}'")

if __name__ == "__main__":
    # You will need to copy the robust `run_anarci` function from our previous conversation here
    main_workflow()