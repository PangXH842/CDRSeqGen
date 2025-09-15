import pandas as pd
import requests
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# The filtered summary CSV file created by the previous script.
INPUT_FILTERED_FILE = "sabdab_summary_filtered.csv"

# The directory where the downloaded PDB files will be saved.
OUTPUT_PDB_DIRECTORY = "data/raw/sabdab/"

# The base URL for the RCSB PDB database.
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/"

# -----------------------------------------------------------------------------
# --- MAIN SCRIPT ---
# -----------------------------------------------------------------------------

def download_pdb_files(filtered_csv_path: str, output_dir: str):
    """
    Downloads PDB files listed in a filtered summary file from the RCSB PDB database.
    
    Args:
        filtered_csv_path (str): Path to the filtered CSV file containing PDB IDs.
        output_dir (str): The directory where PDB files will be saved.
    """
    # 1. Check if the input file exists
    if not os.path.exists(filtered_csv_path):
        print(f"Error: Input file not found at '{filtered_csv_path}'")
        print("Please run the 'filter_sabdab_summary.py' script first.")
        return

    # 2. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"PDB files will be saved to: '{output_dir}'")

    # 3. Read the list of PDB IDs from the CSV file
    try:
        df = pd.read_csv(filtered_csv_path)
        if 'pdb' not in df.columns:
            print("Error: The CSV file must contain a column named 'pdb'.")
            return
        pdb_ids = df['pdb'].unique().tolist()
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return
    
    print(f"Found {len(pdb_ids)} unique PDB IDs to download.")

    # 4. Loop through the PDB IDs and download each file
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Use tqdm for a nice progress bar
    for pdb_id in tqdm(pdb_ids, desc="Downloading PDBs"):
        pdb_id = pdb_id.lower() # PDB IDs are case-insensitive, but filenames are often lowercase
        file_name = f"{pdb_id}.pdb"
        output_path = os.path.join(output_dir, file_name)
        
        # Check if the file already exists to avoid re-downloading
        if os.path.exists(output_path):
            skipped_count += 1
            continue

        # Construct the full download URL
        url = f"{PDB_DOWNLOAD_URL}{file_name}"
        
        try:
            # Make the request to download the file
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            # Write the content to the local file
            with open(output_path, 'w') as f:
                f.write(response.text)
            
            downloaded_count += 1

        except requests.exceptions.RequestException as e:
            # Handle potential network errors or bad responses
            # print(f"\nFailed to download {pdb_id}: {e}")
            failed_count += 1
            continue

    # --- Print Final Summary ---
    print("-" * 30)
    print("Download complete.")
    print(f"Successfully downloaded: {downloaded_count} files")
    print(f"Skipped (already exist): {skipped_count} files")
    print(f"Failed to download: {failed_count} files")
    
    if failed_count > 0:
        print("\nNote: Failures can occur for various reasons, such as temporary network issues or")
        print("if a PDB entry has been superseded or removed. You can try running the script again.")

if __name__ == "__main__":
    download_pdb_files(
        filtered_csv_path=INPUT_FILTERED_FILE,
        output_dir=OUTPUT_PDB_DIRECTORY,
    )