# %% [markdown]
# # Generic Viral Protein Sequence Processing

# %%
from IPython.display import display
import pandas as pd
import re
import numpy as np
import time
import warnings

# Suppress FutureWarnings to prevent output truncation
warnings.filterwarnings('ignore', category=FutureWarning)

from Bio import SeqIO
from Bio.Align import PairwiseAligner

#I forget if this one is needed: (check later)
from Bio import AlignIO

from multiprocessing import Pool, Queue, Process, cpu_count
import os
from tqdm import tqdm

from collections import Counter

# Import our custom functions
from genbank_functions import (
    initialize_aligner, 
    process_record, 
    process_long_sequence, 
    process_short_sequence, 
    create_row, 
    worker, 
    dna_to_amino_acids,
    get_protein_config,
    get_processing_config,
    get_output_config
)

# %% [markdown]
# Files downloaded from genbank website on 01/31/2025 are:
# * ref_seq.fasta
#   * the reference sequence from initial WuHan strain
# * sequences_complete.fasta
#   * this contains genbank sequences that have length >= 29000 nucleotides
#   * the file size is 275 GB
#   * 9,018,239 total sequences
# * sequences_complete_sample.fasta
#   * this contains 2000 random samples of file above.
#   * 43,643 total sequences
# * sequences_NotComplete.fasta
#   * this contains genbank sequences that have length <= 28999 nucleotides
# * sequences_NotHuman.fasta
#   * this contains sequences found in hosts other than humans
#   * 1,316 total sequences
# * data/raw/genbank/sequences_complete_append_4-28-25.fasta
#   * this contains genbank sequences that are uploaded after 01/31/2025 and collected before 04/28/2025 for reliable comparison with gisaid data

# %% [markdown]
# # Configuration and Setup

# %%
# Set working directory to project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)

# Configuration - Change this to analyze different proteins
PROTEIN_NAME = "rbd"  # Options: 'mpro', 'plpro', 'rbd', etc.
CONFIG_PATH = "src/config/protein_configs.yaml"
DATA_SOURCE = 'genbank'

# APPEND DATA HYPERPARAMETERS
APPEND_DATA = False  # Set to True to append additional data, False to run normally
APPEND_DATA_FILENAME = "data/raw/genbank/sequences_complete_append_4-28-25.fasta"

# Load configurations
protein_config = get_protein_config(PROTEIN_NAME, CONFIG_PATH, data_source=DATA_SOURCE)
processing_config = get_processing_config(CONFIG_PATH, data_source=DATA_SOURCE)
output_config = get_output_config(CONFIG_PATH, data_source=DATA_SOURCE)

print(f"  genbank Processing Configuration:")
print(f"  Protein: {PROTEIN_NAME}")
print(f"  Data Source: {DATA_SOURCE}")
print(f"  Append Data: {'Yes' if APPEND_DATA else 'No'}")
if APPEND_DATA:
    print(f"  Append File: {APPEND_DATA_FILENAME}")

# %% [markdown]
# # Helper Functions

# %%
def process_fasta_file(fasta_file, ref_seq, aligner, protein_config, file_label="main"):
    """Process a FASTA file and return valid and invalid DataFrames"""
    print(f" Processing {file_label} FASTA file: {fasta_file}")
    
    if not os.path.exists(fasta_file):
        print(f" File not found: {fasta_file}")
        return None, None
    
    # Create temporary output directory for this processing run
    temp_output_dir = f"temp_results_{file_label}_{PROTEIN_NAME}"
    os.makedirs(temp_output_dir, exist_ok=True)
    
    try:
        # Create a bounded queue to distribute records to workers
        MAX_QUEUE_SIZE = 1000 # processing_config.get('max_queue_size', 1000)
        queue = Queue(maxsize=MAX_QUEUE_SIZE)

        # Start worker processes
        num_workers = processing_config.get('multiprocessing', {}).get('max_workers') or cpu_count()
        print(f" Starting {num_workers} worker processes for {file_label} data...")
        workers = []
        for worker_id in range(num_workers):
            p = Process(target=worker, args=(queue, aligner, ref_seq, temp_output_dir, worker_id, protein_config))
            p.start()
            workers.append(p)

        # Stream the FASTA file and add records to the queue
        start = time.time()
        total_records = 0
        progress_intervals = processing_config.get('progress_report_intervals', [100, 1000, 5000, 10000, 50000])
        progress_frequency = processing_config.get('progress_report_frequency', 1000000)

        with open(fasta_file) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                queue.put(record)
                total_records += 1
                if total_records in progress_intervals or total_records % progress_frequency == 0:
                    print(f"  📊 {file_label} - at seq: {total_records} (time: {time.time() - start:.1f}s)")

        # Add sentinel values to signal workers to stop
        for _ in range(num_workers):
            queue.put(None)

        # Wait for all workers to finish
        for p in workers:
            p.join()

        # Combine results from all workers
        sequence_col = f"{PROTEIN_NAME} Sequence"
        columns = ['Name', 'ID', 'Date', sequence_col, 'Score']
        df_chunks = []
        bad_df_chunks = []

        for worker_id in range(num_workers):
            df_chunk_path = os.path.join(temp_output_dir, f"{PROTEIN_NAME}_df_chunk_{worker_id}.csv")
            bad_df_chunk_path = os.path.join(temp_output_dir, f"{PROTEIN_NAME}_bad_df_chunk_{worker_id}.csv")
            
            if os.path.exists(df_chunk_path):
                chunk = pd.read_csv(df_chunk_path)
                if not chunk.empty:
                    df_chunks.append(chunk)
                os.remove(df_chunk_path)
            if os.path.exists(bad_df_chunk_path):
                bad_chunk = pd.read_csv(bad_df_chunk_path)
                if not bad_chunk.empty:
                    bad_df_chunks.append(bad_chunk)
                os.remove(bad_df_chunk_path)

        # Concatenate all chunks at once
        df = pd.concat(df_chunks, ignore_index=True) if df_chunks else pd.DataFrame(columns=columns)
        bad_df = pd.concat(bad_df_chunks, ignore_index=True) if bad_df_chunks else pd.DataFrame(columns=columns)

        print(f"  {file_label} processing complete:")
        print(f"  Valid sequences: {len(df):,}")
        print(f"  Invalid sequences: {len(bad_df):,}")
        print(f"  Time taken: {time.time() - start:.2f} seconds")
        
        return df, bad_df
        
    finally:
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)

def check_duplicate_ids(df, bad_df, label="combined"):
    """Check for duplicate IDs in the datasets"""
    print(f"  Checking for duplicate IDs in {label} data...")
    
    # Check duplicates in valid sequences
    df_duplicates = df[df.duplicated(subset=['ID'], keep=False)]
    bad_df_duplicates = bad_df[bad_df.duplicated(subset=['ID'], keep=False)]
    
    # Check for IDs that appear in both valid and invalid datasets
    cross_duplicates = set(df['ID']) & set(bad_df['ID'])
    
    total_duplicates = len(df_duplicates) + len(bad_df_duplicates) + len(cross_duplicates)
    
    if total_duplicates > 0:
        print(f"  Found {total_duplicates} duplicate issues:")
        
        if len(df_duplicates) > 0:
            print(f"  - Valid sequences with duplicate IDs: {len(df_duplicates)}")
            print(f"    Sample duplicate IDs: {df_duplicates['ID'].unique()[:5].tolist()}")
        
        if len(bad_df_duplicates) > 0:
            print(f"  - Invalid sequences with duplicate IDs: {len(bad_df_duplicates)}")
            print(f"    Sample duplicate IDs: {bad_df_duplicates['ID'].unique()[:5].tolist()}")
        
        if len(cross_duplicates) > 0:
            print(f"  - IDs appearing in both valid and invalid: {len(cross_duplicates)}")
            print(f"    Sample cross-duplicate IDs: {list(cross_duplicates)[:5]}")
        
        return False, {
            'valid_duplicates': df_duplicates,
            'invalid_duplicates': bad_df_duplicates,
            'cross_duplicates': cross_duplicates
        }
    else:
        print(f"  No duplicate IDs found in {label} data")
        return True, None

# %% [markdown]
# # Record-by-record opening and processing of file

# %%
with open("data/reference/ref_seq.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        ref_seq = str(record.seq)[protein_config['start_position']:protein_config['start_position'] + protein_config['length']]
print(f"Reference {PROTEIN_NAME} sequence:")
print(ref_seq)
print(f"Length: {len(ref_seq)}")

# %%
# Initialize aligner
aligner = initialize_aligner(PROTEIN_NAME, CONFIG_PATH)

if APPEND_DATA:
    print(f"\n{'='*60}")
    print(f"🧬ROCESSING APPEND genbank DATA ONLY")
    print(f"{'='*60}")
    print(f" Processing append file: {APPEND_DATA_FILENAME}")
    
    df_main, bad_df_main = process_fasta_file(
        APPEND_DATA_FILENAME, 
        ref_seq, 
        aligner, 
        protein_config, 
        "append"
    )
    
    if df_main is None:
        print(" Failed to process append data file")
        exit(1)
    
    # When append data is enabled, we only process the append file
    final_df = df_main
    final_bad_df = bad_df_main
    is_unique = True
    
    print(f"\n APPEND_DATA mode - processed {APPEND_DATA_FILENAME} only")
    
else:
    # Process main data file
    print(f"\n{'='*60}")
    print(f"🧬ROCESSING MAIN genbank DATA")
    print(f"{'='*60}")
    print(f" Processing main file: data/raw/genbank/sequences_complete.fasta")

    df_main, bad_df_main = process_fasta_file(
        "data/raw/genbank/sequences_complete.fasta", 
        ref_seq, 
        aligner, 
        protein_config, 
        "main"
    )

    if df_main is None:
        print(" Failed to process main data file")
        exit(1)
    
    final_df = df_main
    final_bad_df = bad_df_main
    is_unique = True
    
    print(f"\n APPEND_DATA is False - processed main data only")

print(f"\nFinal dataset summary:")
print(f"  Total valid sequences: {len(final_df):,}")
print(f"  Total invalid sequences: {len(final_bad_df):,}")
print(f"  Success rate: {len(final_df)/(len(final_df)+len(final_bad_df))*100:.2f}%")

# %% [markdown]
# # Save Results

# %%
# Only save if no duplicate IDs detected
if is_unique:
    print(f"\n Saving results to CSV files...")
    
    results_dir_template = output_config.get('results_directory', 'results/genbank/{protein}/tables')
    results_dir = results_dir_template.format(protein=PROTEIN_NAME)
    os.makedirs(results_dir, exist_ok=True)

    # Get file naming templates from config
    file_prefixes = output_config.get('file_prefixes', {})
    valid_file_template = file_prefixes.get('valid_sequences', 'genbank_{protein}_sequences')
    invalid_file_template = file_prefixes.get('invalid_sequences', 'genbank_{protein}_bad_sequences')

    # Generate filenames
    valid_filename = f"{results_dir}/{valid_file_template.format(protein=PROTEIN_NAME)}.csv"
    invalid_filename = f"{results_dir}/{invalid_file_template.format(protein=PROTEIN_NAME)}.csv"

    final_df.to_csv(valid_filename, index=False)
    final_bad_df.to_csv(invalid_filename, index=False)

    print(f"\n {PROTEIN_NAME.upper()} genbank processing complete!")
    print(f" Results saved to:")
    print(f"   Valid sequences: {valid_filename} ({len(final_df):,} sequences)")
    print(f"   Invalid sequences: {invalid_filename} ({len(final_bad_df):,} sequences)")
    
    if APPEND_DATA:
        print(f" Processed append data from: {APPEND_DATA_FILENAME}")
    else:
        print(f" Processed main data from: data/raw/genbank/sequences_complete.fasta")
        
else:
    print(f"\n RESULTS NOT SAVED DUE TO DUPLICATE IDs")
    print(f" Please resolve duplicate issues before saving")
    print(f" Data is available in variables:")
    print(f"   - final_df: {len(final_df):,} valid sequences")
    print(f"   - final_bad_df: {len(final_bad_df):,} invalid sequences")
    
    # Show some duplicate examples for troubleshooting
    if 'duplicate_info' in locals() and duplicate_info:
        print(f"\n Sample duplicate information for troubleshooting:")
        if len(duplicate_info['valid_duplicates']) > 0:
            print(f"Valid duplicates sample:")
            print(duplicate_info['valid_duplicates'][['ID', 'Name', 'Date']].head())
        if len(duplicate_info['invalid_duplicates']) > 0:
            print(f"Invalid duplicates sample:")
            print(duplicate_info['invalid_duplicates'][['ID', 'Name', 'Date']].head())

# %%
print(f"   Valid sequences: {valid_filename} ({len(final_df):,} sequences)")
print(f"   Invalid sequences: {invalid_filename} ({len(final_bad_df):,} sequences)")


# %% [markdown]
# # Combine protein data into one file

# %%
# Combine protein data into one file
print(f"\n{'='*60}")
print(f"🧬OMBINING PROTEIN DATA INTO ONE FILE")
print(f"{'='*60}")

# Load the three protein files
print(f" Loading protein sequence files...")
rbd_file = "results/genbank/rbd/tables/genbank_rbd_sequences.csv"

try:
    rbd_df = pd.read_csv(rbd_file)
    print(f" Successfully loaded all protein files")
except Exception as e:
    print(f" Error loading protein files: {e}")
    raise
# %%
print(rbd_df.columns)
# %%
# Convert sequences to amino acids and handle N's
print(f"\n🧬Converting sequences to amino acids...")

# Load reference sequence
with open("data/reference/ref_seq.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        ref_seq = str(record.seq)

# Function to fill N's with reference sequence
def fill_n_with_ref(seq, ref_seq, start_pos, length):
    """Fill N's in sequence with reference sequence"""
    ref_segment = ref_seq[start_pos:start_pos + length]
    filled_seq = list(seq)
    for i in range(len(seq)):
        if seq[i] == 'N':
            filled_seq[i] = ref_segment[i]
    return ''.join(filled_seq)

# Function to process each protein's sequences
def process_protein_sequences(df, protein_name, ref_seq):
    """Process sequences for a specific protein"""
    if f"{protein_name} Sequence" not in df.columns:
        print(f" No sequence column found for {protein_name}")
        return df
    
    # Get protein config
    protein_config = get_protein_config(protein_name, CONFIG_PATH, data_source=DATA_SOURCE)
    start_pos = protein_config['start_position']
    length = protein_config['length']
    
    print(f"\nProcessing {protein_name} sequences...")
    print(f"  Start position: {start_pos}")
    print(f"  Length: {length}")
    
    # Fill N's and convert to amino acids
    sequence_col = f"{protein_name} Sequence"
    aa_col = f"{protein_name} aa"
    
    # Handle N's first
    print(f"  Filling N's with reference sequence...")
    df[sequence_col] = df[sequence_col].apply(
        lambda x: fill_n_with_ref(x, ref_seq, start_pos, length) if isinstance(x, str) else x
    )
    
    # Convert to amino acids
    print(f"  Converting to amino acids...")
    df[aa_col] = dna_to_amino_acids(df[sequence_col].tolist())
    
    print(f" {protein_name} processing complete")
    return df

# Process each protein's sequences
rbd_df = process_protein_sequences(rbd_df, "rbd", ref_seq)

# %%
# Rename columns to match desired output format
print(f"\n Renaming columns...")
})
rbd_df = rbd_df.rename(columns={
    'rbd aa': 'rbd_aa_sequence',
    'Score': 'rbd_score'
})
# %%
print(rbd_df.columns)
display(rbd_df.head())

# %%
from datetime import datetime
# Add DateTime_Ordinal column
print(f"\n Adding DateTime_Ordinal column...")
def convert_to_ordinal(date_str):
    """Convert date string to ordinal date"""
    try:
        
        date_str = str(date_str).strip()
        # Common date formats in GenBank data
        formats = [
            '%Y-%m-%d',      # 2021-11-24
            '%Y-%m',         # 2021-11 (assume day 1)
            '%Y',            # 2021 (assume Jan 1)
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).toordinal()
            except ValueError:
                continue
        
        return None
    except:
        return None

# Add DateTime_Ordinal to each dataframe
for df in [rbd_df]:
    df['DateTime_Ordinal'] = df['Date'].apply(convert_to_ordinal)

# %%
display(rbd_df.head(2))
# %%
# Check for NA in DateTime_Ordinal column and print the length
rbd_df = rbd_df[rbd_df['DateTime_Ordinal'].notna()]
print(f"\n NA values in rbd DateTime_Ordinal column: {rbd_df['DateTime_Ordinal'].isna().sum()}")

# %%
# Merge dataframes on ID
print(f"\n Merging protein dataframes...")
combined_df = (rbd_df[['ID', 'rbd_aa_sequence', 'rbd_score']], 
    on='ID', 
    how='outer'
)
# %%
# Reorder columns to match desired format
desired_columns = [
    'Name', 'ID', 'Date', 'DateTime_Ordinal',
    'rbd_aa_sequence', 'rbd_score'
]
combined_df = combined_df[desired_columns]

# %%
display(combined_df.head())

# %%
# Save combined file
print(f"\n Saving combined data...")
output_dir = "results/genbank/tables"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/multi_protein_genbank_combined_data.csv"
combined_df.to_csv(output_file, index=False)

print(f"\n Successfully combined protein data:")
print(f"   Total sequences: {len(combined_df):,}")
print(f"   Saved to: {output_file}")
print(f"   Columns: {', '.join(combined_df.columns)}")

# Print some statistics
print(f"\n Data completeness:")
print(f"  RBD sequences: {combined_df['rbd_aa_sequence'].notna().sum():,}")
print(f"  Complete records (all proteins): {combined_df.notna().all(axis=1).sum():,}")

print(f"\n NA values in rbd_aa_sequence column: {combined_df['rbd_aa_sequence'].isna().sum()}")

# %%
