"""
Generic Functions for Viral Protein Sequence Analysis
Supports analysis of any viral protein (MPro, PLPro, etc.) with configurable parameters
"""

import warnings
import pandas as pd
import os
import yaml
from Bio.Align import PairwiseAligner
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
import matplotlib.image as mpimg
from matplotlib import font_manager
import matplotlib.pylab as pylab
from Bio import SeqIO

# Suppress specific matplotlib warnings
warnings.filterwarnings('ignore', message='Ignoring fixed.*limits.*data aspect')


def initialize_aligner(protein_name=None, config_path="../config/protein_configs.yaml", 
                      mode=None, mismatch_score=None, gap_open_score=None):
    """
    Initialize sequence aligner with configurable parameters from YAML or manual override
    
    Args:
        protein_name: Name of protein to get alignment config from YAML (optional)
        config_path: Path to the YAML configuration file
        mode: Alignment mode ('local' or 'global') - overrides YAML if provided
        mismatch_score: Penalty for mismatches - overrides YAML if provided
        gap_open_score: Penalty for gap opening - overrides YAML if provided
    
    Returns:
        PairwiseAligner object
    """
    # Get defaults from YAML if protein_name provided
    if protein_name:
        protein_config = get_protein_config(protein_name, config_path)
        alignment_config = protein_config.get('alignment', {})
        mode = mode or alignment_config.get('mode', 'local')
        mismatch_score = mismatch_score or alignment_config.get('mismatch_score', -1)
        gap_open_score = gap_open_score or alignment_config.get('open_gap_score', -2)
    else:
        # Use provided values or defaults
        mode = mode or 'local'
        mismatch_score = mismatch_score or -1
        gap_open_score = gap_open_score or -2
    
    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = gap_open_score
    return aligner


def process_record(record, aligner, ref_seq, protein_config):
    """
    Process a sequence record using appropriate method based on length
    
    Args:
        record: BioPython sequence record
        aligner: PairwiseAligner object
        ref_seq: Reference sequence string
        protein_config: Protein configuration dictionary from YAML
    
    Returns:
        tuple: (query_start, query_end, alignment_score)
    """
    try: 
        expected_length = protein_config['length']
        long_seq_threshold = protein_config['min_sequence_length']
        
        if len(str(record.seq)) > long_seq_threshold:
            return process_long_sequence(record, aligner, ref_seq, protein_config)
        else:
            return process_short_sequence(record, aligner, ref_seq)
    except Exception as e:
        print(f"Error processing record {record.id}: {e}")
        return None, None, None


def process_long_sequence(record, aligner, ref_seq, protein_config, step_size=500, max_iterations=5):
    """
    Process long sequences using windowed alignment approach
    
    Args:
        record: BioPython sequence record
        aligner: PairwiseAligner object
        ref_seq: Reference sequence string
        protein_config: Protein configuration dictionary from YAML
        step_size: Step size for expanding search window
        max_iterations: Maximum iterations before falling back to full alignment
    
    Returns:
        tuple: (query_start, query_end, alignment_score)
    """
    try: 
        expected_length = protein_config['length']
        # Calculate initial start and end from YAML config
        start_position = protein_config['start_position']
        start_offset = protein_config['search_window']['start_offset']
        end_offset = protein_config['search_window']['end_offset']
        initial_start = start_position + start_offset
        initial_end = start_position + end_offset
        
        j = 0
        while True:
            start_idx = initial_start - j * step_size
            end_idx = initial_end + j * step_size
            alignments = aligner.align(ref_seq, str(record.seq)[start_idx:end_idx])
            best_alignment = alignments[0]
            query_start = best_alignment.coordinates[1, 0] + start_idx
            query_end = best_alignment.coordinates[1, -1] + start_idx
    
            if (query_end - query_start == expected_length):
                break
            j += 1
            if j == max_iterations:
                # Fall back to full sequence alignment
                alignments = aligner.align(ref_seq, str(record.seq))
                best_alignment = alignments[0]
                query_start = best_alignment.coordinates[1, 0]
                query_end = best_alignment.coordinates[1, -1]
                break
        return query_start, query_end, best_alignment.score
    except Exception as e:
        print(f"Error processing long sequence {record.id}: {e}")
        return None, None, None


def process_short_sequence(record, aligner, ref_seq):
    """
    Process short sequences using full alignment
    
    Args:
        record: BioPython sequence record
        aligner: PairwiseAligner object
        ref_seq: Reference sequence string
    
    Returns:
        tuple: (query_start, query_end, alignment_score)
    """
    try: 
        alignments = aligner.align(ref_seq, str(record.seq))
        best_alignment = alignments[0]
        query_start = best_alignment.coordinates[1, 0]
        query_end = best_alignment.coordinates[1, -1]
        return query_start, query_end, best_alignment.score
    except Exception as e:
        print(f"Error processing short sequence {record.id}: {e}")
        return None, None, None


def create_row(record, query_start, query_end, score, protein_name, bad=False):
    """
    Create a dictionary row from processed sequence data
    
    Args:
        record: BioPython sequence record
        query_start: Start position of aligned sequence
        query_end: End position of aligned sequence
        score: Alignment score
        protein_name: Name of target protein (e.g., 'mpro', 'plpro')
        bad: Whether this is a "bad" alignment
    
    Returns:
        dict: Row data for DataFrame
    """
    try: 
        match = record.description.split("|")
        sequence_col = f"{protein_name} Sequence"
        
        if not bad: 
            return {
                "Name": match[1],
                "ID": match[0],
                "Date": match[2],
                sequence_col: str(record.seq)[query_start:query_end], 
                "Score": score
            }
        else: 
            return {
                "Name": match[1],
                "ID": match[0],
                "Date": match[2],
                sequence_col: str(record.seq), 
                "Score": score
            }
    except Exception as e:
        print(f"Error creating row for record {record.id}: {e}")
        return None


def worker(queue, aligner, ref_seq, output_dir, worker_id, protein_config):
    """
    Worker function to process records from the queue in parallel
    
    Args:
        queue: Multiprocessing queue containing sequence records
        aligner: PairwiseAligner object
        ref_seq: Reference sequence string
        output_dir: Directory to save temporary results
        worker_id: Unique identifier for this worker
        protein_config: Protein configuration dictionary from YAML
    """
    protein_name = protein_config['name']
    expected_length = protein_config['length']
    sequence_col = f"{protein_name} Sequence"
    columns = ['Name', 'ID', 'Date', sequence_col, 'Score']
    
    # Use lists to collect rows, then create DataFrame at the end
    good_rows = []
    bad_rows = []
    
    while True:
        record = queue.get()
        if record is None:  # Sentinel value to indicate no more records
            break
        
        query_start, query_end, score = process_record(record, aligner, ref_seq, protein_config)
        if query_start is None or query_end is None:
            continue
            
        if len(str(record.seq)[query_start:query_end]) == expected_length:
            row = create_row(record, query_start, query_end, score, protein_name)
            if row is not None:
                good_rows.append(row)
        else:
            row = create_row(record, query_start, query_end, score, protein_name, bad=True)
            if row is not None:
                bad_rows.append(row)

    # Create DataFrames from collected rows
    df_chunk = pd.DataFrame(good_rows, columns=columns) if good_rows else pd.DataFrame(columns=columns)
    bad_df_chunk = pd.DataFrame(bad_rows, columns=columns) if bad_rows else pd.DataFrame(columns=columns)
    
    # Save results to temporary files with protein-specific naming
    df_chunk.to_csv(os.path.join(output_dir, f"{protein_name}_df_chunk_{worker_id}.csv"), index=False)
    bad_df_chunk.to_csv(os.path.join(output_dir, f"{protein_name}_bad_df_chunk_{worker_id}.csv"), index=False)


def dna_to_amino_acids(dna_sequences):
    """
    Convert DNA sequences to amino acid sequences using standard genetic code
    
    Args:
        dna_sequences: List of DNA sequence strings
    
    Returns:
        list: List of amino acid sequence strings
    """
    # Standard genetic code table
    aa_table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', # '_' means STOP codon
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }

    amino_acid_sequences = []

    for dna_sequence in dna_sequences:
        # Convert to uppercase and remove any whitespace
        dna_sequence = dna_sequence.upper().replace(" ", "")

        # Translate DNA to amino acids
        amino_acids = ""
        for i in range(0, len(dna_sequence), 3):
            codon_aa = dna_sequence[i:i+3]
            if len(codon_aa) == 3:
                amino_acid = aa_table.get(codon_aa, 'X')  # 'X' for unknown amino acids
                amino_acids += amino_acid

        amino_acid_sequences.append(amino_acids)

    return amino_acid_sequences


def load_protein_config(config_path="src/config/protein_configs.yaml"):
    """
    Load protein configuration from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f" Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f" Error parsing YAML file: {e}")
        return {}


def get_protein_config(protein_name, config_path="src/config/protein_configs.yaml", data_source='genbank'):
    """
    Get configuration for a specific protein
    
    Args:
        protein_name: Name of the protein (mpro, plpro, rbd)
        config_path: Path to the YAML configuration file
        data_source: Data source ('genbank' or 'gisaid')
    
    Returns:
        dict: Protein configuration
    """
    full_config = load_protein_config(config_path)
    
    # Get the data source config
    if data_source in full_config and 'proteins' in full_config[data_source]:
        proteins_config = full_config[data_source]['proteins']
        if protein_name in proteins_config:
            return proteins_config[protein_name]
    
    print(f" Protein '{protein_name}' configuration not found in {data_source}")
    return {}


def get_processing_config(config_path="src/config/protein_configs.yaml", data_source='genbank'):
    """
    Get processing configuration parameters from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        data_source: Data source ('genbank' or 'gisaid')
    
    Returns:
        dict: Processing configuration parameters
    """
    full_config = load_protein_config(config_path)
    
    # Get the data source config
    if data_source in full_config and 'processing' in full_config[data_source]:
        return full_config[data_source]['processing']
    
    print(f" Processing configuration not found for {data_source}")
    return {}


def get_output_config(config_path="src/config/protein_configs.yaml", data_source='genbank'):
    """
    Get output configuration parameters from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        data_source: Data source to get config for ('genbank' or 'gisaid')
    
    Returns:
        dict: Output configuration parameters
    """
    full_config = load_protein_config(config_path)
    
    # Get the data source config
    if data_source in full_config and 'output' in full_config[data_source]:
        return full_config[data_source]['output']
    
    print(f" Output configuration not found for {data_source}")
    return {}
