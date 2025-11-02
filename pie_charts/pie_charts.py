# %% [markdown]
# # Make figures for the paper
# This notebook makes pie charts for the paper. 
# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import glob
from matplotlib.colors import LinearSegmentedColormap
#from PyPDF2 import PdfMerger  # Add this import
import json

print("Imported libraries")

# Set working directory to project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
print(f" Set working directory to project root: {project_root}")
#color_key_path = 'data/genbank_rbd_all_sequences_colors.json'
#color_key_path = 'data/VOC_colorkey.json'
color_key_path = 'data/VOCafterB2_colorkey.json'
# %%
# Functions: 
def format_to_3_sig_figs(number):
    """Format a number to 3 significant figures"""
    if number == 0:
        return "0.00%"
    if number >= 100:
        return "100%"
    if number >= 10:
        return f"{number:.1f}%"  # One decimal place for numbers >= 10
    import math
    magnitude = math.floor(math.log10(abs(number)))
    decimal_places = max(0, 2 - magnitude)
    return f"{number:.{decimal_places}}%"  # Add % sign directly here

def adjust_mutation_position(mutation):
    """Adjust mutation position by adding 330 to match RBD numbering"""
    import re
    match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
    if match:
        ref, pos, mut = match.groups()
        new_pos = int(pos) + 330
        return f"{ref}{new_pos}{mut}"
    return mutation

def create_prettier_pie_labels(ax, wedges, labels, sizes, angle_threshold=15, 
                           radius=1.5, distance_increment=0.40,
                           num_distance_levels=3):
    """
    Create prettier pie chart labels with connecting lines using angle threshold logic
    """
    current_level = 0  # Track current distance level
    cumulative_angle = 0  # Track cumulative angle
    
    for i, (wedge, label) in enumerate(zip(wedges, labels)):
        # Calculate angle span of the wedge
        angle_span = abs(wedge.theta2 - wedge.theta1)
        cumulative_angle += angle_span
        
        # Reset cycle if we've accumulated enough angle
        if cumulative_angle >= angle_threshold or current_level == num_distance_levels:
            current_level = 0
            cumulative_angle = 0
        
        # Get the center angle of the wedge
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        ang_rad = np.deg2rad(ang)
        
        # Wedge edge position for connecting line start
        wedge_x = 1.1 * np.cos(ang_rad)
        wedge_y = 1.1 * np.sin(ang_rad)
        
        # Calculate label radius based on current level
        label_radius = radius + (current_level * distance_increment)
        
        # Calculate final position keeping radial direction
        label_x = label_radius * np.cos(ang_rad)
        label_y = label_radius * np.sin(ang_rad)
        
        # Draw connecting line
        ax.plot([wedge_x, label_x], [wedge_y, label_y], 
                color='gray', linewidth=1, alpha=0.7, zorder=1)
        
        # Add label with box
        bbox_props = dict(boxstyle="round,pad=0.42", fc="white", ec="gray", alpha=0.8)
        ax.text(label_x, label_y, label, ha='center', va='center', 
                fontsize=13, bbox=bbox_props, zorder=10)  
        
        # Update level for next label
        current_level = current_level + 1

def get_mutations(sequence, reference_sequence):
    """Get mutations between a sequence and reference sequence."""
    mutations = []
    for i, (ref, mut) in enumerate(zip(reference_sequence, sequence)):
        if ref != mut:
            mutations.append(f"{ref}{i+1}{mut}")
    return mutations

def create_mutation_df(sequences, reference_sequence):
    """Create a DataFrame of mutation counts from sequences."""
    total_sequences = len(sequences)
    all_mutations = []
    for seq in sequences:
        mutations = get_mutations(seq, reference_sequence)
        # Adjust positions in mutations
        mutations = [adjust_mutation_position(mut) for mut in mutations]
        all_mutations.extend(mutations)
    
    mutation_counts = Counter(all_mutations)
    df = pd.DataFrame(list(mutation_counts.items()), columns=['Mutation', 'Count'])
    df = df.sort_values('Count', ascending=False)
    
    # Calculate frequencies
    total_mutations = df['Count'].sum()
    df['frequency'] = df['Count'] / total_sequences
    return df

def aggregate_mutations_by_position(mutation_df):
    """Aggregate mutations by position and calculate total counts and frequencies."""
    # Extract positions from mutations
    mutation_df['Position'] = mutation_df['Mutation'].str[1:-1]
    
    # Group by position and sum counts
    position_df = mutation_df.groupby('Position').agg({
        'Count': 'sum',
        'frequency': 'sum'
    }).reset_index()
    
    # Sort by total count
    position_df = position_df.sort_values('Count', ascending=False)
    # For each position, get all mutations at that position
    position_df['Mutations'] = position_df['Position'].apply(
        lambda pos: ', '.join(
            [f"{mut}({format_to_3_sig_figs(freq*100)})" 
             for mut, freq in zip(
                 mutation_df[mutation_df['Position'] == pos]['Mutation'],
                 mutation_df[mutation_df['Position'] == pos]['frequency']
             )]
        )
    )
    return position_df

def create_single_pie_chart(mutation_df, title_prefix, output_dir, use_color_key=False, score_threshold=-100):
    """Create a single pie chart showing top 20 mutations."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load color key if needed
    color_key = None
    if use_color_key:
        import json
        try:
            with open(color_key_path, 'r') as f:
                color_key = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load color key file: {e}")
            use_color_key = False
    
    # Always use top 20 individual mutations (not positions)
    top_data = mutation_df.sort_values('Count', ascending=False).head(20)
    
    # Get sizes for pie chart
    sizes = top_data['Count'].tolist()
    colors = []
    labels_data = []
    
    # Prepare colors based on mutation positions
    for i, row in enumerate(top_data.itertuples()):
        if use_color_key and color_key is not None:
            position = row.Mutation[1:-1]  # Extract position from mutation
            if position and position in color_key:
                colors.append(color_key[position])
            else:
                colors.append(color_key.get('others', '#CCCCCC'))
        else:
            colors.append(plt.cm.cool(1 - (i / (len(top_data) - 1) if len(top_data) > 1 else 0)))
        labels_data.append(row)
    
    # Add "Others" wedge for GenBank data if there are remaining mutations
    start_angle = 0
    is_genbank = 'GenBank' in title_prefix
    if is_genbank and len(mutation_df) > 20:
        remaining_mutations = mutation_df.iloc[20:]  # Get mutations beyond top 20
        others_count = remaining_mutations['Count'].sum()
        
        # Relative frequency: for slice sizing within this chart
        total_mutations_in_chart = top_data['Count'].sum() + others_count
        others_relative_freq_for_sizing = others_count / total_mutations_in_chart
        
        # Use the pie slice percentage for the label
        others_percentage = others_relative_freq_for_sizing * 100
        
        # Add Others to the data
        sizes.append(others_count)
        colors.append('lightgray')
        
        # Create a dummy row for Others labeling (use pie slice percentage)
        from collections import namedtuple
        OthersRow = namedtuple('OthersRow', ['Mutation', 'frequency'])
        others_row = OthersRow('Others', others_relative_freq_for_sizing)  # Use the actual slice proportion
        labels_data.append(others_row)
        
        # Calculate start angle to position "Others" wedge pointing right
        others_angle_span = others_relative_freq_for_sizing * 360
        start_angle = others_angle_span / 2  # Center "Others" at 0 degrees
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Create pie chart with calculated start angle
    wedges, _ = ax.pie(sizes, colors=colors, startangle=start_angle,
                      radius=1.2, wedgeprops={'edgecolor': 'white', 'linewidth': 0.5})
    
    # Add labels directly on wedges instead of using pretty labels
    total_size = sum(sizes)
    for i, (wedge, label_data) in enumerate(zip(wedges, labels_data)):
        percentage = sizes[i] / total_size
        
        # Only show label if wedge is big enough (>2% of total)
        if percentage > 0.0000:
            # Calculate angle at center of wedge
            ang = (wedge.theta2 + wedge.theta1) / 2.
            ang_rad = np.deg2rad(ang)
            
            # Calculate text position (slightly more inward)
            radius = 0.9  # Distance from center (moved inward from 0.8)
            x = radius * np.cos(ang_rad)
            y = radius * np.sin(ang_rad)
            
            # Format label
            if hasattr(label_data, 'Mutation') and label_data.Mutation == 'Others':
                label = f"Others - {format_to_3_sig_figs(label_data.frequency*100)}"
            else:
                label = f"{label_data.Mutation} - {format_to_3_sig_figs(label_data.frequency*100)}"
            
            # Calculate rotation angle
            rotation = ang
            if ang > 90 and ang <= 270:
                rotation += 180
            
            # Add text
            ax.text(x, y, label,
                   rotation=rotation,
                   rotation_mode='anchor',
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=10)
    
    # Set title with more padding and include score threshold if needed
    if 'Train' in title_prefix:
        title = f"{title_prefix}\nTop 20 Mutations\nScore Threshold ≥ {score_threshold}"
    else:
        title = f"{title_prefix}\nTop 20 Mutations"
    ax.set_title(title, pad=20, fontsize=24)
    
    # Set consistent bounds
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    
    # Save figure
    base_filename = title_prefix.replace(' ', '_').title()
    plt.savefig(f"{output_dir}/{base_filename}.png", 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.savefig(f"{output_dir}/{base_filename}.pdf", 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close(fig)

def create_nested_pie_chart(mutation_df, title_prefix, output_dir, use_color_key=False, score_threshold=-100):
    """Create a nested pie chart with positions in inner ring and mutations in outer ring."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load color key if needed
    color_key = None
    if use_color_key:
        import json
        try:
            with open(color_key_path, 'r') as f:
                color_key = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load color key file: {e}")
            use_color_key = False
    
    # Aggregate mutations by position first
    position_df = aggregate_mutations_by_position(mutation_df)
    
    # Take top 20 positions (to keep visualization manageable)
    top_positions = position_df.head(20)
    
    # Prepare data for inner ring (positions)
    inner_sizes = top_positions['Count'].tolist()
    inner_labels = [f"{pos}" for pos in top_positions['Position']]
    
    # Prepare colors for positions
    position_colors = []
    for pos in top_positions['Position']:
        if use_color_key and color_key is not None and pos in color_key:
            position_colors.append(color_key[pos])
        else:
            position_colors.append('#CCCCCC')
    
    # Prepare data for outer ring (mutations)
    outer_sizes = []
    outer_labels = []
    outer_colors = []
    position_boundaries = []  # Track position boundaries for largest wedge logic
    mutation_frequencies = []  # Store mutation frequencies for comparison
    
    # For each position, get its mutations
    for pos in top_positions['Position']:
        # Mark start of this position's mutations
        start_idx = len(outer_sizes)
        
        # Get mutations for this position
        pos_mutations = mutation_df[mutation_df['Mutation'].str[1:-1] == pos]
        pos_mutations = pos_mutations.sort_values('Count', ascending=False)
        
        # Add each mutation's count and label
        for _, mut_row in pos_mutations.iterrows():
            outer_sizes.append(mut_row['Count'])
            mutation_frequencies.append(mut_row['frequency'])
            outer_labels.append(f"{mut_row['Mutation']} - {format_to_3_sig_figs(mut_row['frequency']*100)}")
            # Use the same color as the position
            if use_color_key and color_key is not None and pos in color_key:
                outer_colors.append(color_key[pos])
            else:
                outer_colors.append('#CCCCCC')
        
        # Mark end of this position's mutations
        end_idx = len(outer_sizes)
        position_boundaries.append((start_idx, end_idx))
    
    # Print comparison for the first mutation of the first position
    total_outer_size = sum(outer_sizes)
    first_mutation_pie_percentage = (outer_sizes[0] / total_outer_size) * 100
    first_mutation_frequency_percentage = mutation_frequencies[0] * 100
    print(f"\nPercentage comparison for first mutation ({outer_labels[0].split(' - ')[0]}):")
    print(f"  Percentage of pie (relative to shown mutations): {first_mutation_pie_percentage:.2f}%")
    print(f"  Frequency in dataset (relative to all sequences): {first_mutation_frequency_percentage:.2f}%")
    print(f"  These numbers should be different as they use different denominators")
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))  # Larger figure for nested chart
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Create inner pie (positions)
    inner_wedges, _ = ax.pie(inner_sizes, colors=position_colors, radius=.7,
                            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1))
    
    # Create outer pie (mutations)
    outer_wedges, _ = ax.pie(outer_sizes, colors=outer_colors, radius=1,
                            wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1))

    # Add labels to inner wedges
    for i, (wedge, label) in enumerate(zip(inner_wedges, inner_labels)):
        ang = (wedge.theta2 + wedge.theta1)/2.
        ang_rad = np.deg2rad(ang)
        radius = 0.7  # radius of inner ring
        wedge_width = 0.5  # width of inner ring
        text_radius = radius - (wedge_width / 2)  # place text in middle of wedge
        x = text_radius * np.cos(ang_rad)
        y = text_radius * np.sin(ang_rad)
        """rotation = ang
        if ang > 90 and ang <= 270:
            rotation += 180"""
        ax.text(x, y, label, 
               #rotation=rotation,
               rotation_mode='anchor',
               horizontalalignment='center',
               verticalalignment='center',
               fontsize=12)
    # Add labels to outer wedges special logic for largest wedge per position
    total_outer_size = sum(outer_sizes)
    for i, (wedge, label) in enumerate(zip(outer_wedges, outer_labels)):
        should_show_label = False
        
        # Check if this wedge is >= 1% of total
        if outer_sizes[i] / total_outer_size >= 0.01:
            should_show_label = True
        else:
            # Check if this is the largest wedge for its position
            for start_idx, end_idx in position_boundaries:
                if start_idx <= i < end_idx:
                    # This wedge belongs to this position
                    # Check if it's the largest in this position (first one since sorted by Count desc)
                    if i == start_idx:
                        should_show_label = True
                    break
        
        if should_show_label:
            ang = (wedge.theta2 + wedge.theta1)/2.
            
            # Convert angle to radians for text placement
            ang_rad = np.deg2rad(ang)
            
            # Calculate position for text (in the middle of the wedge)
            radius = 1.0  # radius of outer ring
            wedge_width = 0.3  # width of outer ring
            text_radius = radius - (wedge_width / 2)  # place text in middle of wedge
            
            x = text_radius * np.cos(ang_rad)
            y = text_radius * np.sin(ang_rad)
            
            # Determine rotation angle for text
            rotation = ang
            if ang > 90 and ang <= 270:
                rotation += 180
            
            # Add text
            ax.text(x, y, label, 
                   rotation=rotation,
                   rotation_mode='anchor',
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=12)
    
    # Set title
    if 'Train' in title_prefix:
        title = f"{title_prefix}\nTop 20 Positions\nScore Threshold ≥ {score_threshold}"
    else:
        title = f"{title_prefix}\nTop 20 Positions"
    ax.set_title(title, pad=20, fontsize=24)
    
    # Save figure
    base_filename = title_prefix.replace(' ', '_').title()
    plt.savefig(f"{output_dir}/{base_filename}_Nested.png", 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.savefig(f"{output_dir}/{base_filename}_Nested.pdf", 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close(fig)

def read_fasta_sequence(fasta_file):
    """Read a FASTA file and return the sequence."""
    with open(fasta_file) as f:
        lines = f.readlines()
    # Skip header line and join all other lines, removing whitespace
    sequence = ''.join(line.strip() for line in lines[1:])
    return sequence

def load_and_filter_data(file_path, score_threshold=-100):
    """Load CSV data and filter out sequences with score below threshold."""
    try:
        # Load data with proper column names
        data = pd.read_csv(file_path, header=None)
        data.columns = ['id', 'sequence', 'score']
        
        # Filter out sequences with score below threshold
        original_count = len(data)
        data = data[data['score'] >= score_threshold]
        filtered_count = len(data)
        
        print(f"  Loaded {original_count} sequences, filtered to {filtered_count} sequences (score >= {score_threshold})")
        return data
    except Exception as e:
        print(f"  Error loading {file_path}: {str(e)}")
        return None

def find_simulation_files(data_dir):
    """Find all simulation files in the given directory."""
    pattern = os.path.join(data_dir, "TESTEVAL_*.csv")
    sim_files = glob.glob(pattern)
    return sorted(sim_files)

def extract_simulation_name(file_path):
    """Extract a meaningful name from simulation file path."""
    filename = os.path.basename(file_path)
    
    # Extract key parameters from filename
    if 'alldata' in filename:
        return 'alldata'
    elif 'baseonly' in filename:
        return 'baseonly'
    else:
        # Fallback: use part of filename
        base_name = filename.replace('TESTEVAL_', '').replace('_sim_table.csv', '')
        # Take first meaningful part
        parts = base_name.split('_')
        if len(parts) > 2:
            return f"{parts[1]}_{parts[2]}"  # e.g., "mlp_l[8,8,8]"
        else:
            return base_name[:20]  # First 20 chars

def threshold_comparison_train(train_types=['baseonly', 'alldata'], 
                             thresholds=[-100, -1.74, -1.5, -1.0, -0.5, 0.0, 0.1, 0.2, 0.3]):
    """Create a subplot comparing different score thresholds for each training data type."""
    print("\nCreating threshold comparison subplots...")
    
    # Set up the figure
    fig = plt.figure(figsize=(45, 15))  # Increased size to accommodate labels
    gs = fig.add_gridspec(3, 9, wspace=0, hspace=0)  # Increased spacing
    
    # Get reference sequence once
    reference_sequence = read_fasta_sequence('data/rbd_ref_aa.fasta')
    
    # Load color key
    try:
        import json
        with open(color_key_path, 'r') as f:
            color_key = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load color key file: {e}")
        color_key = None
    
    # Process each training type
    for row, train_type in enumerate(train_types):
        print(f"\nProcessing {train_type} for threshold comparison...")
        
        # Get all CSV files for this training type
        train_dir = f"data/train_data/{train_type}"
        
        # Get all CSV files including train_data.csv
        csv_files = glob.glob(os.path.join(train_dir, "*.csv"))
        
        if not csv_files:
            print(f"Warning: No CSV files found for {train_type}")
            continue
            
        print(f"Found {len(csv_files)} files for {train_type}:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
            
        # Load all sequences first
        all_sequences = []
        all_scores = []
        for file in csv_files:
            try:
                # Load data without filtering
                data = pd.read_csv(file, header=None)
                data.columns = ['id', 'sequence', 'score']
                all_sequences.extend(data['sequence'].tolist())
                all_scores.extend(data['score'].tolist())
                print(f"    Loaded {len(data)} sequences from {os.path.basename(file)}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
        
        if not all_sequences:
            print(f"No valid sequences found for {train_type}")
            continue
            
        print(f"Total sequences for {train_type}: {len(all_sequences)}")
            
        # Convert to DataFrame for easier filtering
        combined_data = pd.DataFrame({
            'sequence': all_sequences,
            'score': all_scores
        })
        
        # Process each threshold
        for col, threshold in enumerate(thresholds):
            print(f"  Processing threshold {threshold}")
            
            # Filter data for this threshold
            filtered_data = combined_data[combined_data['score'] >= threshold]
            filtered_sequences = filtered_data['sequence'].tolist()
            
            if not filtered_sequences:
                print(f"  No sequences pass threshold {threshold}")
                continue
                
            # Create mutation DataFrame
            mutations_df = create_mutation_df(filtered_sequences, reference_sequence)
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Take top 20 mutations
            top_mutations = mutations_df.sort_values('Count', ascending=False).head(20)
            
            # Prepare pie chart data
            sizes = []
            colors = []
            
            for i, (_, mutation_row) in enumerate(top_mutations.iterrows()):
                sizes.append(mutation_row['Count'])
                
                # Get color from color key based on mutation position
                if color_key is not None:
                    position = mutation_row['Mutation'][1:-1]
                    if position and position in color_key:
                        colors.append(color_key[position])
                    else:
                        colors.append(color_key.get('others', '#CCCCCC'))
                else:
                    colors.append(plt.cm.cool(1 - (i / (len(top_mutations) - 1) if len(top_mutations) > 1 else 0)))
            
            # Create pie chart
            wedges, _ = ax.pie(sizes, colors=colors, startangle=0,
                             radius=1.0, wedgeprops={'edgecolor': 'white', 'linewidth': 0.5})
            
            # Improved text placement
            total_size = sum(sizes)
            cumsum = np.cumsum(sizes)
            cumsum = np.insert(cumsum, 0, 0)  # Add starting point
            
            # Calculate angles for each wedge
            angles = 2 * np.pi * cumsum / total_size
            
            for i, (wedge, (_, mutation_row)) in enumerate(zip(wedges, top_mutations.iterrows())):
                percentage = sizes[i] / total_size
                
                # Only show label if wedge is big enough (>2% of total)
                if percentage > 0.0000:
                    # Calculate angle at center of wedge
                    theta = (angles[i] + angles[i + 1]) / 2
                    
                    # Calculate text position (slightly outside the wedge)
                    radius = 0.85  # Consistent distance from center
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    
                    # Calculate rotation angle
                    rotation_angle = np.degrees(theta)
                    if rotation_angle > 90 and rotation_angle <= 270:
                        rotation_angle += 180
                    
                    # Add mutation text with consistent alignment
                    ax.text(x, y, f"   {mutation_row['Mutation']}   ",
                           rotation=rotation_angle,
                           size=8,
                           horizontalalignment='center',
                           verticalalignment='center',
                           rotation_mode='anchor')
            
            # Add title
            ax.set_title(f"{train_type.capitalize()}\nScore Threshold: {threshold}\nN={len(filtered_sequences)}", 
                        fontsize=12, pad=0)
            
            # Set aspect ratio to be equal (circular pie)
            ax.set_aspect('equal')
            
            # Set consistent bounds for all subplots
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
    
    # Save the comparison figure
    output_dir = "figures/pie_charts/train/afterB2/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(f"{output_dir}/threshold_comparison_train.png", 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.savefig(f"{output_dir}/threshold_comparison_train.pdf", 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close(fig)
    print("\nThreshold comparison plots saved!")


def threshold_comparison_train_eval(train_types=['baseonly', 'alldata'], 
                                   thresholds=[-100, 0.0]):
    """Create a subplot comparing different score thresholds for train eval data."""
    print("\nCreating train eval threshold comparison subplots...")
    
    # Set up the figure - 2 rows, 9 columns
    fig = plt.figure(figsize=(45, 10))  # Wide figure for 9 thresholds, 2 rows
    gs = fig.add_gridspec(2, 9, wspace=0, hspace=0)
    
    # Get reference sequence
    reference_sequence = read_fasta_sequence('data/rbd_ref_aa.fasta')
    
    # Load color key
    try:
        import json
        with open(color_key_path, 'r') as f:
            color_key = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load color key file: {e}")
        color_key = None
    
    # Define the eval files for each train type
    eval_files = {
        'baseonly': 'data/train_data/train_eval/TRAINEVAL_mlp_l[16,32,16]_wdecay0.0001_learningrate0.0001_baseonly_nulltuner_multiattempt_train_pred_table.csv',
        'alldata': 'data/train_data/train_eval/TRAINEVAL_mlp_l[8,8,8]_wdecay0.0005_learningrate0.0001_alldata_fan8_multiattempt_train_pred_table.csv'
    }
    
    # Load the train data once (shared between both)
    train_data_file = 'data/train_data/train_eval/train_data.csv'
    try:
        train_data = pd.read_csv(train_data_file, header=None)
        train_data.columns = ['id', 'sequence', 'score1']
        print(f"Loaded {len(train_data)} sequences from train_data.csv")
    except Exception as e:
        print(f"Error loading train data: {e}")
        return
    
    # Process each training type
    for row, train_type in enumerate(train_types):
        print(f"\nProcessing {train_type} train eval...")
        
        # Load the eval prediction file
        eval_file = eval_files[train_type]
        try:
            eval_data = pd.read_csv(eval_file, header=None)
            print(f"Loaded eval data with columns: {eval_data.columns.tolist()}")
            
            # Use the 3th column as the score (index 2)
            score_column = eval_data.columns[2]
            print(f"Using score column: {score_column}")
            
            # Combine train data sequences with eval scores
            combined_data = pd.DataFrame({
                'sequence': train_data['sequence'],
                'score': eval_data[score_column]
            })
            
        except Exception as e:
            print(f"Error loading {train_type} eval data: {e}")
            continue
        
        # Process each threshold
        for col, threshold in enumerate(thresholds):
            print(f"  Processing threshold {threshold}")
            
            # Filter data for this threshold
            filtered_data = combined_data[combined_data['score'] >= threshold]
            filtered_sequences = filtered_data['sequence'].tolist()
            
            if not filtered_sequences:
                print(f"  No sequences pass threshold {threshold}")
                continue
                
            # Create mutation DataFrame
            mutations_df = create_mutation_df(filtered_sequences, reference_sequence)
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Take top 20 mutations
            top_mutations = mutations_df.sort_values('Count', ascending=False).head(20)
            
            # Prepare pie chart data
            sizes = []
            colors = []
            
            for i, (_, mutation_row) in enumerate(top_mutations.iterrows()):
                sizes.append(mutation_row['Count'])
                
                # Get color from color key based on mutation position
                if color_key is not None:
                    position = mutation_row['Mutation'][1:-1]
                    if position and position in color_key:
                        colors.append(color_key[position])
                    else:
                        colors.append(color_key.get('others', '#CCCCCC'))
                else:
                    colors.append(plt.cm.cool(1 - (i / (len(top_mutations) - 1) if len(top_mutations) > 1 else 0)))
            
            # Create pie chart
            wedges, _ = ax.pie(sizes, colors=colors, startangle=0,
                             radius=1.0, wedgeprops={'edgecolor': 'white', 'linewidth': 0.5})
            
            # Improved text placement
            total_size = sum(sizes)
            cumsum = np.cumsum(sizes)
            cumsum = np.insert(cumsum, 0, 0)  # Add starting point
            
            # Calculate angles for each wedge
            angles = 2 * np.pi * cumsum / total_size
            
            for i, (wedge, (_, mutation_row)) in enumerate(zip(wedges, top_mutations.iterrows())):
                percentage = sizes[i] / total_size
                
                # Only show label if wedge is big enough (>1.5% of total)
                if percentage > 0.0000:
                    # Calculate angle at center of wedge
                    theta = (angles[i] + angles[i + 1]) / 2
                    
                    # Calculate text position (slightly outside the wedge)
                    radius = 0.85  # Consistent distance from center
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    
                    # Calculate rotation angle
                    rotation_angle = np.degrees(theta)
                    if rotation_angle > 90 and rotation_angle <= 270:
                        rotation_angle += 180
                    
                    # Add mutation text with consistent alignment
                    ax.text(x, y, f"   {mutation_row['Mutation']}   ",
                           rotation=rotation_angle,
                           size=8,
                           horizontalalignment='center',
                           verticalalignment='center',
                           rotation_mode='anchor')
            
            # Add title
            ax.set_title(f"{train_type.capitalize()} Eval\nScore Threshold: {threshold}\nN={len(filtered_sequences)}", 
                        fontsize=12, pad=0)
            
            # Set aspect ratio to be equal (circular pie)
            ax.set_aspect('equal')
            
            # Set consistent bounds for all subplots
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
    
    # Save the comparison figure
    output_dir = "figures/pie_charts/train_eval"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(f"{output_dir}/threshold_comparison_train_eval.png", 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.savefig(f"{output_dir}/threshold_comparison_train_eval.pdf", 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close(fig)
    print("\nTrain eval threshold comparison plots saved!")

def create_train_eval_individual_charts():
    """Create individual pie charts for train eval data with score threshold >= 0."""
    print("\nCreating individual train eval pie charts...")
    
    # Get reference sequence
    reference_sequence = read_fasta_sequence('data/rbd_ref_aa.fasta')
    
    # Create output directory
    output_dir = "figures/pie_charts/train_eval"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define the eval files for each train type
    eval_files = {
        'baseonly': 'data/train_data/train_eval/TRAINEVAL_mlp_l[16,32,16]_wdecay0.0001_learningrate0.0001_baseonly_nulltuner_multiattempt_train_pred_table.csv',
        'alldata': 'data/train_data/train_eval/TRAINEVAL_mlp_l[8,8,8]_wdecay0.0005_learningrate0.0001_alldata_fan8_multiattempt_train_pred_table.csv'
    }
    
    # Load the train data once (shared between both)
    train_data_file = 'data/train_data/train_eval/train_data.csv'
    try:
        train_data = pd.read_csv(train_data_file, header=None)
        train_data.columns = ['id', 'sequence', 'score1']
        print(f"Loaded {len(train_data)} sequences from train_data.csv")
    except Exception as e:
        print(f"Error loading train data: {e}")
        return
    
    # Process each training type
    for train_type in ['baseonly', 'alldata']:
        print(f"\nProcessing {train_type} train eval individual chart...")
        
        # Load the eval prediction file
        eval_file = eval_files[train_type]
        try:
            eval_data = pd.read_csv(eval_file, header=None)
            
            # Use the 3th column as the score (index 2)
            score_column = eval_data.columns[2]
            print(f"Using score column: {score_column}")
            
            # Combine train data sequences with eval scores
            combined_data = pd.DataFrame({
                'sequence': train_data['sequence'],
                'score': eval_data[score_column]
            })
            
            # Filter for score >= 0
            filtered_data = combined_data[combined_data['score'] >= 0]
            print(f"Filtered to {len(filtered_data)} sequences with score >= 0")
            
            if len(filtered_data) > 0:
                # Create mutation DataFrame
                mutations_df = create_mutation_df(filtered_data['sequence'].tolist(), reference_sequence)
                
                # Create pie chart
                title = f"{train_type.capitalize()} Train Eval"
                create_single_pie_chart(mutations_df, title, output_dir, use_color_key=True, score_threshold=0)
                print(f"Created {train_type} train eval pie chart")
                create_nested_pie_chart(mutations_df, title, output_dir, use_color_key=True, score_threshold=0)
                print(f"Created {train_type} train eval nested pie chart")
            else:
                print(f"No sequences with score >= 0 for {train_type}")
                
        except Exception as e:
            print(f"Error processing {train_type} eval data: {e}")
            continue
    
    print("Individual train eval pie charts complete!")

def create_genbank_nested_pie_chart():
    """Create a nested pie chart from GenBank RBD point mutation frequencies."""
    print("\nCreating GenBank nested pie chart...")
    csv_path = 'data/genbank/genbank_rbd_point_mutations.csv'
    if not os.path.exists(csv_path):
        print(f"  GenBank CSV not found at {csv_path}")
        return

    try:
        genbank_df = pd.read_csv(csv_path)
        # Prepare DataFrame to match expected schema
        # Adjust position to RBD numbering by adding 330
        genbank_df['Mutation'] = genbank_df['mutation'].apply(adjust_mutation_position)
        genbank_df['Count'] = genbank_df['sequence_count']
        mutation_df = genbank_df[['Mutation', 'Count', 'frequency']].copy()

        output_dir = 'figures/pie_charts/genbank'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        create_nested_pie_chart(
            mutation_df,
            title_prefix='GenBank RBD Point Mutations',
            output_dir=output_dir,
            use_color_key=True,
        )
        #create_single_pie_chart(
        #    mutation_df,
        #    title_prefix='GenBank RBD Point Mutations',
        #    output_dir=output_dir,
        #    use_color_key=True
        #)
        print("GenBank nested pie chart created!")
    except Exception as e:
        print(f"  Error while creating GenBank nested pie chart: {str(e)}")

def process_training_data(train_type, score_threshold=-100):
    """Process training data for a specific type (alldata, batest, or baseonly)."""
    print(f"\nProcessing {train_type} training data...")
    
    # Set up paths
    train_dir = f"data/train_data/{train_type}"
    output_dir = "figures/pie_charts/train/afterB2/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all CSV files including train_data.csv
    csv_files = glob.glob(os.path.join(train_dir, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files for {train_type}")
    
    # Load and combine all data
    all_data = []
    for file in csv_files:
        data = load_and_filter_data(file, score_threshold)
        if data is not None:
            all_data.append(data)
            print(f"  Loaded data from {os.path.basename(file)}")
    
    if not all_data:
        print(f"No valid data found for {train_type}")
        return
    
    # Combine all dataframes
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined data has {len(combined_data)} sequences")
    
    # Get reference sequence
    reference_sequence = read_fasta_sequence('data/rbd_ref_aa.fasta')
    
    # Create mutation DataFrame
    mutation_df = create_mutation_df(combined_data['sequence'].tolist(), reference_sequence)
    
    # Create pie charts
    title = f"{train_type.capitalize()} Train Data"
    # Create regular pie charts (commented out as requested)
    #create_single_pie_chart(mutation_df, title, output_dir, use_color_key=True, 
    #                      score_threshold=score_threshold, sort_by_position=False)
    #create_single_pie_chart(mutation_df, title, output_dir, use_color_key=True, 
    #                      score_threshold=score_threshold, sort_by_position=True)
    
    # Create nested pie chart with score threshold 0
    # Filter data for score threshold 0
    filtered_data_for_nested = combined_data[combined_data['score'] >= 0]
    if len(filtered_data_for_nested) > 0:
        nested_mutation_df = create_mutation_df(filtered_data_for_nested['sequence'].tolist(), reference_sequence)
        create_nested_pie_chart(nested_mutation_df, title, output_dir, use_color_key=True,
                              score_threshold=0)
        create_single_pie_chart(nested_mutation_df, title, output_dir, use_color_key=True,
                              score_threshold=0)
    print(f"Created pie charts for {train_type} training data")

def process_simulation_dataset(ref_name):
    """Process simulations."""
    print(f"\n{'='*60}")
    print(f"PROCESSING {ref_name.upper()} REFERENCE DATASET")
    print(f"{'='*60}")
    
    # Set up directories
    data_dir = f"data/{ref_name}"
    output_dir = f"figures/pie_charts/{ref_name}/afterB2"
    
    # Determine if this is BA.1 or WT dataset
    variant_prefix = "BA.1" if "ba1" in ref_name.lower() else "WT"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist. Skipping {ref_name}.")
        return
    
    # Get reference sequence
    print("Loading reference sequence...")
    reference_sequence = read_fasta_sequence('data/rbd_ref_aa.fasta')
    print(f"Reference sequence length: {len(reference_sequence)}")
    
    # Process simulation data
    print(f"\nProcessing simulation data...")
    simulation_files = find_simulation_files(data_dir)
    
    if not simulation_files:
        print(f"  No simulation files found in {data_dir}")
        return
    
    print(f"  Found {len(simulation_files)} simulation files:")
    for sim_file in simulation_files:
        print(f"    {os.path.basename(sim_file)}")
        
        # Extract simulation type from filename
        sim_type = None
        if 'baseonly' in sim_file:
            sim_type = 'baseonly'
        elif 'alldata' in sim_file:
            sim_type = 'alldata'
        
        if sim_type is None:
            print(f"  Warning: Could not determine simulation type for {os.path.basename(sim_file)}")
            continue
            
        print(f"\n  Processing simulation: {sim_type}")
        
        try:
            # Load simulation data
            simulation_data = pd.read_csv(sim_file)
            print(f"    Columns: {simulation_data.columns.tolist()[:5]}...")
            
            # Process simulation data - combine position columns into sequences
            if any(col.startswith('p') and col[1:].isdigit() for col in simulation_data.columns):
                position_cols = [col for col in simulation_data.columns if col.startswith('p') and col[1:].isdigit()]
            else:
                position_cols = [col for col in simulation_data.columns if any(
                    pattern in col.lower() for pattern in ['pos_', 'position_', 'pos', 'position']
                ) and any(c.isdigit() for c in col)]
            
            if not position_cols:
                raise ValueError(f"No position columns found in {sim_type} data")
            
            print(f"    Found {len(position_cols)} position columns")
            
            # Sort position columns numerically
            position_cols.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            
            # Create sequences
            simulation_sequences = simulation_data[position_cols].astype(str).apply(lambda row: ''.join(row), axis=1).tolist()
            
            print(f"    Number of sequences: {len(simulation_sequences)}")
            print(f"    Sequence length: {len(simulation_sequences[0])}")
            
            if len(simulation_sequences[0]) != len(reference_sequence):
                print(f"    WARNING: Sequence length ({len(simulation_sequences[0])}) doesn't match reference sequence length ({len(reference_sequence)})")
            
            # Create mutation DataFrame
            print(f"    Analyzing mutations...")
            simulation_mutations_df = create_mutation_df(simulation_sequences, reference_sequence)
            
            if len(simulation_mutations_df) == 0:
                print(f"    WARNING: No mutations found in {sim_type} dataset!")
            else:
                print(f"    Found {len(simulation_mutations_df)} unique mutations")
                
                # Create pie charts
                print(f"    Creating pie charts...")
                title = f"{variant_prefix} Simulation {sim_type.capitalize()} Color Key"
                # Create regular pie charts
                #create_single_pie_chart(simulation_mutations_df, title, output_dir, 
                #                      use_color_key=True, sort_by_position=False)
                #create_single_pie_chart(simulation_mutations_df, title, output_dir, 
                #                      use_color_key=True, sort_by_position=True)
                # Create nested pie chart
                create_nested_pie_chart(simulation_mutations_df, title, output_dir, 
                                       use_color_key=True)
                create_single_pie_chart(simulation_mutations_df, title, output_dir, 
                                      use_color_key=True)
                print(f"    Completed {sim_type}")
        except Exception as e:
            print(f"    Error processing {sim_type}: {str(e)}")
            import traceback
            traceback.print_exc()


def create_color_key_legend():
    """Create a PDF page showing all colors in the color key with their positions."""
    # Load color key
    with open(color_key_path, 'r') as f:
        color_key = json.load(f)
    
    # Sort positions numerically
    positions = sorted([int(pos) for pos in color_key.keys() if pos != 'others'])
    positions.append('others')

    print("positions: ", positions, "type: ", type(positions))
    # Calculate grid dimensions - use 4 rows and 5 columns for better layout
    n_items = len(positions)
    n_cols = 5  # Fixed 5 columns
    n_rows = 4  # Fixed 4 rows
    
    # Create figure with appropriate aspect ratio
    fig_width = 10
    fig_height = fig_width * (n_rows / n_cols) * 0.5  # Make it wider than tall
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create grid of squares (only for items that exist)
    for i, pos in enumerate(positions):
        if i >= n_rows * n_cols:  # Stop if we exceed the grid size
            break
            
        row = i // n_cols
        col = i % n_cols
        
        # Calculate square position
        x = col / n_cols
        y = 1 - (row + 1) / n_rows  # Start from top
        width = 1 / n_cols
        height = 1 / n_rows
        
        # Add colored square
        rect = plt.Rectangle((x + width*0.1, y + height*0.1), 
                           width*0.8, height*0.6,  # Make squares shorter to leave room for text
                           facecolor=color_key[str(pos)],
                           edgecolor='black',
                           linewidth=1)
        plt.gca().add_patch(rect)
        
        # Add position label
        plt.text(x + width/2, y + height*0.5,
                str(pos),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=10)
    
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gca().axis('off')
    if "genbank" in color_key_path:
        plt.title('Color Key for RBD Positions based on top GenBank Mutations', pad=20, fontsize=16)
    elif "VOC" in color_key_path:
        plt.title('Color Key for RBD Positions based on VOC Mutations', pad=20, fontsize=16)
    
    # Save to a temporary file
    temp_file = "figures/pie_charts/temp_color_key_legend.pdf"
    plt.savefig(temp_file, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    return temp_file

# %%
# Main execution
def main():
    #combine_pdf_outputs()
    #flag = False
    flag = True
    if flag:
        """Main execution function."""
        print("Starting pie chart generation...")
        
        # Process training data for each type
        train_types = ['alldata', 'baseonly']
        for train_type in train_types:
            process_training_data(train_type)
        
        # Create threshold comparison plots
        threshold_comparison_train()
        threshold_comparison_train_eval()
        
        # Create train eval individual charts
        create_train_eval_individual_charts()

        # Create GenBank nested pie chart
        create_genbank_nested_pie_chart()
        
        # Process both simulation datasets
        reference_datasets = ['wuhan_sim', 'ba1_sim']
        for ref_name in reference_datasets:
            try:
                process_simulation_dataset(ref_name)
            except Exception as e:
                print(f"Error processing {ref_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\nAll processing complete!")

# Execute main function
if __name__ == "__main__":
    main()

# %%
