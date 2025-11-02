import matplotlib.pyplot as plt
import numpy as np
import json

# VOC positions
#voc_positions = [339,346,356,368,371,373,375,376,403,405,408,417,440,445,446,450,452,455,460,477,478,481,484,486,490,498,501,505]
voc_positions = [346,356,368,403,445,450,452,455,460,481,486,490] # after BA.2
voc_positions.sort()  # Sort positions for consistent color assignment

# Create color map using cool gradient
n_colors = len(voc_positions)
colors = plt.cm.cool(np.linspace(0, 1, n_colors))

# Convert RGB colors to hex format
color_dict = {}
for pos, color in zip(voc_positions, colors):
    # Convert RGB to hex, excluding alpha channel
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(color[0] * 255),
        int(color[1] * 255),
        int(color[2] * 255)
    )
    color_dict[str(pos)] = hex_color

# Add 'others' color
color_dict['others'] = '#d3d3d3'  # Light gray for consistency

# Save to JSON file
with open('../../data/VOCafterB2_colorkey.json', 'w') as f:
    json.dump(color_dict, f, indent=2)

print("Created VOC color key with the following mappings:")
for pos, color in color_dict.items():
    print(f"  Position {pos}: {color}") 
