import os
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from matplotlib.colors import LogNorm

# Define the shape of the conductivity data (e.g., (100, 500, 60, 30))
# Replace with the actual shape of your data
array_shape = (50, 501, 100, 100)  # Set the correct shape here

# Use memory-mapping to load the large file
conductivity_data = np.load('../data/combined_conductivity_maps.npy', mmap_mode='r').reshape(array_shape)

# Flip columns and convert to resistivity in a memory-efficient way
num_sequences, timesteps, grid_size_y, grid_size_x = array_shape
conductivity_data = conductivity_data[..., ::-1]
resistivity_data = 1 / conductivity_data
print("Resistivity data shape:", resistivity_data.shape)

# Define physical domain dimensions
domain_height = 3  # Vertical length in meters
domain_width = 1   # Horizontal length in meters

# Define the grid boundaries based on the physical dimensions
x_min, x_max = 0, domain_width
y_min, y_max = 0, domain_height

# Update electrode positions along the left edge within the new height
electrode_positions = np.linspace(0.9, 2.1, 32)

# Flags for controlling visualization
SAVE_ELECTRODE_FIGURES = False
SAVE_TRIANGULAR_MATRIX_FIGURES = False

# Create output directory
output_folder = "visualizations_large"
os.makedirs(output_folder, exist_ok=True)

# Define the measurement scheme
scheme = "wa"

# Function to create a refined mesh near the electrodes
def create_refined_mesh(domain_width, domain_height, electrode_positions, x_min=0, x_max=1, y_min=0, y_max=3):
    world = mt.createRectangle(start=[x_min, y_min], end=[x_max, y_max], marker=1)
    for y_pos in electrode_positions:
        world.createNode([x_min, y_pos])
        world.createNode([x_min + 0.05, y_pos])  # Small refinement near electrodes
    mesh = mt.createMesh(world, quality=34)
    return mesh

def calculate_apparent_resistivity_for_time_step(resistivity_map, scheme):
    mesh = create_refined_mesh(domain_width, domain_height, electrode_positions)
    resistivity_model = np.zeros(mesh.cellCount())

    # Assign resistivity values from the map to the mesh cells
    for cell in mesh.cells():
        center = cell.center()
        i = int(center[0] / domain_width * resistivity_map.shape[1])
        j = int(center[1] / domain_height * resistivity_map.shape[0])
        resistivity_model[cell.id()] = resistivity_map[j, i]

    # Setup dipole-dipole ERT configuration with the electrodes along the left edge
    electrode_coordinates = [[x_min, y_pos] for y_pos in electrode_positions]
    data = ert.createData(elecs=electrode_coordinates, schemeName=scheme)

    # Simulate apparent resistivity with geometric factor
    data = ert.simulate(mesh, scheme=data, res=resistivity_model, noiseLevel=0.5, noiseAbs=1e-6, seed=1337)

    # Replace invalid values with NaN
    rhoa = data['rhoa']
    rhoa[rhoa < 0] = np.nan
    data['rhoa'] = rhoa

    return data, mesh, resistivity_model, electrode_coordinates

# Function to reformulate and return the triangular matrix
def reformulate_triangular_matrix(apparent_resistivity_data, scheme):
    row_sizes = np.arange(29, 0, -3) if scheme == "wa" else np.arange(29, 0, -1)
    array = np.array(apparent_resistivity_data['rhoa'])
    if array.size != np.sum(row_sizes):
        raise ValueError("Array size does not match the required triangular shape.")
    
    # Initialize the triangular matrix with NaNs
    triangular_matrix = np.full((len(row_sizes), row_sizes[0]), np.nan)
    start_idx = 0
    for i, size in enumerate(row_sizes):
        triangular_matrix[i, :size] = array[start_idx:start_idx + size]
        start_idx += size

    return triangular_matrix

# Function to process each sequence of resistivity maps in chunks
def process_sequence(seq_idx):
    triM_sequence = []
    
    # Process each timestep individually
    for idx in range(timesteps):
        resistivity_map = resistivity_data[seq_idx, idx]
        apparent_resistivity_data, mesh, resistivity_model, electrode_coordinates = calculate_apparent_resistivity_for_time_step(resistivity_map, scheme)
        
        # Reformulate and save the triangular matrix
        triM = reformulate_triangular_matrix(apparent_resistivity_data, scheme)
        triM_sequence.append(triM)
    
    # Save this sequence's triangular matrices directly to a file to avoid memory issues
    triM_sequence = np.array(triM_sequence)
    np.save(os.path.join(output_folder, f"triangular_matrix_seq_{seq_idx}.npy"), triM_sequence)
    
    return True  # Indicate successful completion

# Parallel execution using multiprocessing.Pool with progress bar
if __name__ == "__main__":
    # Use multiprocessing to process each sequence independently
    with Pool(processes=6) as pool:
        list(tqdm(pool.imap(process_sequence, range(num_sequences)), total=num_sequences, desc="Processing sequences"))

    print("All triangular matrix files saved in the output folder.")
