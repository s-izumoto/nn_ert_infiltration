# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:05:01 2024

@author: izumoto
"""

import numpy as np
import os
from tqdm import tqdm

# Folder containing the individual triangular matrix files for each sequence
input_folder = "visualizations_large"
output_file = "united_triangular_matrices.npy"

# Get a list of all .npy files in the input folder, assuming they follow a consistent naming pattern
# Sort the list to maintain order by sequence index
npy_files = sorted([f for f in os.listdir(input_folder) if f.startswith("triangular_matrix_seq_") and f.endswith(".npy")])

# Load the first file to determine the shape of each matrix (e.g., timesteps, matrix_height, matrix_width)
first_matrix = np.load(os.path.join(input_folder, npy_files[0]))
timesteps, matrix_height, matrix_width = first_matrix.shape

# Initialize an empty list to store each sequence's matrices
sequences = []

# Loop through each file and load its data, then add it to the sequences list
for npy_file in tqdm(npy_files, desc="Loading triangular matrix files"):
    file_path = os.path.join(input_folder, npy_file)
    sequence_data = np.load(file_path)
    sequences.append(sequence_data)

# Stack all sequences along the first dimension to create the final array
united_array = np.stack(sequences, axis=0)  # Shape will be (sequence, timesteps, matrix_height, matrix_width)

# Save the final array to a single .npy file
np.save(output_file, united_array)

print(f"United array saved as '{output_file}' with shape {united_array.shape}")
