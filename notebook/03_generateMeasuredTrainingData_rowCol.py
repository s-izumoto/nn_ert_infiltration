# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:09:41 2024

@author: izumoto
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Function to convert difference map into a probability distribution
def convert_to_probability_distribution(difference_map):
    exp_diff = np.abs(difference_map)
    total = np.sum(exp_diff)
    if total == 0:
        return np.zeros_like(exp_diff)
    probability_map = exp_diff / total
    return probability_map

# Function to create overlapping sequences of specified length
def create_sequences(data, seq_length):
    sequences = []
    for i in range(data.shape[0]):
        for start in range(data.shape[1] - seq_length + 1):
            sequences.append(data[i, start:start + seq_length])
    return np.array(sequences)

# Load true resistivity data
true_resistivity_data = np.load('united_triangular_matrices.npy')
true_resistivity_data = np.nan_to_num(true_resistivity_data, nan=0)

grid_size_x = true_resistivity_data.shape[-1]
grid_size_y = true_resistivity_data.shape[-2]
num_measurements = 1

true_resistivity_data = true_resistivity_data[:, 0::num_measurements, :, :]

# Create a folder to save the images
output_folder = 'frames_training_data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Prepare arrays and lists
X = []
y_probabilities = []
Xseq = []
y_probabilities_seq = []
previous_measurement_indices = np.zeros((grid_size_y, grid_size_x))
measured = []
measurement_indices = []  # List to store the (col, row) pairs

for seq in range(true_resistivity_data.shape[0]):
    measured_resistivity_map = true_resistivity_data[seq, 0, :, :].copy()
    Xseq = []
    y_probabilities_seq = []
    measured_seq = []
    indices_seq = []  # List to store (col, row) pairs for each time step

    for t in range(1, true_resistivity_data.shape[1]):
        true_resistivity_map_t = true_resistivity_data[seq, t, :, :].copy()
        difference_map = np.abs(true_resistivity_map_t - measured_resistivity_map)
        
        # Determine measurement points based on the largest difference
        rows, cols = np.unravel_index(np.argsort(difference_map, axis=None)[-num_measurements:], true_resistivity_map_t.shape)
        
        # Update the measured resistivity map and store indices
        for r, c in zip(rows, cols):
            measured_resistivity_map[r, c] = true_resistivity_map_t[r, c]
            indices_seq.append((c, r))  # Save as (col, row)

        feedback_input = y_probabilities_seq[-1] if t > 1 else np.zeros((grid_size_y, grid_size_x))
        combined_input = np.stack([measured_resistivity_map, feedback_input], axis=-1)
        Xseq.append(combined_input)
        measured_seq.append(measured_resistivity_map.copy())
        
        probability_map = convert_to_probability_distribution(difference_map)
        y_probabilities_seq.append(probability_map)
        
        if seq == 5:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            im1 = plt.imshow(measured_resistivity_map, cmap='hot', interpolation='nearest')
            plt.title(f'Measured Resistivity Map at Time {t+1}')
            cbar1 = plt.colorbar(im1, orientation='horizontal', pad=0.2)

            plt.subplot(1, 2, 2)
            im3 = plt.imshow(true_resistivity_data[seq, t, :, :], cmap='hot', interpolation='nearest')
            plt.title(f'True Resistivity Map at Time {t+1}')
            cbar3 = plt.colorbar(im3, orientation='horizontal', pad=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'frame_{t:04d}.png'))
            plt.close()

    X.append(Xseq)
    y_probabilities.append(y_probabilities_seq)
    measured.append(measured_seq)
    measurement_indices.append(indices_seq)

# Convert lists to arrays and save
X = np.array(X)
y_probabilities = np.array(y_probabilities)
measured = np.array(measured)
measurement_indices = np.array(measurement_indices)

np.save('measured_training_data.npy', measured)
np.save('measurement_indices.npy', measurement_indices)  # Save (col, row) pairs
