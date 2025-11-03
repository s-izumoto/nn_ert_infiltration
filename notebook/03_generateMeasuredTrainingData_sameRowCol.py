# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:38:24 2024

@author: izumoto
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# Function to convert difference map into a probability distribution
def convert_to_probability_distribution(difference_map):
    exp_diff = np.abs(difference_map)  # Apply an exponential function to differences
    total = np.sum(exp_diff)  # Sum of the differences
    if total == 0:  # Check for zero total to avoid division by zero
        return np.zeros_like(exp_diff)  # Return a zero map if the total is zero
    probability_map = exp_diff / total  # Normalize to create a probability distribution
    return probability_map

# Function to create overlapping sequences of specified length
def create_sequences(data, seq_length):
    sequences = []
    for i in range(data.shape[0]):  # For each sequence in the dataset
        for start in range(data.shape[1] - seq_length + 1):
            sequences.append(data[i, start:start + seq_length])
    return np.array(sequences)

# Load true resistivity data
true_resistivity_data = np.load('united_triangular_matrices.npy')
true_resistivity_data = true_resistivity_data[:, :, :, :]  # Remove the first time step if needed
true_resistivity_data = np.nan_to_num(true_resistivity_data, nan=0)
# Set the number of time steps
#num_time_steps = 10
grid_size_x = true_resistivity_data.shape[-1]
grid_size_y = true_resistivity_data.shape[-2]
#num_measurements = grid_size_x*grid_size_y//16  # Number of measurement places at a time
num_measurements = 1

true_resistivity_data = true_resistivity_data[:,0::num_measurements,:,:]

# Create a folder to save the images
output_folder = 'frames_training_data_checkRC'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Prepare X and y
X = []
y_probabilities = []
Xseq = []
y_probabilities_seq = []
previous_measurement_indices = np.zeros((grid_size_y, grid_size_x))
measured = []

# Set the fixed range for color bars (you can adjust these based on your data)
vmin_resistivity = 0  # minimum value for resistivity color range
vmax_resistivity = 1  # maximum value for resistivity color range

positions = np.load("measurement_indices.npy")


seq_for_rc = 31
positions_seq = positions[seq_for_rc,:,:]


for seq in range(true_resistivity_data.shape[0]):
    measured_resistivity_map = true_resistivity_data[seq, 0, :, :].copy()  # Start with the initial resistivity map
    Xseq = []
    y_probabilities_seq = []
    measured_seq = []
    
    for t in range(1, true_resistivity_data.shape[1]):
        true_resistivity_map_t = true_resistivity_data[seq, t, :, :].copy()
        difference_map = np.abs(true_resistivity_map_t - measured_resistivity_map)
        
        # Determine measurement points based on the largest difference
        #rows, cols = np.unravel_index(np.argsort(difference_map, axis=None)[-num_measurements:], true_resistivity_map_t.shape)
        col = [positions_seq[t-1,0]]
        row = [positions_seq[t-1,1]]
        # Update the measured resistivity map with the new measurements
        for r, c in zip(row, col):
            measured_resistivity_map[r, c] = true_resistivity_map_t[r, c]
        """
        # Create the feedback input by combining measured resistivity map and previous output
        if t == 1:
            # No previous output at the first time step
            feedback_input = np.zeros((grid_size_y, grid_size_x))
        else:
            feedback_input = y_probabilities_seq[-1]  # Use the last probability map as feedback
        
        combined_input = np.stack([measured_resistivity_map, feedback_input], axis=-1)  # Shape (grid_size, grid_size, 2)
        Xseq.append(combined_input)
        """
        Xseq.append(measured_resistivity_map)
        
        measured_seq.append(measured_resistivity_map.copy()) # .copy() is necessary
        
        # Convert difference map to a probability distribution for the target output
        probability_map = convert_to_probability_distribution(difference_map)
        y_probabilities_seq.append(probability_map)
        """
        if seq == 5:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            im1 = plt.imshow(measured_resistivity_map, cmap='hot', interpolation='nearest')
            plt.title(f'Measured Resistivity Map at Time {t+1}')
            cbar1 = plt.colorbar(im1, orientation='horizontal', pad=0.2)

            plt.subplot(1, 2, 2)
            im3 = plt.imshow(true_resistivity_data[seq, t, :, :], cmap='hot', interpolation='nearest')
            plt.title(f'True Resistivity Map at Time {t}')
            cbar3 = plt.colorbar(im3, orientation='horizontal', pad=0.2)
            
            # Automatically adjust the layout to prevent overlap
            plt.tight_layout()
            
            # Save the figure as a PNG file in the 'frames' folder
            plt.savefig(os.path.join(output_folder, f'frame_{t:04d}.png'))
            plt.close()
"""

    X.append(Xseq)
    y_probabilities.append(y_probabilities_seq)
    measured.append(measured_seq)

# Convert lists to arrays
X = np.array(X)
y_probabilities = np.array(y_probabilities)
measured = np.array(measured)
np.save('measured_training_data_sameRowColSeq31.npy', measured)

