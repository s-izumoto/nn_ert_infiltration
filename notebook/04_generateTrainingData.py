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

# --- メドイド（中央値に最も近いシーケンス）を自動選択 ---
S = positions.shape[0]
med = np.median(positions, axis=0)                 # (T,2) 実数
d = np.abs(positions - med).sum(axis=(1,2))        # (S,) 各シーケンスのL1距離合計
seq_for_rc = int(np.argmin(d))                      # 最小距離のシーケンス
# 範囲ガード & フォールバック
if not (0 <= seq_for_rc < S):
    seq_for_rc = 31 if S > 50 else 0

# 任意: ログとして保存（後工程で再利用したい場合）
try:
    np.save("chosen_seq_index.npy", np.array(seq_for_rc, dtype=int))
except Exception:
    pass

positions_seq = positions[seq_for_rc, :, :]        # (T,2)

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

        Xseq.append(measured_resistivity_map)
        
        measured_seq.append(measured_resistivity_map.copy()) # .copy() is necessary
        
        # Convert difference map to a probability distribution for the target output
        probability_map = convert_to_probability_distribution(difference_map)
        y_probabilities_seq.append(probability_map)

    X.append(Xseq)
    y_probabilities.append(y_probabilities_seq)
    measured.append(measured_seq)

# Convert lists to arrays
X = np.array(X)
y_probabilities = np.array(y_probabilities)
measured = np.array(measured)
np.save(f"measured_training_data_sameRowColSeq{seq_for_rc}.npy", measured)

