# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:21:06 2024

@author: izumoto
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, Dense, RepeatVector, TimeDistributed
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import load_model


model = load_model('simplified_seq2seq_lstm.keras')
#model = load_model('seq2seq_lstm_with_teacher_forcing.keras')

time_steps = 30
output_seq_length = 29

series = 1

# Data Preparation (same as your code above)
early = True
chooseIndex = False
sparce = True
diff = True
timeContext = False
normalization = False
meanCentered = True


def create_array(data):
    row_sizes = np.arange(29, 0, -3)
    filled_data = []
    for i, size in enumerate(row_sizes):
        filled_data.extend(data[i, :size])
    return np.array(filled_data)

def de_create_array(flat_data):
    row_sizes = np.arange(29, 0, -3)
    max_row_size = row_sizes[0]
    matrix = np.zeros((len(row_sizes), max_row_size))
    start_idx = 0
    for i, size in enumerate(row_sizes):
        end_idx = start_idx + size
        matrix[i, :size] = flat_data[start_idx:end_idx]
        start_idx = end_idx
    return matrix

input_data = 1/np.load('measured_training_data_sameRowColSeq31.npy')*1000
initial_data = 1/np.load('united_triangular_matrices.npy')[:,0,:,:]*1000
initial_data = np.expand_dims(initial_data, axis=1)
input_data = np.concatenate((initial_data, input_data), axis=1)
input_data = np.nan_to_num(input_data, nan=0)

output_data = 1/np.load('united_triangular_matrices.npy')*1000
#output_data = np.concatenate((initial_data, output_data), axis=1)
output_data = np.nan_to_num(output_data, nan=0)

if early:
    output_data = output_data[:,:310,:,:]
    input_data = input_data[:,:310,:,:]
    
if chooseIndex:
    index = [26,37,31,19,36,28,38,18,15]
    output_data = [output_data[x,:,:,:] for x in index]
    input_data  = [input_data [x,:,:,:] for x in index]
    output_data = np.array(output_data)
    input_data  = np.array(input_data)

if sparce:
    input_data = input_data[:,::10,:,:]
    output_data = output_data[:,::10,:,:]

if diff:
    input_data = np.diff(input_data,axis=1)
    output_data = np.diff(output_data,axis=1)

if normalization:
        # Load the normalization factors
    norm_factors = np.load('normalization_factors.npz')
    time_step_min = norm_factors['time_step_min']
    time_step_max = norm_factors['time_step_max']
    
    # Apply the saved scaling to new input data
    input_data = (input_data - time_step_min) / (time_step_max - time_step_min)


# Loop through all the series
for series in range(len(input_data)):
    input_sequence = []
    output_sequence = []

    if timeContext:
        for ts in range(0, time_steps):
            resistivity_flat = create_array(input_data[series, ts, :, :])
            time_context = np.full(resistivity_flat.shape, ts / (input_data.shape[1] - 1))
            resistivity_with_time = np.concatenate([resistivity_flat, time_context])
            input_sequence.append(resistivity_with_time)
        for ts in range(1, time_steps):
            resistivity_flat = create_array(output_data[series, ts, :, :])
            output_sequence.append(resistivity_flat)
    else:
        for ts in range(0, time_steps):
            resistivity_flat = create_array(input_data[series, ts, :, :])
            input_sequence.append(resistivity_flat)
        for ts in range(1, time_steps):
            resistivity_flat = create_array(output_data[series, ts, :, :])
            output_sequence.append(resistivity_flat)

    input_sequence = np.array(input_sequence)
    output_sequence = np.array(output_sequence)

    if meanCentered:
        data = np.load('mean_values.npz')
        Xmean_loaded = data['Xmean']
        ymean_loaded = data['ymean']
        input_sequence = input_sequence - Xmean_loaded

    # Initialize the decoder input
    decoder_input = np.zeros((1, output_seq_length, output_sequence.shape[1]))

    # Predict the output sequence
    predicted_output_sequence = []
    for t in range(output_seq_length):
        prediction = model.predict([np.expand_dims(input_sequence, axis=0), decoder_input])
        predicted_output_sequence.append(prediction[:, t, :])
        decoder_input[:, t, :] = prediction[:, t, :]

    if meanCentered:
        for i in range(len(predicted_output_sequence)):
            predicted_output_sequence[i] = predicted_output_sequence[i] + ymean_loaded[i, :]

    # Convert predictions back to images
    output_image = []
    true_image = []

    for arr, true_arr in zip(predicted_output_sequence, output_sequence):
        output_image.append(de_create_array(arr.squeeze()))
        true_image.append(de_create_array(true_arr.squeeze()))

    output_image = np.array(output_image)
    true_image = np.array(true_image)

    # Save the output images
    np.save(f"output_image_1_4_series_long2_{series}.npy", output_image)

    # Visualization
    # Summed images for visualization
    output_image_end = np.sum(output_image, axis=0)
    true_image_end = np.sum(true_image, axis=0)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Display the predicted image
    axes[0].imshow(output_image_end, cmap='viridis')
    axes[0].set_title(f"Predicted Image (Series {series})")
    axes[0].axis('off')

    # Display the true image
    axes[1].imshow(true_image_end, cmap='viridis')
    axes[1].set_title(f"True Image (Series {series})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    """