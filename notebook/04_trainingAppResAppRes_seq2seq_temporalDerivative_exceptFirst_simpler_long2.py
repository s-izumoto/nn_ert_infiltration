# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:49:12 2024

@author: izumoto
"""


import numpy as np
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Input, Dense, TimeDistributed


# Data Preparation (same as your code above)
early = True
chooseIndex = False
sparce = True
diff = True
timeContext = False
normalization = False
meanCentered = True
        
time_steps = 30
output_seq_length = 29

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

#plt.imshow(input_data[1,30,:,:])

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
        # Calculate the min and max for each time step across all sequences
    time_step_min = np.min(input_data, axis=(0, 2, 3), keepdims=True)  # Shape: (1, time_steps, 1, 1)
    time_step_max = np.max(input_data, axis=(0, 2, 3), keepdims=True)  # Shape: (1, time_steps, 1, 1)
    
    # Scale each time step to the range [0, 1]
    scaled_input_data = (input_data - time_step_min) / (time_step_max - time_step_min)  # Add small value to avoid division by zero
    
    # Repeat for output_data if needed
    time_step_min_output = np.min(output_data, axis=(0, 2, 3), keepdims=True)
    time_step_max_output = np.max(output_data, axis=(0, 2, 3), keepdims=True)
    scaled_output_data = (output_data - time_step_min_output) / (time_step_max_output - time_step_min_output)
    
    # Use scaled data for training
    input_data = scaled_input_data
    output_data = scaled_output_data
    
        # Save the normalization factors
    np.savez('normalization_factors.npz', time_step_min=time_step_min, time_step_max=time_step_max)
    np.savez('normalization_factors_output.npz', time_step_min_output=time_step_min_output, time_step_max_output=time_step_max_output)


X, y = [], []

if timeContext:    
    for series in range(input_data.shape[0]):
        for i in range(input_data.shape[1] - time_steps + 1):
            input_sequence = []
            output_sequence = []
            for ts in range(i, i + time_steps):
                resistivity_flat = create_array(input_data[series, ts, :, :])
                time_context = np.full(resistivity_flat.shape, ts / (input_data.shape[1] - 1))
                resistivity_with_time = np.concatenate([resistivity_flat, time_context])
                input_sequence.append(resistivity_with_time)
            for ts in range(i + (time_steps - output_seq_length), i + time_steps):
                resistivity_flat = create_array(output_data[series, ts, :, :])
                output_sequence.append(resistivity_flat)
                #time_context = np.full(resistivity_flat.shape, ts / (output_data.shape[1] - 1))
                #resistivity_with_time = np.concatenate([resistivity_flat, time_context])
                #output_sequence.append(resistivity_with_time)
                
            X.append(np.array(input_sequence))
            y.append(np.array(output_sequence))
    
    X = np.array(X)
    y = np.array(y)
    
else:
        # Generate training data without time context
    for series in range(input_data.shape[0]):
        for i in range(input_data.shape[1] - time_steps +1):
            input_sequence = []
            output_sequence = []
            
            for ts in range(i, i + time_steps):
                # Process input data
                resistivity_flat = create_array(input_data[series, ts, :, :])
                input_sequence.append(resistivity_flat)
            for ts in range(i + (time_steps - output_seq_length),i + time_steps):
                # Process output data
                resistivity_flat = create_array(output_data[series, ts, :, :])
                output_sequence.append(resistivity_flat)
                
            # Append processed sequences to X and y
            X.append(np.array(input_sequence))
            y.append(np.array(output_sequence))
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
if meanCentered:
    Xmean = []
    ymean = []

    for i in range(X.shape[1]):
        mean = np.mean(X[:,i,:],axis=0)
        Xmean.append(mean)
    for i in range(y.shape[1]):
        mean = np.mean(y[:,i,:],axis=0)
        ymean.append(mean)
        
    Xmean = np.array(Xmean)
    ymean = np.array(ymean)

    X = X - Xmean
    y = y - ymean
    
    np.savez('mean_values.npz', Xmean=Xmean, ymean=ymean)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
# Define a simpler model
encoder_inputs = Input(shape=(time_steps, X.shape[2]))

# Single-layer encoder
encoder_outputs, state_h, state_c = LSTM(512, return_state=True)(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(output_seq_length, y.shape[2]))

# Single-layer decoder
decoder_lstm = LSTM(512, return_sequences=True)
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(y.shape[2]))
decoder_outputs = decoder_dense(decoder_outputs)
"""

# Encoder with multiple LSTM layers
encoder_inputs = Input(shape=(time_steps, X.shape[2]))
encoder_lstm1 = LSTM(512, return_sequences=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(512, return_state=True)(encoder_lstm1)
encoder_states = [state_h, state_c]

# Decoder with multiple LSTM layers
decoder_inputs = Input(shape=(output_seq_length, y.shape[2]))
decoder_lstm1 = LSTM(512, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
decoder_lstm2 = LSTM(512, return_sequences=True)(decoder_outputs)
decoder_outputs = TimeDistributed(Dense(y.shape[2]))(decoder_lstm2)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Prepare decoder input data for teacher forcing
decoder_input_data_train = np.concatenate([np.zeros_like(y_train[:, :1, :]), y_train[:, :-1, :]], axis=1)
decoder_input_data_test = np.concatenate([np.zeros_like(y_test[:, :1, :]), y_test[:, :-1, :]], axis=1)

# Train the model
history = model.fit(
    [X_train, decoder_input_data_train], y_train,
    epochs=40,
    batch_size=5,
    validation_data=([X_test, decoder_input_data_test], y_test)
)

# Plot training and validation loss
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Save the model
model.save('simplified_seq2seq_lstm.keras')