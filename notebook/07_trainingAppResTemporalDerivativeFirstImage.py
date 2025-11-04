# -*- coding: utf-8 -*-
"""
PyTorch rewrite of 07_trainingAppResAppRes_seq2seq_temporalDerivative_onlyFirst.py
- Keeps data processing and network design identical to the original Keras code.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------
# Config flags (same semantics)
# -----------------------------
early = True
chooseIndex = False
sparce = True
diff = True
timeContext = False
normalization = False
meanCentered = True

time_steps = 4  # same as original

# -----------------------------
# Utils: triangular flattening
# -----------------------------
def create_array(data: np.ndarray) -> np.ndarray:
    """Flatten triangular rows in sizes 29, 26, 23, ..., 2, 1 (step -3)."""
    row_sizes = np.arange(29, 0, -3)
    filled_data = []
    for i, size in enumerate(row_sizes):
        filled_data.extend(data[i, :size])
    return np.array(filled_data)

def de_create_array(flat_data: np.ndarray) -> np.ndarray:
    """Inverse of create_array (for completeness / debugging)."""
    row_sizes = np.arange(29, 0, -3)
    max_row_size = row_sizes[0]
    matrix = np.zeros((len(row_sizes), max_row_size))
    start_idx = 0
    for i, size in enumerate(row_sizes):
        end_idx = start_idx + size
        matrix[i, :size] = flat_data[start_idx:end_idx]
        start_idx = end_idx
    return matrix

# -----------------------------
# Data Loading (same paths)
# -----------------------------
# input_data: measured + prepend initial true map at t0
input_data = 1.0 / np.load('measured_training_data_sameRowColSeq43.npy') * 1000.0
initial_data = 1.0 / np.load('united_triangular_matrices.npy')[:, 0, :, :] * 1000.0
initial_data = np.expand_dims(initial_data, axis=1)
input_data = np.concatenate((initial_data, input_data), axis=1)
input_data = np.nan_to_num(input_data, nan=0.0)

# output_data: true maps
output_data = 1.0 / np.load('united_triangular_matrices.npy') * 1000.0
output_data = np.nan_to_num(output_data, nan=0.0)

# -----------------------------
# Preprocessing flags (same)
# -----------------------------
if early:
    input_data = input_data[:, :50, :, :]
    output_data = output_data[:, :50, :, :]

if chooseIndex:
    index = [26, 37, 31, 19, 36, 28, 38, 18, 15]
    input_data = np.array([input_data[x, :, :, :] for x in index])
    output_data = np.array([output_data[x, :, :, :] for x in index])

if sparce:
    input_data = input_data[:, ::10, :, :]
    output_data = output_data[:, ::10, :, :]

if diff:
    input_data = np.diff(input_data, axis=1)
    output_data = np.diff(output_data, axis=1)

# Optional per-time normalization (disabled by default to match original flags)
if normalization:
    # input
    time_step_min = np.min(input_data, axis=(0, 2, 3), keepdims=True)
    time_step_max = np.max(input_data, axis=(0, 2, 3), keepdims=True)
    input_data = (input_data - time_step_min) / (time_step_max - time_step_min + 1e-12)

    # output
    time_step_min_output = np.min(output_data, axis=(0, 2, 3), keepdims=True)
    time_step_max_output = np.max(output_data, axis=(0, 2, 3), keepdims=True)
    output_data = (output_data - time_step_min_output) / (time_step_max_output - time_step_min_output + 1e-12)

    # Save factors (same file names)
    np.savez('normalization_factors.npz', time_step_min=time_step_min, time_step_max=time_step_max)
    np.savez('normalization_factors_output.npz',
             time_step_min_output=time_step_min_output, time_step_max_output=time_step_max_output)

# -----------------------------
# Build X, y sequences
# -----------------------------
X_list, y_list = [], []

if timeContext:
    # (*disabled by default, but implemented equivalently*)
    for series in range(input_data.shape[0]):
        for i in range(input_data.shape[1] - time_steps + 1):
            inp_seq = []
            out_seq = []
            for ts in range(i, i + time_steps):
                flat_in = create_array(input_data[series, ts, :, :]).astype(np.float32)
                time_ctx = np.full_like(flat_in, ts / float(input_data.shape[1] - 1), dtype=np.float32)
                with_time = np.concatenate([flat_in, time_ctx], axis=0)
                inp_seq.append(with_time)
            for ts in range(i, i + time_steps):
                flat_out = create_array(output_data[series, ts, :, :]).astype(np.float32)
                out_seq.append(flat_out)
            X_list.append(np.array(inp_seq, dtype=np.float32))
            y_list.append(np.array(out_seq, dtype=np.float32))
else:
    for series in range(input_data.shape[0]):
        for i in range(input_data.shape[1] - time_steps + 1):
            inp_seq = []
            out_seq = []
            for ts in range(i, i + time_steps):
                flat_in = create_array(input_data[series, ts, :, :]).astype(np.float32)
                inp_seq.append(flat_in)
            for ts in range(i, i + time_steps):
                flat_out = create_array(output_data[series, ts, :, :]).astype(np.float32)
                out_seq.append(flat_out)
            X_list.append(np.array(inp_seq, dtype=np.float32))
            y_list.append(np.array(out_seq, dtype=np.float32))

X = np.array(X_list, dtype=np.float32)  # (N, T, F)
y = np.array(y_list, dtype=np.float32)  # (N, T, F)

# Mean centering per time step (same logic)
if meanCentered:
    Xmean = []
    ymean = []
    for i in range(X.shape[1]):
        Xmean.append(np.mean(X[:, i, :], axis=0))
    for i in range(y.shape[1]):
        ymean.append(np.mean(y[:, i, :], axis=0))
    Xmean = np.array(Xmean, dtype=np.float32)
    ymean = np.array(ymean, dtype=np.float32)

    X = X - Xmean  # broadcast OK: (N,T,F) - (T,F)
    y = y - ymean

    np.savez('mean_values_first.npz', Xmean=Xmean, ymean=ymean)

# Keras code takes only the first time step of y
y = y[:, 0, :]  # shape (N, F_out)

# -----------------------------
# Train/val split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Torch Dataset / DataLoader
# -----------------------------
class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, T, F)
        self.y = torch.from_numpy(y)  # (N, F_out)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 5
train_ds = XYDataset(X_train, y_train)
val_ds = XYDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

# -----------------------------
# Model (same design)
# LSTM(512) x2 -> Dense(256)->Dense(128)->Dense(F_out)
# -----------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 512, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0  # Keras LSTM default unless specified
        )
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, T, F)
        out, (hn, cn) = self.lstm(x)     # out: (B,T,H), hn: (num_layers,B,H)
        last = out[:, -1, :]             # last time step hidden (B,H) -> matches Keras LSTM(return_sequences=False)
        z = self.relu(self.fc1(last))
        z = self.relu(self.fc2(z))
        y = self.out(z)                  # (B, F_out)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X.shape[2]
output_dim = y.shape[1]

model = LSTMRegressor(input_dim=input_dim, output_dim=output_dim, hidden_size=512, num_layers=2).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Keras default Adam lr≈0.001

# -----------------------------
# Training loop
# -----------------------------
epochs = 100
train_losses = []
val_losses = []

for epoch in range(1, epochs + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)   # (B,T,F)
        yb = yb.to(device)   # (B,F_out)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_ds)

    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            running_val += loss.item() * xb.size(0)
    val_loss = running_val / len(val_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"[epoch {epoch:03d}] train={train_loss:.6f}  val={val_loss:.6f}")

# -----------------------------
# Plot and save losses
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
# plt.show()  # non-interactive env なら保存のみ

# -----------------------------
# Save model (state_dict + meta)
# -----------------------------
save_payload = {
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'output_dim': output_dim,
    'time_steps': time_steps,
    'mean_centered': meanCentered,
    'flags': {
        'early': early, 'chooseIndex': chooseIndex, 'sparce': sparce,
        'diff': diff, 'timeContext': timeContext, 'normalization': normalization
    }
}
torch.save(save_payload, 'single_output_lstm_model.pt')
print("Saved model to single_output_lstm_model.pt and loss plot to loss_curve.png")
