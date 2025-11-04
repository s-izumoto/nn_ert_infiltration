# -*- coding: utf-8 -*-
"""
Training script for the first-stage LSTM regression model
---------------------------------------------------------

This script performs end-to-end training of an LSTM-based regressor for
temporal evolution of apparent resistivity maps. 

Main features:
    - Configuration through a YAML file (see --config argument)
    - Folder-based input and output management
    - Optional k-fold cross validation and grid search over learning rates
      and batch sizes
    - Optional retraining using the best hyperparameter combination
    - Automatic saving of model weights, mean values, normalization factors,
      and training logs

Workflow overview:
    1. Load configuration (paths, preprocessing, model, and training settings)
    2. Load measured and true resistivity data (converted to conductivity)
    3. Apply preprocessing (cropping, differencing, normalization, etc.)
    4. Create sequence data for LSTM
    5. Train and evaluate with k-fold CV
    6. Retrain the model using the best settings and save all outputs

"""

import os
import csv
import math
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import time, datetime
from datetime import timedelta

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. Please install it using 'pip install pyyaml'.") from e


# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed: int = 42):
    """Ensure reproducibility by fixing all random seeds."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_list_cli(s: str, cast):
    """Parse comma- or semicolon-separated CLI arguments into a list."""
    s = (s or "").strip()
    if not s:
        return []
    return [cast(p.strip()) for p in s.replace(";", ",").split(",") if p.strip()]


# Triangular row sizes for flattening the 2D matrix
ROW_SIZES = np.arange(29, 0, -3)

def create_array(data: np.ndarray) -> np.ndarray:
    """
    Flatten the triangular matrix (e.g., ERT measurement layout)
    by concatenating only the valid parts of each row.
    """
    filled = []
    for i, size in enumerate(ROW_SIZES):
        filled.extend(data[i, :size])
    return np.array(filled, dtype=np.float32)


# ============================================================
# Dataset and Model Definitions
# ============================================================

class XYDataset(Dataset):
    """Simple PyTorch Dataset wrapping input (X) and target (y) arrays."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    """
    Multi-layer LSTM followed by fully connected layers for regression.
    Takes a sequence input and predicts a single flattened output map.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 512, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        last = out[:, -1, :]           # take only the last timestep
        z = self.relu(self.fc1(last))
        z = self.relu(self.fc2(z))
        y = self.out(z)                # final output
        return y


# ============================================================
# Core Functions
# ============================================================

def load_and_preprocess(cfg: dict):
    """
    Load input data and apply preprocessing steps as defined in YAML config.
    Includes options for early cropping, sparsity, differencing, normalization,
    and mean-centering.
    """
    p_in = cfg["inputs"]
    p_pp = cfg["preprocess"]
    p_out = cfg["outputs"]

    measured_path = Path(p_in["measured_dir"]) / p_in["measured_file"]
    united_path   = Path(p_in["united_dir"]) / p_in["united_file"]

    # Input: measured conductivity (1/ρ * 1000) with true t0 appended at front
    input_data = 1.0 / np.load(str(measured_path)) * 1000.0
    initial_data = 1.0 / np.load(str(united_path))[:, 0, :, :] * 1000.0
    initial_data = np.expand_dims(initial_data, axis=1)
    input_data = np.concatenate((initial_data, input_data), axis=1)
    input_data = np.nan_to_num(input_data, nan=0.0)

    # Output: true conductivity (1/ρ * 1000)
    output_data = 1.0 / np.load(str(united_path)) * 1000.0
    output_data = np.nan_to_num(output_data, nan=0.0)

    # --- Preprocessing options ---
    # Early cropping
    if p_pp.get("early", False):
        L = int(p_pp.get("early_limit", 50))
        input_data = input_data[:, :L, :, :]
        output_data = output_data[:, :L, :, :]

    # Select specific indices (e.g., for debugging or subset runs)
    choose = p_pp.get("choose_index", []) or []
    if len(choose) > 0:
        input_data = np.array([input_data[i] for i in choose])
        output_data = np.array([output_data[i] for i in choose])

    # Temporal sparsity (sampling every N steps)
    if p_pp.get("sparse", False):
        s = int(p_pp.get("sparse_stride", 10))
        input_data = input_data[:, ::s, :, :]
        output_data = output_data[:, ::s, :, :]

    # Temporal differencing
    if p_pp.get("diff", False):
        input_data = np.diff(input_data, axis=1)
        output_data = np.diff(output_data, axis=1)

    # Normalization (optional)
    if p_pp.get("normalization", False):
        time_step_min = np.min(input_data, axis=(0, 2, 3), keepdims=True)
        time_step_max = np.max(input_data, axis=(0, 2, 3), keepdims=True)
        input_data = (input_data - time_step_min) / (time_step_max - time_step_min + 1e-12)

        time_step_min_output = np.min(output_data, axis=(0, 2, 3), keepdims=True)
        time_step_max_output = np.max(output_data, axis=(0, 2, 3), keepdims=True)
        output_data = (output_data - time_step_min_output) / (time_step_max_output - time_step_min_output + 1e-12)

        out_dir = Path(p_out["base_dir"]) ; out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / p_out["norm_in_npz"], time_step_min=time_step_min, time_step_max=time_step_max)
        np.savez(out_dir / p_out["norm_out_npz"],
                 time_step_min_output=time_step_min_output, time_step_max_output=time_step_max_output)

    # --- Sequence creation ---
    T = int(p_pp.get("time_steps", 4))
    use_time_ctx = bool(p_pp.get("time_context", False))

    X_list, y_list = [], []
    if use_time_ctx:
        # Add normalized time context channel
        for series in range(input_data.shape[0]):
            for i in range(input_data.shape[1] - T + 1):
                inp_seq, out_seq = [], []
                for ts in range(i, i + T):
                    flat_in = create_array(input_data[series, ts, :, :]).astype(np.float32)
                    time_ctx = np.full_like(flat_in, ts / float(input_data.shape[1] - 1), dtype=np.float32)
                    inp_seq.append(np.concatenate([flat_in, time_ctx], axis=0))
                for ts in range(i, i + T):
                    flat_out = create_array(output_data[series, ts, :, :]).astype(np.float32)
                    out_seq.append(flat_out)
                X_list.append(np.array(inp_seq, dtype=np.float32))
                y_list.append(np.array(out_seq, dtype=np.float32))
    else:
        # Without time context
        for series in range(input_data.shape[0]):
            for i in range(input_data.shape[1] - T + 1):
                inp_seq, out_seq = [], []
                for ts in range(i, i + T):
                    inp_seq.append(create_array(input_data[series, ts, :, :]).astype(np.float32))
                for ts in range(i, i + T):
                    out_seq.append(create_array(output_data[series, ts, :, :]).astype(np.float32))
                X_list.append(np.array(inp_seq, dtype=np.float32))
                y_list.append(np.array(out_seq, dtype=np.float32))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Mean centering per time step
    Xmean = None; ymean = None
    if p_pp.get("mean_centered", True):
        Xmean = np.array([np.mean(X[:, i, :], axis=0) for i in range(X.shape[1])], dtype=np.float32)
        ymean = np.array([np.mean(y[:, i, :], axis=0) for i in range(y.shape[1])], dtype=np.float32)
        X = X - Xmean
        y = y - ymean
        out_dir = Path(p_out["base_dir"]) ; out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / p_out["mean_values_npz"], Xmean=Xmean, ymean=ymean)

    # As in the Keras version, train only on the first output step
    y = y[:, 0, :]

    return X, y, Xmean, ymean


def save_loss_curve_png(train_losses, val_losses, out_path: Path):
    """Plot and save loss curves for train and validation."""
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend(); plt.title("Training Loss Curve"); plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()
