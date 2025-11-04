# -*- coding: utf-8 -*-
"""
LSTM training (first-stage regressor) for time-evolving conductivity maps
========================================================================

Overview
--------
This script trains an LSTM-based regressor that predicts the *next* flattened
conductivity map from a short input sequence of past maps. It is a cleaned-up,
YAML-driven version of the original:
    07_trainingAppResTemporalDerivativeFirstImage_kfold.py

Typical use case:
- You have time-lapse apparent resistivity data for multiple sequences/fields.
- You convert resistivity ρ [Ω·m] to conductivity σ [mS·m⁻¹] via σ = 1000 / ρ.
- You want to learn short-term temporal dynamics with a compact triangular
  layout (flattened per row sizes 29, 26, 23, …, 2, 1).

Key features
------------
1) **Config by YAML**: All paths, preprocessing flags, model sizes, and
   training hyperparameters are read from a YAML file passed via `--config`.
   Command-line options can override selected YAML fields (LRs, batch sizes,
   k-folds, epochs, seed, retrain toggle).

2) **Folder I/O**: Inputs are read from folder + filename pairs; all artifacts
   (normalization stats, mean-centering stats, CV logs, best model, curves)
   are saved under an output `base_dir` specified in the YAML.

3) **Preprocessing pipeline** (configurable):
   - **Unit conversion**: loads measured and true ρ as NumPy arrays and
     converts them to σ = 1000 / ρ. NaNs are replaced by zeros.
   - **Initial frame prepend**: the true t=0 frame is prepended to the measured
     stack so the model can “see” the initial state.
   - **Early crop / sparse sampling**: limit the maximum time or subsample time.
   - **Temporal differencing**: optional `np.diff` on the time axis.
   - **Normalization**: per-time-step min–max normalization (input/output
     separately), saved to NPZ for later reuse.
   - **Mean centering**: per-time-step mean removal for both X and y; means are
     saved to NPZ (required to invert centering at inference time).

4) **Sequence building**:
   - **Input X**: for each series and window of length `T`, we flatten each
     2D frame with a triangular mask (row sizes = 29, 26, 23, …, 2, 1) into
     a 1D vector. If `time_context: true`, a normalized time channel
     (scalar replicated to the same length) is concatenated per frame.
   - **Target y**: built with the same flattening. For training, only the
     *first* step of the output window is used (matching the original Keras
     behavior).

5) **Model**:
   - A small **unidirectional LSTM** (num_layers, hidden_size configurable)
     followed by two MLP layers and a linear head that outputs one flattened
     map (same length as the triangular flattening).

6) **Training/CV & Logging**:
   - **K-fold cross validation** over the grid of (LR × batch size).
   - Per-epoch train/validation loss logged to CSV; per-combo per-fold logs
     also saved. Summary grid CSV consolidates all folds and means.
   - The best (mean validation) hyperparameters can be **retrained** on an
     80/20 split; final weights, single-file payload, and loss curves are saved.

7) **Determinism**:
   - `seed` controls NumPy and PyTorch RNGs (CPU + CUDA).
   - Note: cuDNN kernels can still introduce slight nondeterminism unless
     further flags are set; this script opts for speed by default.

Input expectations
------------------
- **Measured file**: NumPy array with shape `(N, Tm, H, W)`, where Tm is the
  measured time length. Values are resistivity ρ [Ω·m]; converted internally
  to σ [mS·m⁻¹].
- **United (true) file**: NumPy array with shape `(N, Tu, H, W)` that contains
  the corresponding true time series (ρ), also converted to σ. The `t=0` frame
  is used to prepend the measured stack.

After preprocessing and windowing:
- `X` has shape `(num_windows, T, F)` where `F` is the triangular flattened
  length (with `+1` if `time_context: true`).
- `y` has shape `(num_windows, F)` (only the first output step is used).

Saved artifacts
---------------
Saved under `outputs.base_dir` (see YAML):
- `norm_in_npz`, `norm_out_npz`: min/max per time step for inputs/outputs.
- `mean_values_npz`: arrays `Xmean`, `ymean` for time-step-wise centering.
- `grid_search_csv`: summary of CV results per (LR, batch).
- `best_config_txt`: the best LR, batch size, and mean validation score.
- `best_cv_indices.npz`: indices used in each CV fold.
- (If `retrain_best: true`) final:
  - `best_model_pt`: `state_dict` of the best model after 80/20 retraining.
  - `single_output_lstm_model.pt`: payload with `model_state_dict`,
    `input_dim`, `output_dim`, and `time_steps`.
  - `loss_history_best_retrain.csv` and `loss_curve_best_retrain_first.png`.

YAML quick reference
--------------------
```yaml
inputs:
  measured_dir: path/to/dir
  measured_file: measured_training_data.npy
  united_dir: path/to/dir
  united_file: united_triangular_matrices.npy

preprocess:
  early: false          # if true, keep only [:early_limit] timesteps
  early_limit: 50
  choose_index: []      # optional subset of series indices
  sparse: false
  sparse_stride: 10
  diff: false
  normalization: false
  mean_centered: true
  time_steps: 4         # window length T
  time_context: false   # add normalized time channel

model:
  hidden_size: 512
  num_layers: 2

training:
  lrs: [0.001, 0.0005, 0.0001]
  batch_sizes: [4, 8, 16]
  kfolds: 5
  epochs: 100
  seed: 42
  num_workers: 0
  retrain_best: true

outputs:
  base_dir: results_first
  norm_in_npz: norm_input.npz
  norm_out_npz: norm_output.npz
  mean_values_npz: mean_values.npz
  grid_search_csv: grid_search_results.csv
  best_config_txt: best_config.txt
  cv_indices_npz: best_cv_indices.npz
  retrain_indices_npz: best_retrain_indices.npz
  best_model_pt: best_model_first.pt
  single_payload_pt: single_output_lstm_model.pt
  loss_history_csv: loss_history_best_retrain.csv
  loss_curve_png: loss_curve_best_retrain_first.png
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
