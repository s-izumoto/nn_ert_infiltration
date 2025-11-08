# -*- coding: utf-8 -*-
"""
Sequence-to-Sequence training with YAML configuration and elapsed-time logging
-----------------------------------------------------------------------------

Overview
--------
This script trains a 2-layer LSTM encoder–decoder (seq2seq) model to predict
time-evolving *triangular* conductivity maps from short history windows.
It reads inputs/targets from .npy/.npz files **or** from folders that contain
multiple .npy/.npz files, builds sliding-window sequences, runs a small grid
search over (learning rate × batch size) with K-fold CV, and optionally
re-trains the best setting on an 80/20 split. Loss histories and indices are
saved for full reproducibility, and an optional PNG loss curve is rendered.

What the script does (high level)
---------------------------------
1) Load arrays from a file or from every .npy/.npz inside a folder.
   - Arrays are normalized to shape (N, T, H, W). If an array is (T, H, W),
     we add a singleton N=1.
   - For .npz we load either the provided key or the first available key.

2) Build model inputs/targets:
   - `measured` and `united` are converted to conductivity units via (1000 / x)
     and NaNs/±Inf are sanitized.
   - `initial_data` is the first time step of the true field (from `united`).
     The model input is the concatenation of `initial_data` (T=1) and
     `measured` (T>1) along time.
   - Optional time cropping (`early`), sub-sampling (`sparce`), difference
     over time (`diff_flag`), per-time-step min–max normalization
     (`normalization`), and mean-centering (`meanCentered`) are supported.

3) Create sliding sequences of length `time_steps` (encoder) that predict
   the last `output_seq_length` steps (decoder). Because the spatial grid is
   triangular, each (H, W) map is flattened with `create_array(...)` into a
   compact 1D vector whose layout matches the Wenner-style triangular mask.

4) Train & evaluate:
   - K-fold CV over (learning rates × batch sizes); pick the configuration
     with the best mean validation loss.
   - Optionally retrain with the best hyperparameters on an 80/20 split,
     save the final weights and a loss-curve PNG (if Pillow is available).

Expected data shapes & triangular layout
----------------------------------------
- Raw arrays: (N, T, H, W) or (T, H, W).
- Triangular flattening: the grid is not rectangular; only the first
  `row_sizes = [29, 26, 23, ..., 2]` entries of each row are valid and are
  concatenated row by row. `create_array` produces a 1D feature vector
  F = sum(row_sizes). `de_create_array` reverses this for visualization.

# ===========================
# YAML Configuration Guide — 05_trainingSequence.py
# ===========================
# Keys for training a 2-layer LSTM encoder–decoder (seq2seq) on triangular maps.
# Paste this block at the top of trainingSequence.yml (no example config below).

# --- paths ---
# paths.measured (str): File or folder of measured stacks (.npy/.npz; shape N×T×H×W or T×H×W).
# paths.united   (str): File or folder of true stacks (.npy/.npz; same ordering/shapes as measured).
# paths.results_dir (str): Output directory for logs, models, and artifacts.

# --- preprocess ---
# preprocess.early (bool): Use only the early portion of each time series.
# preprocess.chooseIndex (bool): Keep only samples listed in choose_indices.
# preprocess.sparce (bool): Subsample time steps (e.g., use ::10 internally).
# preprocess.diff (bool): Replace series with time differences (applied to X and y).
# preprocess.timeContext (bool): Append a normalized time channel [0,1] to encoder inputs.
# preprocess.normalization (bool): Per-time-step min–max over (N,H,W) for inputs/targets.
# preprocess.meanCentered (bool): Subtract per-time-step means and save them.

# --- choose_indices ---
# choose_indices ([]int): Sample indices to keep (effective when chooseIndex=true).

# --- sequence ---
# sequence.time_steps (int): Encoder window length (default 30).
# sequence.output_seq_length (int): Decoder output length (default 29).

# --- model ---
# model.enc_hidden (int): Encoder hidden size (reserved; current implementation uses 512).
# model.dec_hidden (int): Decoder hidden size (reserved; current implementation uses 512).

# --- train ---
# train.lrs ([float]): Learning rates to grid-search.
# train.batch_sizes ([int]): Batch sizes to grid-search.
# train.kfolds (int): Number of CV folds.
# train.epochs (int): Epochs per fold.
# train.seed (int): RNG seed for reproducibility.
# train.retrain_best (bool): Retrain best (lr, batch) on an 80/20 split and save artifacts.

# --- artifacts ---
# artifacts.save_loss_curve_png (bool): Save retrain loss curve as PNG (if Pillow is available).

# --- notes ---
# • Inputs are converted to conductivity (σ = 1000 / ρ) and NaN/Inf are sanitized.
# • The triangular (H,W) maps are flattened to a 1D feature via row sizes [29,26,23,...,2].
# • Folder inputs: file counts and sort order must match between measured and united.


Outputs (artifacts written to `results_dir`)
--------------------------------------------
- Grid search logs:
  - `grid_search_results.csv` (all folds for each (lr, bs) and the mean)
  - `best_config.txt` (best lr/bs and its mean val loss)
  - per-fold CSV: `{combo_tag}_{fold_tag}.csv` with train/val loss per epoch
  - `best_cv_indices.npz` (train/val indices for each fold, seed, and summary)

- Best retrain (80/20):
  - `best_model.pt` (state_dict of the seq2seq model)
  - `loss_history_best_retrain.csv`
  - `loss_curve_best_retrain.png` (if Pillow is available)
  - `best_retrain_indices.npz` (train/val indices for the 80/20 split)

- Preprocessing metadata (optional, depending on flags):
  - `normalization_factors_in.npz`, `normalization_factors_out.npz`
  - `mean_values.npz` (per-time-step means for encoder/decoder features)

Device & performance
--------------------
- Uses CUDA if available; otherwise falls back to CPU.
- Set `train.batch_sizes` conservatively if you see OOM on GPU.
- `pin_memory=True` is enabled automatically for CUDA.

How to run
----------
Example:
    python 05_trainingSequence.py --config configs/trainingSequence.yml

Common pitfalls
---------------
- Folder inputs: the **number and order** of files in `measured` and `united`
  must match (same N and consistent sorting). A mismatch will raise an error.
- When `diff=True`, both inputs and targets become *differences* along time;
  adjust downstream evaluation accordingly.
- If `timeContext=True`, a normalized time channel in [0,1] is concatenated
  to the encoder features; keep the same setting for inference pipelines.
"""

import os
import math
import random
import numpy as np
from pathlib import Path
import csv
import time
from datetime import datetime

# Pillow is optional, used only for saving loss-curve images
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
import yaml


# ============================================================
# Utility: Load arrays from file or folder
# ============================================================
def load_array_or_folder(path_or_dir: str, npz_key: str | None = None) -> np.ndarray:
    """
    Load either a single .npy/.npz file or all .npy/.npz files inside a directory.
    Concatenate along the first axis (N dimension).

    Each array is expected to have shape (T, H, W) or (N, T, H, W).
    If (T, H, W) is given, it is automatically expanded to (1, T, H, W).

    For .npz files, the specified key (npz_key) is used; otherwise, the first key is loaded.
    """
    p = Path(path_or_dir)
    files: list[Path] = []

    # Accept folder or single file
    if p.is_dir():
        files = sorted(list(p.glob("*.npy")) + list(p.glob("*.npz")))
        if not files:
            raise FileNotFoundError(f"No npy/npz found in {p}")
    else:
        files = [p]
        if not p.exists():
            raise FileNotFoundError(str(p))

    batches = []
    ref_shape = None

    for f in files:
        if f.suffix.lower() == ".npz":
            with np.load(f) as nz:
                key = npz_key if npz_key else nz.files[0]
                arr = nz[key]
        else:
            arr = np.load(f)

        arr = np.asarray(arr, dtype=np.float32)
        # Normalize shape to (1, T, H, W)
        if arr.ndim == 3:
            arr = arr[None, ...]
        elif arr.ndim != 4:
            raise ValueError(f"Unsupported array shape {arr.shape} in {f} (expect 3D or 4D)")

        if ref_shape is None:
            ref_shape = arr.shape[1:]  # (T, H, W)
        elif arr.shape[1:] != ref_shape:
            raise ValueError(f"Shape mismatch: {arr.shape} vs ref {ref_shape} in {f}")

        batches.append(arr)

    out = np.concatenate(batches, axis=0)
    return out


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Data conversion utilities
# ============================================================
def create_array(data_2d: np.ndarray) -> np.ndarray:
    """Flatten the triangular 2D map into a 1D array."""
    row_sizes = np.arange(29, 0, -3)
    filled_data = []
    for i, size in enumerate(row_sizes):
        filled_data.extend(data_2d[i, :size])
    return np.array(filled_data, dtype=np.float32)


def de_create_array(flat_data: np.ndarray) -> np.ndarray:
    """Reconstruct the triangular 2D map from a flattened 1D array."""
    row_sizes = np.arange(29, 0, -3)
    max_row_size = row_sizes[0]
    matrix = np.zeros((len(row_sizes), max_row_size), dtype=np.float32)
    start_idx = 0
    for i, size in enumerate(row_sizes):
        end_idx = start_idx + size
        matrix[i, :size] = flat_data[start_idx:end_idx]
        start_idx = end_idx
    return matrix


def _kfold_indices(n_samples: int, n_splits: int, seed: int = 42):
    """Generate reproducible k-fold indices."""
    idx = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    cur = 0
    for fs in fold_sizes:
        va = idx[cur:cur+fs]
        tr = np.concatenate([idx[:cur], idx[cur+fs:]])
        cur += fs
        yield tr, va


# ============================================================
# Dataset class
# ============================================================
class Seq2SeqDataset(Dataset):
    """Dataset for sequence-to-sequence learning."""
    def __init__(self, X, dec_in, y):
        super().__init__()
        self.X = X
        self.dec_in = dec_in
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),       # (30, Fin)
            torch.from_numpy(self.dec_in[idx]),  # (29, Fout)
            torch.from_numpy(self.y[idx])        # (29, Fout)
        )


# ============================================================
# Model definitions (Encoder, Decoder, Seq2Seq)
# ============================================================
class Encoder(nn.Module):
    """Two-layer LSTM encoder."""
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, (h, c) = self.lstm2(out1)
        return (h, c)


class Decoder(nn.Module):
    """Two-layer LSTM decoder with linear projection."""
    def __init__(self, out_dim, hidden=512):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=out_dim, hidden_size=hidden, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.proj  = nn.Linear(hidden, out_dim)

    def forward(self, y_in, h0, c0):
        out1, _ = self.lstm1(y_in, (h0, c0))
        out2, _ = self.lstm2(out1)
        y_hat = self.proj(out2)
        return y_hat


class Seq2Seq(nn.Module):
    """Full encoder-decoder (sequence-to-sequence) model."""
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden)
        self.decoder = Decoder(out_dim, hidden)

    def forward(self, x_enc, y_dec_in):
        h, c = self.encoder(x_enc)
        y_hat = self.decoder(y_dec_in, h, c)
        return y_hat


# ============================================================
# Visualization: save training curve as PNG using Pillow
# ============================================================
def save_loss_curve_png_pillow(train_losses, val_losses, out_path="loss_curve_pytorch.png",
                               size=(1200, 480), margin=60):
    if not _HAS_PIL:
        print("[warn] Pillow not found. Skipping PNG output (CSV will still be saved).")
        return
    W, H = size
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    x0, y0 = margin, margin
    x1, y1 = W - margin, H - margin

    # Draw curves
    xs = list(range(1, max(len(train_losses), len(val_losses)) + 1))
    all_vals = []
    if len(train_losses): all_vals += list(train_losses)
    if len(val_losses):   all_vals += list(val_losses)
    if not all_vals:
        all_vals = [0.0, 1.0]
    vmin = float(min(all_vals)); vmax = float(max(all_vals))
    if abs(vmax - vmin) < 1e-12: vmax = vmin + 1.0

    def to_xy(i, v):
        n = len(xs)
        t = (i - 1) / max(1, n - 1)
        X = x0 + t * (x1 - x0)
        Y = y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)
        return (X, Y)

    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)
    for k in range(6):
        ty = y1 - k * (y1 - y0) / 5
        draw.line([(x0, ty), (x1, ty)], fill=(230, 230, 230), width=1)

    if len(train_losses) >= 2:
        pts = [to_xy(i, v) for i, v in zip(xs[:len(train_losses)], train_losses)]
        draw.line(pts, fill=(30, 144, 255), width=3)
    if len(val_losses) >= 2:
        pts = [to_xy(i, v) for i, v in zip(xs[:len(val_losses)], val_losses)]
        draw.line(pts, fill=(220, 20, 60), width=3)

    # Add labels and legend
    try:
        font = ImageFont.load_default()
        draw.text((x0, y0 - 20), "Loss", fill=(0, 0, 0), font=font)
        draw.text((x1 - 80, y1 + 8), "Epoch", fill=(0, 0, 0), font=font)
        legend_y = y0 + 8
        draw.rectangle([x1 - 180, legend_y - 8, x1 - 20, legend_y + 40], outline=(0,0,0))
        draw.line([(x1 - 170, legend_y), (x1 - 130, legend_y)], fill=(30,144,255), width=3)
        draw.text((x1 - 125, legend_y - 8), "Train", fill=(0, 0, 0), font=font)
        draw.line([(x1 - 170, legend_y + 22), (x1 - 130, legend_y + 22)], fill=(220,20,60), width=3)
        draw.text((x1 - 125, legend_y + 14), "Val", fill=(0, 0, 0), font=font)
    except Exception:
        pass

    img.save(out_path)
    print(f"[done] saved {out_path}")
