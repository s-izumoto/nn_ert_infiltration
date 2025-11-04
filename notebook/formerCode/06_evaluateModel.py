# -*- coding: utf-8 -*-
"""
PyTorch rewrite of: 05_useModel_AppResAppRes_seq2seq_exceptFirst_long2_loop.py
- Mirrors data preparation flags/flows
- Implements a simple LSTM encoder–decoder (no attention) for sequence-to-sequence
- Performs greedy decoding by feeding the previous prediction to the decoder

⚠️ Note:
- Keras .keras weights are NOT compatible with PyTorch. Train a PyTorch model
  with the same architecture to use this script for inference, or adapt a
  conversion pipeline separately. See `CHECKPOINT_PATH`.
"""

from __future__ import annotations
import os
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# =====================
# Config (match TF code)
# =====================
MODEL_NAME = "simplified_seq2seq_lstm"
CHECKPOINT_PATH = Path("seq2seq_lstm.pt")  # <- set to your .pt or .pth

time_steps = 30
output_seq_length = 29

# Data flags
early = True
chooseIndex = False
sparce = True
use_diff = True
use_time_context = False
use_normalization = False
meanCentered = True

# Model hyperparams (adjust to match your TF model if known)
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.0
BIDIR = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Helpers: triangular (de)flatten
# =====================

def create_array(data: np.ndarray) -> np.ndarray:
    """Flatten triangular rows: sizes 29, 26, 23, ..., 2? actually down by 3."""
    row_sizes = np.arange(29, 0, -3)
    filled = []
    for i, size in enumerate(row_sizes):
        filled.extend(data[i, :size])
    return np.array(filled, dtype=np.float32)


def de_create_array(flat: np.ndarray) -> np.ndarray:
    row_sizes = np.arange(29, 0, -3)
    max_row = row_sizes[0]
    mat = np.zeros((len(row_sizes), max_row), dtype=np.float32)
    start = 0
    for i, size in enumerate(row_sizes):
        end = start + size
        mat[i, :size] = flat[start:end]
        start = end
    return mat


# =====================
# Model
# =====================
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_layers: int = 1, dropout: float = 0.0, bidir: bool = False):
        super().__init__()
        self.hidden = hidden
        self.bidir = bidir
        self.num_directions = 2 if bidir else 1
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0.0),
                           batch_first=True, bidirectional=bidir)

    def forward(self, x):
        # x: (B, T, F)
        out, (h, c) = self.rnn(x)  # h,c: (num_layers*num_directions, B, H)
        if self.bidir:
            # concat directions for last layer
            # Convert to (B, H*2) then project back to (B, H)
            h_last = torch.cat([h[-2], h[-1]], dim=-1)
            c_last = torch.cat([c[-2], c[-1]], dim=-1)
            return out, (h_last.unsqueeze(0), c_last.unsqueeze(0))
        return out, (h, c)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0.0),
                           batch_first=True)
        self.proj = nn.Linear(hidden, input_dim)

    def forward(self, y_prev, hidden):
        # y_prev: (B, 1, F)
        out, hidden = self.rnn(y_prev, hidden)
        y_hat = self.proj(out)  # (B, 1, F)
        return y_hat, hidden


class Seq2Seq(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_layers: int = 1, dropout: float = 0.0, bidir: bool = False):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden, num_layers, dropout, bidir)
        self.decoder = Decoder(input_dim, hidden, num_layers, dropout)

    def forward(self, src: torch.Tensor, tgt_len: int) -> torch.Tensor:
        # src: (B, Tsrc, F)
        B, _, F = src.shape
        _, hidden = self.encoder(src)
        # start with zeros (like the TF script's decoder_input)
        y_prev = src.new_zeros((B, 1, F))
        outs = []
        for _ in range(tgt_len):
            y_hat, hidden = self.decoder(y_prev, hidden)
            outs.append(y_hat)
            y_prev = y_hat.detach()  # greedy
        return torch.cat(outs, dim=1)  # (B, tgt_len, F)


# =====================
# Data loading (mirror TF code)
# =====================

def load_data():
    # scale 1/x * 1000 as in the TF script
    input_data = 1.0 / np.load('measured_training_data_sameRowColSeq43.npy') * 1000.0
    initial_data = 1.0 / np.load('united_triangular_matrices.npy')[:, 0, :, :] * 1000.0
    initial_data = np.expand_dims(initial_data, axis=1)
    input_data = np.concatenate((initial_data, input_data), axis=1)
    input_data = np.nan_to_num(input_data, nan=0.0).astype(np.float32)

    output_data = 1.0 / np.load('united_triangular_matrices.npy') * 1000.0
    output_data = np.nan_to_num(output_data, nan=0.0).astype(np.float32)

    if early:
        output_data = output_data[:, :310, :, :]
        input_data = input_data[:, :310, :, :]

    if chooseIndex:
        index = [26, 37, 31, 19, 36, 28, 38, 18, 15]
        output_data = np.array([output_data[x, :, :, :] for x in index])
        input_data = np.array([input_data[x, :, :, :] for x in index])

    if sparce:
        input_data = input_data[:, ::10, :, :]
        output_data = output_data[:, ::10, :, :]

    if use_diff:
        input_data = np.diff(input_data, axis=1)
        output_data = np.diff(output_data, axis=1)

    if use_normalization:
        norm = np.load('normalization_factors.npz')
        tmin = norm['time_step_min']
        tmax = norm['time_step_max']
        input_data = (input_data - tmin) / (tmax - tmin)

    return input_data, output_data


# =====================
# Mean-centering helpers
# =====================

def apply_mean_centering(x_seq_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract Xmean from each time step features; return (x_mc, Xmean, ymean)."""
    if meanCentered:
        data = np.load('mean_values.npz')
        Xmean = data['Xmean']
        ymean = data['ymean']
        x_mc = x_seq_2d - Xmean
        return x_mc, Xmean, ymean
    return x_seq_2d, None, None


# =====================
# Inference over all series
# =====================

def main():
    # Load data
    input_data, output_data = load_data()

    # Prepare one series at a time (as in TF)
    # Determine feature dimension by flattening one frame
    sample_flat = create_array(input_data[0, 0, :, :])
    feat_dim = sample_flat.shape[0]

    # Build model
    model = Seq2Seq(
        input_dim=feat_dim + (1 if use_time_context else 0),
        hidden=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidir=BIDIR,
    ).to(DEVICE)

    if CHECKPOINT_PATH.exists():
        state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"[info] Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"[warn] Checkpoint not found: {CHECKPOINT_PATH}. The model has random weights.")

    model.eval()
    os.makedirs(".", exist_ok=True)

    n_series = len(input_data)
    for series in range(n_series):
        # Build encoder input sequence
        enc_steps = []
        dec_targets = []
        if use_time_context:
            for ts in range(0, time_steps):
                flat = create_array(input_data[series, ts, :, :]).astype(np.float32)
                time_ctx = np.full_like(flat, ts / float(input_data.shape[1] - 1), dtype=np.float32)
                with_time = np.concatenate([flat, time_ctx], axis=0)
                enc_steps.append(with_time)
            for ts in range(1, time_steps):
                dec_targets.append(create_array(output_data[series, ts, :, :]).astype(np.float32))
        else:
            for ts in range(0, time_steps):
                enc_steps.append(create_array(input_data[series, ts, :, :]).astype(np.float32))
            for ts in range(1, time_steps):
                dec_targets.append(create_array(output_data[series, ts, :, :]).astype(np.float32))

        enc_np = np.stack(enc_steps, axis=0)     # (T, F[+1])
        tgt_np = np.stack(dec_targets, axis=0)   # (T-1, F)

        # Mean centering (input only), keep ymean to add back to predictions per step
        enc_np_mc, Xmean, ymean = apply_mean_centering(enc_np)

        # Torch tensors
        src = torch.from_numpy(enc_np_mc).unsqueeze(0).to(DEVICE)  # (1, T, F)

        with torch.no_grad():
            pred = model(src, tgt_len=output_seq_length)  # (1, 29, F)
            pred_np = pred.squeeze(0).cpu().numpy()       # (29, F)

        # Add means back step-wise if enabled
        if meanCentered and ymean is not None:
            # ymean expected shape: (29, F)
            if ymean.shape != pred_np.shape:
                raise ValueError(f"ymean shape {ymean.shape} != predictions {pred_np.shape}")
            pred_np = pred_np + ymean

        # Convert to triangular images
        output_imgs = []
        true_imgs = []
        for t in range(output_seq_length):
            output_imgs.append(de_create_array(pred_np[t]))
            true_imgs.append(de_create_array(tgt_np[t]))
        output_imgs = np.stack(output_imgs, axis=0)  # (29, rows, cols)
        true_imgs = np.stack(true_imgs, axis=0)

        out_name = f"output_image_1_4_series_long2_{series}.npy"
        np.save(out_name, output_imgs.astype(np.float32))
        print(f"[save] {out_name}  shape={output_imgs.shape}")

        # Optional: quick summaries (sum over time like TF commented viz)
        out_sum = output_imgs.sum(axis=0)
        true_sum = true_imgs.sum(axis=0)
        print(f"[summary] series={series}  out_sum(min/max)=({out_sum.min():.4g}/{out_sum.max():.4g})  true_sum(min/max)=({true_sum.min():.4g}/{true_sum.max():.4g})")


if __name__ == "__main__":
    main()
