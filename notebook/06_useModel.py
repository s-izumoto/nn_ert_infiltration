# -*- coding: utf-8 -*-
"""
Evaluate a trained seq2seq LSTM and:
  - save ALL series predictions into a single .npy file
  - save PNG images for quick visual inspection

Based on your original 06_evaluateModel.py (PyTorch seq2seq).
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =====================
# Config (match TF code)
# =====================
MODEL_NAME = "simplified_seq2seq_lstm"
CHECKPOINT_PATH = Path("best_model.pt")  # <- set to your .pt or .pth

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

# Model hyperparams
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.0
BIDIR = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Helpers: triangular (de)flatten
# =====================
def create_array(data: np.ndarray) -> np.ndarray:
    """Flatten triangular rows: sizes 29, 26, 23, ..., down by 3."""
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
# Encoder / Decoder を差し替え
# ===== Model =====
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, (h, c) = self.lstm2(out)
        return (h, c)

class Decoder(nn.Module):
    def __init__(self, out_dim, hidden):
        super().__init__()
        self.lstm1 = nn.LSTM(out_dim, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.proj  = nn.Linear(hidden, out_dim)

    # 変更：h/c を2層分受け取り、更新して返す
    def forward(self, dec_in, state1=None, state2=None):
        # state* は (h, c) or None（None のときは自動でゼロ初期化）
        out1, state1 = self.lstm1(dec_in, state1)
        out2, state2 = self.lstm2(out1, state2)
        y_hat = self.proj(out2)
        return y_hat, state1, state2


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=512, num_layers=2, dropout=0.0, bidir=False):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden)
        self.decoder = Decoder(output_dim, hidden)
        self.hidden  = hidden
        self.output_dim = output_dim

    def forward(self, src: torch.Tensor, tgt_len: int) -> torch.Tensor:
        B = src.size(0)

        # エンコーダの最終状態（1層ぶん）を取得
        h_enc, c_enc = self.encoder(src)        # 形状: (1, B, H)

        # デコーダ初期状態
        state1 = (h_enc, c_enc)                 # LSTM1 は encoder の状態で初期化
        state2 = None                           # LSTM2 はゼロ初期化（Keras と同等のことが多い）

        # 最初の入力（<bos> 的にゼロでも良いが、現行仕様にならう）
        y_prev = torch.zeros(B, 1, self.output_dim, device=src.device, dtype=src.dtype)

        outs = []
        for _ in range(tgt_len):
            # ★ ここが“連続状態”のポイント：decoder から更新後の state を受け取り次ステップへ渡す
            y_hat, state1, state2 = self.decoder(y_prev, state1, state2)
            outs.append(y_hat)                  # (B, 1, F)
            y_prev = y_hat                      # 次ステップの入力にフィードバック

        return torch.cat(outs, dim=1)           # (B, tgt_len, F)

def safe_inverse_k(arr, scale=1000.0, eps=1e-8, clip=1e6):
    # asarray(copy=...) は不可 → array(copy=False) を使う
    a = np.array(arr, dtype=np.float64, copy=False)

    # ゼロ割り防止：|a|>eps の所だけ  scale/a を計算
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(scale, a, out=out, where=np.abs(a) > eps)

    # NaN / ±inf を数値に置換してからクリップ
    out = np.nan_to_num(out, nan=0.0, posinf=clip, neginf=-clip)
    np.clip(out, -clip, clip, out=out)

    return out.astype(np.float32, copy=False)
# =====================
# Data loading (mirror TF code)
# =====================
def load_data():
    # scale 1/x * 1000 as in the TF script
    raw_measured = np.load('measured_training_data_sameRowColSeq34_test.npy')
    input_data = (1000.0 / raw_measured).astype(np.float32)
    input_data = np.nan_to_num(input_data, nan=0.0)

    united = np.load('united_triangular_matrices_test.npy')
    initial_data = (1000.0 / united[:, 0, :, :]).astype(np.float32)
    initial_data = np.nan_to_num(initial_data, nan=0.0)
    initial_data = np.expand_dims(initial_data, axis=1)
    input_data = np.concatenate((initial_data, input_data), axis=1)
    input_data = np.nan_to_num(input_data, nan=0.0).astype(np.float32)

    output_data = (1000.0 / united).astype(np.float32)
    output_data = np.nan_to_num(output_data, nan=0.0)
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
# Plot helpers
# =====================
def save_png_images(out_dir: Path, series: int, imgs: np.ndarray, vmin=None, vmax=None):
    """
    Save:
      - time-sum image (sum over t)
      - the first and last time-step images for a quick glance
    imgs: (T, rows, cols)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sum over time
    fig = plt.figure()
    plt.imshow(imgs.sum(axis=0), aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f"Series {series} - Sum over time")
    fig.savefig(out_dir / f"series{series:03d}_sum.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # First step
    fig = plt.figure()
    plt.imshow(imgs[0], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f"Series {series} - t0")
    fig.savefig(out_dir / f"series{series:03d}_t0.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Last step
    fig = plt.figure()
    plt.imshow(imgs[-1], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f"Series {series} - t{imgs.shape[0]-1}")
    fig.savefig(out_dir / f"series{series:03d}_tLast.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

# =====================
# Inference over all series
# =====================
def main():
    # Load data
    input_data, output_data = load_data()

    # Determine feature dimension by flattening one frame
    sample_flat = create_array(input_data[0, 0, :, :])
    feat_dim = sample_flat.shape[0]

    # Build model
    model = Seq2Seq(
        input_dim=feat_dim + (1 if use_time_context else 0),
        output_dim=feat_dim,                   # ★これを追加
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
    all_outputs = []  # will become (Nseries, 29, rows, cols)

    png_dir = Path("pred_png")
    png_dir.mkdir(exist_ok=True)

    # Optionally set a fixed color scale using ground-truth ranges
    # You can set vmin/vmax to None for auto-scaling per image
    vmin = None
    vmax = None

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
            if ymean.shape != pred_np.shape:
                raise ValueError(f"ymean shape {ymean.shape} != predictions {pred_np.shape}")
            pred_np = pred_np + ymean

        # Convert to triangular images
        output_imgs = []
        for t in range(output_seq_length):
            output_imgs.append(de_create_array(pred_np[t]))
        output_imgs = np.stack(output_imgs, axis=0).astype(np.float32)  # (29, rows, cols)

        # Accumulate
        all_outputs.append(output_imgs)

        # Save quick-view PNGs for this series
        save_png_images(png_dir, series, output_imgs, vmin=vmin, vmax=vmax)

        # Console summary
        out_sum = output_imgs.sum(axis=0)
        print(f"[summary] series={series}  out_sum(min/max)=({out_sum.min():.4g}/{out_sum.max():.4g})  shape={output_imgs.shape}")

    # === Save ALL series into one .npy ===
    all_outputs = np.stack(all_outputs, axis=0)  # (Nseries, 29, rows, cols)
    out_npy = "outputs_pred_all_series.npy"
    np.save(out_npy, all_outputs)
    print(f"[save] {out_npy}  shape={all_outputs.shape}")

    print(f"[done] PNGs saved under: {png_dir.resolve()}")

if __name__ == "__main__":
    main()
