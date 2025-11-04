# -*- coding: utf-8 -*-
"""
Reconstruct full maps from a seq2seq model that was trained on
(mean-centered) temporal differences, and compare to ground-truth
for selected field indices.

Key points:
- Adds back y-mean per time step (de-centering) before integration.
- Reconstructs maps by cumulatively summing diffs starting from the
  TRUE initial map (t=0) for the target field(s).
- Saves per-field PNGs and a CSV with per-step MAE/RMSE.

Usage examples:
  python 06_evaluateModel_recon.py --checkpoint best_model.pt --fields 0 2
  python 06_evaluateModel_recon.py --checkpoint best_model.pt --field 5

Assumptions:
- Training used: early=True, sparce=True (every 10th step), use_diff=True,
  meanCentered=True, time_steps=30, output_seq_length=29.
- Files present:
    - measured_training_data_sameRowColSeq43.npy (unused for reconstruction
      baseline, but loaded to mirror the training input path)
    - united_triangular_matrices.npy
    - mean_values.npz  (contains Xmean, ymean)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =====================
# Config matching training
# =====================
time_steps = 30
output_seq_length = 29

# Data flags (must match training)
early = True
chooseIndex = False
sparce = True
use_diff = True
use_time_context = False
use_normalization = False
meanCentered = True

# Model hyperparams (structure must match the trained checkpoint)
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.0
BIDIR = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Triangular (de)flatten helpers (Wenner-α rows: 29,26,23,...)
# =====================
def create_array(data: np.ndarray) -> np.ndarray:
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
# Seq2Seq model (must match training architecture)
# =====================
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
    def forward(self, dec_in, h0, c0):
        out1, _ = self.lstm1(dec_in, (h0, c0))
        out2, _ = self.lstm2(out1)
        y_hat = self.proj(out2)
        return y_hat

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=512, num_layers=2, dropout=0.0, bidir=False):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden)
        self.decoder = Decoder(output_dim, hidden)
        self.output_dim = output_dim
    def forward(self, src: torch.Tensor, tgt_len: int) -> torch.Tensor:
        B = src.size(0)
        h, c = self.encoder(src)
        y_prev = torch.zeros(B, 1, self.output_dim, device=src.device, dtype=src.dtype)
        outs = []
        for _ in range(tgt_len):
            y_hat = self.decoder(y_prev, h, c)
            outs.append(y_hat)
            y_prev = y_hat.detach()
        return torch.cat(outs, dim=1)  # (B, tgt_len, output_dim)

# =====================
# Data loading & transforms (mirror training)
# =====================
def safe_inverse_k(arr, scale=1000.0, eps=1e-6, clip=1e6):
    a = np.array(arr, dtype=np.float64, copy=False)
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(scale, a, out=out, where=np.abs(a) > eps)
    out = np.nan_to_num(out, nan=0.0, posinf=clip, neginf=-clip)
    np.clip(out, -clip, clip, out=out)
    return out.astype(np.float32, copy=False)

def load_inputs_and_diffs():
    # Input path to mirror training pipeline (measured → for encoder only)
    raw_measured = np.load('measured_training_data_sameRowColSeq43.npy')
    input_data = safe_inverse_k(raw_measured)

    # Ground-truth full series (united)
    united = np.load('united_triangular_matrices.npy')
    united_safe = safe_inverse_k(united)

    # initial t=0 true map for each series (after safe_inverse_k)
    initial_true = united_safe[:, 0:1, :, :]  # (N,1,R,C)

    # Mirror training slicing
    if early:
        input_data = input_data[:, :310, :, :]
        united_safe = united_safe[:, :310, :, :]
        initial_true = initial_true[:, :1, :, :]

    if chooseIndex:
        index = [26, 37, 31, 19, 36, 28, 38, 18, 15]
        input_data = np.array([input_data[x] for x in index])
        united_safe = np.array([united_safe[x] for x in index])
        initial_true = np.array([initial_true[x] for x in index])

    if sparce:
        input_data = input_data[:, ::10, :, :]
        united_safe = united_safe[:, ::10, :, :]
        # initial_true is already t=0 only

    # For the decoder target during training we used diffs of united
    gt_diff = np.diff(united_safe, axis=1)  # (N, T-1, R, C)

    # Encoder input during training: diffs of [initial_true, measured]
    enc_stack = np.concatenate([initial_true, input_data], axis=1)
    enc_diff = np.diff(enc_stack, axis=1)   # (N, T-1, R, C)

    return enc_diff.astype(np.float32), gt_diff.astype(np.float32), initial_true.squeeze(1)  # (N,29,R,C), (N,29,R,C), (N,R,C)

# =====================
# Mean-centering helpers
# =====================
def load_means():
    if not meanCentered:
        return None, None
    data = np.load('mean_values.npz')
    # Shapes expected: (time_steps, F) for Xmean and ymean (after flatten)
    return data['Xmean'], data['ymean']

# =====================
# Utility: metrics and plots
# =====================
def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def plot_last_step(truth_last, pred_last, out_path: Path, title_prefix: str = ""):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    im0 = axs[0].imshow(truth_last, aspect='auto')
    axs[0].set_title(f"{title_prefix}Truth (last)")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred_last, aspect='auto')
    axs[1].set_title(f"{title_prefix}Pred (last)")
    fig.colorbar(im1, ax=axs[1])

    diff = pred_last - truth_last
    im2 = axs[2].imshow(diff, aspect='auto')
    axs[2].set_title(f"{title_prefix}Pred-Truth (last)")
    fig.colorbar(im2, ax=axs[2])

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_step_errors_csv(per_step_mae, per_step_rmse, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, len(per_step_mae) + 1)
    hdr = "step,mae,rmse\n"
    with out_csv.open('w', encoding='utf-8') as f:
        f.write(hdr)
        for s, a, r in zip(steps, per_step_mae, per_step_rmse):
            f.write(f"{s},{a:.6g},{r:.6g}\n")

# =====================
# Main
# =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default='best_model.pt')
    ap.add_argument('--outdir', type=str, default='eval_out')
    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument('--field', type=int, help='Single field index to evaluate')
    grp.add_argument('--fields', type=int, nargs='+', help='Multiple field indices to evaluate')
    ns = ap.parse_args()

    out_dir = Path(ns.outdir)

    # Load transformed inputs & ground-truth diffs + initial true maps
    enc_diff, gt_diff, initial_true = load_inputs_and_diffs()  # shapes: (N,29,R,C), (N,29,R,C), (N,R,C)

    # Determine feature dimension by flattening one frame
    sample_flat = create_array(enc_diff[0, 0])
    feat_dim = sample_flat.shape[0]

    # Build / load model
    model = Seq2Seq(
        input_dim=feat_dim + (1 if use_time_context else 0),
        output_dim=feat_dim,
        hidden=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidir=BIDIR,
    ).to(DEVICE)

    ckpt_path = Path(ns.checkpoint)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"[info] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found: {ckpt_path}. Using random weights.")

    model.eval()

    # Mean centering vectors (flattened per time step)
    Xmean, ymean = load_means()
    if meanCentered:
        if ymean is None:
            raise FileNotFoundError("mean_values.npz (ymean) not found while meanCentered=True")
        if ymean.shape[0] != output_seq_length or ymean.shape[1] != feat_dim:
            raise ValueError(f"ymean shape mismatch: got {ymean.shape}, expected ({output_seq_length},{feat_dim})")

    # Pick target fields
    n_series = enc_diff.shape[0]
    if ns.field is not None:
        target_fields = [ns.field]
    elif ns.fields is not None and len(ns.fields) > 0:
        target_fields = list(ns.fields)
    else:
        target_fields = list(range(n_series))  # default: all

    all_preds = {}

    for series in target_fields:
        if not (0 <= series < n_series):
            print(f"[skip] field {series} is out of range (0..{n_series-1})")
            continue

        # === Build encoder input sequence (flatten & optional time context) ===
        enc_steps = []  # (T, F)
        for ts in range(time_steps):  # ★ 29→30 に変更
            flat = create_array(enc_diff[series, ts]).astype(np.float32)
            enc_steps.append(flat)

        enc_np = np.stack(enc_steps, axis=0)  # (30, F)
        # === Mean centering on encoder input ===
        assert Xmean is not None, "Xmean is required"
        assert Xmean.shape == enc_np.shape, f"Xmean shape {Xmean.shape} != encoder input {enc_np.shape}"
        enc_np_mc = enc_np - Xmean

        # === Inference (predict centered diffs) ===
        src = torch.from_numpy(enc_np_mc).unsqueeze(0).to(DEVICE)  # (1, 29, F)
        with torch.no_grad():
            pred_centered = model(src, tgt_len=output_seq_length)  # (1,29,F)
            pred_centered = pred_centered.squeeze(0).cpu().numpy()  # (29,F)

        # === De-centering (add back ymean per step) ===
        if meanCentered and ymean is not None:
            if ymean.shape != pred_centered.shape:
                raise ValueError(f"ymean shape {ymean.shape} != predictions {pred_centered.shape}")
            pred_diff_flat = pred_centered + ymean
        else:
            pred_diff_flat = pred_centered

        # === Convert to triangular 2D diffs ===
        pred_diff_2d = np.stack([de_create_array(pred_diff_flat[t]) for t in range(output_seq_length)], axis=0)  # (29,R,C)

        # === Reconstruct maps by cumulative sum starting from TRUE initial map ===
        g0 = initial_true[series]  # (R,C)
        pred_maps = np.empty((output_seq_length, *g0.shape), dtype=np.float32)
        acc = g0.copy()
        for t in range(output_seq_length):
            acc = acc + pred_diff_2d[t]
            pred_maps[t] = acc

        # === Reconstruct ground-truth maps from GT diffs (fair comparison) ===
        gt_maps = np.empty_like(pred_maps)
        acc_gt = g0.copy()
        for t in range(output_seq_length):
            acc_gt = acc_gt + gt_diff[series, t]
            gt_maps[t] = acc_gt

        # === Metrics per step ===
        per_step_mae = [mae(pred_maps[t], gt_maps[t]) for t in range(output_seq_length)]
        per_step_rmse = [rmse(pred_maps[t], gt_maps[t]) for t in range(output_seq_length)]

        # === Save artifacts ===
        fld_dir = out_dir / f"field{series:03d}"
        fld_dir.mkdir(parents=True, exist_ok=True)

        # last-step panel
        plot_last_step(gt_maps[-1], pred_maps[-1], fld_dir / "last_step_panel.png", title_prefix=f"fld{series:03d} ")

        # per-step error curves
        fig = plt.figure()
        plt.plot(np.arange(1, output_seq_length+1), per_step_mae, label='MAE')
        plt.plot(np.arange(1, output_seq_length+1), per_step_rmse, label='RMSE')
        plt.xlabel('Step (sparse index)')
        plt.ylabel('Error')
        plt.title(f'Field {series:03d} per-step errors')
        plt.legend()
        fig.savefig(fld_dir / 'errors_per_step.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

        # numpy dumps for further analysis
        np.save(fld_dir / 'pred_maps.npy', pred_maps)
        np.save(fld_dir / 'gt_maps.npy', gt_maps)
        save_step_errors_csv(per_step_mae, per_step_rmse, fld_dir / 'errors.csv')

        # Aggregate handle
        all_preds[series] = {
            'pred_maps': pred_maps,
            'gt_maps': gt_maps,
            'mae': per_step_mae,
            'rmse': per_step_rmse,
        }

        print(f"[done] field {series:03d}: last-step MAE={per_step_mae[-1]:.4g}, RMSE={per_step_rmse[-1]:.4g}")

    # Optional: save a compact index of evaluated fields
    idx_txt = out_dir / 'evaluated_fields.txt'
    with idx_txt.open('w', encoding='utf-8') as f:
        for k in sorted(all_preds.keys()):
            f.write(f"{k}\n")
    print(f"[save] Results under: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
