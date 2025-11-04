# -*- coding: utf-8 -*-
"""
K-Fold grid search version of:
  07_trainingAppResTemporalDerivativeFirstImage.py

What changed:
- Adds argparse flags: --lrs, --batch-sizes, --kfolds, --epochs, --seed, --retrain-best/--no-retrain-best
- Wraps training into run_one_fold(...)
- Loops over (lr, batch_size) × k-fold, writes per-fold CSV to results/<combo>_foldXX.csv
- Saves grid_search_results.csv, best_config.txt, and results/best_cv_indices.npz
- Optionally retrains best config on an 80/20 split and saves best_model.pt, loss_history_best_retrain.csv, loss_curve_best_retrain.png

Data processing and model architecture are untouched.
"""

import os
import math
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

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
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# Utils
# -----------------------------
def create_array(data: np.ndarray) -> np.ndarray:
    """Flatten triangular rows in sizes 29, 26, 23, ..., 2, 1 (step -3)."""
    row_sizes = np.arange(29, 0, -3)
    filled_data = []
    for i, size in enumerate(row_sizes):
        filled_data.extend(data[i, :size])
    return np.array(filled_data, dtype=np.float32)

def de_create_array(flat_data: np.ndarray) -> np.ndarray:
    """Inverse of create_array (for completeness / debugging)."""
    row_sizes = np.arange(29, 0, -3)
    max_row_size = row_sizes[0]
    matrix = np.zeros((len(row_sizes), max_row_size), dtype=np.float32)
    start_idx = 0
    for i, size in enumerate(row_sizes):
        end_idx = start_idx + size
        matrix[i, :size] = flat_data[start_idx:end_idx]
        start_idx = end_idx
    return matrix

def _parse_list(s: str, cast):
    s = (s or "").strip()
    if not s:
        return []
    return [cast(p.strip()) for p in s.replace(";", ",").split(",") if p.strip()]

def _kfold_indices(n_samples: int, n_splits: int, seed: int = 42):
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

def save_loss_curve_png_pillow(train_losses, val_losses, out_path="loss_curve_best_retrain.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# -----------------------------
# Data Loading (same paths)
# -----------------------------
def load_and_preprocess():
    # input_data: measured + prepend initial true map at t0
    input_data = 1.0 / np.load('measured_training_data_sameRowColSeq34.npy') * 1000.0
    initial_data = 1.0 / np.load('united_triangular_matrices.npy')[:, 0, :, :] * 1000.0
    initial_data = np.expand_dims(initial_data, axis=1)
    input_data = np.concatenate((initial_data, input_data), axis=1)
    input_data = np.nan_to_num(input_data, nan=0.0)

    # output_data: true maps
    output_data = 1.0 / np.load('united_triangular_matrices.npy') * 1000.0
    output_data = np.nan_to_num(output_data, nan=0.0)

    # Flags
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

    # Optional normalization (off by default)
    if normalization:
        time_step_min = np.min(input_data, axis=(0, 2, 3), keepdims=True)
        time_step_max = np.max(input_data, axis=(0, 2, 3), keepdims=True)
        input_data = (input_data - time_step_min) / (time_step_max - time_step_min + 1e-12)

        time_step_min_output = np.min(output_data, axis=(0, 2, 3), keepdims=True)
        time_step_max_output = np.max(output_data, axis=(0, 2, 3), keepdims=True)
        output_data = (output_data - time_step_min_output) / (time_step_max_output - time_step_min_output + 1e-12)

        np.savez('normalization_factors_first.npz', time_step_min=time_step_min, time_step_max=time_step_max)
        np.savez('normalization_factors_output_first.npz',
                 time_step_min_output=time_step_min_output, time_step_max_output=time_step_max_output)

    # Build sequences X, y
    X_list, y_list = [], []
    if timeContext:
        for series in range(input_data.shape[0]):
            for i in range(input_data.shape[1] - time_steps + 1):
                inp_seq, out_seq = [], []
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
                inp_seq, out_seq = [], []
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

    Xmean = None
    ymean = None
    if meanCentered:
        Xmean = np.array([np.mean(X[:, i, :], axis=0) for i in range(X.shape[1])], dtype=np.float32)
        ymean = np.array([np.mean(y[:, i, :], axis=0) for i in range(y.shape[1])], dtype=np.float32)
        X = X - Xmean
        y = y - ymean
        np.savez('mean_values_first.npz', Xmean=Xmean, ymean=ymean)

    # Keras code takes only the first time step of y
    y = y[:, 0, :]  # (N, F_out)

    return X, y, Xmean, ymean

# -----------------------------
# Torch Dataset
# -----------------------------
class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, T, F)
        self.y = torch.from_numpy(y)  # (N, F_out)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
            dropout=0.0
        )
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)     # out: (B,T,H)
        last = out[:, -1, :]             # last time step hidden (B,H)
        z = self.relu(self.fc1(last))
        z = self.relu(self.fc2(z))
        y = self.out(z)                  # (B, F_out)
        return y

# -----------------------------
# One-fold trainer
# -----------------------------
def run_one_fold(X, y, train_idx, val_idx, lr, batch_size, epochs, device, combo_tag: str, fold_tag: str):
    ds = XYDataset(X, y)
    pin = (device.type == "cuda")
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, pin_memory=pin)

    input_dim = X.shape[2]
    output_dim = y.shape[1]
    model = LSTMRegressor(input_dim=input_dim, output_dim=output_dim, hidden_size=512, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0; ntr = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            run += loss.item() * xb.size(0); ntr += xb.size(0)
        train_loss = run / max(1, ntr)

        model.eval()
        runv = 0.0; nva = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                runv += loss.item() * xb.size(0); nva += xb.size(0)
        val_loss = runv / max(1, nva)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss

    os.makedirs("results_first", exist_ok=True)
    with open(f"results_first/{combo_tag}_{fold_tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
            w.writerow([i, tr, va])
    print(f"[log] saved results_first/{combo_tag}_{fold_tag}.csv")
    return best_val

# -----------------------------
# Retrain best config on 80/20
# -----------------------------
def retrain_and_save_best(X, y, lr, batch_size, epochs, device, seed=42):
    N = X.shape[0]
    n_val = int(np.ceil(N * 0.2))
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    train_idx, val_idx = idx[:-n_val], idx[-n_val:]

    ds = XYDataset(X, y)
    pin = (device.type == "cuda")
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, pin_memory=pin)

    input_dim = X.shape[2]
    output_dim = y.shape[1]
    model = LSTMRegressor(input_dim=input_dim, output_dim=output_dim, hidden_size=512, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0; ntr = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            run += loss.item() * xb.size(0); ntr += xb.size(0)
        train_loss = run / max(1, ntr)

        model.eval()
        runv = 0.0; nva = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                runv += loss.item() * xb.size(0); nva += xb.size(0)
        val_loss = runv / max(1, nva)

        train_losses.append(train_loss); val_losses.append(val_loss)
        print(f"[retrain {ep:03d}] train={train_loss:.6f}  val={val_loss:.6f}")

    # Save artifacts
    torch.save(model.state_dict(), "best_model_first.pt")
    with open("loss_history_best_retrain.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_loss"])
        for i,(tr,va) in enumerate(zip(train_losses, val_losses), start=1):
            w.writerow([i, tr, va])
    save_loss_curve_png_pillow(train_losses, val_losses, out_path="loss_curve_best_retrain_first.png")

    # Optional: also save payload with meta like original file did
    payload = {
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
    torch.save(payload, 'single_output_lstm_model.pt')
    print("Saved model to best_model_first.pt and single_output_lstm_model.pt and loss plot to loss_curve_best_retrain.png")
    return train_idx, val_idx, train_losses, val_losses

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", type=str, default="1e-3,5e-4,1e-4")
    ap.add_argument("--batch-sizes", type=str, default="4,8,16")
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retrain-best", action="store_true", help="Retrain best config on 80/20 and save artifacts")
    ap.add_argument("--no-retrain-best", dest="retrain_best", action="store_false")
    ap.set_defaults(retrain_best=True)
    ns = ap.parse_args()

    set_seed(ns.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[info] device:", device)

    # Load & preprocess
    X, y, Xmean, ymean = load_and_preprocess()
    print(f"[shape] X: {X.shape}  y: {y.shape}")

    lrs = _parse_list(ns.lrs, float)
    bss = _parse_list(ns.batch_sizes, int)
    if not lrs or not bss:
        raise ValueError("Provide --lrs and --batch-sizes (comma or semicolon separated)")

    splits = list(_kfold_indices(X.shape[0], ns.kfolds, seed=ns.seed))

    os.makedirs("results_first", exist_ok=True)
    results = []
    best_mean = float("inf")
    best_cfg = (None, None)

    for lr in lrs:
        for bs in bss:
            combo_tag = f"lr{lr}_bs{bs}".replace(".", "p")
            print(f"\n[grid] lr={lr}  batch_size={bs}  k={ns.kfolds}")
            fold_vals = []
            for k, (tr, va) in enumerate(splits, start=1):
                v = run_one_fold(X, y, tr, va, lr=lr, batch_size=bs, epochs=ns.epochs,
                                 device=device, combo_tag=combo_tag, fold_tag=f"fold{k:02d}")
                fold_vals.append(v)
                print(f"  fold {k}/{ns.kfolds} best_val={v:.6f}")
            mean_v = float(np.mean(fold_vals))
            print(f"[grid] lr={lr} bs={bs} mean_best_val={mean_v:.6f}")
            results.append((lr, bs, *fold_vals, mean_v))
            if mean_v < best_mean:
                best_mean = mean_v
                best_cfg = (lr, bs)

    # Save grid summary
    header = ["lr","batch_size"]+[f"fold{i+1}_best_val" for i in range(ns.kfolds)]+["mean_best_val"]
    with open("grid_search_results.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for row in results:
            w.writerow(row)
    with open("best_config.txt","w") as f:
        f.write(f"best_lr={best_cfg[0]}\\n")
        f.write(f"best_batch_size={best_cfg[1]}\\n")
        f.write(f"mean_best_val={best_mean}\\n")
    print(f"[best] lr={best_cfg[0]}  batch_size={best_cfg[1]}  mean_best_val={best_mean:.6f}")

    # Save CV splits
    save_dict = {"n_samples": X.shape[0], "kfolds": ns.kfolds, "seed": ns.seed,
                 "best_lr": best_cfg[0], "best_batch_size": best_cfg[1], "mean_best_val": best_mean}
    for k, (tr, va) in enumerate(splits, start=1):
        save_dict[f"train_idx_fold{k:02d}"] = np.asarray(tr, dtype=np.int64)
        save_dict[f"val_idx_fold{k:02d}"]   = np.asarray(va, dtype=np.int64)
    np.savez("results_first/best_cv_indices.npz", **save_dict)

    # Optional retrain
    if ns.retrain_best and best_cfg[0] is not None:
        print("\n[retrain] best config on 80/20 split...")
        tr80, va20, tr_hist, va_hist = retrain_and_save_best(X, y,
                                                             lr=best_cfg[0], batch_size=best_cfg[1],
                                                             epochs=ns.epochs, device=device, seed=ns.seed)
        np.savez("results_first/best_retrain_indices.npz",
                 train_idx=np.asarray(tr80, dtype=np.int64),
                 val_idx=np.asarray(va20, dtype=np.int64),
                 best_lr=best_cfg[0], best_batch_size=best_cfg[1], seed=ns.seed)

if __name__ == "__main__":
    main()
