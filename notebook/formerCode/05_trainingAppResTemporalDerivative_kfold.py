
# -*- coding: utf-8 -*-
"""
K-fold grid search for (lr × batch_size) on the seq2seq LSTM model.

This file is a drop-in upgrade of 05_trainingAppResTemporalDerivative.py:
- Keeps the same data preprocessing pipeline (safe_inverse, diff, mean-centered).
- Adds argparse flags:
    --lrs "1e-3,5e-4,1e-4"
    --batch-sizes "4,8,16"
    --kfolds 5
    --epochs 40
    --retrain-best (store_true) ← after selection, retrain once on an 80/20 split
    --seed 42
- For each (lr, batch_size), runs K-fold CV and records the *best val loss per fold* (min across epochs).
  Selects the setting with the lowest mean(best_val_loss_per_fold).
- Saves:
    grid_search_results.csv                 (all combos, fold losses, mean)
    best_config.txt                         (the chosen lr & batch_size)
    loss_history_best_retrain.csv           (if --retrain-best is enabled)
    loss_curve_best_retrain.png             (if Pillow available)
    best_model.pt                           (state_dict of best retrained model)
"""

import os
import math
import random
import argparse
import warnings
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# ====== Reproducibility ======
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ====== Simple KFold (no sklearn dependency) ======
def kfold_indices(n_samples: int, n_splits: int, seed: int = 42):
    """Yield (train_idx, val_idx) for each fold."""
    idx = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        current = stop
        yield train_idx, val_idx

# ====== Utilities ======
def parse_list(s, cast):
    s = s.strip()
    if not s:
        return []
    parts = [p for p in s.replace(";", ",").split(",") if p.strip()]
    return [cast(p) for p in parts]

def create_array(data_2d: np.ndarray) -> np.ndarray:
    """Pack the triangular matrix into a 1D vector (same as original)."""
    row_sizes = np.arange(29, 0, -3)
    filled_data = []
    for i, size in enumerate(row_sizes):
        filled_data.extend(data_2d[i, :size])
    return np.array(filled_data, dtype=np.float32)

def save_loss_curve_png_pillow(train_losses, val_losses, out_path="loss_curve_best_retrain.png",
                               size=(1200, 480), margin=60):
    if not _HAS_PIL:
        print("[warn] Pillow not found; skip PNG export.")
        return
    W, H = size
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    x0, y0 = margin, margin
    x1, y1 = W - margin, H - margin

    xs = list(range(1, max(len(train_losses), len(val_losses)) + 1))
    all_vals = []
    if len(train_losses): all_vals += list(train_losses)
    if len(val_losses):   all_vals += list(val_losses)
    if not all_vals:
        all_vals = [0.0, 1.0]
    vmin = float(min(all_vals))
    vmax = float(max(all_vals))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    def to_xy(i, v):
        n = len(xs)
        t = (i - 1) / max(1, n - 1)
        X = x0 + t * (x1 - x0)
        Y = y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)
        return (X, Y)

    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)
    for k in range(6):
        ty = y1 - k * (y1 - y0) / 5
        draw.line([(x0, ty), (x1, ty)], fill=(230,230,230), width=1)

    if len(train_losses) >= 2:
        pts = [to_xy(i, v) for i, v in enumerate(train_losses, start=1)]
        draw.line(pts, fill=(30,144,255), width=3)
    if len(val_losses) >= 2:
        pts = [to_xy(i, v) for i, v in enumerate(val_losses, start=1)]
        draw.line(pts, fill=(220,20,60), width=3)
    try:
        font = ImageFont.load_default()
        draw.text((x0, y0 - 20), "Loss", fill=(0, 0, 0), font=font)
        draw.text((x1 - 80, y1 + 8), "Epoch", fill=(0, 0, 0), font=font)
    except Exception:
        pass
    img.save(out_path)
    print(f"[done] saved {out_path}")

# ====== Model ======
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, (h, c) = self.lstm2(out1)
        return (h, c)

class Decoder(nn.Module):
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
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden)
        self.decoder = Decoder(out_dim, hidden)
    def forward(self, x_enc, y_dec_in):
        h, c = self.encoder(x_enc)
        y_hat = self.decoder(y_dec_in, h, c)
        return y_hat

class Seq2SeqDataset(Dataset):
    def __init__(self, X, dec_in, y):
        super().__init__()
        self.X = X
        self.dec_in = dec_in
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.dec_in[idx]),
            torch.from_numpy(self.y[idx])
        )

# ====== Data pipeline (same as original) ======
def load_and_build_xy(
    early=True, chooseIndex=False, sparce=True, diff=True,
    normalization=False, meanCentered=True,
    timeContext=False, time_steps=30, output_seq_length=29,
):
    # Input files
    _measured_raw = np.load('measured_training_data_sameRowColSeq43.npy').astype(np.float32)
    _united_raw   = np.load('united_triangular_matrices.npy').astype(np.float32)

    eps = 1e-6
    def safe_inverse_k(x):
        out = np.empty_like(x, dtype=np.float32)
        np.divide(1000.0, x, out=out, where=np.abs(x) > eps)
        out[np.abs(x) <= eps] = 0.0
        out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        out = np.clip(out, -1e6, 1e6)
        return out

    measured = safe_inverse_k(_measured_raw)
    united   = safe_inverse_k(_united_raw)
    initial_data = safe_inverse_k(_united_raw[:, 0:1, :, :])

    input_data  = np.concatenate((initial_data, measured), axis=1).astype(np.float32)
    input_data  = np.nan_to_num(input_data,  nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    output_data = np.nan_to_num(united,      nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    if early:
        input_data = input_data[:, :310, :, :]
        output_data = output_data[:, :310, :, :]
    if chooseIndex:
        index = [26, 37, 31, 19, 36, 28, 38, 18, 15]
        input_data  = np.array([input_data[x, :, :, :] for x in index])
        output_data = np.array([output_data[x, :, :, :] for x in index])
    if sparce:
        input_data  = input_data[:, ::10, :, :]
        output_data = output_data[:, ::10, :, :]
    if diff:
        input_data  = np.diff(input_data,  axis=1)
        output_data = np.diff(output_data, axis=1)
    if normalization:
        tmin_in = np.min(input_data, axis=(0, 2, 3), keepdims=True)
        tmax_in = np.max(input_data, axis=(0, 2, 3), keepdims=True)
        input_data = (input_data - tmin_in) / (tmax_in - tmin_in + 1e-12)
        tmin_out = np.min(output_data, axis=(0, 2, 3), keepdims=True)
        tmax_out = np.max(output_data, axis=(0, 2, 3), keepdims=True)
        output_data = (output_data - tmin_out) / (tmax_out - tmin_out + 1e-12)
        np.savez('normalization_factors.npz', time_step_min=tmin_in, time_step_max=tmax_in)
        np.savez('normalization_factors_output.npz', time_step_min_output=tmin_out, time_step_max_output=tmax_out)

    X_list, y_list = [], []
    T_total = input_data.shape[1]
    for series in range(input_data.shape[0]):
        for i in range(T_total - time_steps + 1):
            inp_seq = []
            out_seq = []
            for ts in range(i, i + time_steps):
                resist_flat = create_array(input_data[series, ts, :, :])
                inp_seq.append(resist_flat)
            for ts in range(i + (time_steps - output_seq_length), i + time_steps):
                resist_flat = create_array(output_data[series, ts, :, :])
                out_seq.append(resist_flat)
            X_list.append(np.stack(inp_seq, axis=0))
            y_list.append(np.stack(out_seq, axis=0))
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    # teacher forcing input
    dec_in_train = np.concatenate([np.zeros_like(y[:, :1, :], dtype=np.float32), y[:, :-1, :]], axis=1)

    # mean centering (per-time-step mean over samples)
    if meanCentered:
        Xmean = np.asarray([np.mean(X[:, t, :], axis=0) for t in range(X.shape[1])], dtype=np.float32)
        ymean = np.asarray([np.mean(y[:, t, :], axis=0) for t in range(y.shape[1])], dtype=np.float32)
        X = X - Xmean[None, :, :]
        y = y - ymean[None, :, :]
        np.savez('mean_values.npz', Xmean=Xmean, ymean=ymean)

    print("[shape] X:", X.shape, "y:", y.shape)
    return X, y, dec_in_train

# ====== One training run (for a given split) ======
def run_train_val(X, y, dec_in, train_idx, val_idx, lr, batch_size, epochs, device,
                  fold_id: int, combo_tag: str):
    ds = Seq2SeqDataset(X, dec_in, y)
    pin = (device.type == "cuda")
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, pin_memory=pin)

    in_dim  = X.shape[2]
    out_dim = y.shape[2]
    model = Seq2Seq(in_dim, out_dim, hidden=512).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    train_losses, val_losses = [], []
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        ntr = 0
        for x_enc, y_dec_in, y_gt in train_loader:
            bsz = x_enc.size(0)
            x_enc   = x_enc.to(device)
            y_dec_in= y_dec_in.to(device)
            y_gt    = y_gt.to(device)
            opt.zero_grad()
            y_hat = model(x_enc, y_dec_in)
            loss = crit(y_hat, y_gt)
            loss.backward()
            opt.step()
            tl += loss.item() * bsz
            ntr += bsz
        tl /= max(1, ntr)             # ← ここでエポック平均

        # validation
        model.eval()
        vl = 0.0
        nva = 0
        with torch.no_grad():
            for x_enc, y_dec_in, y_gt in val_loader:
                bsz = x_enc.size(0)
                x_enc   = x_enc.to(device)
                y_dec_in= y_dec_in.to(device)
                y_gt    = y_gt.to(device)
                y_hat = model(x_enc, y_dec_in)
                loss = crit(y_hat, y_gt)
                vl += loss.item() * bsz
                nva += bsz
        vl /= max(1, nva)

        train_losses.append(tl)
        val_losses.append(vl)
        if vl < best_val:
            best_val = vl


    os.makedirs("results", exist_ok=True)
    fname = f"results/{combo_tag}_fold{fold_id:02d}.csv"   # 例: lr0.001_bs8_fold01.csv
    import csv
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, vl) in enumerate(zip(train_losses, val_losses), start=1):
            w.writerow([i, tr, vl])
    print(f"[log] saved {fname}")

    return best_val

# ====== Retrain best on 80/20 split for artifact saving ======
def retrain_and_save(X, y, dec_in, lr, batch_size, epochs, device):
    from math import ceil
    N = X.shape[0]
    n_val = int(ceil(N * 0.2))
    n_tr  = N - n_val
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    train_idx, val_idx = idx[:n_tr], idx[n_tr:]

    ds = Seq2SeqDataset(X, dec_in, y)
    pin = (device.type == "cuda")
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, pin_memory=pin)

    in_dim  = X.shape[2]
    out_dim = y.shape[2]
    model = Seq2Seq(in_dim, out_dim, hidden=512).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    train_losses, val_losses = [], []
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        ntr = 0
        for x_enc, y_dec_in, y_gt in train_loader:
            bsz = x_enc.size(0)
            x_enc   = x_enc.to(device)
            y_dec_in= y_dec_in.to(device)
            y_gt    = y_gt.to(device)
            opt.zero_grad()
            y_hat = model(x_enc, y_dec_in)
            loss = crit(y_hat, y_gt)
            loss.backward()
            opt.step()
            tl += loss.item() * bsz
            ntr += bsz
        tl /= max(1, ntr)
        model.eval()
        vl = 0.0
        nva = 0
        with torch.no_grad():
            for x_enc, y_dec_in, y_gt in val_loader:
                bsz = x_enc.size(0)
                x_enc   = x_enc.to(device)
                y_dec_in= y_dec_in.to(device)
                y_gt    = y_gt.to(device)
                y_hat = model(x_enc, y_dec_in)
                loss = crit(y_hat, y_gt)
                vl += loss.item() * bsz
                nva += bsz
        vl /= max(1, nva)
        train_losses.append(tl); val_losses.append(vl)
        print(f"[retrain epoch {ep:02d}] train_loss={tl:.6f}  val_loss={vl:.6f}")

    # save artifacts
    import csv
    with open("loss_history_best_retrain.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses), start=1):
            w.writerow([i, tl, vl])
    save_loss_curve_png_pillow(train_losses, val_losses, out_path="loss_curve_best_retrain.png")
    torch.save(model.state_dict(), "best_model.pt")
    print("[done] saved best_model.pt, loss_history_best_retrain.csv, loss_curve_best_retrain.png")
    return train_idx, val_idx 


# ====== Main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", type=str, default="1e-3,5e-4,1e-4",
                    help='Comma-separated learning rates, e.g. "1e-3,5e-4"')
    ap.add_argument("--batch-sizes", type=str, default="4,8,16",
                    help='Comma-separated batch sizes, e.g. "4,8,16"')
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retrain-best", action="store_true",
                    help="After selecting best config by mean val loss, retrain once and save artifacts.")
    ap.add_argument(
        "--no-retrain-best",
        dest="retrain_best",
        action="store_false",
        help="Disable retraining the best config (default: on).",
    )
    ap.set_defaults(retrain_best=True)
    ns = ap.parse_args()

    set_seed(ns.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[info] device:", device)

    # Build data (same steps/flags as original)
    X, y, dec_in = load_and_build_xy(
        early=True, chooseIndex=False, sparce=True, diff=True,
        normalization=False, meanCentered=True,
        timeContext=False, time_steps=30, output_seq_length=29,
    )
    # Sanity (should match: X: (50, 30, 155) y: (50, 29, 155) in user's note)
    print(f"[sanity] shapes -> X: {X.shape}, y: {y.shape}")
    splits = list(kfold_indices(X.shape[0], ns.kfolds, seed=ns.seed))


    lrs = parse_list(ns.lrs, float)
    bss = parse_list(ns.batch_sizes, int)
    if not lrs or not bss:
        raise ValueError("Provide at least one lr and one batch size via --lrs and --batch-sizes")

    results = []  # rows for CSV
    best_mean = float("inf")
    best_cfg = (None, None)

    for lr in lrs:
        for bs in bss:
            fold_losses = []
            combo_tag = f"lr{lr}_bs{bs}".replace(".", "p")
            print(f"\n[grid] lr={lr}  batch_size={bs}  k={ns.kfolds}")

            for k, (tr_idx, va_idx) in enumerate(splits, start=1):
                best_val = run_train_val(
                    X, y, dec_in, tr_idx, va_idx,
                    lr=lr, batch_size=bs, epochs=ns.epochs, device=device,
                    fold_id=k, combo_tag=combo_tag
                )
                fold_losses.append(best_val)
                print(f"  fold {k}/{ns.kfolds}  best_val={best_val:.6f}")

            mean_val = float(np.mean(fold_losses))
            print(f"[grid] lr={lr} bs={bs}  mean_best_val={mean_val:.6f}")

            results.append((lr, bs, *fold_losses, mean_val))

            # ★★★ ここが肝：ベストを更新 ★★★
            if mean_val < best_mean:
                best_mean = mean_val
                best_cfg  = (lr, bs)

    if not results:
        raise RuntimeError("[fatal] No results produced — check data shapes, kfolds, or lr/bs parsing.")

    # Save CSV summary
    import csv
    header = ["lr", "batch_size"] + [f"fold{i+1}_best_val" for i in range(ns.kfolds)] + ["mean_best_val"]
    with open("grid_search_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in results:
            w.writerow(row)
    print("[done] saved grid_search_results.csv")

    # Save best config
    with open("best_config.txt", "w") as f:
        f.write(f"best_lr={best_cfg[0]}\n")
        f.write(f"best_batch_size={best_cfg[1]}\n")
        f.write(f"mean_best_val={best_mean}\n")
    print(f"[best] lr={best_cfg[0]}  batch_size={best_cfg[1]}  mean_best_val={best_mean:.6f}")
    print("[done] saved best_config.txt")

        # === ベスト構成の評価に用いた KFold 分割インデックスを保存 ===
    os.makedirs("results", exist_ok=True)
    save_dict = {
        "n_samples": X.shape[0],
        "kfolds": ns.kfolds,
        "seed": ns.seed,
        "best_lr": best_cfg[0],
        "best_batch_size": best_cfg[1],
        "mean_best_val": best_mean,
    }
    # 各foldのtrain/valを展開して npz に保存（train_idx_fold01 など）
    for k, (tr, va) in enumerate(splits, start=1):
        save_dict[f"train_idx_fold{k:02d}"] = tr.astype(np.int64)
        save_dict[f"val_idx_fold{k:02d}"] = va.astype(np.int64)
    np.savez("results/best_cv_indices.npz", **save_dict)
    print("[done] saved results/best_cv_indices.npz")


    # retrain best (optional)
    if ns.retrain_best and best_cfg[0] is not None:
            print("\n[retrain] training best config on an 80/20 split for artifact export...")
            tr80, va20 = retrain_and_save(
                X, y, dec_in,
                lr=best_cfg[0], batch_size=best_cfg[1],
                epochs=ns.epochs, device=device
            )
            # 再学習で用いた80/20のインデックスも保存
            np.savez(
                "results/best_retrain_indices.npz",
                train_idx=np.asarray(tr80, dtype=np.int64),
                val_idx=np.asarray(va20, dtype=np.int64),
                best_lr=best_cfg[0],
                best_batch_size=best_cfg[1],
                seed=ns.seed
            )
            print("[done] saved results/best_retrain_indices.npz")


if __name__ == "__main__":
    main()
