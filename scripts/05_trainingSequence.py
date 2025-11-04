# -*- coding: utf-8 -*-
"""
YAML設定対応版 + 経過時間出力付き
(元: 05_trainingAppResTemporalDerivative.py)
"""

import os
import math
import random
import numpy as np
from pathlib import Path
import csv
import time
from datetime import datetime

# Pillowは任意
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

def load_array_or_folder(path_or_dir: str, npz_key: str | None = None) -> np.ndarray:
    """
    - .npy / .npz のファイルパス もしくは ディレクトリ（中の *.npy / *.npz を全部）を受け取り、
      先頭次元で連結して返す。
    - 各配列は (T, H, W) または (N, T, H, W) を想定（(T, H, W) は自動で N=1 を付与）。
    - npz は npz_key があればそれを、なければ先頭のキーを使う。
    """
    p = Path(path_or_dir)
    files: list[Path] = []
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
        # (T,H,W) → (1,T,H,W) に揃える
        if arr.ndim == 3:
            arr = arr[None, ...]
        elif arr.ndim != 4:
            raise ValueError(f"Unsupported array shape {arr.shape} in {f} (expect 3D or 4D)")

        if ref_shape is None:
            ref_shape = arr.shape[1:]  # (T,H,W)
        elif arr.shape[1:] != ref_shape:
            raise ValueError(f"Shape mismatch: {arr.shape} vs ref {ref_shape} in {f}")

        batches.append(arr)

    out = np.concatenate(batches, axis=0)  # N を縦に積む
    return out

# ====== 再現性 ======
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ====== ユーティリティ ======
def create_array(data_2d: np.ndarray) -> np.ndarray:
    row_sizes = np.arange(29, 0, -3)
    filled_data = []
    for i, size in enumerate(row_sizes):
        filled_data.extend(data_2d[i, :size])
    return np.array(filled_data, dtype=np.float32)

def de_create_array(flat_data: np.ndarray) -> np.ndarray:
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

# ====== Dataset ======
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
            torch.from_numpy(self.X[idx]),       # (30, Fin)
            torch.from_numpy(self.dec_in[idx]),  # (29, Fout)
            torch.from_numpy(self.y[idx])        # (29, Fout)
        )

# ====== モデル ======
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

def save_loss_curve_png_pillow(train_losses, val_losses, out_path="loss_curve_pytorch.png",
                               size=(1200, 480), margin=60):
    if not _HAS_PIL:
        print("[warn] Pillow が見つかりませんでした。PNG は作らず CSV のみ保存します。")
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

def run_one_fold(X, y, dec_in, train_idx, val_idx, lr, batch_size, epochs, device,
                 fold_tag, combo_tag, results_dir):
    ds = Seq2SeqDataset(X, dec_in, y)
    pin = (device.type == "cuda")
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, pin_memory=pin)

    in_dim, out_dim = X.shape[2], y.shape[2]
    model = Seq2Seq(in_dim, out_dim, hidden=512).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.MSELoss()

    best_val = float("inf")
    train_losses, val_losses = [], []

    for ep in range(1, epochs + 1):
        model.train()
        tl, ntr = 0.0, 0
        for x_enc, y_dec_in, y_gt in train_loader:
            bsz = x_enc.size(0)
            x_enc = x_enc.to(device); y_dec_in = y_dec_in.to(device); y_gt = y_gt.to(device)
            opt.zero_grad()
            y_hat = model(x_enc, y_dec_in)
            loss = crit(y_hat, y_gt)
            loss.backward()
            opt.step()
            tl += loss.item() * bsz; ntr += bsz
        tl /= max(1, ntr)

        model.eval()
        vl, nva = 0.0, 0
        with torch.no_grad():
            for x_enc, y_dec_in, y_gt in val_loader:
                bsz = x_enc.size(0)
                x_enc = x_enc.to(device); y_dec_in = y_dec_in.to(device); y_gt = y_gt.to(device)
                y_hat = model(x_enc, y_dec_in)
                loss = crit(y_hat, y_gt)
                vl += loss.item() * bsz; nva += bsz
        vl /= max(1, nva)

        train_losses.append(tl); val_losses.append(vl)
        if vl < best_val:
            best_val = vl

    os.makedirs(results_dir, exist_ok=True)
    with open(Path(results_dir, f"{combo_tag}_{fold_tag}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
            w.writerow([i, tr, va])

    print(f"[log] saved {Path(results_dir, f'{combo_tag}_{fold_tag}.csv')}")
    return best_val

def retrain_and_save_best(X, y, dec_in, lr, batch_size, epochs, device,
                          results_dir, save_png=True):
    N = X.shape[0]
    n_val = int(np.ceil(N * 0.2))
    idx = np.arange(N)
    rng = np.random.default_rng(42); rng.shuffle(idx)
    train_idx, val_idx = idx[:-n_val], idx[-n_val:]

    ds = Seq2SeqDataset(X, dec_in, y)
    pin = (device.type == "cuda")
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, pin_memory=pin)

    in_dim, out_dim = X.shape[2], y.shape[2]
    model = Seq2Seq(in_dim, out_dim, hidden=512).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.MSELoss()

    train_losses, val_losses = [], []
    for ep in range(1, epochs + 1):
        model.train()
        tl, ntr = 0.0, 0
        for x_enc, y_dec_in, y_gt in train_loader:
            bsz = x_enc.size(0)
            x_enc = x_enc.to(device); y_dec_in = y_dec_in.to(device); y_gt = y_gt.to(device)
            opt.zero_grad()
            y_hat = model(x_enc, y_dec_in)
            loss = crit(y_hat, y_gt)
            loss.backward()
            opt.step()
            tl += loss.item() * bsz; ntr += bsz
        tl /= max(1, ntr)

        model.eval()
        vl, nva = 0.0, 0
        with torch.no_grad():
            for x_enc, y_dec_in, y_gt in val_loader:
                bsz = x_enc.size(0)
                x_enc = x_enc.to(device); y_dec_in = y_dec_in.to(device); y_gt = y_gt.to(device)
                y_hat = model(x_enc, y_dec_in)
                loss = crit(y_hat, y_gt)
                vl += loss.item() * bsz; nva += bsz
        vl /= max(1, nva)

        train_losses.append(tl); val_losses.append(vl)
        print(f"[retrain {ep:02d}] train={tl:.6f} val={vl:.6f}")

    torch.save(model.state_dict(), Path(results_dir, "best_model.pt"))
    with open(Path(results_dir, "loss_history_best_retrain.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","train_loss","val_loss"])
        for i,(tr,va) in enumerate(zip(train_losses,val_losses), start=1):
            w.writerow([i,tr,va])

    if save_png:
        save_loss_curve_png_pillow(
            train_losses, val_losses,
            out_path=str(Path(results_dir, "loss_curve_best_retrain.png"))
        )
    return train_idx, val_idx

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="YAML設定ファイルパス")
    ns = p.parse_args()

    # === 設定読込 ===
    with open(ns.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    measured_path = paths.get("measured", "measured_training_data_sameRowColSeq34.npy")
    united_path   = paths.get("united",   "united_triangular_matrices.npy")
    results_dir   = paths.get("results_dir", "results")

    preprocess = cfg.get("preprocess", {})
    early         = bool(preprocess.get("early", True))
    chooseIndex   = bool(preprocess.get("chooseIndex", False))
    sparce        = bool(preprocess.get("sparce", True))
    diff_flag     = bool(preprocess.get("diff", True))
    timeContext   = bool(preprocess.get("timeContext", False))
    normalization = bool(preprocess.get("normalization", False))
    meanCentered  = bool(preprocess.get("meanCentered", True))

    choose_indices = cfg.get("choose_indices", [26, 37, 31, 19, 36, 28, 38, 18, 15])

    sequence = cfg.get("sequence", {})
    time_steps = int(sequence.get("time_steps", 30))
    output_seq_length = int(sequence.get("output_seq_length", 29))

    model_cfg = cfg.get("model", {})
    enc_hidden = int(model_cfg.get("enc_hidden", 512))
    dec_hidden = int(model_cfg.get("dec_hidden", 512))
    # （上のhiddenは現在モデル内部で512固定、必要なら渡して使うように拡張可）

    train_cfg = cfg.get("train", {})
    lrs         = list(train_cfg.get("lrs", [1e-3, 5e-4, 1e-4]))
    batch_sizes = list(train_cfg.get("batch_sizes", [4, 8, 16]))
    kfolds      = int(train_cfg.get("kfolds", 5))
    epochs      = int(train_cfg.get("epochs", 40))
    seed        = int(train_cfg.get("seed", 42))
    retrain_best = bool(train_cfg.get("retrain_best", True))

    artifacts = cfg.get("artifacts", {})
    save_png = bool(artifacts.get("save_loss_curve_png", True))

    # ====== ランタイム情報 ======
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[info]", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("[info] device:", device)
    os.makedirs(results_dir, exist_ok=True)

    t0_total = time.perf_counter()

    # ====== データ読み込み & 前処理 ======
    _measured_raw = load_array_or_folder(measured_path)  # (N, T, H, W) に正規化
    _united_raw   = load_array_or_folder(united_path)    # (N, T, H, W)

    # N の整合チェック（同じ並びでマッチしているか）
    if _measured_raw.shape[0] != _united_raw.shape[0]:
        raise ValueError(
            f"N mismatch: measured N={_measured_raw.shape[0]} vs united N={_united_raw.shape[0]}. "
            "フォルダー入力のときは、同数のファイルを同じソート順で並べてください。"
    )
    
    measured = (1000.0 / _measured_raw).astype(np.float32)
    measured = np.nan_to_num(measured, nan=0.0)

    united = (1000.0 / _united_raw).astype(np.float32)
    united = np.nan_to_num(united, nan=0.0)

    initial_data = (1000.0 / _united_raw[:, 0:1, :, :]).astype(np.float32)
    initial_data = np.nan_to_num(initial_data, nan=0.0)

    input_data  = np.concatenate((initial_data, measured), axis=1).astype(np.float32)
    input_data  = np.nan_to_num(input_data,  nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    output_data = np.nan_to_num(united,      nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    if early:
        input_data = input_data[:, :310, :, :]
        output_data = output_data[:, :310, :, :]

    if chooseIndex:
        input_data  = np.array([input_data[x, :, :, :] for x in choose_indices])
        output_data = np.array([output_data[x, :, :, :] for x in choose_indices])

    if sparce:
        input_data  = input_data[:, ::10, :, :]
        output_data = output_data[:, ::10, :, :]

    if diff_flag:
        input_data  = np.diff(input_data,  axis=1)
        output_data = np.diff(output_data, axis=1)

    if normalization:
        tmin_in = np.min(input_data, axis=(0, 2, 3), keepdims=True)
        tmax_in = np.max(input_data, axis=(0, 2, 3), keepdims=True)
        input_data = (input_data - tmin_in) / (tmax_in - tmin_in + 1e-12)

        tmin_out = np.min(output_data, axis=(0, 2, 3), keepdims=True)
        tmax_out = np.max(output_data, axis=(0, 2, 3), keepdims=True)
        output_data = (output_data - tmin_out) / (tmax_out - tmin_out + 1e-12)

        np.savez(Path(results_dir, 'normalization_factors_in.npz'),
                 time_step_min=tmin_in, time_step_max=tmax_in)
        np.savez(Path(results_dir, 'normalization_factors_out.npz'),
                 time_step_min_output=tmin_out, time_step_max_output=tmax_out)

    # X, y を構成
    X_list, y_list = [], []
    T_total = input_data.shape[1]

    if timeContext:
        for series in range(input_data.shape[0]):
            for i in range(T_total - time_steps + 1):
                inp_seq = []
                out_seq = []
                for ts in range(i, i + time_steps):
                    resist_flat = create_array(input_data[series, ts, :, :])
                    time_ctx = np.full_like(resist_flat, fill_value=ts / (T_total - 1), dtype=np.float32)
                    inp_seq.append(np.concatenate([resist_flat, time_ctx], axis=0))
                for ts in range(i + (time_steps - output_seq_length), i + time_steps):
                    resist_flat = create_array(output_data[series, ts, :, :])
                    out_seq.append(resist_flat)
                X_list.append(np.stack(inp_seq, axis=0))
                y_list.append(np.stack(out_seq, axis=0))
    else:
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

    X = np.asarray(X_list, dtype=np.float32)   # (N, 30, Fin)
    y = np.asarray(y_list, dtype=np.float32)   # (N, 29, Fout)

    if meanCentered:
        Xmean = []
        ymean = []
        for t in range(X.shape[1]):
            Xmean.append(np.mean(X[:, t, :], axis=0))
        for t in range(y.shape[1]):
            ymean.append(np.mean(y[:, t, :], axis=0))
        Xmean = np.asarray(Xmean, dtype=np.float32)  # (30, Fin)
        ymean = np.asarray(ymean, dtype=np.float32)  # (29, Fout)
        X = X - Xmean[None, :, :]
        y = y - ymean[None, :, :]
        np.savez(Path(results_dir, 'mean_values.npz'), Xmean=Xmean, ymean=ymean)

    print("[shape] X:", X.shape, "y:", y.shape)

    dec_in_train = np.concatenate([np.zeros_like(y[:, :1, :], dtype=np.float32), y[:, :-1, :]], axis=1)

    # ====== グリッドサーチ ======
    splits = list(_kfold_indices(X.shape[0], kfolds, seed=seed))
    results = []
    best_mean = float("inf")
    best_cfg = (None, None)

    for lr in lrs:
        for bs in batch_sizes:
            combo_tag = f"lr{lr}_bs{bs}".replace(".", "p")
            print(f"\n[grid] lr={lr}  batch_size={bs}  k={kfolds}")
            t0_combo = time.perf_counter()
            fold_vals = []
            for k, (tr, va) in enumerate(splits, start=1):
                v = run_one_fold(X, y, dec_in_train, tr, va,
                                 lr=lr, batch_size=bs, epochs=epochs,
                                 device=device, fold_tag=f"fold{k:02d}",
                                 combo_tag=combo_tag, results_dir=results_dir)
                fold_vals.append(v)
                print(f"  fold {k}/{kfolds} best_val={v:.6f}")
            mean_v = float(np.mean(fold_vals))
            dt_combo = time.perf_counter() - t0_combo
            print(f"[grid] lr={lr} bs={bs} mean_best_val={mean_v:.6f}  (elapsed {dt_combo:.1f}s)")
            results.append((lr, bs, *fold_vals, mean_v))
            if mean_v < best_mean:
                best_mean = mean_v
                best_cfg = (lr, bs)

    # 集計保存
    header = ["lr", "batch_size"] + [f"fold{i+1}_best_val" for i in range(kfolds)] + ["mean_best_val"]
    with open(Path(results_dir, "grid_search_results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for row in results:
            w.writerow(row)
    with open(Path(results_dir, "best_config.txt"), "w") as f:
        f.write(f"best_lr={best_cfg[0]}\n")
        f.write(f"best_batch_size={best_cfg[1]}\n")
        f.write(f"mean_best_val={best_mean}\n")
    print(f"[best] lr={best_cfg[0]}  batch_size={best_cfg[1]}  mean_best_val={best_mean:.6f}")

    # KFold index 保存（再現性）
    save_dict = {"n_samples": X.shape[0], "kfolds": kfolds, "seed": seed,
                 "best_lr": best_cfg[0], "best_batch_size": best_cfg[1], "mean_best_val": best_mean}
    for k, (tr, va) in enumerate(splits, start=1):
        save_dict[f"train_idx_fold{k:02d}"] = np.asarray(tr, dtype=np.int64)
        save_dict[f"val_idx_fold{k:02d}"]   = np.asarray(va, dtype=np.int64)
    np.savez(Path(results_dir, "best_cv_indices.npz"), **save_dict)

    # ベスト設定で80/20再学習（任意・YAMLで制御）
    if retrain_best and best_cfg[0] is not None:
        print("\n[retrain] best config on 80/20 split...")
        tr80, va20 = retrain_and_save_best(
            X, y, dec_in_train,
            lr=best_cfg[0], batch_size=best_cfg[1],
            epochs=epochs, device=device,
            results_dir=results_dir, save_png=save_png
        )
        np.savez(Path(results_dir, "best_retrain_indices.npz"),
                 train_idx=np.asarray(tr80, dtype=np.int64),
                 val_idx=np.asarray(va20, dtype=np.int64),
                 best_lr=best_cfg[0], best_batch_size=best_cfg[1], seed=seed)

    # === 合計経過時間 ===
    dt_total = time.perf_counter() - t0_total
    print(f"\n[time] total elapsed: {dt_total:.1f}s ({dt_total/60.0:.1f} min)")

if __name__ == "__main__":
    main()
