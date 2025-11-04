# -*- coding: utf-8 -*-
"""
YAML分割＋フォルダー入出力対応版。
元ファイル: 07_trainingAppResTemporalDerivativeFirstImage_kfold.py
  - データ前処理とモデル設計は同じ意味を保持
  - 入出力はすべて YAML から取得
  - 生成物も YAML で指定した base_dir 配下に整理
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
    raise RuntimeError("PyYAML が必要です: pip install pyyaml") from e

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _parse_list_cli(s: str, cast):
    s = (s or "").strip()
    if not s:
        return []
    return [cast(p.strip()) for p in s.replace(";", ",").split(",") if p.strip()]

ROW_SIZES = np.arange(29, 0, -3)

def create_array(data: np.ndarray) -> np.ndarray:
    """Flatten triangular rows in sizes 29, 26, 23, ..., 2, 1 (step -3)."""
    filled = []
    for i, size in enumerate(ROW_SIZES):
        filled.extend(data[i, :size])
    return np.array(filled, dtype=np.float32)

# -----------------------------
# Dataset / Model
# -----------------------------
class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMRegressor(nn.Module):
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
        out, _ = self.lstm(x)          # (B,T,H)
        last = out[:, -1, :]           # (B,H)
        z = self.relu(self.fc1(last))
        z = self.relu(self.fc2(z))
        y = self.out(z)                # (B,F)
        return y

# -----------------------------
# Core
# -----------------------------

def load_and_preprocess(cfg: dict):
    p_in = cfg["inputs"]
    p_pp = cfg["preprocess"]
    p_out = cfg["outputs"]

    measured_path = Path(p_in["measured_dir"]) / p_in["measured_file"]
    united_path   = Path(p_in["united_dir"]) / p_in["united_file"]

    # 入力: measured（1/ρ*1000）＋ t0 真値 を先頭に付加
    input_data = 1.0 / np.load(str(measured_path)) * 1000.0
    initial_data = 1.0 / np.load(str(united_path))[:, 0, :, :] * 1000.0
    initial_data = np.expand_dims(initial_data, axis=1)
    input_data = np.concatenate((initial_data, input_data), axis=1)
    input_data = np.nan_to_num(input_data, nan=0.0)

    # 出力: 真値（1/ρ*1000）
    output_data = 1.0 / np.load(str(united_path)) * 1000.0
    output_data = np.nan_to_num(output_data, nan=0.0)

    # early / choose_index / sparse / diff
    if p_pp.get("early", False):
        L = int(p_pp.get("early_limit", 50))
        input_data = input_data[:, :L, :, :]
        output_data = output_data[:, :L, :, :]

    choose = p_pp.get("choose_index", []) or []
    if len(choose) > 0:
        input_data = np.array([input_data[i] for i in choose])
        output_data = np.array([output_data[i] for i in choose])

    if p_pp.get("sparse", False):
        s = int(p_pp.get("sparse_stride", 10))
        input_data = input_data[:, ::s, :, :]
        output_data = output_data[:, ::s, :, :]

    if p_pp.get("diff", False):
        input_data = np.diff(input_data, axis=1)
        output_data = np.diff(output_data, axis=1)

    # 正規化（任意）
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

    # シーケンス作成（X: 入力、y: 出力）
    T = int(p_pp.get("time_steps", 4))
    use_time_ctx = bool(p_pp.get("time_context", False))

    X_list, y_list = [], []
    if use_time_ctx:
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

    # 時刻ごと平均センタリング
    Xmean = None; ymean = None
    if p_pp.get("mean_centered", True):
        Xmean = np.array([np.mean(X[:, i, :], axis=0) for i in range(X.shape[1])], dtype=np.float32)
        ymean = np.array([np.mean(y[:, i, :], axis=0) for i in range(y.shape[1])], dtype=np.float32)
        X = X - Xmean
        y = y - ymean
        out_dir = Path(p_out["base_dir"]) ; out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / p_out["mean_values_npz"], Xmean=Xmean, ymean=ymean)

    # Keras実装と同じく、yは先頭タイムステップのみ学習
    y = y[:, 0, :]

    return X, y, Xmean, ymean


def save_loss_curve_png(train_losses, val_losses, out_path: Path):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend(); plt.title("Loss"); plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def kfold_indices(n_samples: int, n_splits: int, seed: int = 42):
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


def run_one_fold(X, y, train_idx, val_idx, lr, batch_size, epochs, device, combo_tag: str, fold_tag: str,
                  out_dir: Path, num_workers: int, model_cfg: dict):
    ds = XYDataset(X, y)
    pin = (device.type == "cuda")
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, pin_memory=pin,
                              num_workers=num_workers)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, pin_memory=pin,
                              num_workers=num_workers)

    input_dim = X.shape[2]
    output_dim = y.shape[1]
    model = LSTMRegressor(input_dim=input_dim, output_dim=output_dim,
                          hidden_size=int(model_cfg.get("hidden_size",512)),
                          num_layers=int(model_cfg.get("num_layers",2))).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0; ntr = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            run += loss.item() * xb.size(0); ntr += xb.size(0)
        train_loss = run / max(1, ntr)

        model.eval(); runv = 0.0; nva = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                loss = criterion(model(xb), yb)
                runv += loss.item() * xb.size(0); nva += xb.size(0)
        val_loss = runv / max(1, nva)

        train_losses.append(train_loss); val_losses.append(val_loss)
        if val_loss < best_val: best_val = val_loss

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{combo_tag}_{fold_tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","train_loss","val_loss"])
        for i,(tr,va) in enumerate(zip(train_losses,val_losses), start=1):
            w.writerow([i,tr,va])
    print(f"[log] saved {csv_path}")
    return best_val


def retrain_and_save_best(X, y, lr, batch_size, epochs, device, seed: int, out_paths: dict,
                           num_workers: int, model_cfg: dict):
    N = X.shape[0]
    n_val = int(np.ceil(N * 0.2))
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    train_idx, val_idx = idx[:-n_val], idx[-n_val:]

    ds = XYDataset(X, y)
    pin = (device.type == "cuda")
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, pin_memory=pin,
                              num_workers=num_workers)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, pin_memory=pin,
                              num_workers=num_workers)

    input_dim = X.shape[2]
    output_dim = y.shape[1]
    model = LSTMRegressor(input_dim=input_dim, output_dim=output_dim,
                          hidden_size=int(model_cfg.get("hidden_size",512)),
                          num_layers=int(model_cfg.get("num_layers",2))).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0; ntr = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            run += loss.item() * xb.size(0); ntr += xb.size(0)
        train_loss = run / max(1, ntr)

        model.eval(); runv = 0.0; nva = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                loss = criterion(model(xb), yb)
                runv += loss.item() * xb.size(0); nva += xb.size(0)
        val_loss = runv / max(1, nva)

        train_losses.append(train_loss); val_losses.append(val_loss)
        print(f"[retrain {ep:03d}] train={train_loss:.6f}  val={val_loss:.6f}")

    # 保存
    torch.save(model.state_dict(), out_paths["best_model_pt"])
    payload = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'time_steps': X.shape[1],
    }
    torch.save(payload, out_paths["single_payload_pt"])

    with open(out_paths["loss_history_csv"], "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","train_loss","val_loss"])
        for i,(tr,va) in enumerate(zip(train_losses,val_losses), start=1):
            w.writerow([i,tr,va])
    save_loss_curve_png(train_losses, val_losses, Path(out_paths["loss_curve_png"]))

    print(f"Saved model to {out_paths['best_model_pt']} and payload to {out_paths['single_payload_pt']}")
    return train_idx, val_idx, train_losses, val_losses


# -----------------------------
# Main
# -----------------------------

def main():
    start_wall = datetime.datetime.now()
    start = time.perf_counter()
    print(f"[time] start: {start_wall:%Y-%m-%d %H:%M:%S}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML設定ファイル")
    # 追加の上書きCLI（省略可）
    ap.add_argument("--lrs", type=str, default=None)
    ap.add_argument("--batch-sizes", type=str, default=None)
    ap.add_argument("--kfolds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--retrain-best", dest="retrain_best", action="store_true")
    ap.add_argument("--no-retrain-best", dest="retrain_best", action="store_false")
    ap.set_defaults(retrain_best=None)  # None のとき YAML を採用
    ns = ap.parse_args()

    # YAML読み込み
    with open(ns.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # CLI で上書き（指定された場合のみ）
    if ns.lrs is not None:
        cfg["training"]["lrs"] = _parse_list_cli(ns.lrs, float)
    if ns.batch_sizes is not None:
        cfg["training"]["batch_sizes"] = _parse_list_cli(ns.batch_sizes, int)
    if ns.kfolds is not None:
        cfg["training"]["kfolds"] = int(ns.kfolds)
    if ns.epochs is not None:
        cfg["training"]["epochs"] = int(ns.epochs)
    if ns.seed is not None:
        cfg["training"]["seed"] = int(ns.seed)
    if ns.retrain_best is not None:
        cfg["training"]["retrain_best"] = bool(ns.retrain_best)

    trn = cfg["training"]; outs = cfg["outputs"]

    set_seed(int(trn.get("seed", 42)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[info] device:", device)

    # データ作成
    X, y, Xmean, ymean = load_and_preprocess(cfg)
    print(f"[shape] X: {X.shape}  y: {y.shape}")

    # 出力ベース
    base_dir = Path(outs.get("base_dir", "results_first"))
    base_dir.mkdir(parents=True, exist_ok=True)

    lrs = list(trn.get("lrs", [1e-3, 5e-4, 1e-4]))
    bss = list(trn.get("batch_sizes", [4, 8, 16]))
    kfolds = int(trn.get("kfolds", 5))
    epochs = int(trn.get("epochs", 100))
    seed = int(trn.get("seed", 42))
    num_workers = int(trn.get("num_workers", 0))

    splits = list(kfold_indices(X.shape[0], kfolds, seed=seed))

    results = []
    best_mean = float("inf")
    best_cfg = (None, None)

    for lr in lrs:
        for bs in bss:
            combo_tag = f"lr{lr}_bs{bs}".replace(".", "p")
            print(f"\n[grid] lr={lr}  batch_size={bs}  k={kfolds}")
            fold_vals = []
            for k, (tr, va) in enumerate(splits, start=1):
                v = run_one_fold(
                    X, y, tr, va, lr=float(lr), batch_size=int(bs), epochs=epochs, device=device,
                    combo_tag=combo_tag, fold_tag=f"fold{k:02d}", out_dir=base_dir,
                    num_workers=num_workers, model_cfg=cfg.get("model", {})
                )
                fold_vals.append(v)
                print(f"  fold {k}/{kfolds} best_val={v:.6f}")
            mean_v = float(np.mean(fold_vals))
            print(f"[grid] lr={lr} bs={bs} mean_best_val={mean_v:.6f}")
            results.append((lr, bs, *fold_vals, mean_v))
            if mean_v < best_mean:
                best_mean = mean_v; best_cfg = (lr, bs)

    # まとめCSV/テキスト/分割情報
    grid_csv = base_dir / outs.get("grid_search_csv", "grid_search_results.csv")
    with grid_csv.open("w", newline="") as f:
        header = ["lr","batch_size"]+[f"fold{i+1}_best_val" for i in range(kfolds)]+["mean_best_val"]
        w = csv.writer(f); w.writerow(header)
        for row in results: w.writerow(row)

    best_txt = base_dir / outs.get("best_config_txt", "best_config.txt")
    with best_txt.open("w") as f:
        f.write(f"best_lr={best_cfg[0]}\n")
        f.write(f"best_batch_size={best_cfg[1]}\n")
        f.write(f"mean_best_val={best_mean}\n")
    print(f"[best] lr={best_cfg[0]}  batch_size={best_cfg[1]}  mean_best_val={best_mean:.6f}")

    cv_npz = base_dir / outs.get("cv_indices_npz", "best_cv_indices.npz")
    save_dict = {"n_samples": X.shape[0], "kfolds": kfolds, "seed": seed,
                 "best_lr": best_cfg[0], "best_batch_size": best_cfg[1], "mean_best_val": best_mean}
    for k, (tr, va) in enumerate(splits, start=1):
        save_dict[f"train_idx_fold{k:02d}"] = np.asarray(tr, dtype=np.int64)
        save_dict[f"val_idx_fold{k:02d}"]   = np.asarray(va, dtype=np.int64)
    np.savez(cv_npz, **save_dict)

    # ベスト再学習（任意）
    if bool(trn.get("retrain_best", True)) and best_cfg[0] is not None:
        print("\n[retrain] best config on 80/20 split...")
        # 出力パスを作成
        out_paths = {
            "best_model_pt":      str(base_dir / outs.get("best_model_pt", "best_model_first.pt")),
            "single_payload_pt":  str(base_dir / outs.get("single_payload_pt", "single_output_lstm_model.pt")),
            "loss_history_csv":   str(base_dir / outs.get("loss_history_csv", "loss_history_best_retrain.csv")),
            "loss_curve_png":     str(base_dir / outs.get("loss_curve_png", "loss_curve_best_retrain_first.png")),
        }
        tr80, va20, tr_hist, va_hist = retrain_and_save_best(
            X, y, lr=float(best_cfg[0]), batch_size=int(best_cfg[1]), epochs=epochs, device=device,
            seed=seed, out_paths=out_paths, num_workers=num_workers, model_cfg=cfg.get("model", {})
        )
        np.savez(base_dir / outs.get("retrain_indices_npz", "best_retrain_indices.npz"),
                 train_idx=np.asarray(tr80, dtype=np.int64),
                 val_idx=np.asarray(va20, dtype=np.int64),
                 best_lr=best_cfg[0], best_batch_size=best_cfg[1], seed=seed)
    
    end_wall = datetime.datetime.now()
    elapsed = time.perf_counter() - start
    print(f"[time] end:   {end_wall:%Y-%m-%d %H:%M:%S}")
    print(f"[time] elapsed: {elapsed:.1f}s ({timedelta(seconds=int(elapsed))})")


if __name__ == "__main__":
    main()