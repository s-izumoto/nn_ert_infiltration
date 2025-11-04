# 03_generateMeasDesign.py  — progress 可視化版
# -*- coding: utf-8 -*-
"""
Split version with progress bars and timing.

Usage:
    python 03_generateMeasDesign.py --config configs/generate_meas_design.yml

YAML の値が最優先です。
"""

import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError as e:
    raise SystemExit(
        "[error] PyYAML が見つかりません。`pip install pyyaml` または `conda install pyyaml` を実行してください。"
    ) from e

# --- optional: tqdm for progress bars ---
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None  # tqdm 未インストールでも動くように

def _list_input_npys(input_path: str):
    p = Path(input_path)
    if p.is_dir():
        return sorted(p.glob("*.npy"))
    if p.suffix.lower() == ".npy" and p.exists():
        return [p]
    raise FileNotFoundError(f"[error] input not found or not an .npy/.dir: {input_path}")

def _resolve_out(path_or_dir: str, stem: str, default_name: str) -> Path:
    p = Path(path_or_dir)
    if p.suffix.lower() == ".npy":
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{default_name}__{stem}.npy"

def convert_to_probability_distribution(difference_map: np.ndarray) -> np.ndarray:
    exp_diff = np.abs(difference_map)
    total = np.sum(exp_diff)
    if total == 0:
        return np.zeros_like(exp_diff)
    return exp_diff / total

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_from_cfg(cfg: dict):
    # ---- 基本設定（YAML 最優先）----
    input_file       = cfg.get("input_file", "united_triangular_matrices.npy")
    nan_fill_value   = float(cfg.get("nan_fill_value", 0.0))
    time_stride      = int(cfg.get("time_stride", 1))

    num_measurements = int(cfg.get("num_measurements", 1))
    save_probability = bool(cfg.get("save_probability", False))

    save_frames      = bool(cfg.get("save_frames", True))
    frame_output_dir = str(cfg.get("frame_output_dir", "frames_training_data"))
    frame_seq_index  = int(cfg.get("frame_seq_index", 5))
    cmap             = str(cfg.get("cmap", "hot"))

    measured_output  = str(cfg.get("measured_output", "measured_training_data.npy"))
    indices_output   = str(cfg.get("indices_output", "measurement_indices.npy"))

    # ---- 進捗表示の設定 ----
    progress_mode      = str(cfg.get("progress", "bar")).lower()     # "bar" | "print" | "none"
    progress_leave     = bool(cfg.get("progress_leave", False))       # True なら終了後もバーを残す
    progress_seq_every = int(cfg.get("progress_seq_every", 5))        # print モード時の表示間隔

    inputs = _list_input_npys(input_file)

    t0_all = time.perf_counter()
    print(f"[start] {_now()} | inputs={len(inputs)} file(s) | progress='{progress_mode}'")

    # 入力ファイルごとのループ
    file_iter = enumerate(inputs, start=1)
    if progress_mode == "bar" and tqdm is not None:
        file_iter = tqdm(file_iter, total=len(inputs), desc="Inputs", leave=progress_leave)

    for i_file, in_path in file_iter:
        t0_file = time.perf_counter()
        true_resistivity_data = np.load(in_path)
        true_resistivity_data = np.nan_to_num(true_resistivity_data, nan=nan_fill_value)

        if true_resistivity_data.ndim != 4:
            raise ValueError(f"[error] unexpected shape for input {in_path}: {true_resistivity_data.shape} (expect 4D: seq,time,y,x)")

        if time_stride < 1:
            time_stride = 1
        data = true_resistivity_data[:, 0::time_stride, :, :]
        n_seq, n_time, grid_size_y, grid_size_x = data.shape

        # 可視化フォルダ（入力ごとにサブフォルダ）
        if save_frames:
            out_frames_dir = Path(frame_output_dir) / in_path.stem
            out_frames_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_frames_dir = None

        X = []
        y_probabilities = []
        measured = []
        measurement_indices = []

        # シーケンスのループ
        seq_iter = range(n_seq)
        if progress_mode == "bar" and tqdm is not None:
            seq_iter = tqdm(seq_iter, total=n_seq, desc=f"{in_path.stem} | seq", leave=progress_leave, position=1)

        for s in seq_iter:
            measured_resistivity_map = data[s, 0, :, :].copy()
            Xseq, y_probabilities_seq, measured_seq, indices_seq = [], [], [], []

            # 時刻ループ
            time_iter = range(1, n_time)
            if progress_mode == "bar" and tqdm is not None:
                time_iter = tqdm(time_iter, total=n_time-1, desc=f"t", leave=False, position=2)

            for t in time_iter:
                true_t = data[s, t, :, :].copy()
                difference_map = np.abs(true_t - measured_resistivity_map)
                flat_idx = np.argsort(difference_map, axis=None)[-num_measurements:]
                rows, cols = np.unravel_index(flat_idx, true_t.shape)

                for r, c in zip(rows, cols):
                    measured_resistivity_map[r, c] = true_t[r, c]
                    indices_seq.append((int(c), int(r)))  # (col, row)

                feedback_input = y_probabilities_seq[-1] if (t > 1 and y_probabilities_seq) else np.zeros((grid_size_y, grid_size_x))
                combined_input = np.stack([measured_resistivity_map, feedback_input], axis=-1)
                Xseq.append(combined_input)
                measured_seq.append(measured_resistivity_map.copy())

                probability_map = convert_to_probability_distribution(difference_map)
                y_probabilities_seq.append(probability_map)

                if save_frames and s == frame_seq_index:
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    im1 = plt.imshow(measured_resistivity_map, cmap=cmap, interpolation='nearest')
                    plt.title(f'{in_path.stem} | Seq {s} | Measured @ t={t+1}')
                    plt.colorbar(im1, orientation='horizontal', pad=0.2)

                    plt.subplot(1, 2, 2)
                    im3 = plt.imshow(true_t, cmap=cmap, interpolation='nearest')
                    plt.title(f'{in_path.stem} | Seq {s} | True @ t={t+1}')
                    plt.colorbar(im3, orientation='horizontal', pad=0.2)
                    plt.tight_layout()
                    plt.savefig(Path(out_frames_dir) / f'seq{s:03d}_t{t:04d}.png')
                    plt.close()

                if progress_mode == "print" and (t % 50 == 0 or t == n_time - 1):
                    print(f"[{in_path.stem}] seq {s+1}/{n_seq} t {t}/{n_time-1}")

            X.append(Xseq)
            y_probabilities.append(y_probabilities_seq)
            measured.append(measured_seq)
            measurement_indices.append(indices_seq)

            if progress_mode == "print" and ( (s+1) % max(1, progress_seq_every) == 0 or (s+1)==n_seq ):
                print(f"[{in_path.stem}] seq {s+1}/{n_seq} done")

        measured_arr = np.array(measured, dtype=np.float32)
        indices_arr  = np.array(measurement_indices, dtype=np.int32)

        measured_out_path = _resolve_out(measured_output, in_path.stem, "measured_training_data")
        indices_out_path  = _resolve_out(indices_output,  in_path.stem, "measurement_indices")

        np.save(measured_out_path, measured_arr)
        np.save(indices_out_path, indices_arr)

        if save_probability:
            prob_out_path = _resolve_out("y_probabilities.npy", in_path.stem, "y_probabilities")
            np.save(prob_out_path, np.array(y_probabilities, dtype=np.float32))

        dt_file = timedelta(seconds=round(time.perf_counter() - t0_file, 2))
        print(f"[done] {_now()} {in_path.name}  elapsed={dt_file}")
        print(f"  measured -> {measured_out_path}  shape={measured_arr.shape}")
        print(f"  indices  -> {indices_out_path}   shape={indices_arr.shape}")
        if save_frames:
            print(f"  frames   -> {out_frames_dir}")

    dt_all = timedelta(seconds=round(time.perf_counter() - t0_all, 2))
    print(f"[finish] {_now()} | total elapsed={dt_all}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/generate_meas_design.yml",
                    help="設定YAMLのパス（YAMLが最優先）")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[error] config YAML not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    run_from_cfg(cfg)

if __name__ == "__main__":
    main()
