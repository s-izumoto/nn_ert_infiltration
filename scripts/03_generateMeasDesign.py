# 03_generateMeasDesign.py — with progress visualization
# -*- coding: utf-8 -*-
"""
Generate synthetic "measured" datasets from true resistivity maps by
sequentially updating a measured map with the locations that exhibit the
largest mismatch from the ground truth at each time step.

SUMMARY
-------
This script reads one or more 4-D NumPy arrays of true resistivity
(e.g., shape = (N_seq, N_time, H, W)). For each sequence, it initializes
a "measured" map at t=0 from the true map, then for t=1..T-1 it:
  1) Computes the absolute difference between the true map and the current
     measured map.
  2) Selects the top-K pixels with the largest difference (K = num_measurements).
  3) Copies the true values at those pixels into the measured map (simulating
     targeted measurements).
  4) (Optional) Produces a probability map by normalizing the difference map.
  5) (Optional) Saves side-by-side frames (measured vs. true) for visualization.

The process yields a time-evolving "measured" dataset that reflects a
greedy measurement policy focused on the most discrepant locations. This
is useful for downstream training/evaluation of models that learn where
to measure next, or that forecast full fields from sparse updates.

INPUTS
------
- input_file : str
    Path to a single .npy file OR a directory containing multiple .npy files.
    Each .npy must be a 4-D array with axes (sequence, time, y, x).
- YAML configuration (recommended)
    All runtime parameters are read from a YAML file and have priority over
    command-line defaults (see CONFIG KEYS below).

OUTPUTS
-------
- measured_* .npy
    Array of shape (N_seq, N_time-1, H, W) storing the evolving measured maps
    after each update step (starting from t=1).
- measurement_indices_* .npy
    Array of shape (N_seq, (N_time-1) * K, 2) with (col, row) pairs of the
    selected pixels at each time step (K = num_measurements).
- y_probabilities_* .npy  (optional, if save_probability: true)
    Probability maps (same H×W per time step) obtained by L1-normalizing the
    absolute differences.
- frames/ (optional)
    PNGs for a single sequence (frame_seq_index) showing Measured vs. True at
    each time step, helpful for quick visual QA.

SHAPES & CONVENTIONS
--------------------
- Input truth: (N_seq, N_time, H, W)
- Measured output: (N_seq, N_time-1, H, W)
- Indices output: (N_seq, (N_time-1)*K, 2)  where each pair is (col, row)
- Time subsampling: If time_stride > 1, the script uses t = 0, 0+stride, ...
- NaNs in input are replaced by `nan_fill_value` before processing.

ALGORITHM (PER SEQUENCE)
------------------------
Initialize: measured_map = true[:, t=0]
For t = 1..T-1:
  diff = |true[:, t] - measured_map|
  pick top-K pixels by diff
  measured_map[picked] = true[:, t][picked]
  probability_map = diff / diff.sum()          # if save_probability
  save visualization frame (optional)

# ===========================
# YAML Configuration Guide — 03_generateMeasDesign.py
# ===========================
# Each key defines the input type and its purpose for generating
# "measured" time series by greedily updating pixels with the largest
# true–measured mismatch at each time step.

# === Input ===
# input_file (str): Path to a single 4-D .npy OR a directory of .npy files (seq, time, H, W).
# nan_fill_value (float): Value used to replace NaNs before processing.
# time_stride (int): Temporal subsampling stride (use 1 for every time step).

# === Measurement policy ===
# num_measurements (int): K — number of top-difference pixels updated per time step.
# save_probability (bool): If true, save L1-normalized difference maps as probability maps.

# === Visualization ===
# save_frames (bool): If true, save side-by-side PNGs (Measured vs True) for one sequence.
# frame_output_dir (str): Output folder for visualization frames (created per input stem).
# frame_seq_index (int): Sequence index to visualize (0-based).
# cmap (str): Matplotlib colormap name for saved frames (e.g., "hot").

# === Outputs ===
# measured_output (str): File or directory for measured maps (shape: N_seq × (T-1) × H × W).
# indices_output  (str): File or directory for (col,row) measurement indices per step.
#   # Note: If a path ends with ".npy", write exactly there; if it is a directory,
#   # files will be named "<default>__<input_stem>.npy" to avoid collisions.
#   # Probability maps use an auto path ("y_probabilities.npy") when save_probability=true.

# === Progress display ===
# progress (str): "bar" | "print" | "none" — choose progress reporting mode.
# progress_leave (bool): Keep progress bars after completion (when using "bar").
# progress_seq_every (int): Print interval (in sequences) when progress="print".

OUTPUT PATH RULES
-----------------
- If an output path ends with ".npy", the file is written exactly there.
- If an output path is a directory, the script writes a file named
  "<default_name>__<input_stem>.npy" inside it. This allows batching multiple
  inputs from a directory without collisions.

USAGE
-----
    python 03_generateMeasDesign.py --config configs/generateMeasDesign.yml

NOTES
-----
- Requires: numpy, matplotlib, pyyaml; optional: tqdm (for progress bars).
- Designed to be deterministic given the same inputs and config.
- Measurement indices are stored as (col, row) to match typical image (x, y)
  conventions and ease plotting/mesh mapping later.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

import numpy as np
import matplotlib.pyplot as plt

# --- YAML config loader ---
try:
    import yaml
except ImportError as e:
    raise SystemExit(
        "[error] PyYAML not found. Please install with `pip install pyyaml` or `conda install pyyaml`."
    ) from e

# --- optional: tqdm for progress bars ---
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None  # fallback if tqdm is not installed


# ============================================================
# Utility functions
# ============================================================

def _list_input_npys(input_path: str):
    """Return a list of .npy input files (single file or directory)."""
    p = Path(input_path)
    if p.is_dir():
        return sorted(p.glob("*.npy"))
    if p.suffix.lower() == ".npy" and p.exists():
        return [p]
    raise FileNotFoundError(f"[error] Input not found or not a .npy file/dir: {input_path}")


def _resolve_out(path_or_dir: str, stem: str, default_name: str) -> Path:
    """Generate consistent output file paths."""
    p = Path(path_or_dir)
    if p.suffix.lower() == ".npy":
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{default_name}__{stem}.npy"


def convert_to_probability_distribution(difference_map: np.ndarray) -> np.ndarray:
    """Convert a difference map into a normalized probability distribution."""
    exp_diff = np.abs(difference_map)
    total = np.sum(exp_diff)
    if total == 0:
        return np.zeros_like(exp_diff)
    return exp_diff / total


def _now():
    """Return current time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Main process
# ============================================================

def run_from_cfg(cfg: dict):
    # ---- Configuration (YAML has priority) ----
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

    # ---- Progress display settings ----
    progress_mode      = str(cfg.get("progress", "bar")).lower()   # "bar" | "print" | "none"
    progress_leave     = bool(cfg.get("progress_leave", False))     # Keep bars after completion
    progress_seq_every = int(cfg.get("progress_seq_every", 5))      # Print interval for "print" mode

    # ---- Collect input files ----
    inputs = _list_input_npys(input_file)
    t0_all = time.perf_counter()
    print(f"[start] {_now()} | inputs={len(inputs)} file(s) | progress='{progress_mode}'")

    # ---- Loop over input files ----
    file_iter = enumerate(inputs, start=1)
    if progress_mode == "bar" and tqdm is not None:
        file_iter = tqdm(file_iter, total=len(inputs), desc="Inputs", leave=progress_leave)

    for i_file, in_path in file_iter:
        t0_file = time.perf_counter()
        true_resistivity_data = np.load(in_path)
        true_resistivity_data = np.nan_to_num(true_resistivity_data, nan=nan_fill_value)

        if true_resistivity_data.ndim != 4:
            raise ValueError(f"[error] Unexpected shape: {true_resistivity_data.shape} (expected 4D: seq,time,y,x)")

        if time_stride < 1:
            time_stride = 1
        data = true_resistivity_data[:, 0::time_stride, :, :]
        n_seq, n_time, grid_y, grid_x = data.shape

        # Visualization folder (per input file)
        if save_frames:
            out_frames_dir = Path(frame_output_dir) / in_path.stem
            out_frames_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_frames_dir = None

        X, y_probabilities, measured, measurement_indices = [], [], [], []

        # ---- Sequence loop ----
        seq_iter = range(n_seq)
        if progress_mode == "bar" and tqdm is not None:
            seq_iter = tqdm(seq_iter, total=n_seq, desc=f"{in_path.stem} | seq", leave=progress_leave, position=1)

        for s in seq_iter:
            measured_resistivity_map = data[s, 0, :, :].copy()
            Xseq, y_probabilities_seq, measured_seq, indices_seq = [], [], [], []

            # ---- Time loop ----
            time_iter = range(1, n_time)
            if progress_mode == "bar" and tqdm is not None:
                time_iter = tqdm(time_iter, total=n_time-1, desc=f"t", leave=False, position=2)

            for t in time_iter:
                true_t = data[s, t, :, :].copy()
                difference_map = np.abs(true_t - measured_resistivity_map)

                # Pick top-N largest difference pixels
                flat_idx = np.argsort(difference_map, axis=None)[-num_measurements:]
                rows, cols = np.unravel_index(flat_idx, true_t.shape)

                # Update measured map
                for r, c in zip(rows, cols):
                    measured_resistivity_map[r, c] = true_t[r, c]
                    indices_seq.append((int(c), int(r)))  # store as (col, row)

                # Create model input: measured map + previous probability (feedback)
                feedback_input = y_probabilities_seq[-1] if (t > 1 and y_probabilities_seq) else np.zeros((grid_y, grid_x))
                combined_input = np.stack([measured_resistivity_map, feedback_input], axis=-1)
                Xseq.append(combined_input)
                measured_seq.append(measured_resistivity_map.copy())

                # Compute new probability map
                probability_map = convert_to_probability_distribution(difference_map)
                y_probabilities_seq.append(probability_map)

                # Save visualization frame (optional)
                if save_frames and s == frame_seq_index:
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    im1 = plt.imshow(measured_resistivity_map, cmap=cmap, interpolation='nearest')
                    plt.title(f'{in_path.stem} | Seq {s} | Measured @ t={t+1}')
                    plt.colorbar(im1, orientation='horizontal', pad=0.2)

                    plt.subplot(1, 2, 2)
                    im2 = plt.imshow(true_t, cmap=cmap, interpolation='nearest')
                    plt.title(f'{in_path.stem} | Seq {s} | True @ t={t+1}')
                    plt.colorbar(im2, orientation='horizontal', pad=0.2)

                    plt.tight_layout()
                    plt.savefig(Path(out_frames_dir) / f'seq{s:03d}_t{t:04d}.png')
                    plt.close()

                # Optional console progress
                if progress_mode == "print" and (t % 50 == 0 or t == n_time - 1):
                    print(f"[{in_path.stem}] seq {s+1}/{n_seq} t {t}/{n_time-1}")

            X.append(Xseq)
            y_probabilities.append(y_probabilities_seq)
            measured.append(measured_seq)
            measurement_indices.append(indices_seq)

            if progress_mode == "print" and ((s+1) % max(1, progress_seq_every) == 0 or (s+1) == n_seq):
                print(f"[{in_path.stem}] seq {s+1}/{n_seq} done")

        # ---- Save results ----
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


# ============================================================
# Entry point
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=str, default="configs/generate_meas_design.yml",
        help="Path to YAML configuration file (YAML values take precedence)."
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[error] Config YAML not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    run_from_cfg(cfg)


if __name__ == "__main__":
    main()
