# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
04_generateTrainingData.py
==========================
Purpose
-------
Generate *measured* resistivity datasets from *true* resistivity maps by
simulating a sequential measurement process at predefined (column, row)
positions. The same position sequence can be reused for both training and
test sets so that model evaluation is consistent with training.

High-level Workflow
-------------------
1) Load a 4D true-resistivity tensor (N, T, H, W) and a set of candidate
   measurement position sequences (S, T, 2).
2) Select ONE representative position sequence:
   - "median": pick the sequence closest to the per-time-step median (robust).
   - "fixed":  use the provided fixed_index.
3) For each sequence n=0..N-1:
   - Initialize measured_map = true[n, t=0].
   - For t = 1..T' (clipped by position length):
       * Read (c, r) from the selected position sequence at t-1.
       * Update measured_map[r, c] = true[n, t][r, c] (only that pixel).
       * Record the updated measured_map for time t.
4) Save the stacked measured maps as a NumPy array:
   shape = (N, T'-1, H, W). Optionally do the same for a test set using
   the *same* position sequence.

Input Data & Shapes
-------------------
- True resistivity (train): (N, T, H, W) np.float32/np.float64
- True resistivity (test):  (N_test, T_test, H, W) [optional]
- Positions (candidates):   (S, T, 2) with 0-based integer coordinates
  * positions[s, t] = [col, row]
  * Only one sequence (S=1) is also allowed.

Key Behaviors & Assumptions
---------------------------
- NaN values in true data are replaced with 0.0 to avoid propagation.
- Only the single pixel (r, c) is updated per time step (num_measurements=1).
- Time dimension may be clipped: T' = min(T, positions_seq_len + 1).
- Output time length is T'-1 because t=0 initializes the measured map.
- File naming embeds the chosen sequence index for traceability.

# ===========================
# YAML Configuration Guide — 04_generateTrainingData.py
# ===========================
# Each key defines the input type and its purpose for generating
# *measured* time series from *true* resistivity maps using a fixed
# per-time measurement position sequence.

# === Required paths & filenames ===
# input_dir (str): Directory containing true data .npy files.
# positions_dir (str): Directory containing candidate position sequences .npy.
# output_dir (str): Directory where outputs are written (created if missing).
# train_file (str): Filename of training true data (shape: N × T × H × W).
# positions_file (str): Filename of candidate positions (shape: S × T × 2; [col,row]).

# === Optional inputs ===
# test_file (str | null): Filename of test true data; if absent, train only.

# === Sequence selection policy ===
# sequence_selection.mode (str): "median" (default) or "fixed" — how to pick one sequence.
# sequence_selection.fixed_index (int): Index to use when mode="fixed".

# === Measurement settings ===
# num_measurements (int): Measurements per time step (default 1; current logic assumes 1).

# === Output naming ===
# save_basename (str): Base name for output files (default "measured_training_data_sameRowColSeq").
# save_seq_index (bool): Save chosen_seq_index.npy for traceability (default true).

# --- Notes ---
# • The chosen sequence provides one (col,row) per time step; only that pixel is updated.
# • Time is clipped to available positions: T' = min(T, len(sequence)+1); outputs have length T'−1.
# • NaNs in true data are replaced with 0.0 before processing.


CLI Example
-----------
python 04_generateTrainingData.py --config configs/generateTrainingData.yml

Performance Tips
----------------
- This is memory-bound for large (N, T, H, W). If RAM is constrained:
  * Process sequences in chunks (extend code to stream or memmap).
  * Ensure .npy files are saved with float32 when possible.
"""

import os
from pathlib import Path
import numpy as np
import yaml


def convert_to_probability_distribution(difference_map: np.ndarray) -> np.ndarray:
    """
    Convert a difference map into a normalized probability distribution.

    Parameters
    ----------
    difference_map : np.ndarray
        Map of absolute differences between true and measured resistivity.

    Returns
    -------
    np.ndarray
        Probability distribution (same shape as difference_map).
    """
    exp_diff = np.abs(difference_map)
    total = np.sum(exp_diff)
    if total == 0:
        return np.zeros_like(exp_diff)
    return exp_diff / total


def process_dataset(true_resistivity_data: np.ndarray,
                    positions_seq: np.ndarray,
                    num_measurements: int = 1) -> np.ndarray:
    """
    Simulate the process of sequential measurements based on true data.

    Parameters
    ----------
    true_resistivity_data : np.ndarray
        True resistivity data of shape (N, T, H, W),
        where N = number of sequences, T = timesteps.
    positions_seq : np.ndarray
        Measurement positions of shape (T, 2),
        used to update the measured map step by step.
    num_measurements : int, optional
        Number of measurements per time step (default = 1).

    Returns
    -------
    np.ndarray
        Measured resistivity dataset of shape (N, T-1, H, W).
    """
    # Replace NaN with zeros to avoid numerical issues
    true_resistivity_data = np.nan_to_num(true_resistivity_data, nan=0.0)

    # Currently equivalent to selecting every time step (extendable)
    data = true_resistivity_data[:, 0::num_measurements, :, :]

    N, T, H, W = data.shape
    measured_all = []

    # Align time steps with available position sequence length
    Tmax = min(T, positions_seq.shape[0] + 1)

    for seq in range(N):
        measured_resistivity_map = data[seq, 0].copy()  # initial map at t=0
        measured_seq = []

        for t in range(1, Tmax):
            true_map_t = data[seq, t].copy()
            diff_map = np.abs(true_map_t - measured_resistivity_map)

            # Use predefined (column, row) positions
            c = int(positions_seq[t - 1, 0])
            r = int(positions_seq[t - 1, 1])

            # Update measured map at the selected location
            measured_resistivity_map[r, c] = true_map_t[r, c]

            # Store updated map
            measured_seq.append(measured_resistivity_map.copy())

            # Compute a probability map if needed (currently unused)
            _ = convert_to_probability_distribution(diff_map)

        measured_all.append(measured_seq)

    # Convert to array (N, Tmax-1, H, W)
    measured_all = np.array(measured_all, dtype=np.float32)
    return measured_all


def pick_sequence_index(positions: np.ndarray, mode: str, fixed_index):
    """
    Select a representative measurement sequence index.

    Parameters
    ----------
    positions : np.ndarray
        All candidate position sequences, shape (S, T, 2).
    mode : str
        Selection mode: "median" or "fixed".
    fixed_index : int or None
        Used if mode == "fixed".

    Returns
    -------
    int
        Selected sequence index.
    """
    S = positions.shape[0]
    if mode == "fixed":
        if fixed_index is None:
            raise ValueError("sequence_selection.mode='fixed' but fixed_index not set.")
        if not (0 <= int(fixed_index) < S):
            raise ValueError(f"fixed_index={fixed_index} is out of range (0..{S-1}).")
        return int(fixed_index)

    # "median" mode: select the sequence closest to the median coordinates
    med = np.median(positions, axis=0)            # (T, 2)
    d = np.abs(positions - med).sum(axis=(1, 2))  # (S,)
    seq_for_rc = int(np.argmin(d))
    return seq_for_rc


def main(yaml_path: str):
    """Main routine: load config, process training and test datasets."""
    # === Load YAML configuration ===
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # === Resolve paths (directory-based) ===
    input_dir = Path(cfg["input_dir"]).expanduser()
    positions_dir = Path(cfg["positions_dir"]).expanduser()
    output_dir = Path(cfg["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / cfg["train_file"]
    test_path  = input_dir / cfg["test_file"] if cfg.get("test_file") else None
    positions_path = positions_dir / cfg["positions_file"]

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not positions_path.exists():
        raise FileNotFoundError(f"Positions file not found: {positions_path}")

    # === Load input data ===
    train_data = np.load(str(train_path))  # (N, T, H, W)
    test_data = None
    if test_path and test_path.exists():
        test_data = np.load(str(test_path))

    positions = np.load(str(positions_path))  # (S, T, 2)

    # === Select representative sequence ===
    sel_cfg = cfg.get("sequence_selection", {}) or {}
    mode = sel_cfg.get("mode", "median")
    fixed_index = sel_cfg.get("fixed_index", None)
    seq_for_rc = pick_sequence_index(positions, mode, fixed_index)

    # Safety check: clamp index to range
    S = positions.shape[0]
    if not (0 <= seq_for_rc < S):
        seq_for_rc = 31 if S > 50 else 0

    positions_seq = positions[seq_for_rc]  # (T, 2)

    # === Process training dataset ===
    num_measurements = int(cfg.get("num_measurements", 1))
    measured_train = process_dataset(train_data, positions_seq, num_measurements=num_measurements)

    save_basename = cfg.get("save_basename", "measured_training_data_sameRowColSeq")
    out_train = output_dir / f"{save_basename}{seq_for_rc}.npy"
    np.save(str(out_train), measured_train)
    print(f"[save] {out_train}  shape={measured_train.shape}")

    # Save chosen sequence index for reference
    if cfg.get("save_seq_index", True):
        np.save(str(output_dir / "chosen_seq_index.npy"), np.array(seq_for_rc, dtype=int))

    # === Process test dataset (if provided) ===
    if test_data is not None:
        measured_test = process_dataset(test_data, positions_seq, num_measurements=num_measurements)
        out_test = output_dir / f"{save_basename}{seq_for_rc}_test.npy"
        np.save(str(out_test), measured_test)
        print(f"[save] {out_test}  shape={measured_test.shape}")
    else:
        print("[info] No test dataset found (training data only).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate measured resistivity maps using predefined measurement positions.")
    ap.add_argument("--config", "-c", required=True, help="Path to YAML configuration file.")
    ns = ap.parse_args()
    main(ns.config)
