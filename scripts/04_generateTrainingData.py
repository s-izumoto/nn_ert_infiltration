# -*- coding: utf-8 -*-
"""
04_generateTrainingData.py
--------------------------
Generate measured resistivity datasets from true resistivity maps using
a predefined sequence of measurement positions. The same sequence can
be applied to both training and test datasets for consistency.
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
