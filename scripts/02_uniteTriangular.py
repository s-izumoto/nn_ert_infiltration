# -*- coding: utf-8 -*-
"""
Combine multiple `triangular_matrix_seq_*.npy` files into unified training and test datasets,
with **YAML configuration taking precedence** over any other settings.

Overview
--------
This script scans a directory for NumPy arrays that each hold one sequence of
triangular-matrix images (shape `(T, H, W)` per file). It randomly splits the
files into **train** and **test** sets, verifies that all chosen files share the
same shape (if `strict_shape: true`), stacks them along a new leading axis, and
saves the merged arrays as:

- `train_out` → `(N_train, T, H, W)`
- `test_out`  → `(N_test,  T, H, W)`

Optionally, it also writes the corresponding file name lists for reproducibility
and bookkeeping (e.g., `united_train_filenames.txt`, `united_test_filenames.txt`).

Typical Use Case
----------------
You have many per-sequence `.npy` files (e.g., `triangular_matrix_seq_000.npy`,
`triangular_matrix_seq_001.npy`, …) produced by a visualization or preprocessing
pipeline. You want a single compact training tensor and a single test tensor for
downstream ML workflows (e.g., LSTM, CNN, or sequence models), and you want the
split to be reproducible with a fixed random seed.

Key Behaviors / Assumptions
---------------------------
- **YAML-first:** All configuration comes from the YAML file specified by `-c/--config`.
- **Deterministic listing:** Files are sorted before splitting to ensure a stable base order.
- **Reproducible split:** If `seed` is an integer, the train/test split is repeatable.
- **Shape checking:** If `strict_shape` is `true`, all files must match the reference shape
  taken from the first picked training file; otherwise a clear error is raised.
- **Lightweight I/O:** Only the selected files are loaded into memory (once each), then stacked and saved.

YAML Configuration Keys (with defaults)
---------------------------------------
- `input_folder`      (str, default `"visualizations_large"`): Where to search for `.npy` files.
- `filename_pattern`  (str, default `"triangular_matrix_seq_*.npy"`): Glob pattern for files.
- `train_out`         (str, default `"united_triangular_matrices.npy"`): Output path for train stack.
- `test_out`          (str, default `"united_triangular_matrices_test.npy"`): Output path for test stack.
- `train_count`       (int, default `45`): Number of files to include in the training set.
- `test_count`        (int, default `5`): Number of files to include in the test set.
- `seed`              (int|null, default `42`): Random seed; `null` → non-reproducible random split.
- `strict_shape`      (bool, default `true`): Enforce identical `(T, H, W)` across all selected files.
- `save_filelists`    (bool, default `true`): Whether to save train/test filename lists.
- `filelists_prefix`  (str, default `"united"`): Prefix for the filename-list text files.

Input / Output
--------------
**Input:** A folder containing many `.npy` files, each shaped `(T, H, W)`.

**Output:**
- `train_out` (`.npy`): Array shaped `(N_train, T, H, W)`.
- `test_out`  (`.npy`): Array shaped `(N_test,  T, H, W)`.
- Optional filename lists: `<filelists_prefix>_train_filenames.txt` and
  `<filelists_prefix>_test_filenames.txt`.

CLI Usage
---------
    python 02_uniteTriangular.py -c path/to/config.yml

Example YAML
------------
    input_folder: "visualizations_large"
    filename_pattern: "triangular_matrix_seq_*.npy"
    train_out: "data/united/united_triangular_matrices.npy"
    test_out: "data/united/united_triangular_matrices_test.npy"
    train_count: 45
    test_count: 5
    seed: 42              # use null for fully random split
    strict_shape: true
    save_filelists: true
    filelists_prefix: "data/united/united"

Errors & Troubleshooting
------------------------
- **Not enough files:** Raised if the folder does not contain at least `train_count + test_count`.
- **Shape mismatch:** If `strict_shape: true`, a descriptive error shows which file broke the contract.
"""

import os
import sys
import argparse
from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm

# --- Optional YAML dependency check ---
try:
    import yaml
except ImportError:
    print("[error] PyYAML not found. Please run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def _ensure_parent_dir(path_str: str):
    """Create the parent directory if it does not already exist."""
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def load_cfg(yaml_path: Path) -> dict:
    """Load configuration from YAML and set default parameters."""
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Default values (used if not specified in YAML)
    cfg.setdefault("input_folder", "visualizations_large")
    cfg.setdefault("filename_pattern", "triangular_matrix_seq_*.npy")
    cfg.setdefault("train_out", "united_triangular_matrices.npy")
    cfg.setdefault("test_out", "united_triangular_matrices_test.npy")
    cfg.setdefault("save_filelists", True)
    cfg.setdefault("filelists_prefix", "united")
    cfg.setdefault("train_count", 45)
    cfg.setdefault("test_count", 5)
    cfg.setdefault("seed", 42)           # None = fully random split
    cfg.setdefault("strict_shape", True)
    return cfg


def list_npy_files(input_folder: str, pattern: str):
    """Return a sorted list of .npy files matching the given pattern."""
    paths = sorted(glob.glob(str(Path(input_folder) / pattern)))
    return [Path(p) for p in paths]


def stack_files(files, t_ref=None, h_ref=None, w_ref=None, strict=True):
    """
    Load multiple .npy files and stack them into a single array.
    Ensures all arrays share the same shape if strict=True.
    """
    seqs = []
    for p in tqdm(files, desc="Loading"):
        arr = np.load(p)
        if t_ref is None:
            t_ref, h_ref, w_ref = arr.shape
        if strict and arr.shape != (t_ref, h_ref, w_ref):
            raise ValueError(
                f"Shape mismatch: {p.name} has shape {arr.shape} "
                f"(expected {(t_ref, h_ref, w_ref)})"
            )
        seqs.append(arr)
    return np.stack(seqs, axis=0)  # Output shape: (num_sequences, T, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to YAML configuration file (takes highest priority)"
    )
    # No other CLI arguments are provided — YAML config is prioritized
    args = ap.parse_args()

    yaml_path = Path(args.config)
    cfg = load_cfg(yaml_path)

    # --- Load config values ---
    input_folder   = cfg["input_folder"]
    filename_pat   = cfg["filename_pattern"]
    train_out      = cfg["train_out"]
    test_out       = cfg["test_out"]
    train_count    = int(cfg["train_count"])
    test_count     = int(cfg["test_count"])
    seed           = cfg["seed"]
    strict_shape   = bool(cfg["strict_shape"])
    save_filelists = bool(cfg["save_filelists"])
    lists_prefix   = cfg["filelists_prefix"]

    # --- Gather all .npy files ---
    files = list_npy_files(input_folder, filename_pat)
    if len(files) < train_count + test_count:
        raise ValueError(
            f"Not enough files found: {len(files)} available, "
            f"but {train_count + test_count} required."
        )

    # --- Random split into train/test sets ---
    rng = np.random.default_rng(None if seed is None else int(seed))
    perm = rng.permutation(len(files))
    train_idx = perm[:train_count]
    test_idx  = perm[train_count:train_count + test_count]
    train_files = [files[i] for i in train_idx]
    test_files  = [files[i] for i in test_idx]

    # --- Reference shape from the first training file ---
    first = np.load(train_files[0])
    t_ref, h_ref, w_ref = first.shape

    # --- Load and stack arrays ---
    train_array = stack_files(train_files, t_ref, h_ref, w_ref, strict=strict_shape)
    test_array  = stack_files(test_files,  t_ref, h_ref, w_ref, strict=strict_shape)

    # --- Save merged arrays ---
    _ensure_parent_dir(train_out)
    _ensure_parent_dir(test_out)
    np.save(train_out, train_array)
    np.save(test_out,  test_array)

    # --- Optionally save the file name lists ---
    if save_filelists:
        train_list_path = f"{lists_prefix}_train_filenames.txt"
        test_list_path  = f"{lists_prefix}_test_filenames.txt"
        _ensure_parent_dir(train_list_path)
        _ensure_parent_dir(test_list_path)
        np.savetxt(train_list_path, [p.name for p in train_files], fmt="%s", encoding="utf-8")
        np.savetxt(test_list_path,  [p.name for p in test_files],  fmt="%s", encoding="utf-8")

    # --- Summary output ---
    print(f"[done] train -> {train_out}  shape={train_array.shape}  files={len(train_files)}")
    print(f"[done] test  -> {test_out}   shape={test_array.shape}   files={len(test_files)}")


if __name__ == "__main__":
    main()
