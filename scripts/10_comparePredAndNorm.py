# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Compare and visualize Predicted vs. Measured conductivity maps (YAML-driven).

Overview
--------
This script loads two 4-D NumPy arrays—predictions and measurements—and saves
side-by-side PNGs to compare them at corresponding time steps. Most runtime
options are read from a YAML config, but any option can be overridden from the
command line.

Input tensors
-------------
- Predicted conductivity: shape = (N, T_pred, H, W), dtype = float
- Measured  conductivity: shape = (N, T_meas, H, W), dtype = float
  *Both arrays must be 4-D. If the numbers of sequences differ, the script uses
  the minimum N across the two inputs.*

Units and transformations
-------------------------
- The plotted quantity is “conductivity” in mS·m⁻¹ (as labeled in the figures).
- Measured data are transformed as:
    transformed_measured = 1000.0 / measured
  per the original project convention. Zeros and non-finite values in either
  array are treated as NaN prior to plotting. Optionally, NaN/±Inf can be
  replaced with 0 just before visualization (see `nan_to_zero`).

Time alignment
--------------
Frames to compare are determined by:
    measured_t = pred_t * meas_step_factor - 1
for each prediction time index `pred_t` in:
    pred_t ∈ [start_pred, start_pred + max_pred_steps)
A (pred_t, measured_t) pair is kept only when `0 ≤ measured_t < T_meas`.
This reproduces the original code’s temporal pairing rule.

Color scaling
-------------
You can choose to compute a vmin/vmax per sequence (recommended for heterogeneous
series) or compute a single global vmin/vmax once and reuse it for all images:
- `per_sequence_vrange = true`  → vmin/vmax from all frames of that sequence
- `per_sequence_vrange = false` → vmin/vmax from the entire dataset (first pass)

Invalid values and colormap
---------------------------
- Zeros and non-finite values are set to NaN; NaNs render as white in the
  colormap. If `nan_to_zero = true`, NaNs/±Inf are replaced by 0 before plotting.
- The default colormap is "hot", but you can override it (e.g., "viridis").

Configuration & overrides
-------------------------
- File paths can be provided either as full paths (`pred_file`, `measured_file`)
  or as directory + filename pairs (`*_dir` + `*_filename`).
- Any YAML value can be overridden by CLI flags. CLI always takes precedence.

Outputs
-------
- For each valid (pred_t, measured_t) pair and each selected sequence index,
  one PNG is saved to `out_dir`:
    seq{NNN}_predt{TTT}_meast{TTT}.png
- A progress bar shows save progress; a summary is printed at the end.

Typical YAML
------------
# Example: config.yml
pred_file: null
pred_dir: "data/output/whole"
pred_filename: "conductivity.npy"
measured_file: null
measured_dir: "data/training"
measured_filename: "measured_training_data_sameRowColSeq34_test.npy"

out_dir: "compareWithTestData/pred_vs_measured"
seq: [0, 1, 2]        # or null to process all sequences
start_pred: 1
max_pred_steps: 31
meas_step_factor: 10
cmap: "hot"
nan_to_zero: false
per_sequence_vrange: true
dpi: 150

CLI examples
------------
# Basic run using YAML only
python 10_comparePredAndNorm.py --config config.yml

# Override a few options from the command line
python 10_comparePredAndNorm.py --config config.yml \
  --seq 0 5 12 --start_pred 2 --max_pred_steps 20 --cmap viridis

Assumptions & notes
-------------------
- This script focuses on visualization; it does not validate the physical
  correctness of the 1000/measured convention—this is carried over from the
  original project.
- The LaTeX label "mS·m⁻¹" is typeset in titles and colorbars.
- If no valid (pred_t, measured_t) pairs exist under the current settings, the
  script exits with a warning.
"""


import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm


# ----------------------------------------------------------
# Utility functions
# ----------------------------------------------------------

def _resolve_path(file_path, dir_path, filename):
    """Return the file_path if specified, otherwise combine dir_path and filename."""
    if file_path:
        return Path(file_path)
    if not dir_path or not filename:
        raise ValueError("Either file path OR both dir and filename must be provided.")
    return Path(dir_path) / filename


def load_npy(path: Path) -> np.ndarray:
    """Load a 4D NumPy array (N, T, H, W) and ensure correct shape."""
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    arr = np.load(str(path))
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array (N,T,H,W). Got shape={arr.shape} from {path}")
    return arr.astype(np.float32)


def transform_measured(meas: np.ndarray) -> np.ndarray:
    """
    Convert measured data using 1000 / measured.
    Zero or non-finite values are set to NaN.
    """
    meas_f = meas.astype(np.float32, copy=True)
    zeros = (meas_f == 0.0) | ~np.isfinite(meas_f)
    meas_f[zeros] = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        transformed = 1000.0 / meas_f
    return transformed


def transform_pred(pred: np.ndarray) -> np.ndarray:
    """
    Prepare predicted data by setting zero or non-finite values to NaN.
    """
    pred_f = pred.astype(np.float32, copy=True)
    invalid = (pred_f == 0.0) | ~np.isfinite(pred_f)
    pred_f[invalid] = np.nan
    return pred_f


def load_config(cfg_path: Path) -> dict:
    """Load a YAML configuration file."""
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def cli_override(cfg: dict, ns: argparse.Namespace) -> dict:
    """
    Override YAML configuration with CLI arguments when provided.
    """
    def set_if(name):
        v = getattr(ns, name, None)
        if v is not None:
            cfg[name] = v

    set_if("pred_file")
    set_if("pred_dir")
    set_if("pred_filename")
    set_if("measured_file")
    set_if("measured_dir")
    set_if("measured_filename")
    set_if("out_dir")
    set_if("start_pred")
    set_if("max_pred_steps")
    set_if("meas_step_factor")
    set_if("cmap")
    set_if("nan_to_zero")
    set_if("per_sequence_vrange")
    set_if("dpi")
    if ns.seq is not None:
        cfg["seq"] = ns.seq
    return cfg


# ----------------------------------------------------------
# Main process
# ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare predicted vs. measured conductivity maps.")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")

    # Optional CLI overrides for flexibility
    ap.add_argument("--pred_file", type=str)
    ap.add_argument("--pred_dir", type=str)
    ap.add_argument("--pred_filename", type=str)
    ap.add_argument("--measured_file", type=str)
    ap.add_argument("--measured_dir", type=str)
    ap.add_argument("--measured_filename", type=str)
    ap.add_argument("--out_dir", type=str)
    ap.add_argument("--seq", type=int, nargs="*")
    ap.add_argument("--start_pred", type=int)
    ap.add_argument("--max_pred_steps", type=int)
    ap.add_argument("--meas_step_factor", type=int)
    ap.add_argument("--cmap", type=str)
    ap.add_argument("--nan_to_zero", action="store_true")
    ap.add_argument("--per_sequence_vrange", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"))
    ap.add_argument("--dpi", type=int)

    ns = ap.parse_args()

    # Load YAML and apply overrides
    cfg_path = Path(ns.config)
    cfg = load_config(cfg_path)
    cfg = cli_override(cfg, ns)

    # --- Basic settings ---
    seq = cfg.get("seq", None)
    start_pred = int(cfg.get("start_pred", 1))
    max_pred_steps = int(cfg.get("max_pred_steps", 31))
    meas_step_factor = int(cfg.get("meas_step_factor", 10))
    cmap_name = cfg.get("cmap", "hot")
    nan_to_zero = bool(cfg.get("nan_to_zero", False))
    per_sequence_vrange = bool(cfg.get("per_sequence_vrange", True))
    dpi = int(cfg.get("dpi", 150))

    # --- Resolve file paths ---
    pred_path = _resolve_path(cfg.get("pred_file"), cfg.get("pred_dir"), cfg.get("pred_filename"))
    meas_path = _resolve_path(cfg.get("measured_file"), cfg.get("measured_dir"), cfg.get("measured_filename"))
    out_dir = Path(cfg.get("out_dir", "compareWithTestData/pred_vs_measured"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    pred_raw = load_npy(pred_path)   # (N, T_pred, H, W)
    meas_raw = load_npy(meas_path)   # (N, T_meas, H, W)

    # Align number of sequences (N)
    N = min(pred_raw.shape[0], meas_raw.shape[0])
    pred_raw = pred_raw[:N]
    meas_raw = meas_raw[:N]

    # --- Transform data ---
    pred = transform_pred(pred_raw)
    meas_trans = transform_measured(meas_raw)

    if nan_to_zero:
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        meas_trans = np.nan_to_num(meas_trans, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Map predicted to measured time steps ---
    Tp = pred.shape[1]
    Tm = meas_trans.shape[1]
    pred_times = list(range(start_pred, min(start_pred + max_pred_steps, Tp)))
    pairs = []
    for pt in pred_times:
        mt = pt * meas_step_factor - 1
        if 0 <= mt < Tm:
            pairs.append((pt, mt))
    if not pairs:
        print("[Warning] No valid (pred_t, meas_t) pairs found under current settings.")
        return

    # --- Select sequences to visualize ---
    if seq:
        seq_idx = [i for i in seq if 0 <= i < N]
    else:
        seq_idx = list(range(N))

    total_tasks = len(seq_idx) * len(pairs)
    total = 0

    # Prepare colormap
    base_cmap = plt.get_cmap(cmap_name).copy()
    try:
        base_cmap.set_bad(color="white")
    except Exception:
        pass

    # --- Generate and save comparison plots ---
    with tqdm(total=total_tasks, desc="Saving images", unit="img") as pbar:
        for s in seq_idx:
            # Compute vmin/vmax range
            if per_sequence_vrange:
                pred_vals = [pred[s, pt] for (pt, mt) in pairs]
                meas_vals = [meas_trans[s, mt] for (pt, mt) in pairs]
                all_pred = np.stack(pred_vals, axis=0)
                all_meas = np.stack(meas_vals, axis=0)
                vmin = float(np.nanmin([np.nanmin(all_pred), np.nanmin(all_meas)]))
                vmax = float(np.nanmax([np.nanmax(all_pred), np.nanmax(all_meas)]))
            else:
                if total == 0:
                    vmin = float(np.nanmin([np.nanmin(pred), np.nanmin(meas_trans)]))
                    vmax = float(np.nanmax([np.nanmax(pred), np.nanmax(meas_trans)]))

            # Plot per time pair
            for (pt, mt) in pairs:
                fig = plt.figure(figsize=(12, 5))

                # --- Predicted conductivity ---
                ax1 = plt.subplot(1, 2, 1)
                im1 = ax1.imshow(pred[s, pt], aspect="auto", cmap=base_cmap, vmin=vmin, vmax=vmax)
                ax1.set_title(f"Predicted conductivity ($mS\\,m^{{-1}}$, seq {s}, pred {pt})")
                plt.colorbar(im1, ax=ax1, label="$mS\\,m^{{-1}}$")

                # --- Measured conductivity ---
                ax2 = plt.subplot(1, 2, 2)
                im2 = ax2.imshow(meas_trans[s, mt], aspect="auto", cmap=base_cmap, vmin=vmin, vmax=vmax)
                ax2.set_title(f"Measured conductivity ($mS\\,m^{{-1}}$, seq {s}, meas {mt})")
                plt.colorbar(im2, ax=ax2, label="$mS\\,m^{{-1}}$")

                # Save figure
                out_path = out_dir / f"seq{s:03d}_predt{pt:03d}_meast{mt:03d}.png"
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)

                total += 1
                pbar.update(1)
                if (total % 50) == 0:
                    pbar.set_postfix(saved=total)

    print(f"[Done] Saved {total} images to: {out_dir}")
    print(f"[Info] Used time pairs (pred_t → meas_t): {pairs}")


if __name__ == "__main__":
    main()
