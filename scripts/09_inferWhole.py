"""
Script: 09_inferWhole.py
Purpose
    Reconstruct full time-series conductivity/resistivity maps from a model that predicts
    per-timestep *differences* (seq2seq), align them with ground-truth test data, compute
    error metrics (MAPE), and optionally export comparison images.

Context
    Many sequence models predict Δ (difference) rather than absolute values to stabilize
    learning. This script:
      (1) reads a single "initial" prediction for t=1,
      (2) reads a seq2seq tensor of subsequent per-timestep differences Δ,
      (3) reconstructs absolute values by cumulative summation starting from the true t=0,
      (4) converts ground-truth to the same physical scale (safe 1/x if needed),
      (5) evaluates accuracy and saves results.

Inputs (from YAML config)
    inputs:
      seq2seq_dir,  seq2seq_file
          Numpy .npy with predicted differences of shape (N, T_pred, R, C).
          Each [t] represents Δ at timestep t+1 relative to previous.
      initial_dir, initial_file
          Numpy .npy of the "initial" difference image for each sequence,
          shape (N, R, C). Treated as the first Δ after t=0.
      true_dir,    true_file
          Numpy .npy of ground-truth values of shape (N, T_true_full, R, C).
          These are the raw values before any 1/x scaling.

Processing
    - safe_inverse_k(united, scale=SCALE, eps=EPS, clip=CLIP)
        Converts ground-truth "conductivity-like" quantity k to "resistivity-like"
        values via (scale / k), leaving NaN as NaN and clipping extreme magnitudes.
        If your pipeline already uses absolute conductivity, leave SCALE as desired
        visualization scale or set appropriately.

    - Sequence reconstruction
        Given:
            V0  := true value at t=0 (converted by safe_inverse_k)
            D0  := "initial" diff (initial_all[seq])
            D1… := seq2seq predicted diffs
        We build:
            values_series = cumsum( [ V0, D0, D1, D2, ... ], axis=0 )
        If the first predicted diff duplicates the initial diff, we can drop it with
        processing.drop_duplicate_first_diff = true.

    - Temporal alignment
        Ground-truth can be sub-sampled by processing.num_measurements (e.g., when
        only every k-th timestep is measured). The script truncates both predicted and
        true series to the minimum common length before scoring.

Outputs (to output.dir)
    - <mape_txt> (default: mape_values.txt)
        Per-sequence mean absolute percentage error:
            MAPE(%) = mean_over_valid( |pred − true| / |true| ) × 100
        Only finite, non-zero ground-truth pixels are counted.
    - <stack_file> (default: conductivity.npy)
        4-D array (N, Tmin, R, C) of reconstructed absolute values, where Tmin is the
        shortest per-sequence length after alignment.
    - Optional PNGs when viz.enabled = true
        For selected sequences and timesteps:
          (1) Predicted value
          (2) True value
          (3) |Pred − True|
        NaNs are rendered as white. Pred/True share a unified color scale per frame.

Units & scaling
    - If your physical quantity is conductivity (e.g., mS m^{-1}) but downstream analysis
      expects resistivity, set `processing.scale` to your desired display/unit scale
      and let safe_inverse_k do 1/x conversion. If you already operate in the target
      scale, you may adapt or bypass this step.
    - Colormaps: viz.cmap_value for values (default "hot"), viz.cmap_diff for diffs
      (default "coolwarm").

Expected array shapes
    seq2seq_all:      (N, T_pred, R, C)
    initial_all:      (N, R, C)
    united (truth):   (N, T_true_full, R, C)
    reconstructed:    (N, Tmin, R, C)
    
# ===========================
# YAML Configuration Guide — 09_inferWhole.py
# ===========================
# Keys for reconstructing full time-series maps from:
#   (a) an "initial" diff at t=1 and
#   (b) a seq2seq tensor of subsequent per-timestep differences (Δ).
# The script cumulatively sums from true t=0 to produce absolute maps,
# then scores MAPE and optionally saves comparison images.
# Paste at the top of inferWhole.yml.

# --- run ---
# run.verbose (bool): Print file paths, per-sequence MAPE, and save locations.

# --- inputs ---
# inputs.seq2seq_dir (str): Folder containing predicted Δ series (.npy).
# inputs.seq2seq_file (str): Filename of Δ tensor with shape (N, T_pred, R, C).
# inputs.initial_dir (str): Folder containing the "initial" diff at t=1 (.npy).
# inputs.initial_file (str): Filename of initial diff with shape (N, R, C).
# inputs.true_dir (str): Folder containing ground-truth time stacks (.npy).
# inputs.true_file (str): Filename of truth with shape (N, T_true_full, R, C).

# --- processing ---
# processing.num_measurements (int): Temporal stride for sub-sampling truth (e.g., 10 keeps t=0,10,20,…).
# processing.scale (float): Scale used in safe inverse transform (value → scale/value).
# processing.eps (float): Numerical guard; values with |x| ≤ eps are not inverted.
# processing.clip (float): Absolute clamp for inverted values to avoid extreme magnitudes.
# processing.drop_duplicate_first_diff (bool): If first Δ in seq2seq duplicates the initial diff, drop it.

# --- viz ---
# viz.enabled (bool): Save PNGs comparing Pred/True/|Pred−True| for selected sequences.
# viz.chosen_sequences (list[int]): Sequence indices to render (e.g., [0,1,2,3,4,5]).
# viz.cmap_value (str): Colormap for Pred/True (default "hot"); NaNs shown as white.
# viz.cmap_diff (str): Colormap for |Pred−True| (default "coolwarm"); NaNs shown as white.

# --- output ---
# output.dir (str): Destination folder for results (created if missing).
# output.mape_txt (str): Text file name for per-sequence MAPE (e.g., "mape_values.txt").
# output.stack_file (str): Numpy file name for reconstructed stack (e.g., "conductivity.npy").
# output.image_prefix (str): Prefix for saved PNGs (e.g., "measurement_locations_seq").

CLI
    python 09_inferWhole.py --config path/to/config.yml

Notes & gotchas
    - Shape mismatches are validated early; sequences are trimmed to the minimum N.
    - When computing MAPE, pixels with non-finite truth or zero truth are excluded.
    - If you see overly large reconstructed magnitudes, confirm unit conventions and
      whether safe_inverse_k should be applied (and with what `scale`).
"""


import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Optional YAML import (PyYAML is required)
try:
    import yaml
except Exception:
    yaml = None


def load_yaml(path: Path) -> dict:
    """Safely load a YAML configuration file."""
    if yaml is None:
        raise RuntimeError("PyYAML not found. Please install it using `pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_inverse_k(arr, scale=1000.0, eps=1e-8, clip=1e6):
    """
    Compute a safe inverse transformation: 1/x * scale.
    NaN values remain NaN. Only finite elements are inverted and clipped.
    Returns float32 for memory efficiency.
    """
    a = np.array(arr, dtype=np.float64, copy=False)
    valid = np.isfinite(a) & (np.abs(a) > eps)
    out = np.full_like(a, np.nan, dtype=np.float64)
    out[valid] = scale / a[valid]
    out[valid] = np.clip(out[valid], -clip, clip)
    return out.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Compare predicted diffs with true values using a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    start_time = time.time()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    # === Configuration sections ===
    inputs = cfg.get("inputs", {})
    proc = cfg.get("processing", {})
    viz = cfg.get("viz", {})
    out = cfg.get("output", {})
    run = cfg.get("run", {})

    verbose = bool(run.get("verbose", True))

    # === Construct file paths ===
    seq2seq_path = Path(inputs.get("seq2seq_dir", ".")) / inputs.get("seq2seq_file", "outputs_pred_all_series.npy")
    initial_path = Path(inputs.get("initial_dir", ".")) / inputs.get("initial_file", "pred_images_all.npy")
    true_path = Path(inputs.get("true_dir", ".")) / inputs.get("true_file", "united_triangular_matrices_test.npy")

    out_dir = Path(out.get("dir", "compareWithTestData"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[info] seq2seq:", seq2seq_path)
        print("[info] initial:", initial_path)
        print("[info] true   :", true_path)
        print("[info] outdir :", out_dir)

    # === Input existence check ===
    for p in [seq2seq_path, initial_path, true_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    # === Processing parameters ===
    NUM_MEASUREMENTS = int(proc.get("num_measurements", 10))
    SCALE = float(proc.get("scale", 1000.0))
    EPS = float(proc.get("eps", 1e-8))
    CLIP = float(proc.get("clip", 1e6))
    DROP_DUP_FIRST = bool(proc.get("drop_duplicate_first_diff", True))

    # === Visualization settings ===
    VIZ_ENABLED = bool(viz.get("enabled", True))
    CHOSEN_SEQUENCES = list(viz.get("chosen_sequences", [0, 1, 2, 3, 4, 5]))
    CMAP_VALUE = str(viz.get("cmap_value", "hot"))
    CMAP_DIFF = str(viz.get("cmap_diff", "coolwarm"))

    # === Output file settings ===
    MAPE_TXT = out.get("mape_txt", "mape_values.txt")
    STACK_FILE = out.get("stack_file", "conductivity.npy")
    IMG_PREFIX = out.get("image_prefix", "measurement_locations_seq")

    # === Load input arrays ===
    seq2seq_all = np.load(seq2seq_path)  # (N, T, R, C): predicted diffs
    initial_all = np.load(initial_path)  # (N, R, C): initial prediction
    united = np.load(true_path)          # (N, T_full, R, C): true conductivity values

    # === Shape consistency check ===
    N_s2s, T_s2s, R, C = seq2seq_all.shape
    N_init, R2, C2 = initial_all.shape
    if (R, C) != (R2, C2):
        raise ValueError(f"Spatial dimensions mismatch: SEQ2SEQ ({R},{C}), INITIAL ({R2},{C2})")

    if N_init != N_s2s or united.shape[0] != N_s2s:
        N = min(N_s2s, N_init, united.shape[0])
        if verbose:
            print(f"[warn] Adjusting N to {N} due to inconsistent sequence count.")
        seq2seq_all = seq2seq_all[:N]
        initial_all = initial_all[:N]
        united = united[:N]
    else:
        N = N_s2s

    # === Convert to resistivity (safe 1/x scaling) ===
    true_resistivity_all = safe_inverse_k(united, scale=SCALE, eps=EPS, clip=CLIP)

    mape_values = []
    conductivity_stack = []  # Store reconstructed value sequences for each sample

    # === Main per-sequence loop ===
    for seq in range(N):
        # Prepare difference series
        initial_diff = initial_all[seq][None, ...]  # (1, R, C)
        diffs_seq = seq2seq_all[seq]                # (T, R, C)

        if DROP_DUP_FIRST and np.allclose(diffs_seq[0], initial_all[seq], atol=1e-6, rtol=1e-6):
            diffs_rest = diffs_seq[1:]
        else:
            diffs_rest = diffs_seq

        # True base value at t=0
        initial_true_value = safe_inverse_k(united[seq, 0], scale=SCALE, eps=EPS, clip=CLIP)[None, ...]

        # Combine base + diff sequence
        diff_series = np.concatenate([initial_diff, diffs_rest], axis=0)

        # Reconstruct full values by cumulative summation
        values_series = np.cumsum(np.concatenate([initial_true_value, diff_series], axis=0), axis=0)

        # Downsample true data to match prediction stride
        true_seq = true_resistivity_all[seq, 0::NUM_MEASUREMENTS]
        T_pred = values_series.shape[0]
        T_true = true_seq.shape[0]
        T = min(T_pred, T_true)
        values_series = values_series[:T]
        true_seq = true_seq[:T]

        # === Compute MAPE and visualize if enabled ===
        mape_sum = 0.0
        valid_count = 0
        for t in range(T):
            true_vals = true_seq[t]
            pred_vals = values_series[t]
            mask = np.isfinite(true_vals) & (true_vals != 0.0)
            rel = np.zeros_like(true_vals, dtype=np.float32)
            rel[mask] = np.abs((pred_vals[mask] - true_vals[mask]) / true_vals[mask])
            mape_sum += float(rel[mask].sum())
            valid_count += int(mask.sum())

            if VIZ_ENABLED and (seq in CHOSEN_SEQUENCES):
                cmap_val = plt.get_cmap(CMAP_VALUE).copy()
                cmap_val.set_bad('white')  # Display NaN as white
                cmap_diff = plt.get_cmap(CMAP_DIFF).copy()
                cmap_diff.set_bad('white')

                pred_show = np.ma.masked_invalid(pred_vals)
                true_show = np.ma.masked_invalid(true_vals)

                vmin = min(pred_show.min(), true_show.min())
                vmax = max(pred_show.max(), true_show.max())

                diff_map = np.abs(pred_vals - true_vals)
                invalid = ~np.isfinite(pred_vals) | ~np.isfinite(true_vals)
                diff_map = diff_map.astype(np.float32, copy=False)
                diff_map[invalid] = np.nan
                diff_show = np.ma.masked_invalid(diff_map)

                # === Create visualization ===
                fig = plt.figure(figsize=(18, 6))
                fig.patch.set_facecolor('white')

                ax1 = plt.subplot(1, 3, 1)
                im1 = ax1.imshow(pred_show, aspect='auto', cmap=cmap_val, vmin=vmin, vmax=vmax)
                plt.colorbar(im1, ax=ax1, label="Predicted value")
                ax1.set_title(f"Pred t={t} (Seq {seq})")

                ax2 = plt.subplot(1, 3, 2)
                im2 = ax2.imshow(true_show, aspect='auto', cmap=cmap_val, vmin=vmin, vmax=vmax)
                plt.colorbar(im2, ax=ax2, label="True value")
                ax2.set_title(f"True t={t} (Seq {seq})")

                ax3 = plt.subplot(1, 3, 3)
                im3 = ax3.imshow(diff_show, aspect='auto', cmap=cmap_diff)
                plt.colorbar(im3, ax=ax3, label="|Pred - True|")
                ax3.set_title(f"Diff t={t} (Seq {seq})")

                fig.savefig(out_dir / f"{IMG_PREFIX}_{seq}_timestep_{t}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

        mape = (mape_sum / max(valid_count, 1)) * 100.0
        mape_values.append((seq, mape))
        if verbose:
            print(f"Sequence {seq}: MAPE = {mape:.4f}%  (T={T})")

        conductivity_stack.append(values_series)

    # === Save summary results ===
    mape_path = out_dir / MAPE_TXT
    with mape_path.open("w", encoding="utf-8") as f:
        for seq, m in mape_values:
            f.write(f"Sequence {seq}: MAPE = {m}%\n")
    if verbose:
        print(f"[info] MAPE values saved to {mape_path}")

    # Align all sequences to the shortest T for stacking
    min_T = min(arr.shape[0] for arr in conductivity_stack)
    trimmed = [arr[:min_T] for arr in conductivity_stack]
    all_values = np.stack(trimmed, axis=0)

    stack_path = out_dir / STACK_FILE
    np.save(stack_path, all_values)
    if verbose:
        print(f"[info] Saved {stack_path}  shape={all_values.shape}")

    elapsed = time.time() - start_time
    print(f"[time] Elapsed: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
