
# -*- coding: utf-8 -*-
# Plot side-by-side images: Predicted vs Measured(Test)
# Updates:
# - Pred invalid values (0 or non-finite) are converted to NaN so they render as white.
# - Colormap 'bad' color set to white for both panels.
# - Per-sequence value scale and measured_t mapping remain as before.
# - Time mapping: measured_t = pred_t * 10 - 1.

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

def tic():
    return time.perf_counter()

def lap(t0, msg):
    dt = time.perf_counter() - t0
    print(f"[{msg}] {dt:.3f}s")
    return time.perf_counter()

def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    arr = np.load(str(path))
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array (N,T,H,W). Got shape={arr.shape} from {path}")
    return arr.astype(np.float32)

def transform_measured(meas: np.ndarray) -> np.ndarray:
    # Apply 1000 / measured with safe division (0 -> NaN).
    meas_f = meas.astype(np.float32, copy=True)
    zeros = (meas_f == 0.0) | ~np.isfinite(meas_f)
    meas_f[zeros] = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        transformed = 1000.0 / meas_f
    return transformed

def transform_pred(pred: np.ndarray) -> np.ndarray:
    # Treat 0 and non-finite as invalid -> NaN (so they render as white)
    pred_f = pred.astype(np.float32, copy=True)
    invalid = (pred_f == 0.0) | ~np.isfinite(pred_f)
    pred_f[invalid] = np.nan
    return pred_f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, default="../data/output/whole/conductivity.npy",
                    help="Predicted value stack .npy (N,T,H,W)")
    ap.add_argument("--measured", type=str, default="../data/training/training_data34_test.npy",
                    help="Measured (test) .npy (N,T,H,W)")
    ap.add_argument("--outdir", type=str, default="compareWithTestData/pred_vs_measured",
                    help="Output directory for PNGs")
    ap.add_argument("--seq", type=int, nargs="*", default=None,
                    help="Sequence indices to export. Default: all")
    ap.add_argument("--start_pred", type=int, default=1,
                    help="Pred time index to start from (default 1)")
    ap.add_argument("--max_pred_steps", type=int, default=31,
                    help="Maximum number of pred steps to include (from start_pred). Default 31")
    ap.add_argument("--meas_step_factor", type=int, default=10,
                    help="Mapping factor: measured_t = pred_t * factor - 1 (default 10)")
    ap.add_argument("--cmap", type=str, default="hot", help="Colormap for values")
    ap.add_argument("--nan_to_zero", action="store_true",
                    help="Replace NaNs with 0.0 before plotting")
    ns = ap.parse_args()

    t0 = tic()
    pred_path = Path(ns.pred)
    meas_path = Path(ns.measured)
    outdir = Path(ns.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred_raw = load_npy(pred_path)   # (N,Tp,H,W)
    t0 = lap(t0, "load pred")
    meas_raw = load_npy(meas_path)   # (N,Tm,H,W)
    t0 = lap(t0, "load meas")
    print(f"[shape] pred_raw: {pred_raw.shape}, meas_raw: {meas_raw.shape}")

    # Align N to the minimum available
    N = min(pred_raw.shape[0], meas_raw.shape[0])
    pred_raw = pred_raw[:N]
    meas_raw = meas_raw[:N]
    print(f"[shape] aligned N={N}, pred_raw: {pred_raw.shape}, meas_raw: {meas_raw.shape}")
    t0 = lap(t0, "align N")

    # Transform arrays
    pred = transform_pred(pred_raw)
    t0 = lap(t0, "transform_pred")
    meas_trans = transform_measured(meas_raw)
    t0 = lap(t0, "transform_measured")

    if ns.nan_to_zero:
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        meas_trans = np.nan_to_num(meas_trans, nan=0.0, posinf=0.0, neginf=0.0)
        t0 = lap(t0, "nan_to_zero")

    # Build mapped time index pairs (pred_t, meas_t) with measured_t = pred_t * factor - 1
    Tp = pred.shape[1]
    Tm = meas_trans.shape[1]

    pred_times = list(range(ns.start_pred, min(ns.start_pred + ns.max_pred_steps, Tp)))
    pairs = []
    for pt in pred_times:
        mt = pt * ns.meas_step_factor - 1
        if 0 <= mt < Tm:
            pairs.append((pt, mt))
    if not pairs:
        print("[warn] No valid (pred_t, meas_t) pairs under current settings.")
        return
    
    print(f"[time-map] Tp={Tp}, Tm={Tm}, start_pred={ns.start_pred}, "
      f"max_pred_steps={ns.max_pred_steps}, factor={ns.meas_step_factor}")
    print(f"[time-map] pairs={len(pairs)} (例: {pairs[:3]} ... {pairs[-3:] if len(pairs)>3 else pairs})")
    t0 = lap(t0, "make pairs")

    # Choose sequences
    seq_idx = list(range(N)) if ns.seq is None else [i for i in ns.seq if 0 <= i < N]
    print(f"[seq] len={len(seq_idx)} (例: {seq_idx[:10]})")
    t0 = lap(t0, "select seq")

    count = 0
    total_tasks = len(seq_idx) * len(pairs)
    print(f"[plan] total images = {total_tasks}")
    t0 = lap(t0, "pre-loop summary")
    
    for s in seq_idx:
        # Per-sequence vmin/vmax across selected frames (NaNs ignored)
        pred_vals = [pred[s, pt] for (pt, mt) in pairs]
        meas_vals = [meas_trans[s, mt] for (pt, mt) in pairs]
        all_pred = np.stack(pred_vals, axis=0)
        all_meas = np.stack(meas_vals, axis=0)
        vmin = float(np.nanmin([np.nanmin(all_pred), np.nanmin(all_meas)]))
        vmax = float(np.nanmax([np.nanmax(all_pred), np.nanmax(all_meas)]))

        # Prepare colormap with white for NaNs ("bad" values)
        cmap = plt.get_cmap(ns.cmap).copy()
        try:
            cmap.set_bad(color="white")
        except Exception:
            pass

        for (pt, mt) in pairs:
            pred_img = pred[s, pt]
            meas_img = meas_trans[s, mt]

            fig = plt.figure(figsize=(12, 5))
            ax1 = plt.subplot(1, 2, 1)
            im1 = ax1.imshow(pred_img, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax1.set_title(f"Predicted conductivity ($mS\\,m^{{-1}}$, sequence {s}, step {pt})")
            plt.colorbar(im1, ax=ax1, label="$mS\\,m^{{-1}}$")

            ax2 = plt.subplot(1, 2, 2)
            im2 = ax2.imshow(meas_img, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax2.set_title(f"Measured conductivity ($mS\\,m^{{-1}}$, sequence {s}, step {pt})")
            plt.colorbar(im2, ax=ax2, label="$mS\\,m^{{-1}}$")

            out_path = outdir / f"seq{s:03d}_predt{pt:03d}_meast{mt:03d}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            count += 1

    print(f"[done] saved {count} images to: {outdir}")
    print(f"[info] used time pairs (pred_t -> meas_t): {pairs}")

if __name__ == "__main__":
    main()
