# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Seq2Seq LSTM inference driver configured via a YAML file.

Overview
--------
This script predicts a *sequence* of triangular measurement maps (e.g., apparent
conductivity/resistivity arranged in a triangular grid) using a trained
Seq2Seq LSTM. It reads:
  (1) "measured" arrays (.npy) that serve as inputs,
  (2) a "united" ground-truth stack (.npy) for initialization and (optionally)
      for pre-processing steps (e.g., differencing/normalization/mean-centering),
  (3) optional normalization/mean-value .npz files.

You can point "measured" to a single file or to a directory:
  • If it's a file, only that file is processed.
  • If it's a folder, all .npy files matching a pattern (default: "*.npy") are processed.

For each matched input, the script:
  1) Safely converts values if needed (e.g., 1/x with clipping to avoid NaNs/inf),
  2) Optionally trims early time steps, time-subsamples, or takes time-differences,
  3) Optionally applies time-step normalization and/or mean-centering,
  4) Flattens triangular frames to 1D feature vectors (row sizes 29, 26, 23, ...),
  5) Runs the trained Seq2Seq LSTM autoregressively for `output_seq_length` steps,
  6) Reconstructs triangular 2D frames from predictions,
  7) Saves a 4D array per input file (Nseries × Tpred × rows × cols),
  8) Optionally writes quick-look PNGs (sum over time, t0, last frame).

Inputs & Outputs
----------------
Inputs (from YAML):
  • data.measured.path          : str | Path
      - File → one .npy of shape (Nseries, T_meas, rows, cols)
      - Folder → directory containing multiple such .npy files
  • data.measured.pattern       : str (glob), used only when "path" is a folder
  • data.united.path            : str | Path, single .npy (Nseries, T_full, rows, cols)
  • data.use_diff               : bool, use Δt differencing for input/output
  • data.use_normalization      : bool, apply [time_step_min, time_step_max] scaling
  • data.normalization_factors  : str | Path to .npz OR directory containing it
  • data.mean_centered          : bool, subtract means from encoder input; add y-means back to predictions
  • data.mean_values            : str | Path to .npz OR directory containing it
  • data.use_time_context       : bool, append a normalized time scalar to each encoder step
  • data.early / data.early_steps: bool/int, keep only the first K time steps
  • data.sparse_step            : int, temporal stride (1 = no subsampling)
  • data.choose_index / index_list: subset of series indices to keep

Runtime/model (from YAML):
  • runtime.device              : "auto" | "cpu" | "cuda"
  • runtime.time_steps          : int, encoder time length to feed (e.g., 30)
  • runtime.output_seq_length   : int, number of steps to predict (e.g., 29)
  • model.hidden_size           : int, LSTM hidden size (default 512)
  • model.num_layers            : int, currently unused (model uses 2 stacked LSTMs)
  • model.dropout               : float, dropout prob (not used internally)
  • model.bidirectional         : bool, placeholder (encoder/decoder are unidirectional here)
  • model.checkpoint            : file or directory:
      - If a file: load directly.
      - If a directory: prefer files containing "best" (by most recent mtime);
        otherwise, load the most recent .pt/.pth/.ckpt.

Output (from YAML):
  • io.output_dir               : str | Path, where .npy results are saved
  • io.out_npy_name             : str, suffix for the output filename
      - Final filename becomes: "<measured_stem>__<out_npy_name>"
  • io.png_dir                  : str | Path, where quick-look PNGs are saved
  • io.save_png                 : bool, whether to save PNGs

File Resolution Rules
---------------------
• Normalization / mean-value files:
    If you pass a directory, the script first looks for preferred names
    ("normalization_factors.npz" / "mean_values.npz"). If not found, it picks
    the latest *.npz by modification time.
• Checkpoints:
    If you pass a directory, the script picks the newest checkpoint matching:
    *.pt, *.pth, *.ckpt. Files containing "best" are preferred.

Assumptions
-----------
• Triangular frames follow row sizes [29, 26, 23, ..., 2] (step −3).
• Input/output arrays are float-like and finite after the safe inverse transform.
• The checkpoint corresponds to this exact model architecture (two-layer LSTM encoder
  and decoder with a linear projection head).

Typical Usage
-------------
$ python 06_inferSequence.py --config configs/inferSequence.yml


# ===========================
# YAML Configuration Guide — 06_inferSequence.py
# ===========================
# Keys for running Seq2Seq LSTM *inference* on triangular maps.
# Paste this block at the top of inferSequence.yml (no example config below).

# --- data.measured ---
# data.measured.path    (str | Path): A single .npy file OR a folder of .npy files (shape: N×T×H×W).
# data.measured.pattern (str): Glob used only when "path" is a folder (e.g., "*.npy").

# --- data.united ---
# data.united.path      (str | Path): Ground-truth stack .npy (N×T_full×H×W); t=0 is used for initialization.

# --- data: preprocessing switches ---
# data.use_diff             (bool): Use Δt differencing for inputs/targets.
# data.use_normalization    (bool): Apply per-time-step min–max scaling using normalization_factors.npz.
# data.normalization_factors(str | Path): .npz file OR directory containing it (auto-picks preferred/latest).
# data.mean_centered        (bool): Subtract encoder-input means and add y-means back to predictions.
# data.mean_values          (str | Path): .npz file OR directory containing "mean_values.npz".
# data.use_time_context     (bool): Append a normalized time scalar [0,1] to each encoder step.
# data.early                (bool): Keep only the first K steps (set K via early_steps).
# data.early_steps          (int): Number of early steps to keep when early=true.
# data.sparse_step          (int): Temporal stride (1 = no subsampling).
# data.choose_index         (bool): Restrict to a subset of series given by index_list.
# data.index_list           ([int]): Series indices to keep (used when choose_index=true).

# --- runtime ---
# runtime.device            ("auto"|"cpu"|"cuda"): Compute device (auto prefers CUDA if available).
# runtime.time_steps        (int): Encoder input length (e.g., 30).
# runtime.output_seq_length (int): Number of steps to predict (e.g., 29).

# --- model ---
# model.hidden_size         (int): LSTM hidden size (default 512).
# model.num_layers          (int): Placeholder (implementation uses 2 stacked LSTMs).
# model.dropout             (float): Placeholder (not used internally).
# model.bidirectional       (bool): Placeholder (model is unidirectional here).
# model.checkpoint          (str | Path): Checkpoint file OR folder (prefers "*best*.pt/pth/ckpt", else newest).

# --- io ---
# io.output_dir             (str | Path): Folder to save per-input prediction arrays (.npy).
# io.out_npy_name           (str): Suffix for saved filename: "<measured_stem>__<out_npy_name>".
# io.png_dir                (str | Path): Folder to save quick-look PNGs.
# io.save_png               (bool): Save PNGs for each series (sum over time, t0, last).

# --- notes ---
# • Inputs/outputs are treated as triangular frames with row sizes [29,26,23,...,2] (step −3).
# • Units: the loader safely inverts to σ or ρ via 1/x with clipping; NaN/Inf are sanitized.
# • When directories are provided for .npz/ckpt, the script auto-resolves preferred or latest files.
"""


from __future__ import annotations
import argparse, os, time, glob, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml


# =====================
# Helper functions to find files
# =====================
def _resolve_npz(path_or_dir: Path,
                 prefer_names=("normalization_factors.npz", "mean_values.npz")) -> Path:
    """
    Resolve .npz file from a given path or directory:
    - If a file path is given, return it directly.
    - If a directory is given:
        1) Look for preferred filenames (in `prefer_names`).
        2) If not found, use the most recently modified .npz file in the directory.
    """
    if path_or_dir.is_file():
        return path_or_dir
    if path_or_dir.is_dir():
        # Check for preferred filenames first
        for name in prefer_names:
            p = path_or_dir / name
            if p.exists() and p.is_file():
                return p
        # Otherwise, select the latest .npz file
        cands = [Path(p) for p in glob.glob(str(path_or_dir / "*.npz"))]
        if not cands:
            raise FileNotFoundError(f"No .npz found in: {path_or_dir}")
        return max(cands, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"Path not found: {path_or_dir}")


def _resolve_checkpoint(ckpt_path_or_dir: Path, pattern: str = r".*\.(pt|pth|ckpt)$") -> Path:
    """
    Resolve a model checkpoint path:
    - If a file path is given, return it directly.
    - If a directory is given:
        1) Prefer files containing 'best' in the name.
        2) Otherwise, pick the most recent file with a valid extension (.pt, .pth, .ckpt).
    """
    if ckpt_path_or_dir.is_file():
        return ckpt_path_or_dir

    if ckpt_path_or_dir.is_dir():
        candidates = [Path(p) for p in glob.glob(str(ckpt_path_or_dir / "*"))]
        rx = re.compile(pattern, re.IGNORECASE)
        candidates = [p for p in candidates if p.is_file() and rx.match(p.name)]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint files in: {ckpt_path_or_dir}")

        # Prefer 'best' checkpoints
        bests = [p for p in candidates if "best" in p.name.lower()]
        pick_from = bests if bests else candidates
        return max(pick_from, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path_or_dir}")


# =====================
# Triangular matrix utilities
# =====================
def create_array(data: np.ndarray) -> np.ndarray:
    """
    Flatten a triangular matrix into a 1D array (used for Wenner-type geometry encoding).
    """
    row_sizes = np.arange(29, 0, -3)
    filled = []
    for i, size in enumerate(row_sizes):
        filled.extend(data[i, :size])
    return np.array(filled, dtype=np.float32)


def de_create_array(flat: np.ndarray) -> np.ndarray:
    """
    Restore a flattened triangular vector back to its 2D matrix form.
    """
    row_sizes = np.arange(29, 0, -3)
    max_row = row_sizes[0]
    mat = np.zeros((len(row_sizes), max_row), dtype=np.float32)
    start = 0
    for i, size in enumerate(row_sizes):
        end = start + size
        mat[i, :size] = flat[start:end]
        start = end
    return mat


# =====================
# LSTM Model Definitions
# =====================
class Encoder(nn.Module):
    """Two-layer LSTM encoder."""
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, (h, c) = self.lstm2(out)
        return (h, c)


class Decoder(nn.Module):
    """Two-layer LSTM decoder with a linear projection layer."""
    def __init__(self, out_dim, hidden):
        super().__init__()
        self.lstm1 = nn.LSTM(out_dim, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, dec_in, state1=None, state2=None):
        out1, state1 = self.lstm1(dec_in, state1)
        out2, state2 = self.lstm2(out1, state2)
        y_hat = self.proj(out2)
        return y_hat, state1, state2


class Seq2Seq(nn.Module):
    """Full Seq2Seq LSTM model for time-sequential field prediction."""
    def __init__(self, input_dim, output_dim, hidden=512, num_layers=2, dropout=0.0, bidir=False):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden)
        self.decoder = Decoder(output_dim, hidden)
        self.hidden = hidden
        self.output_dim = output_dim

    def forward(self, src: torch.Tensor, tgt_len: int) -> torch.Tensor:
        """Run autoregressive decoding for a given target length."""
        B = src.size(0)
        h_enc, c_enc = self.encoder(src)
        state1 = (h_enc, c_enc)
        state2 = None
        y_prev = torch.zeros(B, 1, self.output_dim, device=src.device, dtype=src.dtype)
        outs = []
        for _ in range(tgt_len):
            y_hat, state1, state2 = self.decoder(y_prev, state1, state2)
            outs.append(y_hat)
            y_prev = y_hat
        return torch.cat(outs, dim=1)  # (B, tgt_len, F)


# =====================
# Numerical Utilities
# =====================
def safe_inverse_k(arr, scale=1000.0, eps=1e-8, clip=1e6):
    """
    Compute inverse (1/x) safely for conductivity/resistivity conversions.
    Zeros and NaNs are handled gracefully, avoiding divide-by-zero errors.
    """
    a = np.array(arr, dtype=np.float64, copy=False)
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(scale, a, out=out, where=np.abs(a) > eps)
    out = np.nan_to_num(out, nan=0.0, posinf=clip, neginf=-clip)
    np.clip(out, -clip, clip, out=out)
    return out.astype(np.float32, copy=False)


# =====================
# General I/O helpers
# =====================
def _resolve_device(opt: str):
    """Return CUDA if available (auto mode), otherwise CPU."""
    opt = (opt or "auto").lower()
    if opt == "cpu":  return torch.device("cpu")
    if opt == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(p: Path):
    """Create a directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def save_png_images(out_dir: Path, series: int, imgs: np.ndarray, vmin=None, vmax=None):
    """Save a few representative images (sum over time, t0, last frame) for visualization."""
    _ensure_dir(out_dir)
    # sum over time
    fig = plt.figure()
    plt.imshow(imgs.sum(axis=0), aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(); plt.title(f"Series {series} - Sum over time")
    fig.savefig(out_dir / f"series{series:03d}_sum.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # first frame
    fig = plt.figure()
    plt.imshow(imgs[0], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(); plt.title(f"Series {series} - t0")
    fig.savefig(out_dir / f"series{series:03d}_t0.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # last frame
    fig = plt.figure()
    plt.imshow(imgs[-1], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(); plt.title(f"Series {series} - t{imgs.shape[0]-1}")
    fig.savefig(out_dir / f"series{series:03d}_tLast.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# =====================
# Data loading and preprocessing
# =====================
def load_pair(measured_file: Path, united_file: Path, dcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Load both measured and true ("united") datasets,
    apply optional preprocessing such as early trimming, sub-sampling, differencing, or normalization.
    """
    raw_measured = np.load(measured_file)
    input_data = safe_inverse_k(raw_measured, scale=1000.0)

    united = np.load(united_file)  # (Nseries, T_full, rows, cols)
    initial = safe_inverse_k(united[:, 0, :, :], scale=1000.0)  # first timestep as initial state
    initial = np.expand_dims(initial, axis=1)
    input_data = np.concatenate((initial, input_data), axis=1)
    output_data = safe_inverse_k(united, scale=1000.0)

    # Optional preprocessing
    if dcfg.get("early", False):
        k = int(dcfg.get("early_steps", input_data.shape[1]))
        input_data = input_data[:, :k, :, :]
        output_data = output_data[:, :k, :, :]

    if dcfg.get("choose_index", False):
        idx = list(map(int, dcfg.get("index_list", [])))
        if idx:
            input_data = np.array([input_data[x] for x in idx])
            output_data = np.array([output_data[x] for x in idx])

    s = int(dcfg.get("sparse_step", 1))
    if s > 1:
        input_data = input_data[:, ::s, :, :]
        output_data = output_data[:, ::s, :, :]

    if dcfg.get("use_diff", False):
        input_data = np.diff(input_data, axis=1)
        output_data = np.diff(output_data, axis=1)

    if dcfg.get("use_normalization", False):
        nf_raw = dcfg.get("normalization_factors", "normalization_factors.npz")
        nf_path = _resolve_npz(Path(nf_raw), prefer_names=("normalization_factors.npz",))
        norm = np.load(nf_path)
        tmin, tmax = norm["time_step_min"], norm["time_step_max"]
        input_data = (input_data - tmin) / (tmax - tmin)

    return input_data.astype(np.float32), output_data.astype(np.float32)


def apply_mean_centering(x_seq_2d: np.ndarray, dcfg: dict):
    """Optionally subtract mean values stored in mean_values.npz."""
    if dcfg.get("mean_centered", False):
        mv_raw = dcfg.get("mean_values", "mean_values.npz")
        mv_path = _resolve_npz(Path(mv_raw), prefer_names=("mean_values.npz",))
        data = np.load(mv_path)
        Xmean, ymean = data["Xmean"], data["ymean"]
        return x_seq_2d - Xmean, Xmean, ymean
    return x_seq_2d, None, None


# =====================
# Inference for a single measured file
# =====================
def run_inference_one(measured_file: Path, cfg: dict):
    """
    Run inference for one measured data file using the provided configuration.
    """
    model_cfg = cfg["model"]
    run_cfg = cfg["runtime"]
    dcfg = cfg["data"]
    io = cfg["io"]

    device = _resolve_device(run_cfg.get("device", "auto"))
    time_steps = int(run_cfg.get("time_steps", 30))
    out_len = int(run_cfg.get("output_seq_length", 29))

    # I/O setup
    output_dir = Path(io.get("output_dir", "outputs/use_model"))
    png_dir = Path(io.get("png_dir", str(output_dir / "pred_png")))
    save_png = bool(io.get("save_png", True))
    out_name = io.get("out_npy_name", "outputs_pred_all_series.npy")
    united_path = Path(dcfg["united"]["path"])

    # Load data
    input_data, output_data = load_pair(measured_file, united_path, dcfg)

    # Determine feature dimensions
    feat_dim = create_array(input_data[0, 0]).shape[0]
    use_time_ctx = bool(dcfg.get("use_time_context", False))
    in_dim = feat_dim + (1 if use_time_ctx else 0)

    # Build and load model
    model = Seq2Seq(
        input_dim=in_dim,
        output_dim=feat_dim,
        hidden=int(model_cfg.get("hidden_size", 512)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        bidir=bool(model_cfg.get("bidirectional", False)),
    ).to(device)

    ckpt_cfg = model_cfg.get("checkpoint", "best_model.pt")
    ckpt = _resolve_checkpoint(Path(ckpt_cfg))
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    print(f"[info] Loaded checkpoint: {ckpt}")

    model.eval()
    n_series = len(input_data)
    all_outputs = []

    # Inference loop over all series
    for series in range(n_series):
        enc_steps = []
        dec_targets = []
        if use_time_ctx:
            for ts in range(time_steps):
                flat = create_array(input_data[series, ts])
                time_ctx = np.full_like(flat, ts / float(input_data.shape[1] - 1), dtype=np.float32)
                enc_steps.append(np.concatenate([flat, time_ctx], axis=0))
            for ts in range(1, time_steps):
                dec_targets.append(create_array(output_data[series, ts]))
        else:
            for ts in range(time_steps):
                enc_steps.append(create_array(input_data[series, ts]))
            for ts in range(1, time_steps):
                dec_targets.append(create_array(output_data[series, ts]))

        enc_np = np.stack(enc_steps, axis=0)
        tgt_np = np.stack(dec_targets, axis=0)
        enc_np_mc, Xmean, ymean = apply_mean_centering(enc_np, dcfg)

        src = torch.from_numpy(enc_np_mc).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(src, tgt_len=out_len)
            pred_np = pred.squeeze(0).cpu().numpy()

        if dcfg.get("mean_centered", False) and ymean is not None:
            if ymean.shape != pred_np.shape:
                raise ValueError(f"ymean shape {ymean.shape} != predictions {pred_np.shape}")
            pred_np = pred_np + ymean

        # Convert back to images
        output_imgs = np.stack([de_create_array(pred_np[t]) for t in range(out_len)], axis=0).astype(np.float32)
        all_outputs.append(output_imgs)

        if save_png:
            save_png_images(png_dir, series, output_imgs)

        out_sum = output_imgs.sum(axis=0)
        print(f"[summary] {measured_file.name} | series={series}  sum(min/max)=({out_sum.min():.4g}/{out_sum.max():.4g})  shape={output_imgs.shape}")

    # Save combined outputs
    all_outputs = np.stack(all_outputs, axis=0)
    stem = measured_file.stem
    outpath = output_dir / f"{stem}__{out_name}"
    _ensure_dir(output_dir)
    np.save(outpath, all_outputs)
    print(f"[save] {outpath}  shape={all_outputs.shape}")


# =====================
# Entry point
# =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML configuration file.")
    args = ap.parse_args()

    t0 = time.time()
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Determine measured file(s)
    mcfg = cfg["data"]["measured"]
    mpath = Path(mcfg["path"])
    if mpath.is_dir():
        pat = mcfg.get("pattern", "*.npy")
        measured_files = [Path(p) for p in sorted(glob.glob(str(mpath / pat)))]
        if not measured_files:
            raise FileNotFoundError(f"No measured files matched: {mpath}/{pat}")
        print(f"[run] measured dir: {mpath}  count={len(measured_files)}")
    else:
        if not mpath.exists():
            raise FileNotFoundError(f"Measured file not found: {mpath}")
        measured_files = [mpath]
        print(f"[run] measured file: {mpath}")

    # Run inference for each measured file
    for mf in measured_files:
        print(f"[run] start: {mf.name}")
        run_inference_one(mf, cfg)
        print(f"[run] done : {mf.name}")

    print(f"[time] elapsed: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
