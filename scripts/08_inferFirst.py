# -*- coding: utf-8 -*-
"""
Script Name: 08_inferFirst.py

This script performs inference for the first-step prediction task using a trained LSTM model.
It was refactored from `08_useModelFirstImage.py` with the following improvements:
- Configuration is loaded from a YAML file.
- Input and output are directory-based (not hard-coded file paths).
- Preprocessing and network architecture are consistent with training.

The model predicts a 2D conductivity (or resistivity) map for the first time step
based on a sequence of input maps. Visualization and saving options are also configurable.
"""

import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ==============================================================
# Helper functions for handling triangular matrix flattening
# ==============================================================

def create_array(data: np.ndarray) -> np.ndarray:
    """Flatten a triangular matrix with row sizes [29, 26, 23, ..., 2, 1]."""
    row_sizes = np.arange(29, 0, -3)
    flattened = []
    for i, size in enumerate(row_sizes):
        flattened.extend(data[i, :size])
    return np.array(flattened, dtype=np.float32)


def de_create_array(flat_data: np.ndarray) -> np.ndarray:
    """Reconstruct a triangular matrix from its flattened representation."""
    row_sizes = np.arange(29, 0, -3)
    max_row_size = row_sizes[0]
    mat = np.zeros((len(row_sizes), max_row_size), dtype=np.float32)
    start = 0
    for i, size in enumerate(row_sizes):
        end = start + size
        mat[i, :size] = flat_data[start:end]
        start = end
    return mat


# ==============================================================
# LSTM Regression Model (must match the one used in training)
# ==============================================================

class LSTMRegressor(nn.Module):
    """A simple LSTM-based regressor for sequential-to-image prediction."""
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 512, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        last = out[:, -1, :]           # take last time step (B, H)
        z = self.relu(self.fc1(last))
        z = self.relu(self.fc2(z))
        y = self.out(z)                # (B, output_dim)
        return y


# ==============================================================
# Utility: Load configuration file
# ==============================================================

def load_yaml(path):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================
# Main entry point
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="LSTM first-step inference script.")
    parser.add_argument("--config", "-c", required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # -----------------------------
    # === General settings ===
    # -----------------------------
    ckpt_path = cfg["model"]["checkpoint"]
    device_opt = cfg["model"].get("device", "auto")
    override_time_steps = cfg["model"].get("override_time_steps", None)

    measured_path = os.path.join(cfg["data"]["measured_dir"], cfg["data"]["measured_file"])
    truth_path    = os.path.join(cfg["data"]["truth_dir"],    cfg["data"]["truth_file"])

    invert_values = bool(cfg["data"].get("invert_values", True))
    unit_scale    = float(cfg["data"].get("unit_scale", 1000.0))
    prepend_init  = bool(cfg["data"].get("prepend_initial_truth", True))
    nan_fill_val  = float(cfg["data"].get("nan_fill_value", 0.0))

    # Normalization settings
    norm_cfg = cfg.get("norm", {})
    norm_enabled = bool(norm_cfg.get("enabled", False))
    norm_dir = norm_cfg.get("dir", ".")
    norm_fac_file = norm_cfg.get("normalization_factors", "normalization_factors_first.npz")
    mean_vals_file = norm_cfg.get("mean_values", "mean_values_first.npz")

    # Optional preprocessing flags
    flags_cfg = cfg.get("flags", {})
    early_override        = flags_cfg.get("early", None)
    early_max_T           = int(flags_cfg.get("early_max_T", 50))
    choose_idx_enabled    = flags_cfg.get("choose_index_enabled", None)
    choose_indices        = flags_cfg.get("choose_indices", [26,37,31,19,36,28,38,18,15])
    sparse_enabled        = flags_cfg.get("sparse_enabled", None)
    sparse_stride         = int(flags_cfg.get("sparse_stride", 10))
    diff_enabled          = flags_cfg.get("diff_enabled", None)
    time_context_enabled  = flags_cfg.get("time_context_enabled", None)
    mean_centered_enabled = flags_cfg.get("mean_centered_enabled", None)

    # I/O options
    io_cfg = cfg.get("io", {})
    out_dir = io_cfg.get("output_dir", "outputs/infer_first_image")
    os.makedirs(out_dir, exist_ok=True)
    save_figures = bool(io_cfg.get("save_figures", True))
    save_pred_stack = bool(io_cfg.get("save_pred_stack", True))
    cmap = io_cfg.get("cmap", "viridis")

    # Visualization options
    vis_cfg = cfg.get("visual", {})
    save_compare = bool(vis_cfg.get("save_compare", True))
    save_input_frames = bool(vis_cfg.get("save_input_frames", True))

    # -----------------------------
    # === Load checkpoint ===
    # -----------------------------
    payload = torch.load(ckpt_path, map_location="cpu")
    flags = payload.get("flags", {})
    time_steps = int(payload.get("time_steps", 4)) if override_time_steps in (None, "") else int(override_time_steps)
    input_dim = int(payload["input_dim"])
    output_dim = int(payload["output_dim"])

    # Resolve preprocessing flags (training defaults can be overridden)
    early         = bool(flags.get("early", True))        if early_override is None        else bool(early_override)
    chooseIndex   = bool(flags.get("chooseIndex", False)) if choose_idx_enabled is None    else bool(choose_idx_enabled)
    sparse        = bool(flags.get("sparce", True))       if sparse_enabled is None        else bool(sparse_enabled)
    diff          = bool(flags.get("diff", True))         if diff_enabled is None          else bool(diff_enabled)
    timeContext   = bool(flags.get("timeContext", False)) if time_context_enabled is None  else bool(time_context_enabled)

    # Handle name variation for mean-centered flag
    payload_mean_centered = bool(payload.get("mean_centered", flags.get("meanCentered", True)))
    meanCentered = payload_mean_centered if mean_centered_enabled is None else bool(mean_centered_enabled)

    print("[flags]", "early=", early, "chooseIndex=", chooseIndex, "sparse=", sparse,
          "diff=", diff, "timeContext=", timeContext, "meanCentered=", meanCentered)
    print("[meta]", "time_steps=", time_steps, "input_dim=", input_dim, "output_dim=", output_dim)

    # -----------------------------
    # === Build model ===
    # -----------------------------
    if device_opt == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_opt == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = LSTMRegressor(input_dim=input_dim, output_dim=output_dim, hidden_size=512, num_layers=2)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    # -----------------------------
    # === Load and preprocess data ===
    # -----------------------------
    measured = np.load(measured_path)      # (N, T-1, H, W)
    united   = np.load(truth_path)         # (N, T,   H, W)

    # Optional inversion (1/value) and unit scaling
    if invert_values:
        measured = 1.0 / measured * unit_scale
        united   = 1.0 / united   * unit_scale

    # Replace NaNs for stability
    measured = np.nan_to_num(measured, nan=nan_fill_val)
    united   = np.nan_to_num(united,   nan=nan_fill_val)

    # Optionally prepend the initial true field (t=0)
    if prepend_init:
        initial_data = united[:, 0, :, :][:, None, :, :]   # (N, 1, H, W)
        input_data = np.concatenate([initial_data, measured], axis=1)
    else:
        input_data = measured

    output_data = united

    # Apply optional flags
    if early:
        input_data  = input_data[:, :early_max_T, :, :]
        output_data = output_data[:, :early_max_T, :, :]

    if chooseIndex:
        idx = choose_indices
        input_data  = np.array([input_data[x, :, :, :] for x in idx])
        output_data = np.array([output_data[x, :, :, :] for x in idx])

    if sparse:
        input_data  = input_data[:, ::sparse_stride, :, :]
        output_data = output_data[:, ::sparse_stride, :, :]

    if diff:
        input_data  = np.diff(input_data, axis=1)
        output_data = np.diff(output_data, axis=1)

    # -----------------------------
    # === Optional normalization ===
    # -----------------------------
    if norm_enabled:
        fac_npz = np.load(os.path.join(norm_dir, norm_fac_file))
        tmin, tmax = fac_npz['time_step_min'], fac_npz['time_step_max']
        input_data = (input_data - tmin) / (tmax - tmin + 1e-12)

    # Load mean values if mean-centered
    if meanCentered:
        mv = np.load(os.path.join(norm_dir, mean_vals_file))
        Xmean_loaded = mv['Xmean']  # (T, F[+1 if timeContext])
        ymean_loaded = mv['ymean']  # (T, F_out)

    # ==============================================================
    # Inference loop over all series
    # ==============================================================
    num_series = input_data.shape[0]
    print(f"[run] Total sequences: {num_series}")
    all_pred_imgs = []

    for series in range(num_series):
        # --- Build a single input sequence (length = time_steps) ---
        inp_seq = []
        if timeContext:
            for ts in range(time_steps):
                flat = create_array(input_data[series, ts, :, :])
                time_ctx = np.full_like(flat, ts / float(input_data.shape[1] - 1), dtype=np.float32)
                with_time = np.concatenate([flat, time_ctx], axis=0)
                inp_seq.append(with_time.astype(np.float32))
        else:
            for ts in range(time_steps):
                flat = create_array(input_data[series, ts, :, :]).astype(np.float32)
                inp_seq.append(flat)

        x = np.stack(inp_seq, axis=0).astype(np.float32)  # (T, F_in)

        # Mean centering
        if meanCentered:
            x = x - Xmean_loaded

        x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, T, F_in)

        with torch.no_grad():
            y_pred = model(x_t).cpu().numpy().squeeze(0)  # (F_out,)

        # Undo mean-centering on output
        if meanCentered:
            y_pred = y_pred + ymean_loaded[0]

        # Restore 2D triangular map
        pred_img = de_create_array(y_pred)
        true_img = de_create_array(create_array(output_data[series, 0, :, :]))  # first-step target

        all_pred_imgs.append(pred_img)

        # -----------------------------
        # Visualization and saving
        # -----------------------------
        if save_figures and save_compare:
            vmin = float(min(pred_img.min(), true_img.min()))
            vmax = float(max(pred_img.max(), true_img.max()))
            fig, axes = plt.subplots(1, 2, figsize=(12, 8))
            im0 = axes[0].imshow(pred_img, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[0].set_title(f"Predicted (Series {series})")
            axes[0].axis('off')

            axes[1].imshow(true_img, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1].set_title(f"True (Series {series})")
            axes[1].axis('off')

            cbar = fig.colorbar(im0, ax=axes, orientation='vertical', fraction=0.02, pad=0.1)
            cbar.set_label('Conductivity (mS/m)')
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"compare_series_{series}.png"), dpi=150)
            plt.close(fig)

        # Optionally visualize input sequence
        if save_figures and save_input_frames:
            fig2, axes2 = plt.subplots(1, time_steps, figsize=(3 * time_steps, 3))
            for i in range(time_steps):
                axes2[i].imshow(input_data[series, i, :, :], cmap=cmap)
                axes2[i].set_title(f"Input t{i}")
                axes2[i].axis('off')
            plt.tight_layout()
            fig2.savefig(os.path.join(out_dir, f"inputs_series_{series}.png"), dpi=150)
            plt.close(fig2)

    # Save all predicted images as one stack
    if save_pred_stack:
        all_pred_arr = np.stack(all_pred_imgs, axis=0)
        np.save(os.path.join(out_dir, "pred_images_all.npy"), all_pred_arr)

    print(f"Done. Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
