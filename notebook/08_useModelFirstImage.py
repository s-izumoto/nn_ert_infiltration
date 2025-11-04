# -*- coding: utf-8 -*-
"""
PyTorch inference for: 08_useModel_AppResAppRes_seq2seq_onlyFirst_loop.py
- Keeps the SAME data processing and network design as training.
- Loads 'single_output_lstm_model.pt' payload (state_dict + meta + flags).
- Predicts one output (the first target step) per series, like the original.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# Helper: triangular flattening
# -----------------------------
def create_array(data: np.ndarray) -> np.ndarray:
    """Flatten triangular rows in sizes 29, 26, 23, ..., 2, 1 (step -3)."""
    row_sizes = np.arange(29, 0, -3)
    filled = []
    for i, size in enumerate(row_sizes):
        filled.extend(data[i, :size])
    return np.array(filled, dtype=np.float32)

def de_create_array(flat_data: np.ndarray) -> np.ndarray:
    """Inverse of create_array (for visualization)."""
    row_sizes = np.arange(29, 0, -3)
    max_row_size = row_sizes[0]
    mat = np.zeros((len(row_sizes), max_row_size), dtype=np.float32)
    start = 0
    for i, size in enumerate(row_sizes):
        end = start + size
        mat[i, :size] = flat_data[start:end]
        start = end
    return mat

# -----------------------------
# Model (must match training)
# -----------------------------
class LSTMRegressor(nn.Module):
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
        # x: (B, T, F)
        out, _ = self.lstm(x)          # (B, T, H)
        last = out[:, -1, :]           # (B, H)
        z = self.relu(self.fc1(last))
        z = self.relu(self.fc2(z))
        y = self.out(z)                # (B, F_out)
        return y

# -----------------------------
# Load saved torch payload
# -----------------------------
ckpt_path = "single_output_lstm_model.pt"
payload = torch.load(ckpt_path, map_location="cpu")
flags = payload.get("flags", {})
time_steps = int(payload.get("time_steps", 4))
input_dim = int(payload["input_dim"])
output_dim = int(payload["output_dim"])

# Flags used at training (we honor these)
early          = bool(flags.get("early", True))
chooseIndex    = bool(flags.get("chooseIndex", False))
sparce         = bool(flags.get("sparce", True))
diff           = bool(flags.get("diff", True))
timeContext    = bool(flags.get("timeContext", False))
normalization  = bool(flags.get("normalization", False))
meanCentered   = bool(payload.get("mean_centered", True))

print("[flags] early=", early, " chooseIndex=", chooseIndex, " sparce=", sparce,
      " diff=", diff, " timeContext=", timeContext, " normalization=", normalization,
      " meanCentered=", meanCentered)
print("[meta] time_steps=", time_steps, " input_dim=", input_dim, " output_dim=", output_dim)

# Build model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegressor(input_dim=input_dim, output_dim=output_dim, hidden_size=512, num_layers=2)
model.load_state_dict(payload["model_state_dict"], strict=True)
model.to(device)
model.eval()

# -----------------------------
# Load data (same as training)
# -----------------------------
# input_data: measured + prepend initial true map at t0
input_data = 1.0 / np.load('measured_training_data_sameRowColSeq34_test.npy') * 1000.0
initial_data = 1.0 / np.load('united_triangular_matrices_test.npy')[:, 0, :, :] * 1000.0
initial_data = np.expand_dims(initial_data, axis=1)
input_data = np.concatenate((initial_data, input_data), axis=1)
input_data = np.nan_to_num(input_data, nan=0.0)

# output_data: true maps
output_data = 1.0 / np.load('united_triangular_matrices_test.npy') * 1000.0
output_data = np.nan_to_num(output_data, nan=0.0)

if early:
    input_data = input_data[:, :50, :, :]
    output_data = output_data[:, :50, :, :]

if chooseIndex:
    index = [26, 37, 31, 19, 36, 28, 38, 18, 15]
    input_data  = np.array([input_data[x, :, :, :] for x in index])
    output_data = np.array([output_data[x, :, :, :] for x in index])

if sparce:
    input_data = input_data[:, ::10, :, :]
    output_data = output_data[:, ::10, :, :]

if diff:
    input_data = np.diff(input_data, axis=1)
    output_data = np.diff(output_data, axis=1)

# Optional normalization (only if it was done at training)
if normalization:
    fac_in = np.load('normalization_factors_first.npz')
    tmin, tmax = fac_in['time_step_min'], fac_in['time_step_max']
    input_data = (input_data - tmin) / (tmax - tmin + 1e-12)

# Mean values (for meanCentered=True)
if meanCentered:
    mv = np.load('mean_values_first.npz')
    Xmean_loaded = mv['Xmean']  # shape: (T, F[+1 if timeContext])
    ymean_loaded = mv['ymean']  # shape: (T, F_out)

# -----------------------------
# Inference loop over series
# -----------------------------
os.makedirs("outputs_infer", exist_ok=True)

num_series = input_data.shape[0]
print(f"[run] series total: {num_series}")
all_pred_imgs = []

for series in range(num_series):
    # Build a single input sequence of length = time_steps
    inp_seq = []
    if timeContext:
        for ts in range(0, time_steps):
            flat = create_array(input_data[series, ts, :, :])  # (F)
            time_ctx = np.full_like(flat, ts / float(input_data.shape[1] - 1), dtype=np.float32)
            with_time = np.concatenate([flat, time_ctx], axis=0)  # (F+F)
            inp_seq.append(with_time.astype(np.float32))
    else:
        for ts in range(0, time_steps):
            flat = create_array(input_data[series, ts, :, :]).astype(np.float32)
            inp_seq.append(flat)

    x = np.stack(inp_seq, axis=0).astype(np.float32)  # (T, F_input)

    # Mean centering (subtract training means)
    if meanCentered:
        # Xmean_loaded has shape (T, F_input), broadcast OK
        x = x - Xmean_loaded

    # Make batch dimension and to torch
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, T, F_input)

    with torch.no_grad():
        y_pred = model(x_t).cpu().numpy().squeeze(0)  # (F_out,)

    # Undo mean-centering on output side (add back ymean for the first target step)
    if meanCentered:
        y_pred = y_pred + ymean_loaded[0]

    # Recompose to 2D triangular image
    pred_img = de_create_array(y_pred)
    true_img = de_create_array(create_array(output_data[series, 0, :, :]))  # true at target first step

    all_pred_imgs.append(pred_img)
    # Save .npy
    # np.save(f"outputs_infer/output_image_initial_series_{series}.npy", pred_img)

    # Plot side-by-side
    vmin = float(min(pred_img.min(), true_img.min()))
    vmax = float(max(pred_img.max(), true_img.max()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    im0 = axes[0].imshow(pred_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Predicted (series {series})")
    axes[0].axis('off')

    axes[1].imshow(true_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"True (series {series})")
    axes[1].axis('off')

    cbar = fig.colorbar(im0, ax=axes, orientation='vertical', fraction=0.02, pad=0.1)
    cbar.set_label('scale')
    plt.tight_layout()
    fig.savefig(f"outputs_infer/compare_series_{series}.png", dpi=150)
    plt.close(fig)

    # Also save the T input frames for quick inspection
    fig2, axes2 = plt.subplots(1, time_steps, figsize=(3 * time_steps, 3))
    for i in range(time_steps):
        axes2[i].imshow(input_data[series, i, :, :], cmap='viridis')
        axes2[i].set_title(f"in t{i}")
        axes2[i].axis('off')
    plt.tight_layout()
    fig2.savefig(f"outputs_infer/inputs_series_{series}.png", dpi=150)
    plt.close(fig2)

all_pred_arr = np.stack(all_pred_imgs, axis=0)  # 形状: (num_series, rows, cols)
np.save("outputs_infer/pred_images_all.npy", all_pred_arr)
print("Done. Saved outputs to ./outputs_infer/")
