# -*- coding: utf-8 -*-
"""
PyTorch rewrite of:
04_trainingAppResAppRes_seq2seq_temporalDerivative_exceptFirst_simpler_long2.py
- 同等のデータ前処理（early/chooseIndex/sparce/diff/normalization/meanCentered）
- Encoder: LSTM(512)×2（1層目 return_sequences、2層目 state 抽出）
- Decoder: LSTM(512)×2 + Linear（TimeDistributed 相当）
- 教師強制: decoder_input = [zeros(1step), y[:, :-1, :]]
- 40 epochs, batch_size=5, Adam, MSE
"""

import os
import math
import random
import numpy as np
from pathlib import Path
import csv

# ← matplotlib はインポートしない
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ====== 再現性 ======
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[info] device:", device)

# ====== ハイパラ/フラグ（元コード準拠） ======
early = True
chooseIndex = False
sparce = True
diff = True
timeContext = False     # 本実装では False を想定（True でも動くよう分岐あり）
normalization = False
meanCentered = True

time_steps = 30
output_seq_length = 29
enc_hidden = 512
dec_hidden = 512
epochs = 40
batch_size = 5
lr = 1e-3

# ====== ユーティリティ ======
def create_array(data_2d: np.ndarray) -> np.ndarray:
    """三角行列を 1 本に詰める（元コード）"""
    row_sizes = np.arange(29, 0, -3)
    filled_data = []
    for i, size in enumerate(row_sizes):
        filled_data.extend(data_2d[i, :size])
    return np.array(filled_data, dtype=np.float32)

def de_create_array(flat_data: np.ndarray) -> np.ndarray:
    """復元（ここでは学習には未使用だが互換のため残す）"""
    row_sizes = np.arange(29, 0, -3)
    max_row_size = row_sizes[0]
    matrix = np.zeros((len(row_sizes), max_row_size), dtype=np.float32)
    start_idx = 0
    for i, size in enumerate(row_sizes):
        end_idx = start_idx + size
        matrix[i, :size] = flat_data[start_idx:end_idx]
        start_idx = end_idx
    return matrix

# ====== データ読み込み & 前処理（元コード忠実） ======
# 期待ファイル:
# - measured_training_data_sameRowColSeq31.npy
# - united_triangular_matrices.npy
# A) 元配列を読み込み（float32化）
_measured_raw = np.load('measured_training_data_sameRowColSeq31.npy').astype(np.float32)
_united_raw   = np.load('united_triangular_matrices.npy').astype(np.float32)

# B) ゼロ割りを避けて安全に「1000/x」を計算（abs(x) <= eps は 0 とみなす）
eps = 1e-6

def safe_inverse_k(x):
    out = np.empty_like(x, dtype=np.float32)
    np.divide(1000.0, x, out=out, where=np.abs(x) > eps)
    out[np.abs(x) <= eps] = 0.0
    # ∞/NaN を明示的に無害化（posinf/neginf を必ず指定）
    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    # 外れ値の安全クリップ（範囲は必要に応じて調整）
    out = np.clip(out, -1e6, 1e6)
    return out

measured = safe_inverse_k(_measured_raw)             # (N, T, H, W)
united   = safe_inverse_k(_united_raw)               # (N, T, H, W)

# 初期データ（t=0 の united を逆数化）
initial_data = safe_inverse_k(_united_raw[:, 0:1, :, :])  # (N,1,H,W)

# C) 入出力テンソルの構成（NaN/∞ は既に除去済みだが念のため二重化OK）
input_data  = np.concatenate((initial_data, measured), axis=1).astype(np.float32)
input_data  = np.nan_to_num(input_data,  nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
output_data = np.nan_to_num(united,      nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

if early:
    input_data = input_data[:, :310, :, :]
    output_data = output_data[:, :310, :, :]

if chooseIndex:
    index = [26, 37, 31, 19, 36, 28, 38, 18, 15]
    input_data  = np.array([input_data[x, :, :, :] for x in index])
    output_data = np.array([output_data[x, :, :, :] for x in index])

if sparce:
    input_data  = input_data[:, ::10, :, :]
    output_data = output_data[:, ::10, :, :]

if diff:
    input_data  = np.diff(input_data,  axis=1)
    output_data = np.diff(output_data, axis=1)

if normalization:
    # 入力
    tmin_in = np.min(input_data, axis=(0, 2, 3), keepdims=True)
    tmax_in = np.max(input_data, axis=(0, 2, 3), keepdims=True)
    input_data = (input_data - tmin_in) / (tmax_in - tmin_in + 1e-12)

    # 出力
    tmin_out = np.min(output_data, axis=(0, 2, 3), keepdims=True)
    tmax_out = np.max(output_data, axis=(0, 2, 3), keepdims=True)
    output_data = (output_data - tmin_out) / (tmax_out - tmin_out + 1e-12)

    np.savez('normalization_factors.npz', time_step_min=tmin_in, time_step_max=tmax_in)
    np.savez('normalization_factors_output.npz', time_step_min_output=tmin_out, time_step_max_output=tmax_out)

# ---- (X, y) を作成（timeContext 無しが標準） ----
X_list, y_list = [], []

if timeContext:
    # オプション: 時間コンテキストを特徴に連結（Keras 版相当）
    T_total = input_data.shape[1]
    for series in range(input_data.shape[0]):
        for i in range(T_total - time_steps + 1):
            inp_seq = []
            out_seq = []
            for ts in range(i, i + time_steps):
                resist_flat = create_array(input_data[series, ts, :, :])
                time_ctx = np.full_like(resist_flat, fill_value=ts / (T_total - 1), dtype=np.float32)
                inp_seq.append(np.concatenate([resist_flat, time_ctx], axis=0))
            for ts in range(i + (time_steps - output_seq_length), i + time_steps):
                resist_flat = create_array(output_data[series, ts, :, :])
                out_seq.append(resist_flat)
            X_list.append(np.stack(inp_seq, axis=0))
            y_list.append(np.stack(out_seq, axis=0))
else:
    # 通常: 時間コンテキストなし
    T_total = input_data.shape[1]
    for series in range(input_data.shape[0]):
        for i in range(T_total - time_steps + 1):
            inp_seq = []
            out_seq = []
            for ts in range(i, i + time_steps):
                resist_flat = create_array(input_data[series, ts, :, :])
                inp_seq.append(resist_flat)
            for ts in range(i + (time_steps - output_seq_length), i + time_steps):
                resist_flat = create_array(output_data[series, ts, :, :])
                out_seq.append(resist_flat)
            X_list.append(np.stack(inp_seq, axis=0))
            y_list.append(np.stack(out_seq, axis=0))

X = np.asarray(X_list, dtype=np.float32)   # (N, 30, Fin)
y = np.asarray(y_list, dtype=np.float32)   # (N, 29, Fout)

# 平均中心化（時刻毎のベクトル平均を引く：元コードと等価）
if meanCentered:
    Xmean = []
    ymean = []
    for t in range(X.shape[1]):
        Xmean.append(np.mean(X[:, t, :], axis=0))
    for t in range(y.shape[1]):
        ymean.append(np.mean(y[:, t, :], axis=0))
    Xmean = np.asarray(Xmean, dtype=np.float32)  # (30, Fin)
    ymean = np.asarray(ymean, dtype=np.float32)  # (29, Fout)
    X = X - Xmean[None, :, :]
    y = y - ymean[None, :, :]
    np.savez('mean_values.npz', Xmean=Xmean, ymean=ymean)

print("[shape] X:", X.shape, "y:", y.shape)

# ====== 教師強制用の decoder 入力（1step シフト, 先頭は0） ======
dec_in_train = np.concatenate([np.zeros_like(y[:, :1, :], dtype=np.float32), y[:, :-1, :]], axis=1)

# ====== Dataset / DataLoader ======
class Seq2SeqDataset(Dataset):
    def __init__(self, X, dec_in, y):
        super().__init__()
        self.X = X
        self.dec_in = dec_in
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),       # (30, Fin)
            torch.from_numpy(self.dec_in[idx]),  # (29, Fout)
            torch.from_numpy(self.y[idx])        # (29, Fout)
        )

# train/test split（元コード: test_size=0.2, random_state=42）
N = X.shape[0]
n_test = int(math.ceil(N * 0.2))
n_train = N - n_test
dataset = Seq2SeqDataset(X, dec_in_train, y)
train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

pin = (device.type == "cuda")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, pin_memory=pin)

# ====== モデル定義 ======
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
    def forward(self, x):
        # x: (B, 30, Fin)
        out1, _ = self.lstm1(x)                     # (B, 30, H)
        out2, (h, c) = self.lstm2(out1)             # (B, 30, H) & (1,B,H)
        return (h, c)                               # 返すのは最終 state（Keras 相当）

class Decoder(nn.Module):
    def __init__(self, out_dim, hidden=512):
        super().__init__()
        # 1段目は state を受ける
        self.lstm1 = nn.LSTM(input_size=out_dim, hidden_size=hidden, batch_first=True)
        # 2段目はシーケンスをさらに通す
        self.lstm2 = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.proj  = nn.Linear(hidden, out_dim)
    def forward(self, y_in, h0, c0):
        # y_in: (B, 29, Fout)  教師強制入力
        out1, _ = self.lstm1(y_in, (h0, c0))        # (B, 29, H)
        out2, _ = self.lstm2(out1)                  # (B, 29, H)
        y_hat = self.proj(out2)                     # (B, 29, Fout)
        return y_hat

class Seq2Seq(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden)
        self.decoder = Decoder(out_dim, hidden)
    def forward(self, x_enc, y_dec_in):
        h, c = self.encoder(x_enc)
        y_hat = self.decoder(y_dec_in, h, c)
        return y_hat

Fin  = X.shape[2]
Fout = y.shape[2]
model = Seq2Seq(in_dim=Fin, out_dim=Fout, hidden=enc_hidden).to(device)

crit = nn.MSELoss()
opt  = torch.optim.Adam(model.parameters(), lr=lr)

# ====== 学習ループ ======
train_losses, val_losses = [], []
best_vloss = float("inf")

for ep in range(1, epochs + 1):
    model.train()
    tloss = 0.0
    for x_enc, y_dec_in, y_gt in train_loader:
        x_enc   = x_enc.to(device)
        y_dec_in= y_dec_in.to(device)
        y_gt    = y_gt.to(device)

        opt.zero_grad()
        y_hat = model(x_enc, y_dec_in)
        loss = crit(y_hat, y_gt)
        loss.backward()
        opt.step()
        tloss += loss.item() * x_enc.size(0)

    tloss /= len(train_loader.dataset)

    model.eval()
    vloss = 0.0
    with torch.no_grad():
        for x_enc, y_dec_in, y_gt in test_loader:
            x_enc   = x_enc.to(device)
            y_dec_in= y_dec_in.to(device)
            y_gt    = y_gt.to(device)
            y_hat = model(x_enc, y_dec_in)
            loss = crit(y_hat, y_gt)
            vloss += loss.item() * x_enc.size(0)
    vloss /= len(test_loader.dataset)

    train_losses.append(tloss)
    val_losses.append(vloss)
    print(f"[epoch {ep:02d}] train_loss={tloss:.6f}  val_loss={vloss:.6f}")

    if vloss < best_vloss:
        best_vloss = vloss
        torch.save(model.state_dict(), "best_model.pt")

        print(f"[save] best_model.pt updated (val_loss={vloss:.6f})")

torch.save({
    "epoch": epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    "val_loss": val_losses[-1] if len(val_losses) else float("nan"),
}, "last_model.pt")
print("[save] last_model.pt saved")


def save_loss_curve_png_pillow(train_losses, val_losses, out_path="loss_curve_pytorch.png",
                               size=(1200, 480), margin=60):
    """matplotlib なしでロス曲線PNGを作る簡易関数（Pillow使用）"""
    if not _HAS_PIL:
        print("[warn] Pillow が見つかりませんでした。PNG は作らず CSV のみ保存します。")
        return

    W, H = size
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)

    # 軸エリア
    x0, y0 = margin, margin
    x1, y1 = W - margin, H - margin

    # データ範囲
    xs = list(range(1, max(len(train_losses), len(val_losses)) + 1))
    all_vals = []
    if len(train_losses): all_vals += list(train_losses)
    if len(val_losses):   all_vals += list(val_losses)
    if not all_vals:
        all_vals = [0.0, 1.0]
    vmin = float(min(all_vals))
    vmax = float(max(all_vals))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    def to_xy(i, v):
        # i: 1..N -> X in [x0,x1], v -> Y in [y0,y1] (下が大きくなるので反転)
        n = len(xs)
        t = (i - 1) / max(1, n - 1)
        X = x0 + t * (x1 - x0)
        Y = y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)
        return (X, Y)

    # 軸
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)

    # 目盛り（簡易）
    for k in range(6):
        ty = y1 - k * (y1 - y0) / 5
        draw.line([(x0, ty), (x1, ty)], fill=(230, 230, 230), width=1)

    # 折れ線（Train）
    if len(train_losses) >= 2:
        pts = [to_xy(i, v) for i, v in zip(xs[:len(train_losses)], train_losses)]
        draw.line(pts, fill=(30, 144, 255), width=3)  # blue-ish

    # 折れ線（Val）
    if len(val_losses) >= 2:
        pts = [to_xy(i, v) for i, v in zip(xs[:len(val_losses)], val_losses)]
        draw.line(pts, fill=(220, 20, 60), width=3)  # red-ish

    # 注記
    try:
        # フォントは環境依存なので失敗しても無視
        font = ImageFont.load_default()
        draw.text((x0, y0 - 20), "Loss", fill=(0, 0, 0), font=font)
        draw.text((x1 - 80, y1 + 8), "Epoch", fill=(0, 0, 0), font=font)
        legend_y = y0 + 8
        draw.rectangle([x1 - 180, legend_y - 8, x1 - 20, legend_y + 40], outline=(0,0,0))
        draw.line([(x1 - 170, legend_y), (x1 - 130, legend_y)], fill=(30,144,255), width=3)
        draw.text((x1 - 125, legend_y - 8), "Train", fill=(0, 0, 0), font=font)
        draw.line([(x1 - 170, legend_y + 22), (x1 - 130, legend_y + 22)], fill=(220,20,60), width=3)
        draw.text((x1 - 125, legend_y + 14), "Val", fill=(0, 0, 0), font=font)
    except Exception:
        pass

    img.save(out_path)
    print(f"[done] saved {out_path}")

# ====== （学習ループの後）可視化・保存 ======
# CSV も保存（後から自由にプロット可能）
with open("loss_history.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "val_loss"])
    for i in range(max(len(train_losses), len(val_losses))):
        tr = train_losses[i] if i < len(train_losses) else ""
        vl = val_losses[i] if i < len(val_losses) else ""
        w.writerow([i + 1, tr, vl])
print("[done] saved loss_history.csv")

# PNG（Pillow）を保存
save_loss_curve_png_pillow(train_losses, val_losses, out_path="loss_curve_pytorch.png")
