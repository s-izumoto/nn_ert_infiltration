# -*- coding: utf-8 -*-
"""
YAML 設定で Seq2Seq LSTM の推論を実行。
- measured がファイル→その1件を処理
- measured がフォルダ→pattern にマッチした全 .npy を処理
- 出力はフォルダにまとめ、PNG も保存可能
元コード: 06_useModel.py を分割・汎用化
"""

from __future__ import annotations
import argparse, os, time, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import re

def _resolve_npz(path_or_dir: Path,
                 prefer_names=("normalization_factors.npz", "mean_values.npz")) -> Path:
    """
    - ファイルならそのまま返す
    - ディレクトリなら:
        1) prefer_names に一致するものがあればそれを優先（存在順で先に見つかった方）
        2) それ以外は *.npz の中で更新時刻が最新のもの
    """
    if path_or_dir.is_file():
        return path_or_dir
    if path_or_dir.is_dir():
        # 既定名優先
        for name in prefer_names:
            p = path_or_dir / name
            if p.exists() and p.is_file():
                return p
        # なければ *.npz 最新
        cands = [Path(p) for p in glob.glob(str(path_or_dir / "*.npz"))]
        if not cands:
            raise FileNotFoundError(f"No .npz found in: {path_or_dir}")
        return max(cands, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"Path not found: {path_or_dir}")

def _resolve_checkpoint(ckpt_path_or_dir: Path, pattern: str = r".*\.(pt|pth|ckpt)$") -> Path:
    """
    - ファイルが渡されたらそのまま返す
    - ディレクトリが渡されたら中から候補を選ぶ
        1) ファイル名に 'best' を含むものを優先（複数なら更新時刻が新しいもの）
        2) それ以外は拡張子 .pt/.pth/.ckpt の中で更新時刻が新しいもの
    """
    if ckpt_path_or_dir.is_file():
        return ckpt_path_or_dir

    if ckpt_path_or_dir.is_dir():
        # 拡張子フィルタ
        candidates = [Path(p) for p in glob.glob(str(ckpt_path_or_dir / "*"))]
        rx = re.compile(pattern, re.IGNORECASE)
        candidates = [p for p in candidates if p.is_file() and rx.match(p.name)]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint files in: {ckpt_path_or_dir}")

        # 'best' 優先
        bests = [p for p in candidates if "best" in p.name.lower()]
        pick_from = bests if bests else candidates
        pick = max(pick_from, key=lambda p: p.stat().st_mtime)  # 更新日時で最大
        return pick

    raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path_or_dir}")

# =====================
# Triangular (de)flatten
# =====================
def create_array(data: np.ndarray) -> np.ndarray:
    row_sizes = np.arange(29, 0, -3)
    filled = []
    for i, size in enumerate(row_sizes):
        filled.extend(data[i, :size])
    return np.array(filled, dtype=np.float32)

def de_create_array(flat: np.ndarray) -> np.ndarray:
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
# Model
# =====================
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, (h, c) = self.lstm2(out)
        return (h, c)

class Decoder(nn.Module):
    def __init__(self, out_dim, hidden):
        super().__init__()
        self.lstm1 = nn.LSTM(out_dim, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.proj  = nn.Linear(hidden, out_dim)
    def forward(self, dec_in, state1=None, state2=None):
        out1, state1 = self.lstm1(dec_in, state1)
        out2, state2 = self.lstm2(out1, state2)
        y_hat = self.proj(out2)
        return y_hat, state1, state2

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=512, num_layers=2, dropout=0.0, bidir=False):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden)
        self.decoder = Decoder(output_dim, hidden)
        self.hidden  = hidden
        self.output_dim = output_dim
    def forward(self, src: torch.Tensor, tgt_len: int) -> torch.Tensor:
        B = src.size(0)
        h_enc, c_enc = self.encoder(src)        # (1, B, H)
        state1 = (h_enc, c_enc)
        state2 = None
        y_prev = torch.zeros(B, 1, self.output_dim, device=src.device, dtype=src.dtype)
        outs = []
        for _ in range(tgt_len):
            y_hat, state1, state2 = self.decoder(y_prev, state1, state2)
            outs.append(y_hat)
            y_prev = y_hat
        return torch.cat(outs, dim=1)           # (B, tgt_len, F)

def safe_inverse_k(arr, scale=1000.0, eps=1e-8, clip=1e6):
    a = np.array(arr, dtype=np.float64, copy=False)
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(scale, a, out=out, where=np.abs(a) > eps)
    out = np.nan_to_num(out, nan=0.0, posinf=clip, neginf=-clip)
    np.clip(out, -clip, clip, out=out)
    return out.astype(np.float32, copy=False)

# =====================
# I/O helpers
# =====================
def _resolve_device(opt: str):
    opt = (opt or "auto").lower()
    if opt == "cpu":  return torch.device("cpu")
    if opt == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_png_images(out_dir: Path, series: int, imgs: np.ndarray, vmin=None, vmax=None):
    _ensure_dir(out_dir)
    # sum over time
    fig = plt.figure()
    plt.imshow(imgs.sum(axis=0), aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(); plt.title(f"Series {series} - Sum over time")
    fig.savefig(out_dir / f"series{series:03d}_sum.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    # first
    fig = plt.figure()
    plt.imshow(imgs[0], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(); plt.title(f"Series {series} - t0")
    fig.savefig(out_dir / f"series{series:03d}_t0.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    # last
    fig = plt.figure()
    plt.imshow(imgs[-1], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(); plt.title(f"Series {series} - t{imgs.shape[0]-1}")
    fig.savefig(out_dir / f"series{series:03d}_tLast.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

# =====================
# Data loading with YAML options
# =====================
def load_pair(measured_file: Path, united_file: Path, dcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    # measured: (Nseries, T?, rows, cols) あるいは既定形
    raw_measured = np.load(measured_file)
    input_data = safe_inverse_k(raw_measured, scale=1000.0)

    united = np.load(united_file)  # (Nseries, T_full, rows, cols)
    initial = safe_inverse_k(united[:, 0, :, :], scale=1000.0)  # 最初の時刻(真値)を初期値に
    initial = np.expand_dims(initial, axis=1)
    input_data = np.concatenate((initial, input_data), axis=1)
    output_data = safe_inverse_k(united, scale=1000.0)

    # --- 前処理フラグ適用 ---
    if dcfg.get("early", False):
        k = int(dcfg.get("early_steps", input_data.shape[1]))
        input_data  = input_data[:, :k, :, :]
        output_data = output_data[:, :k, :, :]

    if dcfg.get("choose_index", False):
        idx = list(map(int, dcfg.get("index_list", [])))
        if idx:
            input_data  = np.array([input_data[x]  for x in idx])
            output_data = np.array([output_data[x] for x in idx])

    s = int(dcfg.get("sparse_step", 1))
    if s and s > 1:
        input_data  = input_data[:, ::s, :, :]
        output_data = output_data[:, ::s, :, :]

    if dcfg.get("use_diff", False):
        input_data  = np.diff(input_data,  axis=1)
        output_data = np.diff(output_data, axis=1)

    if dcfg.get("use_normalization", False):
        nf_raw = dcfg.get("normalization_factors", "normalization_factors.npz")
        nf_path = _resolve_npz(Path(nf_raw), prefer_names=("normalization_factors.npz",))
        norm = np.load(nf_path)
        tmin, tmax = norm["time_step_min"], norm["time_step_max"]
        input_data = (input_data - tmin) / (tmax - tmin)

    return input_data.astype(np.float32), output_data.astype(np.float32)

def apply_mean_centering(x_seq_2d: np.ndarray, dcfg: dict):
    if dcfg.get("mean_centered", False):
        mv_raw = dcfg.get("mean_values", "mean_values.npz")
        mv_path = _resolve_npz(Path(mv_raw), prefer_names=("mean_values.npz",))
        data = np.load(mv_path)
        Xmean, ymean = data["Xmean"], data["ymean"]
        return x_seq_2d - Xmean, Xmean, ymean
    return x_seq_2d, None, None

# =====================
# Inference for one measured file
# =====================
def run_inference_one(measured_file: Path, cfg: dict):
    model_cfg = cfg["model"]; run_cfg = cfg["runtime"]; dcfg = cfg["data"]; io = cfg["io"]

    # device
    device = _resolve_device(run_cfg.get("device", "auto"))
    time_steps = int(run_cfg.get("time_steps", 30))
    out_len    = int(run_cfg.get("output_seq_length", 29))

    # 入出力
    output_dir = Path(io.get("output_dir", "outputs/use_model"))
    png_dir    = Path(io.get("png_dir", str(output_dir / "pred_png")))
    save_png   = bool(io.get("save_png", True))
    out_name   = io.get("out_npy_name", "outputs_pred_all_series.npy")

    # united はファイルの想定
    united_path = Path(dcfg["united"]["path"])

    # データ読み込み
    input_data, output_data = load_pair(measured_file, united_path, dcfg)

    # 特徴次元を推定
    feat_dim = create_array(input_data[0, 0]).shape[0]
    use_time_ctx = bool(dcfg.get("use_time_context", False))
    in_dim = feat_dim + (1 if use_time_ctx else 0)

    # モデル構築＆ロード
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
    all_outputs = []  # (Nseries, out_len, rows, cols)

    # 推論ループ
    for series in range(n_series):
        enc_steps = []
        dec_targets = []  # 使わないが形合わせで保持
        if use_time_ctx:
            for ts in range(0, time_steps):
                flat = create_array(input_data[series, ts])
                time_ctx = np.full_like(flat, ts / float(input_data.shape[1] - 1), dtype=np.float32)
                enc_steps.append(np.concatenate([flat, time_ctx], axis=0))
            for ts in range(1, time_steps):
                dec_targets.append(create_array(output_data[series, ts]))
        else:
            for ts in range(0, time_steps):
                enc_steps.append(create_array(input_data[series, ts]))
            for ts in range(1, time_steps):
                dec_targets.append(create_array(output_data[series, ts]))

        enc_np = np.stack(enc_steps, axis=0)    # (T, F[+1])
        tgt_np = np.stack(dec_targets, axis=0)  # (T-1, F)

        enc_np_mc, Xmean, ymean = apply_mean_centering(enc_np, dcfg)

        src = torch.from_numpy(enc_np_mc).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(src, tgt_len=out_len)     # (1, out_len, F)
            pred_np = pred.squeeze(0).cpu().numpy()

        if dcfg.get("mean_centered", False) and ymean is not None:
            if ymean.shape != pred_np.shape:
                raise ValueError(f"ymean shape {ymean.shape} != predictions {pred_np.shape}")
            pred_np = pred_np + ymean

        # 画像へ
        output_imgs = np.stack([de_create_array(pred_np[t]) for t in range(out_len)], axis=0).astype(np.float32)
        all_outputs.append(output_imgs)

        if save_png:
            save_png_images(png_dir, series, output_imgs)

        out_sum = output_imgs.sum(axis=0)
        print(f"[summary] {measured_file.name} | series={series}  sum(min/max)=({out_sum.min():.4g}/{out_sum.max():.4g})  shape={output_imgs.shape}")

    # まとめ保存（ファイル単位で別名に）
    all_outputs = np.stack(all_outputs, axis=0)
    stem = measured_file.stem
    outpath = output_dir / f"{stem}__{out_name}"
    _ensure_dir(output_dir)
    np.save(outpath, all_outputs)
    print(f"[save] {outpath}  shape={all_outputs.shape}")

# =====================
# Main
# =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML 設定ファイルへのパス")
    args = ap.parse_args()

    t0 = time.time()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # measured の解決
    mcfg = cfg["data"]["measured"]
    mpath = Path(mcfg["path"])
    measured_files: list[Path]
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

    # 各 measured を処理（united は単一ファイル想定）
    for mf in measured_files:
        print(f"[run] start: {mf.name}")
        run_inference_one(mf, cfg)
        print(f"[run] done : {mf.name}")

    dt = time.time() - t0
    print(f"[time] elapsed: {dt:.2f} s")

if __name__ == "__main__":
    main()
