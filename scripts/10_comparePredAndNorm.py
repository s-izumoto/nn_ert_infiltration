# -*- coding: utf-8 -*-
"""
YAML コンフィグで Pred vs Measured を描画保存
- measured_t = pred_t * factor - 1 の時間対応は元コード通り
- 0 / 非有限は NaN 扱い（nan_to_zero=True の場合は描画前に 0 へ置換）
- シーケンス毎に vmin/vmax を計算（per_sequence_vrange=True）
- 入力は pred_file / measured_file が最優先。未指定なら dir+filename を使用
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

def _resolve_path(file_path, dir_path, filename):
    """file_path が指定されていればそれ、なければ dir+filename を返す"""
    if file_path:
        return Path(file_path)
    if not dir_path or not filename:
        raise ValueError("Either file path OR both dir and filename must be provided.")
    return Path(dir_path) / filename

def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    arr = np.load(str(path))
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array (N,T,H,W). Got shape={arr.shape} from {path}")
    return arr.astype(np.float32)

def transform_measured(meas: np.ndarray) -> np.ndarray:
    # 1000 / measured（0 や非有限は NaN）
    meas_f = meas.astype(np.float32, copy=True)
    zeros = (meas_f == 0.0) | ~np.isfinite(meas_f)
    meas_f[zeros] = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        transformed = 1000.0 / meas_f
    return transformed

def transform_pred(pred: np.ndarray) -> np.ndarray:
    pred_f = pred.astype(np.float32, copy=True)
    invalid = (pred_f == 0.0) | ~np.isfinite(pred_f)
    pred_f[invalid] = np.nan
    return pred_f

def load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

def cli_override(cfg: dict, ns: argparse.Namespace) -> dict:
    """CLI で与えた値があれば YAML を上書き"""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config path")

    # 任意の CLI 上書き（必要なものだけどうぞ）
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
    ap.add_argument("--per_sequence_vrange", type=lambda x: str(x).lower() in ("1","true","yes","y"))
    ap.add_argument("--dpi", type=int)
    ns = ap.parse_args()

    cfg_path = Path(ns.config)
    cfg = load_config(cfg_path)
    cfg = cli_override(cfg, ns)

    # 必須/既定値
    seq = cfg.get("seq", None)
    start_pred = int(cfg.get("start_pred", 1))
    max_pred_steps = int(cfg.get("max_pred_steps", 31))
    meas_step_factor = int(cfg.get("meas_step_factor", 10))
    cmap_name = cfg.get("cmap", "hot")
    nan_to_zero = bool(cfg.get("nan_to_zero", False))
    per_sequence_vrange = bool(cfg.get("per_sequence_vrange", True))
    dpi = int(cfg.get("dpi", 150))

    # 入力パス解決
    pred_path = _resolve_path(
        cfg.get("pred_file"),
        cfg.get("pred_dir"),
        cfg.get("pred_filename"),
    )
    meas_path = _resolve_path(
        cfg.get("measured_file"),
        cfg.get("measured_dir"),
        cfg.get("measured_filename"),
    )

    out_dir = Path(cfg.get("out_dir", "compareWithTestData/pred_vs_measured"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ロード
    pred_raw = load_npy(pred_path)   # (N,Tp,H,W)
    meas_raw = load_npy(meas_path)   # (N,Tm,H,W)

    # N を合わせる
    N = min(pred_raw.shape[0], meas_raw.shape[0])
    pred_raw = pred_raw[:N]
    meas_raw = meas_raw[:N]

    # 変換
    pred = transform_pred(pred_raw)
    meas_trans = transform_measured(meas_raw)

    if nan_to_zero:
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        meas_trans = np.nan_to_num(meas_trans, nan=0.0, posinf=0.0, neginf=0.0)

    # 時間マッピング
    Tp = pred.shape[1]
    Tm = meas_trans.shape[1]
    pred_times = list(range(start_pred, min(start_pred + max_pred_steps, Tp)))
    pairs = []
    for pt in pred_times:
        mt = pt * meas_step_factor - 1
        if 0 <= mt < Tm:
            pairs.append((pt, mt))
    if not pairs:
        print("[warn] No valid (pred_t, meas_t) pairs under current settings.")
        return

    # 対象シーケンス
    if seq:
        seq_idx = [i for i in seq if 0 <= i < N]
    else:
        seq_idx = list(range(N))
    # 対象シーケンス
    if seq:
        seq_idx = [i for i in seq if 0 <= i < N]
    else:
        seq_idx = list(range(N))

    # 進捗バー用の総枚数（デフォルトで表示）
    total_tasks = len(seq_idx) * len(pairs)

    total = 0
    base_cmap = plt.get_cmap(cmap_name).copy()
    try:
        base_cmap.set_bad(color="white")
    except Exception:
        pass

    with tqdm(total=total_tasks, desc="Saving images", unit="img") as pbar:
        for s in seq_idx:
            # 値レンジ（vmin/vmax）
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

            for (pt, mt) in pairs:
                fig = plt.figure(figsize=(12, 5))
                ax1 = plt.subplot(1, 2, 1)
                im1 = ax1.imshow(pred[s, pt], aspect="auto", cmap=base_cmap, vmin=vmin, vmax=vmax)
                ax1.set_title(f"Predicted conductivity ($mS\\,m^{{-1}}$, seq {s}, pred {pt})")
                plt.colorbar(im1, ax=ax1, label="$mS\\,m^{{-1}}$")

                ax2 = plt.subplot(1, 2, 2)
                im2 = ax2.imshow(meas_trans[s, mt], aspect="auto", cmap=base_cmap, vmin=vmin, vmax=vmax)
                ax2.set_title(f"Measured conductivity ($mS\\,m^{{-1}}$, seq {s}, meas {mt})")
                plt.colorbar(im2, ax=ax2, label="$mS\\,m^{{-1}}$")

                out_path = out_dir / f"seq{s:03d}_predt{pt:03d}_meast{mt:03d}.png"
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)

                total += 1
                pbar.update(1)                 # ← 1枚保存ごとに進捗
                if (total % 50) == 0:          # ← 任意：時々サマリ表示
                    pbar.set_postfix(saved=total)

    print(f"[done] saved {total} images to: {out_dir}")
    print(f"[info] used time pairs (pred_t -> meas_t): {pairs}")

if __name__ == "__main__":
    main()
