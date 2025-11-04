# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
import yaml

def convert_to_probability_distribution(difference_map: np.ndarray) -> np.ndarray:
    exp_diff = np.abs(difference_map)
    total = np.sum(exp_diff)
    if total == 0:
        return np.zeros_like(exp_diff)
    return exp_diff / total

def process_dataset(true_resistivity_data: np.ndarray,
                    positions_seq: np.ndarray,
                    num_measurements: int = 1) -> np.ndarray:
    """
    true_resistivity_data: (N, T, H, W)
    positions_seq:         (T, 2)  ※ t=1..T-1 で positions_seq[t-1] を参照
    """
    true_resistivity_data = np.nan_to_num(true_resistivity_data, nan=0.0)

    # 将来拡張の余地は残しつつ、現状は num_measurements=1 相当
    data = true_resistivity_data[:, 0::num_measurements, :, :]

    N, T, H, W = data.shape
    measured_all = []

    # 位置列とデータ長の不一致に備えて最小側に合わせる
    Tmax = min(T, positions_seq.shape[0] + 1)

    for seq in range(N):
        measured_resistivity_map = data[seq, 0].copy()   # 初期マップ
        measured_seq = []

        for t in range(1, Tmax):
            true_map_t = data[seq, t].copy()
            diff_map = np.abs(true_map_t - measured_resistivity_map)

            # 位置は選ばれたシーケンスのものをそのまま使用
            c = int(positions_seq[t - 1, 0])
            r = int(positions_seq[t - 1, 1])

            # 測定で更新
            measured_resistivity_map[r, c] = true_map_t[r, c]

            # ログ用に保持
            measured_seq.append(measured_resistivity_map.copy())

            # もし確率分布 y が必要ならここで計算（現状は未保存）
            _ = convert_to_probability_distribution(diff_map)

        measured_all.append(measured_seq)

    measured_all = np.array(measured_all, dtype=np.float32)  # (N, Tmax-1, H, W)
    return measured_all

def pick_sequence_index(positions: np.ndarray, mode: str, fixed_index):
    """
    positions: (S, T, 2)
    mode: "median" | "fixed"
    """
    S = positions.shape[0]
    if mode == "fixed":
        if fixed_index is None:
            raise ValueError("sequence_selection.mode='fixed' ですが fixed_index が未設定です。")
        if not (0 <= int(fixed_index) < S):
            raise ValueError(f"fixed_index={fixed_index} が範囲外です (0..{S-1})。")
        return int(fixed_index)

    # "median": 各時刻・座標の中央値に最も近いシーケンスを選ぶ
    med = np.median(positions, axis=0)            # (T, 2)
    d = np.abs(positions - med).sum(axis=(1, 2))  # (S,)
    seq_for_rc = int(np.argmin(d))
    return seq_for_rc

def main(yaml_path: str):
    # === YAML 読み込み ===
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # === パス解決（フォルダー対応）===
    input_dir = Path(cfg["input_dir"]).expanduser()
    positions_dir = Path(cfg["positions_dir"]).expanduser()
    output_dir = Path(cfg["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / cfg["train_file"]
    test_path  = input_dir / cfg["test_file"] if cfg.get("test_file") else None
    positions_path = positions_dir / cfg["positions_file"]

    if not train_path.exists():
        raise FileNotFoundError(f"Not found train file: {train_path}")
    if not positions_path.exists():
        raise FileNotFoundError(f"Not found positions file: {positions_path}")

    # === ロード ===
    train_data = np.load(str(train_path))  # (N, T, H, W) を想定
    test_data = None
    if test_path and test_path.exists():
        test_data = np.load(str(test_path))

    positions = np.load(str(positions_path))  # (S, T, 2)

    # === 代表シーケンス選択 ===
    sel_cfg = cfg.get("sequence_selection", {}) or {}
    mode = sel_cfg.get("mode", "median")
    fixed_index = sel_cfg.get("fixed_index", None)
    seq_for_rc = pick_sequence_index(positions, mode, fixed_index)

    # 念のため範囲外に対処
    S = positions.shape[0]
    if not (0 <= seq_for_rc < S):
        seq_for_rc = 31 if S > 50 else 0

    positions_seq = positions[seq_for_rc]  # (T, 2)

    # === Train 処理 ===
    num_measurements = int(cfg.get("num_measurements", 1))
    measured_train = process_dataset(train_data, positions_seq, num_measurements=num_measurements)

    save_basename = cfg.get("save_basename", "measured_training_data_sameRowColSeq")
    out_train = output_dir / f"{save_basename}{seq_for_rc}.npy"
    np.save(str(out_train), measured_train)
    print(f"[save] {out_train}  shape={measured_train.shape}")

    # シーケンスIDの保存（任意）
    if cfg.get("save_seq_index", True):
        np.save(str(output_dir / "chosen_seq_index.npy"), np.array(seq_for_rc, dtype=int))

    # === Test 処理（あれば）===
    if test_data is not None:
        measured_test = process_dataset(test_data, positions_seq, num_measurements=num_measurements)
        out_test = output_dir / f"{save_basename}{seq_for_rc}_test.npy"
        np.save(str(out_test), measured_test)
        print(f"[save] {out_test}  shape={measured_test.shape}")
    else:
        print("[info] test データは見つからず（train のみ出力）")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="YAML 設定ファイルへのパス")
    ns = ap.parse_args()
    main(ns.config)
