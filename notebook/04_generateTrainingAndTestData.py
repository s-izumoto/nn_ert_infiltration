# -*- coding: utf-8 -*-
"""
Train/Test を同一の測定シーケンスで処理し、
measured_training_data_sameRowColSeq{seq}.npy（train）
measured_training_data_sameRowColSeq{seq}_test.npy（test）
を出力します。
"""

import numpy as np
import os


def convert_to_probability_distribution(difference_map):
    exp_diff = np.abs(difference_map)
    total = np.sum(exp_diff)
    if total == 0:
        return np.zeros_like(exp_diff)
    return exp_diff / total


def process_dataset(true_resistivity_data: np.ndarray,
                    positions_seq: np.ndarray,
                    num_measurements: int = 1):
    """
    true_resistivity_data: (Nseq, T, H, W)
    positions_seq:         (T, 2)  ※t=1..T-1 で positions_seq[t-1] を参照
    """
    true_resistivity_data = np.nan_to_num(true_resistivity_data, nan=0.0)

    # （将来まとめ測定に拡張する余地を残す）今は num_measurements=1 のまま
    data = true_resistivity_data[:, 0::num_measurements, :, :]

    N, T, H, W = data.shape
    measured_all = []

    # 安全のため、ポジションとデータの時間長が違ってもミニマムに合わせる
    Tmax = min(T, positions_seq.shape[0] + 1)

    for seq in range(N):
        measured_resistivity_map = data[seq, 0, :, :].copy()
        measured_seq = []

        for t in range(1, Tmax):
            true_map_t = data[seq, t, :, :].copy()
            diff_map = np.abs(true_map_t - measured_resistivity_map)

            # 位置は「選んだシーケンス」のものをそのまま使用
            c = int(positions_seq[t - 1, 0])
            r = int(positions_seq[t - 1, 1])

            # 測定で更新
            measured_resistivity_map[r, c] = true_map_t[r, c]

            # ログ用に保持（.copy() 大事）
            measured_seq.append(measured_resistivity_map.copy())

            # もし確率分布 y が必要ならここで計算（返さないが将来拡張用）
            _ = convert_to_probability_distribution(diff_map)

        measured_all.append(measured_seq)

    measured_all = np.array(measured_all, dtype=np.float32)  # (N, Tmax-1, H, W)
    return measured_all


def main():
    # ---- 入力ロード（train は必須、test はあれば処理）----
    train_path = "united_triangular_matrices.npy"
    test_path = "united_triangular_matrices_test.npy"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Not found: {train_path}")

    train_data = np.load(train_path)  # (N, T, H, W) を想定
    # test は任意
    test_data = None
    if os.path.exists(test_path):
        test_data = np.load(test_path)

    # ---- 測定位置（全シーケンス）をロードし、代表シーケンスを 1 本選ぶ ----
    positions = np.load("measurement_indices.npy")  # (S, T, 2)
    S = positions.shape[0]
    med = np.median(positions, axis=0)            # (T, 2)
    d = np.abs(positions - med).sum(axis=(1, 2))  # (S,)
    seq_for_rc = int(np.argmin(d))
    if not (0 <= seq_for_rc < S):
        seq_for_rc = 31 if S > 50 else 0

    # 後工程で再利用したい場合に保存
    try:
        np.save("chosen_seq_index.npy", np.array(seq_for_rc, dtype=int))
    except Exception:
        pass

    positions_seq = positions[seq_for_rc, :, :]   # (T, 2)

    # ---- Train を処理して保存 ----
    measured_train = process_dataset(train_data, positions_seq, num_measurements=1)
    out_train = f"measured_training_data_sameRowColSeq{seq_for_rc}.npy"
    np.save(out_train, measured_train)
    print(f"[save] {out_train}  shape={measured_train.shape}")

    # ---- Test があれば、同じ positions_seq で処理して保存 ----
    if test_data is not None:
        measured_test = process_dataset(test_data, positions_seq, num_measurements=1)
        out_test = f"measured_training_data_sameRowColSeq{seq_for_rc}_test.npy"
        np.save(out_test, measured_test)
        print(f"[save] {out_test}  shape={measured_test.shape}")
    else:
        print("[info] test データ（united_triangular_matrices_test.npy）は見つかりませんでした。train のみ保存。")


if __name__ == "__main__":
    main()
