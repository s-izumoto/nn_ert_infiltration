# 09_loadPredicted_loop_onlyNetwork12_combined.py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ========= 設定（必要なら変更） =========
# 06_evaluateModel.py が出すまとめファイル（diffの時系列）
SEQ2SEQ_FILE = "outputs_pred_all_series.npy"            # shape: (N, T, R, C)

# 08_useModel_AppResAppRes_seq2seq_onlyFirst_loop.py が出すまとめファイル（最初のターゲット時刻のdiff）
INITIAL_FILE = "outputs_infer/pred_images_all.npy"      # shape: (N, R, C)

# 真値の三角配列（値; 1/x*1000）
UNITED_TRUE_FILE = "united_triangular_matrices_test.npy"     # shape: (N, T_full, R, C)

# 何ステップごとに真値を間引くか（元コードに合わせて10）
NUM_MEASUREMENTS = 10

# 画像保存ディレクトリ
OUTPUT_DIR = Path("compareWithTestData")
OUTPUT_DIR.mkdir(exist_ok=True)

# 可視化するシーケンス番号（任意）
CHOSEN_SEQUENCES = [0, 1, 2, 3, 4, 5]

# ========= 安全な 1/x * 1000 =========
def safe_inverse_k(arr, scale=1000.0, eps=1e-8, clip=1e6):
    a = np.array(arr, dtype=np.float64, copy=False)
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(scale, a, out=out, where=np.abs(a) > eps)
    out = np.nan_to_num(out, nan=0.0, posinf=clip, neginf=-clip)
    np.clip(out, -clip, clip, out=out)
    return out.astype(np.float32, copy=False)

# ========= メイン =========
def main():
    # --- 入力の読み込み ---
    if not Path(SEQ2SEQ_FILE).exists():
        raise FileNotFoundError(f"{SEQ2SEQ_FILE} が見つかりません")
    if not Path(INITIAL_FILE).exists():
        raise FileNotFoundError(f"{INITIAL_FILE} が見つかりません")
    if not Path(UNITED_TRUE_FILE).exists():
        raise FileNotFoundError(f"{UNITED_TRUE_FILE} が見つかりません")

    seq2seq_all = np.load(SEQ2SEQ_FILE)          # (N, T, R, C)  diff系列
    initial_all = np.load(INITIAL_FILE)          # (N, R, C)    diffの最初1枚
    united = np.load(UNITED_TRUE_FILE)           # (N, T_full, R, C) 値（電気伝導度のもと→ 1/x*1000 へ）

    # 形状チェック
    N_s2s, T_s2s, R, C = seq2seq_all.shape
    N_init, R2, C2 = initial_all.shape
    if (R, C) != (R2, C2):
        raise ValueError(f"空間サイズが一致しません: SEQ2SEQ (R,C)=({R},{C}), INITIAL (R,C)=({R2},{C2})")
    if N_init != N_s2s:
        # シリーズ数が合わない場合は小さい方に合わせる
        N = min(N_s2s, N_init, united.shape[0])
        seq2seq_all = seq2seq_all[:N]
        initial_all = initial_all[:N]
        united = united[:N]
        print(f"[warn] N が不一致だったため {N} に揃えました")
    else:
        N = min(N_s2s, united.shape[0])

    # 真値（値スケール）に変換して、10ステップおきに間引き
    true_resistivity_all = safe_inverse_k(united)               # (N, T_full, R, C)
    true_resistivity_all = np.nan_to_num(true_resistivity_all, nan=0.0)

    mape_values = []
    conductivity_stack = []   # 各 seq の再構成（値）時系列を積む

    for seq in range(N):
        # --- diff 系列の用意 ---
        # 08 の最初のターゲット時刻（diff 1枚）
        initial_diff = initial_all[seq][None, ...]              # (1, R, C)

        # 06 の時系列（diff の T_s2s 枚）
        diffs_seq = seq2seq_all[seq]                            # (T, R, C)

        # 先頭が initial_diff とほぼ同じなら重複回避で捨てる
        if np.allclose(diffs_seq[0], initial_all[seq], atol=1e-6, rtol=1e-6):
            diffs_rest = diffs_seq[1:]                          # (T-1, R, C)
        else:
            diffs_rest = diffs_seq                              # (T, R, C)

        # t=0 の真の「値」(基準面) を先頭に置く
        initial_true_value = safe_inverse_k(united[seq, 0])     # (R, C)
        initial_true_value = np.nan_to_num(initial_true_value, nan=0.0)
        initial_true_value = initial_true_value[None, ...]      # (1, R, C)

        # diff を時系列に連結: [t0の値] + [t1のdiff(=initial_diff)] + [t2..のdiff]
        diff_series = np.concatenate([initial_diff, diffs_rest], axis=0)  # (T_use, R, C)
        # 値へ再構成（cumsum）。先頭の initial_true_value をオフセットとして足し込む
        # 具体的には: 値(t0) = initial_true_value
        #            値(t1) = 値(t0) + diff(t1)
        #            値(t2) = 値(t1) + diff(t2) ...
        values_series = np.cumsum(np.concatenate([initial_true_value, diff_series], axis=0), axis=0)
        values_series = np.nan_to_num(values_series, nan=0.0)   # (T_use+1, R, C)

        # --- 真値（間引き）と長さ合わせ ---
        true_seq = true_resistivity_all[seq, 0::NUM_MEASUREMENTS]   # (T_true, R, C)
        T_pred = values_series.shape[0]
        T_true = true_seq.shape[0]
        T = min(T_pred, T_true)
        values_series = values_series[:T]
        true_seq = true_seq[:T]

        # --- 誤差（MAPE） ---
        mape_sum = 0.0
        valid_count = 0
        for t in range(T):
            true_vals = true_seq[t]
            pred_vals = values_series[t]
            mask = (true_vals != 0.0)
            rel = np.abs((pred_vals - true_vals) / true_vals)
            mape_sum += float(rel[mask].sum())
            valid_count += int(mask.sum())

            # 可視化
            if seq in CHOSEN_SEQUENCES:
                diff_map = np.abs(pred_vals - true_vals)
                fig = plt.figure(figsize=(18, 6))

                ax1 = plt.subplot(1, 3, 1)
                im1 = ax1.imshow(pred_vals, aspect='auto', cmap='hot')
                plt.colorbar(im1, ax=ax1, label="Predicted conductivity")
                ax1.set_title(f"Predicted at t={t} (Seq {seq})")

                ax2 = plt.subplot(1, 3, 2)
                im2 = ax2.imshow(true_vals, aspect='auto', cmap='hot')
                plt.colorbar(im2, ax=ax2, label="True conductivity")
                ax2.set_title(f"True at t={t} (Seq {seq})")

                ax3 = plt.subplot(1, 3, 3)
                im3 = ax3.imshow(diff_map, aspect='auto', cmap='coolwarm')
                plt.colorbar(im3, ax=ax3, label="|Pred-True|")
                ax3.set_title(f"Diff at t={t} (Seq {seq})")

                fig.savefig(OUTPUT_DIR / f"measurement_locations_seq_{seq}_timestep_{t}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

        mape = (mape_sum / max(valid_count, 1)) * 100.0
        mape_values.append((seq, mape))
        print(f"Sequence {seq}: MAPE = {mape:.4f}%  (T={T})")

        conductivity_stack.append(values_series)  # 値の時系列

    # --- まとめの保存 ---
    with open(OUTPUT_DIR / "mape_values.txt", "w") as f:
        for seq, m in mape_values:
            f.write(f"Sequence {seq}: MAPE = {m}%\n")
    print(f"MAPE values saved to {OUTPUT_DIR / 'mape_values.txt'}")

    # 形を揃えてから保存（可変長Tの可能性があるのでパディング or オブジェクト配列）
    # ここでは最短Tに揃えてスタック
    min_T = min(arr.shape[0] for arr in conductivity_stack)
    trimmed = [arr[:min_T] for arr in conductivity_stack]               # list of (Tmin, R, C)
    all_values = np.stack(trimmed, axis=0)                               # (N, Tmin, R, C)
    np.save("conductivity.npy", all_values)
    print(f"Saved conductivity.npy  shape={all_values.shape}")

if __name__ == "__main__":
    main()
