import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml  # PyYAML
except Exception:
    yaml = None


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML が見つかりません。`pip install pyyaml` を実行してください。")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_inverse_k(arr, scale=1000.0, eps=1e-8, clip=1e6):
    """安全な 1/x * scale。NaN/Inf をゼロorクリップに置換。float32 で返す。"""
    a = np.array(arr, dtype=np.float64, copy=False)
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(scale, a, out=out, where=np.abs(a) > eps)
    out = np.nan_to_num(out, nan=0.0, posinf=clip, neginf=-clip)
    np.clip(out, -clip, clip, out=out)
    return out.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Compare predicted diffs with true values using YAML config.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    args = parser.parse_args()

    t0 = time.time()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    inputs = cfg.get("inputs", {})
    proc = cfg.get("processing", {})
    viz = cfg.get("viz", {})
    out = cfg.get("output", {})
    run = cfg.get("run", {})

    verbose = bool(run.get("verbose", True))

    # === パス構築 ===
    seq2seq_path = Path(inputs.get("seq2seq_dir", ".")) / inputs.get("seq2seq_file", "outputs_pred_all_series.npy")
    initial_path = Path(inputs.get("initial_dir", ".")) / inputs.get("initial_file", "pred_images_all.npy")
    true_path = Path(inputs.get("true_dir", ".")) / inputs.get("true_file", "united_triangular_matrices_test.npy")

    out_dir = Path(out.get("dir", "compareWithTestData"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[info] seq2seq:", seq2seq_path)
        print("[info] initial:", initial_path)
        print("[info] true   :", true_path)
        print("[info] outdir :", out_dir)

    # === 存在チェック ===
    for p in [seq2seq_path, initial_path, true_path]:
        if not p.exists():
            raise FileNotFoundError(f"入力が見つかりません: {p}")

    # === 変換パラメータ ===
    NUM_MEASUREMENTS = int(proc.get("num_measurements", 10))
    SCALE = float(proc.get("scale", 1000.0))
    EPS = float(proc.get("eps", 1e-8))
    CLIP = float(proc.get("clip", 1e6))
    DROP_DUP_FIRST = bool(proc.get("drop_duplicate_first_diff", True))

    VIZ_ENABLED = bool(viz.get("enabled", True))
    CHOSEN_SEQUENCES = list(viz.get("chosen_sequences", [0, 1, 2, 3, 4, 5]))
    CMAP_VALUE = str(viz.get("cmap_value", "hot"))
    CMAP_DIFF = str(viz.get("cmap_diff", "coolwarm"))

    MAPE_TXT = out.get("mape_txt", "mape_values.txt")
    STACK_FILE = out.get("stack_file", "conductivity.npy")
    IMG_PREFIX = out.get("image_prefix", "measurement_locations_seq")

    # === 入力ロード ===
    seq2seq_all = np.load(seq2seq_path)  # (N, T, R, C)  diff系列
    initial_all = np.load(initial_path)  # (N, R, C)     diff 1枚
    united = np.load(true_path)          # (N, T_full, R, C) 値（→ 1/x*scale へ）

    # 形状チェック
    N_s2s, T_s2s, R, C = seq2seq_all.shape
    N_init, R2, C2 = initial_all.shape
    if (R, C) != (R2, C2):
        raise ValueError(f"空間サイズが一致しません: SEQ2SEQ (R,C)=({R},{C}), INITIAL (R,C)=({R2},{C2})")

    if N_init != N_s2s or united.shape[0] != N_s2s:
        N = min(N_s2s, N_init, united.shape[0])
        if verbose:
            print(f"[warn] N が不一致だったため {N} に揃えます")
        seq2seq_all = seq2seq_all[:N]
        initial_all = initial_all[:N]
        united = united[:N]
    else:
        N = N_s2s

    # 真値（値スケール）に変換して、間引き
    true_resistivity_all = safe_inverse_k(united, scale=SCALE, eps=EPS, clip=CLIP)  # (N, T_full, R, C)
    true_resistivity_all = np.nan_to_num(true_resistivity_all, nan=0.0)

    mape_values = []
    conductivity_stack = []  # 各 seq の再構成（値）時系列を積む

    for seq in range(N):
        # --- diff 系列の用意 ---
        initial_diff = initial_all[seq][None, ...]           # (1, R, C)
        diffs_seq = seq2seq_all[seq]                         # (T, R, C)

        if DROP_DUP_FIRST and np.allclose(diffs_seq[0], initial_all[seq], atol=1e-6, rtol=1e-6):
            diffs_rest = diffs_seq[1:]                      # (T-1, R, C)
        else:
            diffs_rest = diffs_seq                          # (T, R, C)

        # t=0 真の「値」(基準面)
        initial_true_value = safe_inverse_k(united[seq, 0], scale=SCALE, eps=EPS, clip=CLIP)  # (R, C)
        initial_true_value = np.nan_to_num(initial_true_value, nan=0.0)[None, ...]  # (1, R, C)

        # diff を時系列に連結: [t0の値] + [t1のdiff(=initial_diff)] + [t2..のdiff]
        diff_series = np.concatenate([initial_diff, diffs_rest], axis=0)  # (T_use, R, C)

        # 値へ再構成（cumsum）
        values_series = np.cumsum(np.concatenate([initial_true_value, diff_series], axis=0), axis=0)
        values_series = np.nan_to_num(values_series, nan=0.0)  # (T_use+1, R, C)

        # --- 真値（間引き）と長さ合わせ ---
        true_seq = true_resistivity_all[seq, 0::NUM_MEASUREMENTS]   # (T_true, R, C)
        T_pred = values_series.shape[0]
        T_true = true_seq.shape[0]
        T = min(T_pred, T_true)
        values_series = values_series[:T]
        true_seq = true_seq[:T]

        # --- 誤差（MAPE） & 可視化 ---
        mape_sum = 0.0
        valid_count = 0
        for t in range(T):
            true_vals = true_seq[t]
            pred_vals = values_series[t]
            mask = (true_vals != 0.0)
            rel = np.abs((pred_vals - true_vals) / true_vals)
            mape_sum += float(rel[mask].sum())
            valid_count += int(mask.sum())

            if VIZ_ENABLED and (seq in CHOSEN_SEQUENCES):
                diff_map = np.abs(pred_vals - true_vals)
                fig = plt.figure(figsize=(18, 6))

                ax1 = plt.subplot(1, 3, 1)
                im1 = ax1.imshow(pred_vals, aspect='auto', cmap=CMAP_VALUE)
                plt.colorbar(im1, ax=ax1, label="Predicted value")
                ax1.set_title(f"Pred t={t} (Seq {seq})")

                ax2 = plt.subplot(1, 3, 2)
                im2 = ax2.imshow(true_vals, aspect='auto', cmap=CMAP_VALUE)
                plt.colorbar(im2, ax=ax2, label="True value")
                ax2.set_title(f"True t={t} (Seq {seq})")

                ax3 = plt.subplot(1, 3, 3)
                im3 = ax3.imshow(diff_map, aspect='auto', cmap=CMAP_DIFF)
                plt.colorbar(im3, ax=ax3, label="|Pred-True|")
                ax3.set_title(f"Diff t={t} (Seq {seq})")

                fig.savefig(out_dir / f"{IMG_PREFIX}_{seq}_timestep_{t}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

        mape = (mape_sum / max(valid_count, 1)) * 100.0
        mape_values.append((seq, mape))
        if verbose:
            print(f"Sequence {seq}: MAPE = {mape:.4f}%  (T={T})")

        conductivity_stack.append(values_series)  # 値の時系列

    # --- まとめの保存 ---
    mape_path = out_dir / MAPE_TXT
    with mape_path.open("w", encoding="utf-8") as f:
        for seq, m in mape_values:
            f.write(f"Sequence {seq}: MAPE = {m}%\n")
    if verbose:
        print(f"[info] MAPE values saved to {mape_path}")

    # 可変長T対策：最短Tに揃えて保存
    min_T = min(arr.shape[0] for arr in conductivity_stack)
    trimmed = [arr[:min_T] for arr in conductivity_stack]        # list of (Tmin, R, C)
    all_values = np.stack(trimmed, axis=0)                       # (N, Tmin, R, C)

    stack_path = out_dir / STACK_FILE
    np.save(stack_path, all_values)
    if verbose:
        print(f"[info] saved {stack_path}  shape={all_values.shape}")

    dt = time.time() - t0
    print(f"[time] elapsed: {dt:.2f} s")


if __name__ == "__main__":
    main()