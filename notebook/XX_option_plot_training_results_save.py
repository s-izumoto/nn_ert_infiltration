# plot_compare_results.py
# 使い方:
#   python plot_compare_results.py --root . --out plots --also-pdf --with-train
#
# - results/*.csv を読み込み (epoch, train_loss, val_loss)
# - ファイル名 lr{…}_bs{…}_fold{..}.csv から lr / batch_size / fold を抽出
# - (lr, batch_size) 別に fold 平均・標準偏差を計算し、同一グラフに重ね描きして保存
#   1) per_lr_*.png : 同じ lr 内で batch_size を比較
#   2) per_bs_*.png : 同じ batch_size 内で lr を比較
#   3) all_combos.png : すべての (lr, batch) を一枚で比較（図が多い時は混雑に注意）

import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def _fmt_lr_token(token: str) -> str:
    # ファイル名の lr0p001 → 表示は 0.001 に
    # 既に "0.001" 形式ならそのまま
    t = token.replace("p", ".")
    try:
        return f"{float(t):g}"
    except Exception:
        return token

def save_fig(fig, out_dir: Path, stem: str, also_pdf: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    if also_pdf:
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png}")

def load_results(root: Path):
    """results/*.csv を読み込み、(lr, bs) → [DataFrame,... per fold] にまとめる"""
    results_dir = root / "results_first"
    pattern = re.compile(r"^lr(?P<lr>[^_]+)_bs(?P<bs>[^_]+)_fold(?P<fold>\d+)\.csv$")
    groups = defaultdict(list)  # key=(lr_str, bs_str) -> list of df

    if not results_dir.exists():
        print(f"[info] {results_dir} がありません（スキップ）")
        return groups

    for csv_path in sorted(results_dir.glob("*.csv")):
        if csv_path.name == "grid_search_results.csv":
            continue
        m = pattern.match(csv_path.name)
        if not m:
            print(f"[skip] 命名規則外: {csv_path.name}")
            continue

        lr_raw = m.group("lr")
        bs_raw = m.group("bs")
        lr_label = _fmt_lr_token(lr_raw)  # 表示用
        try:
            bs_label = str(int(bs_raw))
        except Exception:
            bs_label = bs_raw

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[warn] 読み込み失敗: {csv_path} -> {e}")
            continue

        required = {"epoch", "train_loss", "val_loss"}
        if not required.issubset(df.columns):
            print(f"[skip] 必要列 {required} 不足: {csv_path.name}")
            continue

        # epoch でソート＆重複 epoch は平均
        df = df.groupby("epoch", as_index=False)[["train_loss", "val_loss"]].mean()
        groups[(lr_label, bs_label)].append(df)

    return groups

def aggregate_by_combo(groups):
    """(lr, bs) ごとに fold 平均/標準偏差を返す。
       戻り: dict[(lr,bs)] = {"epoch": array, "val_mean": array, "val_std": array, "train_mean": array or None, "train_std": array or None}
    """
    out = {}
    for (lr, bs), dfs in groups.items():
        # 共通の epoch 軸を作る（全 df の union を取り、欠損は線形補間 or 最近傍で埋める）
        all_epochs = sorted(set(int(e) for df in dfs for e in df["epoch"].values))
        x = np.array(all_epochs, dtype=int)

        vals = []
        trains = []

        for df in dfs:
            # reindex & interpolate
            tmp = pd.DataFrame({"epoch": x})
            merged = tmp.merge(df, on="epoch", how="left").sort_values("epoch")
            # 線形補間 → 端は前後の値で埋める
            merged["val_loss"] = merged["val_loss"].interpolate("linear", limit_direction="both")
            merged["train_loss"] = merged["train_loss"].interpolate("linear", limit_direction="both")
            vals.append(merged["val_loss"].to_numpy())
            trains.append(merged["train_loss"].to_numpy())

        val_arr = np.vstack(vals) if vals else None
        train_arr = np.vstack(trains) if trains else None

        out[(lr, bs)] = {
            "epoch": x,
            "val_mean": val_arr.mean(axis=0) if val_arr is not None else None,
            "val_std":  val_arr.std(axis=0) if val_arr is not None else None,
            "train_mean": train_arr.mean(axis=0) if train_arr is not None else None,
            "train_std":  train_arr.std(axis=0) if train_arr is not None else None,
            "n_folds": len(dfs),
        }
    return out

def plot_group_by_lr(agg, out_dir: Path, also_pdf: bool, with_train: bool):
    # lr ごとに、異なる batch_size の曲線を同一図に
    # 図は val_loss の平均曲線（±1σのバンドは省略/追加したければここで fill_between）
    lrs = sorted({lr for (lr, bs) in agg.keys()}, key=lambda s: float(s) if s.replace('.', '', 1).isdigit() else s)
    for lr in lrs:
        combos = sorted([(bs, agg[(lr, bs)]) for (lr_i, bs) in agg.keys() if lr_i == lr],
                        key=lambda t: int(t[0]) if t[0].isdigit() else t[0])

        if not combos:
            continue

        fig = plt.figure(figsize=(9, 5))
        for bs, dat in combos:
            x = dat["epoch"]
            yv = dat["val_mean"]
            if yv is None: 
                continue
            label = f"bs={bs}"
            plt.plot(x, yv, label=label)
            if with_train and dat["train_mean"] is not None:
                plt.plot(x, dat["train_mean"], linestyle="--", label=f"bs={bs} (train)")

        plt.title(f"Validation loss by batch size (lr={lr})")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        save_fig(fig, out_dir, f"compare_by_lr_{lr}", also_pdf)

def plot_group_by_bs(agg, out_dir: Path, also_pdf: bool, with_train: bool):
    # batch_size ごとに、異なる lr の曲線を同一図に
    bss = sorted({bs for (lr, bs) in agg.keys()}, key=lambda s: int(s) if s.isdigit() else s)
    for bs in bss:
        combos = sorted([(lr, agg[(lr, bs)]) for (lr, bs_i) in agg.keys() if bs_i == bs],
                        key=lambda t: float(t[0]) if t[0].replace('.', '', 1).isdigit() else t[0])

        if not combos:
            continue

        fig = plt.figure(figsize=(9, 5))
        for lr, dat in combos:
            x = dat["epoch"]
            yv = dat["val_mean"]
            if yv is None:
                continue
            label = f"lr={lr}"
            plt.plot(x, yv, label=label)
            if with_train and dat["train_mean"] is not None:
                plt.plot(x, dat["train_mean"], linestyle="--", label=f"lr={lr} (train)")

        plt.title(f"Validation loss by learning rate (bs={bs})")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        save_fig(fig, out_dir, f"compare_by_bs_{bs}", also_pdf)

def plot_all_combos(agg, out_dir: Path, also_pdf: bool, with_train: bool):
    # すべての (lr, bs) を一枚に
    if not agg:
        return
    combos = sorted(agg.items(),
                    key=lambda kv: (float(kv[0][0]) if kv[0][0].replace('.', '', 1).isdigit() else kv[0][0],
                                    int(kv[0][1]) if kv[0][1].isdigit() else kv[0][1]))

    fig = plt.figure(figsize=(10, 6))
    for (lr, bs), dat in combos:
        x = dat["epoch"]
        yv = dat["val_mean"]
        if yv is None:
            continue
        label = f"lr={lr}, bs={bs}"
        plt.plot(x, yv, label=label)
        if with_train and dat["train_mean"] is not None:
            plt.plot(x, dat["train_mean"], linestyle="--", label=f"lr={lr}, bs={bs} (train)")

    plt.title("Validation loss: all (lr, batch_size) combos (fold-mean)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    save_fig(fig, out_dir, "compare_all_combos", also_pdf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="results/ を探す起点ディレクトリ")
    ap.add_argument("--out", default="plots_compare", help="保存先ディレクトリ")
    ap.add_argument("--also-pdf", action="store_true", help="PNG に加えて PDF も保存")
    ap.add_argument("--with-train", action="store_true", help="train_loss も破線で重ねる")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)

    groups = load_results(root)
    if not groups:
        print("[info] 対象CSVが見つかりませんでした。")
        return
    agg = aggregate_by_combo(groups)

    plot_group_by_lr(agg, out_dir, args.also_pdf, args.with_train)
    plot_group_by_bs(agg, out_dir, args.also_pdf, args.with_train)
    plot_all_combos(agg, out_dir, args.also_pdf, args.with_train)

if __name__ == "__main__":
    main()
