# XX_plotTrainingCurves.py
# ------------------------------------------------------------
# Script purpose:
#   This script visualizes and compares training curves (loss vs. epoch)
#   across multiple hyperparameter combinations (learning rate and batch size).
#   It reads CSV log files produced by model training runs, aggregates results
#   across folds, and produces summary plots:
#       1) compare_by_lr_{lr}.png : compares batch sizes for a given learning rate
#       2) compare_by_bs_{bs}.png : compares learning rates for a given batch size
#       3) compare_all_combos.png : compares all (lr, batch size) combinations
#
# Usage example:
#   python XX_plotTrainingCurves.py --root . --out plots --also-pdf --with-train
#
# Input:
#   - CSV files inside ./results_first/ with names like:
#       lr0p001_bs16_fold0.csv
#       lr0p001_bs16_fold1.csv
#     Each file should contain columns: epoch, train_loss, val_loss
#
# Output:
#   - PNG (and optionally PDF) plots under the specified output directory.
#
# ------------------------------------------------------------

import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def _fmt_lr_token(token: str) -> str:
    """
    Format the learning rate token from the filename into a human-readable float.
    Example:
        lr0p001 → "0.001"
    """
    t = token.replace("p", ".")
    try:
        return f"{float(t):g}"
    except Exception:
        return token


def save_fig(fig, out_dir: Path, stem: str, also_pdf: bool):
    """
    Save a Matplotlib figure as PNG (and optionally PDF) to the specified directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    if also_pdf:
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png}")


def load_results(root: Path):
    """
    Load all training result CSV files from the results_first directory.

    Returns:
        dict[(lr, bs)] -> list of DataFrames (one per fold)
    """
    results_dir = root / "results_first"
    pattern = re.compile(r"^lr(?P<lr>[^_]+)_bs(?P<bs>[^_]+)_fold(?P<fold>\d+)\.csv$")
    groups = defaultdict(list)  # key = (lr_str, bs_str)

    if not results_dir.exists():
        print(f"[info] Directory not found: {results_dir} (skipped)")
        return groups

    for csv_path in sorted(results_dir.glob("*.csv")):
        if csv_path.name == "grid_search_results.csv":
            continue

        m = pattern.match(csv_path.name)
        if not m:
            print(f"[skip] Filename does not match pattern: {csv_path.name}")
            continue

        lr_raw = m.group("lr")
        bs_raw = m.group("bs")
        lr_label = _fmt_lr_token(lr_raw)
        try:
            bs_label = str(int(bs_raw))
        except Exception:
            bs_label = bs_raw

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[warn] Failed to read {csv_path}: {e}")
            continue

        required = {"epoch", "train_loss", "val_loss"}
        if not required.issubset(df.columns):
            print(f"[skip] Missing columns {required} in: {csv_path.name}")
            continue

        # Sort by epoch and merge duplicate epochs by averaging
        df = df.groupby("epoch", as_index=False)[["train_loss", "val_loss"]].mean()
        groups[(lr_label, bs_label)].append(df)

    return groups


def aggregate_by_combo(groups):
    """
    Aggregate folds for each (learning rate, batch size) combination.

    Returns:
        dict[(lr, bs)] = {
            "epoch": array,
            "val_mean": array,
            "val_std": array,
            "train_mean": array,
            "train_std": array,
            "n_folds": int
        }
    """
    out = {}
    for (lr, bs), dfs in groups.items():
        # Create a common epoch axis (union of all folds)
        all_epochs = sorted(set(int(e) for df in dfs for e in df["epoch"].values))
        x = np.array(all_epochs, dtype=int)

        vals, trains = [], []

        for df in dfs:
            tmp = pd.DataFrame({"epoch": x})
            merged = tmp.merge(df, on="epoch", how="left").sort_values("epoch")

            # Fill missing values by linear interpolation
            merged["val_loss"] = merged["val_loss"].interpolate("linear", limit_direction="both")
            merged["train_loss"] = merged["train_loss"].interpolate("linear", limit_direction="both")

            vals.append(merged["val_loss"].to_numpy())
            trains.append(merged["train_loss"].to_numpy())

        val_arr = np.vstack(vals) if vals else None
        train_arr = np.vstack(trains) if trains else None

        out[(lr, bs)] = {
            "epoch": x,
            "val_mean": val_arr.mean(axis=0) if val_arr is not None else None,
            "val_std": val_arr.std(axis=0) if val_arr is not None else None,
            "train_mean": train_arr.mean(axis=0) if train_arr is not None else None,
            "train_std": train_arr.std(axis=0) if train_arr is not None else None,
            "n_folds": len(dfs),
        }
    return out


def plot_group_by_lr(agg, out_dir: Path, also_pdf: bool, with_train: bool):
    """
    Plot validation loss curves grouped by learning rate.
    Different batch sizes are compared within each learning rate.
    """
    lrs = sorted({lr for (lr, bs) in agg.keys()},
                 key=lambda s: float(s) if s.replace('.', '', 1).isdigit() else s)

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
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        save_fig(fig, out_dir, f"compare_by_lr_{lr}", also_pdf)


def plot_group_by_bs(agg, out_dir: Path, also_pdf: bool, with_train: bool):
    """
    Plot validation loss curves grouped by batch size.
    Different learning rates are compared within each batch size.
    """
    bss = sorted({bs for (lr, bs) in agg.keys()},
                 key=lambda s: int(s) if s.isdigit() else s)

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
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        save_fig(fig, out_dir, f"compare_by_bs_{bs}", also_pdf)


def plot_all_combos(agg, out_dir: Path, also_pdf: bool, with_train: bool):
    """
    Plot all (learning rate, batch size) combinations on one figure.
    Useful for an overall comparison, though it can become crowded.
    """
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

    plt.title("Validation loss across all (lr, batch size) combinations")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    save_fig(fig, out_dir, "compare_all_combos", also_pdf)


def main():
    """
    Command-line entry point.
    Parses arguments, loads data, aggregates by (lr, batch size),
    and generates summary plots.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Root directory containing 'results_first' folder")
    ap.add_argument("--out", default="plots_compare", help="Output directory for saved plots")
    ap.add_argument("--also-pdf", action="store_true", help="Save plots as both PNG and PDF")
    ap.add_argument("--with-train", action="store_true", help="Overlay train_loss as dashed lines")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)

    groups = load_results(root)
    if not groups:
        print("[info] No CSV files found.")
        return

    agg = aggregate_by_combo(groups)
    plot_group_by_lr(agg, out_dir, args.also_pdf, args.with_train)
    plot_group_by_bs(agg, out_dir, args.also_pdf, args.with_train)
    plot_all_combos(agg, out_dir, args.also_pdf, args.with_train)


if __name__ == "__main__":
    main()
