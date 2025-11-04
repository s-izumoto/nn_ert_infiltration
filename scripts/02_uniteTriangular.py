# -*- coding: utf-8 -*-
"""
YAML設定を最優先にして、triangular_matrix_seq_*.npy を学習/テストにまとめる。
"""

import os
import sys
import argparse
from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm

try:
    import yaml
except ImportError:
    print("[error] PyYAML が見つかりません。pip install pyyaml を実行してください。", file=sys.stderr)
    sys.exit(1)

def _ensure_parent_dir(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)

def load_cfg(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # 必須/デフォルト
    cfg.setdefault("input_folder", "visualizations_large")
    cfg.setdefault("filename_pattern", "triangular_matrix_seq_*.npy")
    cfg.setdefault("train_out", "united_triangular_matrices.npy")
    cfg.setdefault("test_out", "united_triangular_matrices_test.npy")
    cfg.setdefault("save_filelists", True)
    cfg.setdefault("filelists_prefix", "united")
    cfg.setdefault("train_count", 45)
    cfg.setdefault("test_count", 5)
    cfg.setdefault("seed", 42)           # null なら完全ランダム
    cfg.setdefault("strict_shape", True)
    return cfg


def list_npy_files(input_folder: str, pattern: str):
    # 安全のため、ソートして決定的順序に
    paths = sorted(glob.glob(str(Path(input_folder) / pattern)))
    return [Path(p) for p in paths]


def stack_files(files, t_ref=None, h_ref=None, w_ref=None, strict=True):
    seqs = []
    for p in tqdm(files, desc="Loading"):
        arr = np.load(p)
        if t_ref is None:
            t_ref, h_ref, w_ref = arr.shape
        if strict and arr.shape != (t_ref, h_ref, w_ref):
            raise ValueError(f"形状不一致: {p.name} が {arr.shape}（期待 {(t_ref,h_ref,w_ref)}）")
        seqs.append(arr)
    return np.stack(seqs, axis=0)  # (num_seq, t, h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", type=str, required=True,
                    help="YAML 設定ファイルへのパス（YAMLが最優先）")
    # YAML最優先のため、他のCLI引数は用意しません
    args = ap.parse_args()

    yaml_path = Path(args.config)
    cfg = load_cfg(yaml_path)

    input_folder   = cfg["input_folder"]
    filename_pat   = cfg["filename_pattern"]
    train_out      = cfg["train_out"]
    test_out       = cfg["test_out"]
    train_count    = int(cfg["train_count"])
    test_count     = int(cfg["test_count"])
    seed           = cfg["seed"]  # None / int
    strict_shape   = bool(cfg["strict_shape"])
    save_filelists = bool(cfg["save_filelists"])
    lists_prefix   = cfg["filelists_prefix"]

    # ファイル列挙
    files = list_npy_files(input_folder, filename_pat)
    if len(files) < train_count + test_count:
        raise ValueError(f"ファイルが足りません: 見つかった数 {len(files)}, 必要 {train_count + test_count}")

    # 乱数分割
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(int(seed))

    perm = rng.permutation(len(files))
    train_idx = perm[:train_count]
    test_idx  = perm[train_count:train_count + test_count]
    train_files = [files[i] for i in train_idx]
    test_files  = [files[i] for i in test_idx]

    # 形状基準の取得（最初のtrainファイル）
    first = np.load(train_files[0])
    t_ref, h_ref, w_ref = first.shape

    # 読み込み&スタック
    train_array = stack_files(train_files, t_ref, h_ref, w_ref, strict=strict_shape)
    test_array  = stack_files(test_files,  t_ref, h_ref, w_ref, strict=strict_shape)

    # 保存
    _ensure_parent_dir(train_out)
    _ensure_parent_dir(test_out)
    np.save(train_out, train_array)
    np.save(test_out,  test_array)


    # ファイル名リスト（プレフィックスにディレクトリを含めてもOK）
    if save_filelists:
        train_list_path = f"{lists_prefix}_train_filenames.txt"
        test_list_path  = f"{lists_prefix}_test_filenames.txt"
        _ensure_parent_dir(train_list_path)
        _ensure_parent_dir(test_list_path)
        np.savetxt(train_list_path, [p.name for p in train_files], fmt="%s", encoding="utf-8")
        np.savetxt(test_list_path,  [p.name for p in test_files],  fmt="%s", encoding="utf-8")


    print(f"[done] train -> {train_out}  shape={train_array.shape}  files={len(train_files)}")
    print(f"[done] test  -> {test_out}   shape={test_array.shape}   files={len(test_files)}")


if __name__ == "__main__":
    main()
