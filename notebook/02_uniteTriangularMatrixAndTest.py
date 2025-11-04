# -*- coding: utf-8 -*-
"""
visualizations_large にある triangular_matrix_seq_*.npy（全50想定）から
ランダムで 45 を united_triangular_matrices.npy（学習）
残り 5 を united_triangular_matrices_test.npy（テスト）にまとめて保存
"""

import os
import numpy as np
from tqdm import tqdm

# ==== 設定 ====
input_folder = "visualizations_large"
train_out = "united_triangular_matrices.npy"
test_out  = "united_triangular_matrices_test.npy"
train_count = 45
seed = 42   # 再現したいときに固定。完全ランダムで良ければ None に

# ==== ファイル列挙 ====
npy_files = sorted(
    f for f in os.listdir(input_folder)
    if f.startswith("triangular_matrix_seq_") and f.endswith(".npy")
)
if len(npy_files) < train_count + 5:
    raise ValueError(f"ファイルが足りません: 見つかった数 {len(npy_files)}")

# ==== ランダム分割 ====
rng = np.random.default_rng(seed)
perm = rng.permutation(len(npy_files))
train_idx = perm[:train_count]
test_idx  = perm[train_count:train_count+5]

train_files = [npy_files[i] for i in train_idx]
test_files  = [npy_files[i] for i in test_idx]

# ==== 形状チェックのため最初のファイルを読む ====
def load_one(path):
    return np.load(os.path.join(input_folder, path))

first = load_one(train_files[0])
t, h, w = first.shape

# ==== ロード＆スタック ====
def stack_files(files):
    seqs = []
    for f in tqdm(files, desc="Loading"):
        arr = load_one(f)
        if arr.shape != (t, h, w):
            raise ValueError(f"形状不一致: {f} が {arr.shape}（期待 {(t,h,w)}）")
        seqs.append(arr)
    return np.stack(seqs, axis=0)  # (num_seq, t, h, w)

train_array = stack_files(train_files)
test_array  = stack_files(test_files)

# ==== 保存 ====
np.save(train_out, train_array)
np.save(test_out,  test_array)

# どのファイルを使ったかのメモも保存（任意）
np.savetxt("united_train_filenames.txt", train_files, fmt="%s", encoding="utf-8")
np.savetxt("united_test_filenames.txt",  test_files,  fmt="%s", encoding="utf-8")

print(f"[done] train  -> {train_out}  shape={train_array.shape}  files={len(train_files)}")
print(f"[done] test   -> {test_out}   shape={test_array.shape}   files={len(test_files)}")
