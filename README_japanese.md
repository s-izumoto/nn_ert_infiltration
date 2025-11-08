# ERT — 土壌への水浸透 + ニューラル補完

OpenFOAM による物理ベースのシミュレーションと LSTM/Seq2Seq モデルを組み合わせ、電気探査（ERT: Electrical resistivity tomography）測定で得られた**土壌への水浸透中の見かけ比抵抗場**を高解像度・時系列的に再構成します。元の ERT 測定データを補間することで**時間／空間分解能が約15倍向上**することを期待できます。実際の結果例は、`movie/` フォルダー内の動画で確認できます。

---

## 要約 (TL;DR)
- **目的:** 合成データで学習したニューラルネットワークを用いて、**土壌への水浸透**過程の ERT 測定における空間的・時間的ギャップを補完すること。より具体的には、測定範囲のみ更新された測定マップから全領域の真値マップを推定する。
- **前提:** 本ワークフローでは、**後述の測定条件に対応した多様な土壌条件での数値計算結果が必須入力**となる。これらのデータは、例えばオープンソースの流体シミュレーションソフトウェア OpenFOAM を用いて事前にシミュレーションされた浸潤過程の数値計算結果などから得られ、ERT 測定を模擬的に再現するための入力として利用される。
- **pyGIMLi:** オープンソースの地球物理シミュレーションライブラリ。本ワークフローでは、OpenFOAM で得られた導電率分布をもとに、電気探査（ERT: Electrical Resistivity Tomography）を数値的に再現し、見かけ比抵抗マップを生成するために使用する。
- **ワークフロー:** OpenFOAMシミュレーション → 抵抗率マップ → pyGIMLi → 測定設計　→　訓練データ作成 → LSTM 学習・推論。  
- **使用スクリプト:** `01_generateAppRes.py` ～ `10_comparePredAndNorm.py`。
- **設定:** すべてのスクリプトは YAML 設定ファイルを使用。

---

## プロジェクト構成

```
├── scripts/
│   ├── 01_generateAppRes.py        # OpenFOAM出力の各時系列に対し、pyGIMLi（Wenner–alpha）でERTを数値再現し、見かけ比抵抗の真値（基準）マップを計算して三角行列として出力
│   ├── 02_uniteTriangular.py       # 三角行列を統合して学習／テストセットに分割
│   ├── 03_generateMeasDesign.py    # 真値マップと測定マップの差分が最大となる位置を測定点として選定し、各時系列データに対して最適な測定設計を作成
│   ├── 04_generateTrainingData.py  # 測定設計の中から代表を1つ選び、全時系列データに適用して測定マップを再構成
│   ├── 05_trainingSequence.py      # 測定マップの時系列差分 Δ から真値マップの Δ を予測する LSTM Encoder–Decoder（Seq2Seq）を学習
│   ├── 06_inferSequence.py         # 学習済みモデルを用い、Δ の連続データを自己回帰的に推論
│   ├── 07_trainingFirst.py         # 最初の差分 Δ( t=0→1 ) を回帰する単一出力 LSTM を学習
│   ├── 08_inferFirst.py            # 予測した Δ と 初期伝導率を合成し、t=1 の伝導率マップを再構成
│   ├── 09_inferWhole.py            # 初期伝導率と Δ シリーズを累積して全時系列の伝導率マップを再構成
│   ├── 10_comparePredAndNorm.py    # 予測値と測定値を共通カラースケールで比較
│   └── XX_*.py                     # 補助スクリプト（動画変換、描画、バッチ処理など）
├── configs/                        # 各段階の YAML 設定ファイル
├── movies/                         # 出力されたアニメーション（MP4）
├── .gitignore
└── README.md
```

---

## 測定条件
- **電極数:** 32 本  
- **電極間隔:** 4 cm
- **測定間隔:** 5 秒
- **測定パターン:** 04_generateTrainingData.pyで選ばれた測定設計
- **浸潤条件:** 土壌表面中央 30 cm を **0.0885 mol/L NaCl** 溶液で常時飽和状態に維持  

---

## **OpenFOAM 出力 → NumPy 入力（`.npy`）仕様**

本ワークフローでは、OpenFOAM などで事前にシミュレーションされた浸潤過程の結果を、  
NumPy 配列ファイル（`.npy`）として入力します。`01_generateAppRes.py` はこのファイルを読み込み、  
pyGIMLi による Wenner–alpha 配列の順解析を行います。

---

### **ドメイン（物理サイズ）**
- シミュレーション領域は **幅 3.0 m × 高さ 1.0 m**（**y ∈ [0, 3.0] m**, **x ∈ [0, 1.0] m**）です。  
  原点は左下隅 (**x = 0.0 m**, **y = 0.0 m**) を想定します。  
  入力配列の `(Ny, Nx)` はこの物理サイズに対応した等間隔グリッドです。

#### **ファイル形式**
- **ファイル:** 1 つの NumPy 配列ファイル（`.npy`）  
  例: `data/combined_conductivity_maps.npy`（YAML の `data_path` で指定）

### **配列形状**
```
(N_seq, N_time, Ny, Nx)
```
- **N_seq:** 異なるシナリオ／フィールド（ケース）数（1 でも可）  
- **N_time:** 出力する時刻数（例：t = 0 … T）  
- **Ny, Nx:** 2D グリッド（行 = Y〈上向き〉, 列 = X〈右向き〉）の画素数（全時刻・全時系列データで固定）  

### **物理量と単位**
- **電気伝導率 σ [S/m]**（スカラー、正の実数）  
  `01_generateAppRes.py` 内部で  
  $$\\rho = \\frac{1}{\\sigma}$$
  に変換し、**見かけ比抵抗 ρₐ [Ω·m]** を計算します。

---

## インストール方法

### 1️⃣ Conda 環境を作成
```bash
# conda-forge チャンネルを有効にした Mambaforge / Miniconda の使用を推奨
mamba env create -f environment.yml
# または
conda env create -f environment.yml
```

### 2️⃣ 環境を有効化
```bash
# environment.yml に定義された環境名（例：nn-ert-infiltration）を使用
conda activate nn-ert-infiltration
```

### 3️⃣ インストール確認
```bash
python -c "import torch, pygimli, numpy; print('Torch:', torch.__version__); print('PyGIMLi:', pygimli.__version__)"
```

バージョンが表示されれば設定は完了です。

---

## ⚙️ 一連のパイプライン（実行例）

### **01_generateAppRes.py — 順解析による三角行列生成**
OpenFOAM出力の各時系列に対し、pyGIMLiでWenner–alpha配列を用いて見かけ比抵抗マップを算出し、三角行列として出力します。  

**実行例:**
```bash
python 01_generateAppRes.py --config configs/generateAppRes.yml
```

---

### **02_uniteTriangular.py — 三角行列の統合とデータ分割**
出力された三角行列スタックを統合し、学習用とテスト用のデータセットに分割します。  

**実行例:**
```bash
python 02_uniteTriangular.py --config configs/uniteTriangular.yml
```

---

### **03_generateMeasDesign.py — 測定設計の自動選定**
真値マップと測定マップの差分が最大となる位置を測定点として選定し、測定マップを時系列的に更新します。  

**実行例:**
```bash
python 03_generateMeasDesign.py --config configs/generateMeasDesign.yml
```

---

### **04_generateTrainingData.py — 測定系列データの生成**
代表的な測定列（例: 1本）を選び、全時系列に適用して測定マップ系列を構成します。  

**実行例:**
```bash
python 04_generateTrainingData.py --config configs/generateTrainingData.yml
```

---

### **05_trainingSequence.py — LSTM Encoder–Decoder（Seq2Seq）による時系列学習**
測定マップの時系列差分Δから真値マップのΔを予測するLSTM Encoder–Decoder（Seq2Seq）を学習します。  

**実行例:**
```bash
python 05_trainingSequence.py --config configs/trainingSequence.yml
```

---

### **06_inferSequence.py — 学習済Seq2SeqモデルによるΔ系列の推論**
学習済みのSeq2Seqモデルを用いて、測定系列から全時系列のΔを推論します。  

**実行例:**
```bash
python 06_inferSequence.py --config configs/inferSequence.yml
```

---

### **07_trainingFirst.py — 1ステップ先の導電率マップ予測**
初期測定値から次時刻（t=1）の導電率マップを予測する1ステップモデルを学習します。  

**実行例:**
```bash
python 07_trainingFirst.py --config configs/trainingFirst.yml
```

---

### **08_inferFirst.py — 初期Δと導電率マップの再構成**
学習済みの1ステップモデルを用いて、初期測定値からt=1の導電率マップを再構成します。  

**実行例:**
```bash
python 08_inferFirst.py --config configs/inferFirst.yml
```

---

### **09_inferWhole.py — Δ累積による全時系列の復元**
初期導電率と予測Δ系列を累積して、全時系列の導電率マップを再構成します。  

**実行例:**
```bash
python 09_inferWhole.py --config configs/inferWhole.yml
```

---

### **10_comparePredAndNorm.py — 予測値と測定値の比較可視化**
再構成された導電率マップ（予測）と測定マップを共通カラースケールで比較・出力します。  

**実行例:**
```bash
python 10_comparePredAndNorm.py --config configs/comparePredAndNorm.yml
```

---

## 📦 I/O の流れ

1. **OpenFOAM → pyGIMLi（01）**  
　`combined_conductivity_maps.npy` → `triangular_matrix_seq_*.npy`

2. **統合（02）**  
　各シーケンスを結合し、`united_triangular_matrices.npy`（学習）と `_test.npy`（評価）を生成

3. **測定マップ作成（03 / 04）**  
　真値マップから測定マップ系列を生成（最大差分または代表列による）

4. **学習（05 / 07）**  
　LSTM Encoder–Decoder（Seq2Seq）または1ステップモデルを訓練

5. **推論（06 / 08）**  
　学習済モデルを用いてΔ系列または初期Δを推論

6. **復元・評価（09）**  
　初期値とΔ系列を累積して全時系列を再構成し、MAPEなどの誤差を評価

7. **可視化（10）**  
　予測値と測定値を共通スケールで並列表示して比較出力

---

## 各スクリプトの詳細説明
このセクションでは、各スクリプトの目的と主な入出力をまとめています。ファイル名やパスは YAML の設定に従います。

### **01_generateAppRes.py — 順解析による三角行列生成**  
**目的:**  
OpenFOAMで得た2D導電率マップ（時系列）を入力として、pyGIMLiの**Wenner–alpha 配列**で**見かけ比抵抗（真値マップ）**を数値的に再現し、各時刻の結果を**三角行列形式（29, 26, 23, …, 1）**に整形して出力します。  

**処理概要:**  
1. 導電率フィールドを読み込み、抵抗率に変換  
2. 指定した電極配置（32電極・4 cm間隔）でERT順解析を実行  
3. 見かけ比抵抗を取得し、三角行列キャンバスに再配置  
4. 時系列ごとのスタックとして保存（任意でプレビュー出力）  

**入出力:**  
- **入力:** 
  - OpenFOAM由来の導電率フィールド（NumPy, shape = N×T×H×W）  
- **出力:** 
  - 三角行列見かけ比抵抗スタック（真値マップ, `.npy`）  

### **02_uniteTriangular.py — 三角行列データの統合と分割**  
**目的:**  
`01_generateAppRes.py`で生成された複数の**三角行列見かけ比抵抗ファイル**（例：`triangular_matrix_seq_000.npy`）を読み込み、  
形状を確認したうえで**学習用／テスト用データセット**に分割・統合します。  

**処理概要:**  
1. 指定フォルダ内の `.npy` ファイルを探索（例：`data/simulationAppRes/triangular_matrix_seq_*.npy`）  
2. YAML設定に基づいて**ランダムにtrain/testへ分割**（再現性あり）  
3. 各ファイルを読み込み `(T, H, W)` の形でスタック  
4. `(N_train, T, H, W)` および `(N_test, T, H, W)` 形式で保存  
   （オプションでファイル名リストも保存）  

**入出力:**  
- **入力:** 
  - 各シーケンスの三角行列見かけ比抵抗ファイル（`triangular_matrix_seq_*.npy`, shape = T×H×W）  
- **出力:**  
  - 統合済み学習データ (`united_triangular_matrices.npy`, shape = (N_train, T, H, W))
  - 統合済みテストデータ (`united_triangular_matrices_test.npy`, shape = (N_test, T, H, W))
  - （任意）分割に用いたファイル名リスト（`.txt`）  


### **03_generateMeasDesign.py — 測定設計データの生成**  
**目的:**  
`02_uniteTriangular.py`で統合した**真値（基準）比抵抗マップ**をもとに、
時系列ごとに「真値と測定値の差が大きい位置」を選び出し、  
その場所を新たに測定する過程をシミュレーションして  
**測定値データ（measured）**と**測定位置データ（indices）**を生成します。  

**処理概要:**  
1. 真値マップを読み込み、初期測定マップを t=0 の真値で初期化  
2. 各時刻 t において、真値との差分を計算  
3. 差が最大の位置（上位 K 点）を測定点として選定  
4. 選定点の真値を測定マップに反映し更新  
5. 測定マップ、測定位置、確率マップ（任意）を保存  
6. 指定した時系列データについては測定過程を可視化（フレーム出力）  

**入出力:**  
- **入力:** 
  - 真値比抵抗マップ（`united_triangular_matrices.npy`, shape = N_seq×N_time×H×W）
- **出力:**  
  - 測定マップ時系列：`measured_training_data.npy`（shape = N_seq×(N_time−1)×H×W）  
  - 測定位置インデックス：`measurement_indices.npy`（shape = N_seq×(N_time−1)×K×2）  
  - （任意）確率マップ：`y_probabilities.npy`  
  - （任意）可視化フレーム：`frames_training_data/`  

### **04_generateTrainingData.py — 測定マップの再構成（時系列シミュレーション）**  
**目的:**  
`03_generateMeasDesign.py` で得られた**測定設計（位置情報）**と、  
`02_uniteTriangular.py` で統合された**真値比抵抗マップ**を用いて、  
時系列ごとに「同じ測定パターンを適用した測定値マップ（measured）」を生成します。  
これにより、すべての時系列データで一貫した測定プロセスを再現し、  
**学習用データ**を作成します。  

**処理概要:**  
1. YAML設定を読み込み、真値比抵抗データ（train/test）と測定位置データ（positions）をロード  
2. 複数ある候補の中から、代表となる測定位置の時系列を1つ選択  
   - `"median"` モード: 測定位置の全時系列的な中央値に最も近いパターンを自動選択
   - `"fixed"` モード: あらかじめ指定したインデックスを選択  
3. 各時系列データについて、以下をシミュレーション  
   - 初期状態 `t=0` は真値マップそのまま（初期測定）  
   - 以降の各時刻で、選択された座標 (col, row) の値を真値から取得し、測定マップを更新  
   - 逐次的に更新したマップを記録  
4. すべての時系列データに対し、時系列測定マップをスタックして出力  
   （必要に応じて test セットにも同一の測定位置の時系列を適用）  

**入出力:**  
- **入力:**  
  - 真値比抵抗マップ：`united_triangular_matrices.npy`（shape = N×T×H×W）
  - （任意）テスト用真値比抵抗マップ：`united_triangular_matrices_test.npy`（shape = N×T×H×W）
  - 測定位置の時系列：`measurement_indices.npy`（shape = S×T×2, 各要素 [col, row]）

- **出力:**  
  - 測定マップ：`training_data{index}.npy`（shape = N×(T−1)×H×W）  
  - （任意）テスト用測定マップ：`training_data{index}_test.npy`  
  - 選択されたシーケンス番号：`chosen_seq_index.npy`  

**特徴:**  
- 真値データ中の NaN を自動的に 0 へ置換  
- 各タイムステップで1点のみ更新（`num_measurements=1`）  
- 測定位置の選定を「中央値」または「固定値」で統一可能  
- 出力ファイル名にシーケンス番号を付与し再現性を保持  

### **05_trainingSequence.py — LSTM Encoder–Decoder（Seq2Seq）学習**  
**目的:**  
`04_generateTrainingData.py` で作成した **測定マップ（measured）** と、`02_uniteTriangular.py` の **真値マップ（united）** を用いて、  
**短い履歴ウィンドウから将来の真値（三角行列マップ）を予測**する 2 層 LSTM の Encoder–Decoder（Seq2Seq）モデルを学習します。  
- `preprocess.diff: true` のとき：**測定マップの時系列差分 Δ** から **真値の時系列差分 Δ** を予測  
- `preprocess.diff: false` のとき：**測定マップ（レベル）** から **真値（レベル）** を予測

**処理概要:**  
1. **入力データの読み込み**（`.npy/.npz` 単体 or ディレクトリ内一括）し、形状を `(N, T, H, W)` に正規化  
2. **前処理の適用**（YAML 指定）：時間切り出し、間引き、差分化、正規化、平均センタリング、時間コンテキスト付与など  
3. **エンコーダ用の履歴長 `time_steps` とデコーダ出力長 `output_seq_length`** に基づき、  
   各時系列データから **30 ステップ分の過去データを入力として切り出し、直後の 29 ステップを予測する学習サンプルを作成**。  
   デフォルト設定（`trainingSequence.yml`）では、  
   この 1 セットのみを使用して **「30 ステップの履歴から、次の 29 ステップの真値の変化（Δ）を予測する」** 学習を行います。  
   - 画像は **Wenner 配列由来の三角形グリッド** であるため、`row_sizes = [29, 26, 23, …, 2]` に従って各マップを **1 次元ベクトル** に変換（`create_array` 関数）  
   - 可視化や評価時には、`de_create_array` で元の三角行列に復元  
4. **学習と評価**  
   - 学習率やバッチサイズの組み合わせを変えてモデルを複数回学習し、  
     **検証データに対する誤差の平均値が最も小さい設定**を自動的に選択  
   - 選ばれた設定でもう一度（80%で再学習）実行し、最終モデルと学習履歴を保存  

**入出力:**  
- **入力:**  
  - 測定マップ（measured）(`training_data{index}.npy`, shape = (N, T, H, W))  
  - 真値マップ（united）(`united_triangular_matrices.npy`, shape = (N, T, H, W))  
  - ※ フォルダ指定時は **measured と united のファイル数と順序が一致**している必要あり  
- **出力（`results_dir` に保存）:**  
  - グリッドサーチ結果：`grid_search_results.csv`、`best_config.txt`、fold 別 CSV、`best_cv_indices.npz`  
  - ベスト設定での再学習（任意）：`best_model.pt`、`loss_history_best_retrain.csv`、`loss_curve_best_retrain.png`、`best_retrain_indices.npz`  
  - 前処理メタデータ（フラグ有効時）：`normalization_factors_*.npz`、`mean_values.npz`  


### **06_inferSequence.py — 学習済み Seq2Seq による自己回帰推論（Δ の連続予測）**  
**目的:**  
`05_trainingSequence.py` で学習した **LSTM Encoder–Decoder（Seq2Seq）モデル** を用いて、  
**30 ステップの履歴**から **次の 29 ステップの真値マップの変化（Δ）** を  
**自己回帰的（autoreg）** に連続推論します。  
入力は `04_generateTrainingData.py` で生成した **測定マップ（measured）** と、  
初期化および正規化参照のための **真値マップ（united）** を使用します。

**処理概要:**  
1. **入力データと学習済みモデルの読み込み**  
   - measured と united を `.npy` 形式で読み込み  
   - 必要に応じて差分化（Δ）、正規化、平均センタリングを実施  
   - t=0 の真値を初期フレームとして measured の先頭に結合し、モデル入力を構成  
2. **三角行列マップの変換**  
   - Wenner 配列に由来する三角形グリッドを考慮し、`row_sizes = [29, 26, 23, …, 2]` に従って各フレームを **1 次元ベクトル** に変換（`create_array`）  
   - 予測結果は `de_create_array` で再び **三角行列形式（2D マップ）** に復元  
3. **自己回帰的な Δ 予測（Autoregressive Inference）**  
   - エンコーダへ **30 ステップ分の履歴**を入力  
   - デコーダは **ゼロ初期化**で開始し、**29 ステップ**分の Δ を連続生成  
   - 各ステップの出力 Δ を次の入力に再帰的に利用（teacher forcing なし）  
4. **出力の保存・可視化（任意）**  
   - 予測結果を `.npy` ファイルとして保存（形状：`N_series × T_pred × H × W`）  
   - 任意で、合計マップ・t=0・最終時刻などを PNG で自動出力  

**入出力:**  
- **入力:**  
  - **測定マップ（measured）:** `04_generateTrainingData.py` で作成されたファイルまたはフォルダ（例：`training_data{index}.npy`）  
  - **真値マップ（united）:** `02_uniteTriangular.py` で作成された統合真値ファイル（例：`united_triangular_matrices.npy`）  
  - **学習済みモデル:** `05_trainingSequence.py` の出力（例：`best_model.pt` または `checkpoints/` ディレクトリ）  
  - **正規化ファイル（任意）:** 学習時に保存された `normalization_factors_*.npz` や `mean_values.npz`  

- **出力:**  
  - 予測 Δ マップ：`<measured_stem>__pred_seq.npy`（形状：N_series×T_pred×H×W）  
  - （任意）可視化結果：`pred_png/` 内に PNG 出力（合計図・t0・最終時刻など）  

**特徴:**  
- 学習時と同様に Δ ベースでの推論（`use_diff: true`）に対応  
- 入力フォルダ内の複数ファイルを一括処理可能  
- 学習時に用いた正規化や平均値を自動検出・適用  
- CUDA が利用可能な環境では自動的に GPU 推論を実行  

### **07_trainingFirst.py — 初期差分 Δ(t=0→1) の単一出力 LSTM 学習**  
**目的:**  
`04_generateTrainingData.py` で生成した **測定マップ（measured）** と、`02_uniteTriangular.py` の **真値マップ（united）** を用いて、  
**最初の時間差分 Δ(t=0→1)** を予測する **単一出力 LSTM 回帰モデル** を学習します。  
全時系列を扱う `05_trainingSequence.py` の短期版にあたり、最初の変化のみを対象として学習を行います。  

**処理概要:**  
1. **測定・真値データの読み込みと導電率変換**（ρ→σ = 1000/ρ）を行い、NaNを0に置換  
2. **真値の初期フレーム（t=0）を測定データの先頭に追加**し、モデルの入力系列を構成  
3. **前処理を適用**（YAML設定に従い、時間切り出し、間引き、差分化、正規化、平均センタリングなど）  
4. **短い履歴長 `time_steps`（例：4 ステップ）を入力とし、最初の出力 Δ(t=0→1) を予測する学習サンプルを作成**  
   - 各フレームは Wenner 配列由来の三角形グリッド構造を考慮し、`row_sizes = [29, 26, 23, …, 2, 1]` に従って  
     **1 次元ベクトルに変換（`create_array`）**  
   - 時間チャネル（`time_context`）の追加もオプションで対応  
5. **2 層 LSTM + 全結合層による回帰モデルを学習**  
   - 複数の学習率とバッチサイズの組み合わせで **K-fold 交差検証** を実施  
   - **検証誤差の平均値が最も小さい設定**を選択し、80/20 再学習を実施  
   - 最終モデル、学習曲線、損失履歴を保存  

**入出力:**  
- **入力:**  
  - 測定マップ（measured）：`training_data{seq}.npy.npy`（`04_generateTrainingData.py` 出力）  
  - 真値マップ（united）：`united_triangular_matrices.npy`（`02_uniteTriangular.py` 出力）  
- **出力:**  
  - 学習済みモデル：`best_model_first.pt`、`single_output_lstm_model.pt`  
  - 正規化・平均データ：`norm_input.npz`、`norm_output.npz`、`mean_values.npz`  
  - 学習ログ：`grid_search_results.csv`、`best_config.txt`、`best_cv_indices.npz`  
  - 再学習結果（任意）：`loss_history_best_retrain.csv`、`loss_curve_best_retrain_first.png`  

**特徴:**  
- 最初の変化 Δ(t=0→1) に焦点を当てた短期予測モデル  
- Wenner 配列の三角形構造を考慮した 1D フラット化処理  
- YAML による設定一元化と再現性の高い実験設計  
- 軽量 LSTM による高速な学習  


### **08_inferFirst.py — 予測 Δ と初期フレームを合成し、t=1 のマップを再構成**  
**目的:**  
`07_trainingFirst.py` で学習した **単一出力 LSTM** を用いて、予測した **最初の差分 Δ(t=0→1)** と **初期フレーム (t=0)** を合成し、**t=1 の三角行列マップ** を再構成します。学習時と同一の前処理（差分化・正規化・平均センタリング・単位変換など）を再現して推論します。

**処理概要:**  
1. **測定（measured）と真値（united）を読み込み**、必要に応じて単位反転（1/x）やスケーリングを適用  
2. **t=0 の真値フレームを measured の先頭に連結**し、学習時と同じ履歴長で入力系列を構成  
3. **三角行列のフラット化／復元**（`row_sizes = [29, 26, 23, …, 2, 1]`）を行い、学習時の入出力形状に合わせて整形  
4. **学習済み LSTM を読み込み、最初の出力ステップ（Δ）を推論**  
5. **Δ と初期フレームを合成して t=1 マップを再構成**（平均センタリングや正規化を用いた場合は逆変換を適用）  
6. 予測結果を `.npy` として保存し、**Pred vs True** の比較図や入力系列のプレビューを PNG で出力（任意）

**入出力:**  
- **入力:**  
  - 測定マップ（measured）：`training_data{seq}.npy`（`04_generateTrainingData.py` の出力）  
  - 真値マップ（united）：`united_triangular_matrices.npy`（`02_uniteTriangular.py` の出力）  
  - 学習済みモデル：`best_model_first.pt` または `single_output_lstm_model.pt`（`07_trainingFirst.py` の出力）  
  - （任意）正規化・平均ファイル：`normalization_factors_*.npz`、`mean_values_*.npz`（学習時に保存したもの）  

- **出力:**  
  - 予測 t=1 マップ（スタック）：`pred_images_all.npy`（各系列の 2D 三角行列を格納）  
  - （任意）比較図：`compare_series_*.png`（Predicted vs True）  
  - （任意）入力フレーム可視化：`inputs_series_*.png`  

**特徴:**  
- **Δ(t=0→1) のみを再構成**するシンプルな推論フロー  
- 学習時と同一の前処理フラグを再現（差分・間引き・平均センタリング・時間コンテキスト 等）  
- Wenner 配列に合わせた **三角行列のフラット化／復元** を内蔵  
- フォルダ一括処理・比較図自動保存に対応

### **09_inferWhole.py — 初期値と Δ シリーズを累積して、全時系列マップを再構成**  
**目的:**  
`08_inferFirst.py`（t=0→1 の Δ）および `06_inferSequence.py`（以降の Δ 連続予測）の結果を用いて、  
**初期フレーム t=0** から **Δ（差分）を時間方向に累積**し、**全時系列の伝導率／抵抗率マップ**を再構成します。  
再構成結果を**真値データ**と整合させて評価（MAPE）し、必要に応じて比較画像を出力します。

**処理概要:**  
1. **入力の読み込み**：t=1 の「初期 Δ」、その後の **Δ シリーズ**、および **真値の全時系列**をロード  
2. **単位整合（必要な場合）**：安全な 1/x 変換（スケール・クリップ込み）で真値の物理量を合わせる  
3. **時系列再構成**：  
   - 基準値として **真値の t=0** を使用  
   - **初期 Δ（t=0→1）** と **Δ シリーズ（t=1→…）** を順に**累積和**して、絶対値系列（t=0,1,2,…）を生成  
   - 予測列と真値列の**長さを揃えて**比較可能な範囲に切り揃え  
4. **評価と可視化**：  
   - 各時刻の予測と真値から **MAPE（平均絶対百分率誤差）** を計算  
   - 任意で Pred / True / |Pred−True| の比較図を保存  

**入出力:**  
- **入力:**  
  - **Δ シリーズ（seq2seq 予測）**：`outputs_pred_all_series.npy`（形状：N×T_pred×H×W）  
  - **初期 Δ（t=0→1）**：`pred_images_all.npy`（形状：N×H×W）  
  - **真値の全時系列**：`united_triangular_matrices_test.npy`（形状：N×T_true×H×W）  
- **出力:**  
  - **再構成スタック（絶対値）**：`conductivity.npy`（形状：N×T×H×W）※T は整合後の最短長  
  - **評価サマリ**：`mape_values.txt`（系列ごとの MAPE[%] を列挙）  
  - **比較画像（任意）**：`measurement_locations_seq_{seq}_timestep_{time}.png`  

**特徴:**  
- **t=0 の真値**を基点に、**Δ を時間方向に累積**して絶対値を復元  
- 真値と予測の**長さ・単位を自動整合**（1/x 変換・クリップを含む安全処理）  
- **MAPE を標準出力とファイルに保存**、任意でフレーム比較図を自動生成  

### **10_comparePredAndNorm.py — 予測値と測定値を共通カラースケールで比較可視化**  
**目的:**  
全系列・全時刻に対して、**予測マップ（Predicted）** と **測定マップ（Measured）** を**同一カラースケール**で並置し、  
時間対応をとった PNG 画像として一括出力します（単位は図中ラベルどおり mS·m⁻¹）。

**処理概要:**  
1. **予測スタック**（N×T×H×W）と **測定スタック**（N×T×H×W）を読み込み  
2. 測定スタックはプロジェクト慣例どおり **1000/ρ** に変換して導電率表示に統一（無効値は NaN として可視化）  
3. **時間対応**は `measured_t = pred_t × k − 1` の規則に基づき、比較可能なペアのみを採用（範囲外はスキップ）  
4. **カラースケール**はシーケンス単位または全体から自動決定し、Pred/Measured に**共通の vmin/vmax**を適用  
5. 各ペアについて **Pred vs Measured** の並置図を保存（進捗を表示しつつ一括処理）  
*この挙動はスクリプト実装に沿っています。* :contentReference[oaicite:0]{index=0}

**入出力:**  
- **入力:**  
  - 予測スタック（例）：`conductivity.npy`（形状：N×T_pred×H×W）  
  - 測定スタック（例）：`training_data{seq}.npy`（形状：N×T_meas×H×W）  
- **出力:**  
  - 比較画像（PNG）：`out_dir/seq{NNN}_predt{TTT}_meast{TTT}.png`（Pred／Measured を同一カラースケールで並置）

**特徴:**  
- **共通カラースケール**での厳密比較（系列単位 or 全体スキャンで自動レンジ決定）  
- **時間対応のルール化**により Pred と Measured の対応フレームのみを比較  
- **NaN 白表示**で無効値を明示（必要に応じて 0 置換で可視化も可）  
- カラーマップは既定（"hot"）だが、プロジェクト方針に合わせて変更可能


---
