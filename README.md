# ERT — Water Infiltration into Soil + Neural Completion

This workflow combines **physics-based simulations using OpenFOAM** with **LSTM/Seq2Seq neural network models** to reconstruct **apparent resistivity fields during water infiltration into soil** obtained from Electrical Resistivity Tomography (ERT) measurements with high temporal and spatial resolution.  
By interpolating the original ERT measurements, the workflow can potentially enhance the **temporal and spatial resolution by approximately ×15**. Example results can be viewed as videos in the `movie/` folder.

---

## Summary (TL;DR)
- **Objective:** To fill spatial and temporal gaps in ERT measurements of **water infiltration into soil** using a neural network trained on synthetic data.  
  Specifically, the model estimates the full-domain true resistivity map from measured maps in which only a limited area has been updated.
- **Prerequisite:** This workflow requires **numerical simulation results under various soil conditions corresponding to the measurement settings** described later.  
  These data are obtained, for example, from simulations of infiltration processes conducted in advance using the open-source fluid-dynamics software **OpenFOAM**, and serve as inputs for reproducing ERT measurements numerically.
- **pyGIMLi:** An open-source geophysical simulation library.  
  In this workflow, it is used to numerically reproduce Electrical Resistivity Tomography (ERT) based on the conductivity distributions obtained from OpenFOAM and to generate apparent resistivity maps.
- **Workflow:** OpenFOAM simulation → Resistivity map → pyGIMLi → Measurement design → Training data generation → LSTM training/inference  
- **Scripts used:** `01_generateAppRes.py` to `10_comparePredAndNorm.py`  
- **Configuration:** All scripts use YAML configuration files.

---
## Project Structure

```
├── scripts/
│   ├── 01_generateAppRes.py        # For each time step of OpenFOAM output, numerically reproduce ERT using pyGIMLi (Wenner–alpha array), compute the true (reference) apparent resistivity map, and export it as a triangular matrix
│   ├── 02_uniteTriangular.py       # Combine triangular matrices and split them into training and test sets
│   ├── 03_generateMeasDesign.py    # Select measurement points where the difference between the true and measured maps is largest, and create optimized measurement designs for each time series
│   ├── 04_generateTrainingData.py  # Choose one representative measurement design and reconstruct measured maps for all time-series data
│   ├── 05_trainingSequence.py      # Train an LSTM Encoder–Decoder (Seq2Seq) model to predict the Δ of the true map from the Δ of the measured map
│   ├── 06_inferSequence.py         # Perform autoregressive inference of Δ sequences using the trained model
│   ├── 07_trainingFirst.py         # Train a single-output LSTM to regress the first difference Δ( t=0→1 )
│   ├── 08_inferFirst.py            # Combine the predicted Δ with the initial conductivity to reconstruct the conductivity map at t=1
│   ├── 09_inferWhole.py            # Accumulate the initial conductivity and the Δ series to reconstruct the full time-series conductivity maps
│   ├── 10_comparePredAndNorm.py    # Compare predicted and measured values under a common color scale
│   └── XX_*.py                     # Auxiliary scripts (e.g., video conversion, visualization, batch processing)
├── configs/                        # YAML configuration files for each stage
├── movies/                         # Generated animation outputs (MP4)
├── .gitignore
└── README.md
```

---

## Measurement Conditions
- **Number of electrodes:** 32  
- **Electrode spacing:** 4 cm  
- **Measurement interval:** 5 seconds  
- **Measurement pattern:** The measurement design selected in `04_generateTrainingData.py`  
- **Infiltration condition:** The central 30 cm of the soil surface is continuously saturated with a **0.0885 mol/L NaCl** solution  

---
## **OpenFOAM Output → NumPy Input (`.npy`) Specification**

In this workflow, results from pre-simulated infiltration processes using OpenFOAM (or other simulators)  
are provided as NumPy array files (`.npy`). The script `01_generateAppRes.py` reads this file and performs  
forward modeling with pyGIMLi using a Wenner–alpha electrode array configuration.

---

### **Domain (Physical Dimensions)**
- The simulation domain has dimensions **width = 3.0 m × height = 1.0 m** (**y ∈ [0, 3.0] m**, **x ∈ [0, 1.0] m**).  
  The origin is assumed to be at the bottom-left corner (**x = 0.0 m**, **y = 0.0 m**).  
  The input array `(Ny, Nx)` corresponds to an evenly spaced grid over this physical size.

#### **File Format**
- **File:** A single NumPy array file (`.npy`)  
  Example: `data/combined_conductivity_maps.npy` (specified by `data_path` in the YAML configuration)

### **Array Shape**
```
(N_seq, N_time, Ny, Nx)
```
- **N_seq:** Number of different scenarios/fields (cases) — can be 1 or more  
- **N_time:** Number of time steps output (e.g., t = 0 … T)  
- **Ny, Nx:** 2D grid size (rows = Y direction [upward], columns = X direction [rightward]),  
  consistent across all time steps and all sequences  

### **Physical Quantity and Unit**
- **Electrical conductivity σ [S/m]** (scalar, positive real value)  
  Inside `01_generateAppRes.py`, it is converted as:  
  $$\\rho = \\frac{1}{\\sigma}$$
  to compute the **apparent resistivity ρₐ [Ω·m]**.

---

## Installation

### 1️⃣ Create a Conda Environment
```bash
# Recommended: use Mambaforge or Miniconda with the conda-forge channel enabled
mamba env create -f environment.yml
# or
conda env create -f environment.yml
```

### 2️⃣ Activate the Environment
```bash
# Use the environment name defined in environment.yml (e.g., nn-ert-infiltration)
conda activate nn-ert-infiltration
```

### 3️⃣ Verify Installation
```bash
python -c "import torch, pygimli, numpy; print('Torch:', torch.__version__); print('PyGIMLi:', pygimli.__version__)"
```

If the versions are displayed correctly, the setup is complete.

---

## ⚙️ Overall Pipeline (Execution Examples)

### **01_generateAppRes.py — Generate Triangular Matrices by Forward Modeling**
For each time step of OpenFOAM output, pyGIMLi performs forward modeling using the Wenner–alpha array to compute the apparent resistivity maps, which are exported as triangular matrices.

**Example:**
```bash
python 01_generateAppRes.py --config configs/01_generate.yml
```

---

### **02_uniteTriangular.py — Merge and Split Triangular Matrices**
Combine the exported triangular matrix stacks and split them into training and test datasets.

**Example:**
```bash
python 02_uniteTriangular.py --config configs/02_unite.yml
```

---

### **03_generateMeasDesign.py — Automatic Measurement Design Selection**
Select measurement points with the largest difference between the true and measured maps, and update the measured maps sequentially over time.

**Example:**
```bash
python 03_generateMeasDesign.py --config configs/03_meas_design.yml
```

---

### **04_generateTrainingData.py — Generate Measurement Series Data**
Select a representative measurement sequence (e.g., one row) and apply it to all time steps to construct the measured map series.

**Example:**
```bash
python 04_generateTrainingData.py --config configs/04_generate_training.yml
```

---

### **05_trainingSequence.py — LSTM Encoder–Decoder (Seq2Seq) Training**
Train an LSTM Encoder–Decoder (Seq2Seq) model to predict Δ of the true maps from Δ of the measured maps.

**Example:**
```bash
python 05_trainingSequence.py --config configs/05_seq_train.yml
```

---

### **06_inferSequence.py — Δ-Series Inference Using Trained Seq2Seq Model**
Use the trained Seq2Seq model to infer Δ over the entire time series from measured data.

**Example:**
```bash
python 06_inferSequence.py --config configs/06_seq_infer.yml
```

---

### **07_trainingFirst.py — One-Step Conductivity Prediction**
Train a single-step model that predicts the conductivity map at the next time step (t=1) from the initial measurement.

**Example:**
```bash
python 07_trainingFirst.py --config configs/07_first_train.yml
```

---

### **08_inferFirst.py — Reconstruct Initial Δ and Conductivity Map**
Use the trained one-step model to reconstruct the conductivity map at t=1 from the initial measurement.

**Example:**
```bash
python 08_inferFirst.py --config configs/08_first_infer.yml
```

---

### **09_inferWhole.py — Reconstruct the Full Time Series via Δ Accumulation**
Accumulate the initial conductivity and predicted Δ series to reconstruct the full time-series conductivity maps.

**Example:**
```bash
python 09_inferWhole.py --config configs/09_infer_whole.yml
```

---

### **10_comparePredAndNorm.py — Visual Comparison of Predictions and Measurements**
Compare reconstructed (predicted) and measured conductivity maps under a common color scale.

**Example:**
```bash
python 10_comparePredAndNorm.py --config configs/10_compare.yml
```

---

## 📦 I/O Flow

1. **OpenFOAM → pyGIMLi (Step 01)**  
   `combined_conductivity_maps.npy` → `triangular_matrix_seq_*.npy`

2. **Integration (Step 02)**  
   Combine all sequences to create `united_triangular_matrices.npy` (for training) and `_test.npy` (for evaluation).

3. **Measurement Map Generation (Steps 03 / 04)**  
   Generate measured map sequences from the true maps (using either maximum difference or a representative sequence).

4. **Training (Steps 05 / 07)**  
   Train the LSTM Encoder–Decoder (Seq2Seq) or the single-step model.

5. **Inference (Steps 06 / 08)**  
   Use the trained model to infer Δ sequences or the initial Δ.

6. **Reconstruction & Evaluation (Step 09)**  
   Reconstruct the entire time series by accumulating the initial value and Δ sequence, and evaluate errors such as MAPE.

7. **Visualization (Step 10)**  
   Display predicted and measured values side by side under a common color scale for comparison.

---

## **Detailed Description of Each Script**
Below is an explanation of the purpose and main outputs for each script.

---

### **01_generateAppRes.py — Triangular Matrix Generation via Forward Modeling**  
**Purpose:**  
Takes 2D conductivity maps (time series) obtained from OpenFOAM as input and uses pyGIMLi with the **Wenner–alpha array** to numerically reproduce the **apparent resistivity (true/reference maps)**.  
Each time step is reshaped into a **triangular matrix form (29, 26, 23, …, 1)** and exported as output.  

**Process Overview:**  
1. Load the conductivity field and convert it to resistivity  
2. Perform ERT forward modeling using the specified electrode configuration (32 electrodes, 4 cm spacing)  
3. Obtain apparent resistivity values and map them onto a triangular-matrix canvas  
4. Save as a time-series stack (optionally with preview output)  

**Input / Output:**  
- **Input:**  
  - Conductivity fields from OpenFOAM (NumPy, shape = N×T×H×W)  
  - YAML configuration file  
- **Output:**  
  - Triangular-matrix apparent-resistivity stack (true/reference maps, `.npy`)  

---

### **02_uniteTriangular.py — Integration and Splitting of Triangular-Matrix Data**  
**Purpose:**  
Loads multiple **triangular-matrix apparent-resistivity files** (e.g., `triangular_matrix_seq_000.npy`) generated by `01_generateAppRes.py`,  
checks their shapes, and **merges/splits them into training and test datasets**.  

**Process Overview:**  
1. Search the target folder for `.npy` files (e.g., `visualizations_large/triangular_matrix_seq_*.npy`)  
2. Randomly split them into **train/test** sets based on the YAML configuration (reproducible)  
3. Load each file and stack in the form `(T, H, W)`  
4. Save as `(N_train, T, H, W)` and `(N_test, T, H, W)` arrays  
   (optionally save file-name lists used for the split)  

**Input / Output:**  
- **Input:**  
  - Triangular-matrix apparent-resistivity files for each sequence (`.npy`, shape = T×H×W)  
  - YAML configuration file  
- **Output:**  
  - Combined training data: `(N_train, T, H, W)`  
  - Combined test data: `(N_test, T, H, W)`  
  - (Optional) File-name list used for the split (`.txt`)  

---

### **03_generateMeasDesign.py — Generation of Measurement-Design Data**  
**Purpose:**  
Based on the **true (reference) apparent-resistivity maps** merged by `02_uniteTriangular.py`,  
identify locations with the **largest differences between true and measured maps** for each time step.  
Simulate the process of newly measuring those locations to generate **measured-value data (`measured`)**  
and **measurement-location data (`indices`)**.  

**Process Overview:**  
1. Load the true maps and initialize the measured map using the true values at t=0  
2. For each time t, compute the difference between true and measured maps  
3. Select the top K locations with the largest differences as measurement points  
4. Update the measured map by inserting the true values at the selected points  
5. Save the measured maps, measurement indices, and (optionally) probability maps  
6. Optionally visualize the measurement process for selected time-series data (frame output)  

**Input / Output:**  
- **Input:**  
  - True apparent-resistivity maps (`.npy`, shape = N_seq×N_time×H×W)  
  - YAML configuration file  
- **Output:**  
  - Time-series measured maps: `measured_training_data.npy` (shape = N_seq×(N_time−1)×H×W)  
  - Measurement-location indices: `measurement_indices.npy` (shape = N_seq×(N_time−1)×K×2)  
  - (Optional) Probability maps: `y_probabilities.npy`  
  - (Optional) Visualization frames: `frames_training_data/`  

---

### **04_generateTrainingData.py — Reconstruction of Measured Maps (Time-Series Simulation)**  
**Purpose:**  
Using the **measurement design (location information)** obtained in `03_generateMeasDesign.py` and the **true apparent-resistivity maps** merged in `02_uniteTriangular.py`,  
this script generates time-series measured maps (`measured`) by applying the same measurement pattern to each time step.  
This process reproduces a consistent measurement workflow across all time-series data, producing **training data** for model learning.  

**Process Overview:**  
1. Load the YAML configuration and read the true apparent-resistivity data (train/test) and measurement positions (positions)  
2. Select one representative time-series measurement pattern among multiple candidates  
   - `"median"` mode: automatically select the pattern closest to the temporal median across all measurement positions  
   - `"fixed"` mode: use a pre-specified index  
3. For each time-series dataset, simulate the following:  
   - At the initial state `t=0`, use the true map directly (initial measurement)  
   - At each subsequent time step, update the measured map by replacing values at the selected coordinates (col, row) with the corresponding true values  
   - Record each updated map sequentially  
4. Stack all time-series measured maps and save as output  
   (Optionally, apply the same measurement sequence to the test set as well)  

**Input / Output:**  
- **Input:**  
  - True apparent-resistivity maps: `united_triangular_matrices.npy` (shape = N×T×H×W)  
  - Measurement position sequence: `positions_all.npy` (shape = S×T×2, each element [col, row])  
  - YAML configuration file  

- **Output:**  
  - Measured maps: `measured_training_data_sameRowColSeq{index}.npy` (shape = N×(T−1)×H×W)  
  - (Optional) Test measured maps: `measured_training_data_sameRowColSeq{index}_test.npy`  
  - Selected sequence index: `chosen_seq_index.npy`  

**Features:**  
- Automatically replaces NaN values in true data with 0  
- Updates only one pixel per time step (`num_measurements=1`)  
- Measurement locations can be fixed or median-based for consistency  
- Output filenames include sequence indices for reproducibility  

---

### **05_trainingSequence.py — LSTM Encoder–Decoder (Seq2Seq) Training**  
**Purpose:**  
Using the **measured maps** created in `04_generateTrainingData.py` and the **true maps (united)** from `02_uniteTriangular.py`,  
train a two-layer **LSTM Encoder–Decoder (Seq2Seq)** model that predicts future true triangular-matrix maps from short historical windows.  
- When `preprocess.diff: true`: predict **true-map time differences (Δ)** from **measured-map time differences (Δ)**  
- When `preprocess.diff: false`: predict **true-map levels** from **measured-map levels**  

**Process Overview:**  
1. **Load input data** (`.npy/.npz` single file or directory batch) and normalize to shape `(N, T, H, W)`  
2. **Apply preprocessing** (as specified in YAML): time slicing, subsampling, differencing, normalization, mean centering, temporal context addition, etc.  
3. **Based on encoder history length `time_steps` and decoder output length `output_seq_length`**,  
   extract training samples from each sequence, using **30 past steps as input and predicting the following 29 steps**.  
   In the default configuration (`trainingSequence.yml`),  
   one such set is used to train the model to **predict the next 29-step true Δ sequence from a 30-step measured Δ history**.  
   - Since the maps are **triangular grids from the Wenner array**, each frame is converted into a **1D vector** based on `row_sizes = [29, 26, 23, …, 2]` using the `create_array` function.  
   - During visualization or evaluation, the `de_create_array` function restores the original triangular shape.  
4. **Training and evaluation:**  
   - The model is trained multiple times with different learning-rate and batch-size combinations.  
   - The configuration with the **lowest average validation error** is automatically selected.  
   - The model is retrained with 80% of the data using the best configuration, and the final model and training history are saved.  

**Input / Output:**  
- **Input:**  
  - Measured maps: `(N, T, H, W)`  
  - True maps: `(N, T, H, W)`  
  - YAML configuration file (preprocessing/model/training settings)  
  - When specifying folders, **the number and order of measured and united files must match**.  
- **Output (saved in `results_dir`):**  
  - Grid-search results: `grid_search_results.csv`, `best_config.txt`, per-fold CSV, `best_cv_indices.npz`  
  - Retraining with best config: `best_model.pt`, `loss_history_best_retrain.csv`, `loss_curve_best_retrain.png`, `best_retrain_indices.npz`  
  - Preprocessing metadata (if enabled): `normalization_factors_*.npz`, `mean_values.npz`  

---

### **06_inferSequence.py — Autoregressive Inference with Trained Seq2Seq (Δ-Series Prediction)**  
**Purpose:**  
Use the **LSTM Encoder–Decoder (Seq2Seq)** model trained in `05_trainingSequence.py` to perform **autoregressive prediction** of the **next 29-step Δ sequence** from a **30-step history**.  
Inputs are the **measured maps** from `04_generateTrainingData.py` and the **true maps** from `02_uniteTriangular.py` (for initialization and normalization reference).  

**Process Overview:**  
1. **Load input data and trained model**  
   - Load `measured` and `united` `.npy` files  
   - Apply differencing (Δ), normalization, and mean-centering as needed  
   - Concatenate the true map at t=0 to the beginning of `measured` to form the model input  
2. **Triangular-matrix conversion:**  
   - Convert each frame into a **1D vector** following `row_sizes = [29, 26, 23, …, 2]` (via `create_array`)  
   - Restore predictions back to **2D triangular-matrix format** with `de_create_array`  
3. **Autoregressive Δ prediction:**  
   - Feed **30-step history** into the encoder  
   - Start the decoder with zero initialization, and generate **29 Δ-steps** sequentially  
   - Feed each predicted Δ output back into the next decoder input (no teacher forcing)  
4. **Save and visualize outputs (optional):**  
   - Save predictions as `.npy` (shape: `N_series × T_pred × H × W`)  
   - Optionally, output PNGs for total map, t=0, and final step  

**Input / Output:**  
- **Input:**  
  - **Measured maps:** from `04_generateTrainingData.py` (e.g., `measured_training_data_sameRowColSeq31.npy`)  
  - **True maps:** from `02_uniteTriangular.py` (e.g., `united_triangular_matrices.npy`)  
  - **Trained model:** from `05_trainingSequence.py` (e.g., `best_model.pt` or `checkpoints/`)  
  - **Normalization files (optional):** `normalization_factors_*.npz`, `mean_values.npz`  
  - **Configuration:** YAML (e.g., `configs/infer_sequence.yml`)  
- **Output:**  
  - Predicted Δ maps: `<measured_stem>__pred_seq.npy` (shape = N_series×T_pred×H×W)  
  - (Optional) Visualization: `pred_png/` with PNG outputs (total, t=0, final step, etc.)  

**Features:**  
- Supports Δ-based inference (`use_diff: true`) same as in training  
- Batch processing of multiple input files in a folder  
- Automatically detects and applies normalization and mean-centering from training  
- Runs on GPU automatically if CUDA is available  

---

### **07_trainingFirst.py — Single-Output LSTM Training for Initial Difference Δ(t=0→1)**  
**Purpose:**  
Using the **measured maps** generated by `04_generateTrainingData.py` and the **true maps (united)** from `02_uniteTriangular.py`,  
this script trains a **single-output LSTM regression model** to predict the **initial temporal difference Δ(t=0→1)**.  
It serves as a short-term version of `05_trainingSequence.py`, focusing only on the first change in the time series.  

**Process Overview:**  
1. **Load measured and true data**, convert resistivity to conductivity (ρ → σ = 1000/ρ), and replace NaNs with 0  
2. **Append the initial true frame (t=0)** to the beginning of the measured data to form the model’s input sequence  
3. **Apply preprocessing** according to the YAML configuration (time slicing, subsampling, differencing, normalization, mean-centering, etc.)  
4. **Use a short history length `time_steps` (e.g., 4 steps)** as input to create training samples predicting the first output Δ(t=0→1)**  
   - Each frame is flattened into a **1D vector** following the triangular-grid structure of the Wenner array, based on `row_sizes = [29, 26, 23, …, 2, 1]` (via `create_array`)  
   - Optional addition of a time-channel (`time_context`)  
5. **Train a 2-layer LSTM + fully connected regression model**  
   - Perform **K-fold cross-validation** with various learning-rate and batch-size combinations  
   - Select the configuration with the **lowest average validation error**, then retrain with 80/20 split  
   - Save the final model, learning curves, and loss history  

**Input / Output:**  
- **Input:**  
  - Measured maps: `measured_training_data.npy` (from `04_generateTrainingData.py`)  
  - True maps: `united_triangular_matrices.npy` (from `02_uniteTriangular.py`)  
  - YAML configuration: `configs/training_first.yml`  
- **Output:**  
  - Trained models: `best_model_first.pt`, `single_output_lstm_model.pt`  
  - Normalization and mean data: `norm_input.npz`, `norm_output.npz`, `mean_values.npz`  
  - Training logs: `grid_search_results.csv`, `best_config.txt`, `best_cv_indices.npz`  
  - Optional retraining results: `loss_history_best_retrain.csv`, `loss_curve_best_retrain_first.png`  

**Features:**  
- Short-term prediction model focused on the initial change Δ(t=0→1)  
- 1D flattening based on the triangular Wenner-array grid  
- Unified, YAML-driven configuration for high reproducibility  
- Lightweight LSTM for fast training  

---

### **08_inferFirst.py — Combine Predicted Δ with Initial Frame to Reconstruct t=1 Map**  
**Purpose:**  
Using the **single-output LSTM model** trained in `07_trainingFirst.py`,  
this script combines the **predicted initial difference Δ(t=0→1)** with the **initial frame (t=0)**  
to reconstruct the **t=1 triangular-matrix map**.  
It reproduces the same preprocessing steps used in training (differencing, normalization, mean-centering, unit conversion, etc.) for inference consistency.  

**Process Overview:**  
1. **Load measured and true maps**, applying unit inversion (1/x) or scaling as needed  
2. **Concatenate the true t=0 frame** to the beginning of measured data to match the training input sequence length  
3. **Flatten/restore triangular matrices** (based on `row_sizes = [29, 26, 23, …, 2, 1]`) to match input/output shapes used during training  
4. **Load the trained LSTM model** and predict the first output step (Δ)  
5. **Combine Δ with the initial frame** to reconstruct the t=1 map (apply inverse transforms for normalization or centering if used)  
6. Save the prediction as `.npy`, and optionally export **Pred vs True** comparison plots and input previews as PNGs  

**Input / Output:**  
- **Input:**  
  - Measured maps: `measured_training_data*.npy` (from `04_generateTrainingData.py`)  
  - True maps: `united_triangular_matrices.npy` (from `02_uniteTriangular.py`)  
  - Trained model: `best_model_first.pt` or `single_output_lstm_model.pt` (from `07_trainingFirst.py`)  
  - (Optional) Normalization/mean files: `normalization_factors_*.npz`, `mean_values_*.npz`  
  - YAML configuration: `configs/infer_first.yml`  
- **Output:**  
  - Predicted t=1 maps: `pred_images_all.npy` (2D triangular maps per sequence)  
  - (Optional) Comparison images: `compare_series_*.png` (Predicted vs True)  
  - (Optional) Input previews: `inputs_series_*.png`  

**Features:**  
- Simple inference flow focusing solely on Δ(t=0→1) reconstruction  
- Reproduces all training preprocessing options (differencing, subsampling, mean-centering, time-context, etc.)  
- Built-in flattening/restoration for triangular Wenner-array grids  
- Supports batch folder processing and automatic comparison-plot saving  

---

### **09_inferWhole.py — Reconstruct Full Time-Series Maps by Accumulating Initial Value and Δ Series**  
**Purpose:**  
Using the results from `08_inferFirst.py` (Δ for t=0→1) and `06_inferSequence.py` (subsequent Δ predictions),  
this script reconstructs the **entire conductivity/resistivity time-series maps** by **cumulatively summing Δ values** starting from the **initial frame (t=0)**.  
The reconstructed maps are compared to the **true data** and evaluated using **MAPE (Mean Absolute Percentage Error)**, with optional comparison image outputs.  

**Process Overview:**  
1. **Load inputs:** load the initial Δ (t=0→1), the Δ series, and the full true time-series data  
2. **Ensure unit consistency (if necessary):** perform a safe 1/x transformation (with scaling and clipping) to align physical units  
3. **Reconstruct time series:**  
   - Use the **true t=0 frame** as the baseline  
   - Sequentially **accumulate the initial Δ (t=0→1)** and the subsequent **Δ series (t=1→...)** to reconstruct absolute-value time series (t=0,1,2,…)  
   - Align the lengths of the predicted and true sequences for valid comparison  
4. **Evaluation and visualization:**  
   - Compute **MAPE (Mean Absolute Percentage Error)** between predicted and true maps for each time step  
   - Optionally save comparison images of Pred / True / |Pred − True|  

**Input / Output:**  
- **Input:**  
  - **Δ series (Seq2Seq prediction):** `outputs_pred_all_series.npy` (shape = N×T_pred×H×W)  
  - **Initial Δ (t=0→1):** `pred_images_all.npy` (shape = N×H×W)  
  - **True time-series maps:** `united_triangular_matrices_test.npy` (shape = N×T_true×H×W)  
- **Output:**  
  - **Reconstructed stack (absolute values):** `conductivity.npy` (shape = N×T×H×W; T = min aligned length)  
  - **Evaluation summary:** `mape_values.txt` (lists MAPE[%] per sequence)  
  - **Comparison images (optional):** `compareWithTestData/measurement_locations_seq{idx}_timestep_{t}.png`  

**Features:**  
- Reconstructs absolute values by **cumulative summation of Δ over time from the true t=0 frame**  
- Automatically aligns lengths and units (safe 1/x transformation and clipping)  
- Outputs **MAPE** to both console and file, with optional frame-by-frame comparison image generation  

---

### **10_comparePredAndNorm.py — Visualization: Compare Predictions and Measurements under Common Color Scale**  
**Purpose:**  
For all sequences and time steps, this script places **predicted maps (Predicted)** and **measured maps (Measured)** side by side using a **shared color scale**,  
exporting synchronized PNG images (units: mS·m⁻¹).  

**Process Overview:**  
1. Load **predicted stack** (N×T×H×W) and **measured stack** (N×T×H×W)  
2. Convert the measured stack to conductivity representation (1000/ρ) following the project convention (visualize NaN as blank)  
3. Align time correspondence using the rule `measured_t = pred_t × k − 1`, comparing only valid frame pairs (skip out-of-range frames)  
4. Determine the **shared color range (vmin/vmax)** either per sequence or globally, and apply it to both Pred and Measured maps  
5. Save **side-by-side Pred vs Measured** comparison images for each pair while displaying progress  

**Input / Output:**  
- **Input:**  
  - Predicted stack: `conductivity.npy` (shape = N×T_pred×H×W)  
  - Measured stack: `measured_training_data_*.npy` (shape = N×T_meas×H×W)  
  - YAML configuration file: `configs/compare_pred_norm.yml`  
- **Output:**  
  - Comparison images (PNG): `out_dir/seq{NNN}_predt{TTT}_meast{TTT}.png` (Pred vs Measured under shared color scale)  

**Features:**  
- Performs strict comparison under a **common color scale** (auto range detection by sequence or globally)  
- **Time synchronization rule** ensures only corresponding frames are compared  
- **NaN values rendered as white**, with optional zero-fill visualization  
- Default colormap is `"hot"`, but can be modified per project preference  

---
