# ERT — Water Infiltration into Soil + Neural Completion

> Reconstruct high‑resolution, time‑lapse apparent resistivity fields during **water infiltration into soil** by combining physics‑based simulation (OpenFOAM → pyGIMLi ERT) with sequence models (LSTM/Seq2Seq).

---

## TL;DR
- **Goal:** Fill the spatial and temporal gaps of ERT monitoring during **water infiltration into soil** using a neural network trained on synthetic data.
- **Workflow:** OpenFOAM → resistivity maps → pyGIMLi apparent resistivity (Wenner‑alpha) → measurement design → LSTM training/inference.
- **Scripts:** `01_generateAppRes.py` to `10_comparePredAndNorm.py`.
- **Configuration:** All scripts use YAML config files; CLI flags override them if provided.

---

## Project layout
> Final visual outputs (animations and summaries) are saved under the `movies/` folder.

```
├── scripts/
│   ├── 01_generateAppRes.py        # Compute apparent resistivity maps (pyGIMLi, Wenner–alpha)
│   ├── 02_uniteTriangular.py       # Merge per-sequence triangular .npy into unified train/test sets
│   ├── 03_generateMeasDesign.py    # Select measurement positions based on largest Δ(true–measured)
│   ├── 04_generateTrainingData.py  # Build measured stacks (train/test) using chosen positions
│   ├── 05_trainingSequence.py      # Train LSTM (sequence-to-sequence) on temporal differences
│   ├── 06_inferSequence.py         # Predict Δ-series using the trained sequence model
│   ├── 07_trainingFirst.py         # Train model to reconstruct the initial conductivity map
│   ├── 08_inferFirst.py            # Predict the first-step conductivity map
│   ├── 09_inferWhole.py            # Combine first-step and Δ-series into full reconstruction
│   ├── 10_comparePredAndNorm.py    # Compare predicted vs measured maps with unified color scale
│   └── XX_*.py                     # Utility scripts (video conversion, plotting, batch runners)
├── configs/                        # YAML configs for all stages
├── movies/                         # Exported animations (GIF/MP4)
├── .gitignore
└── README.md
```

---

## Measurement conditions
- **Electrodes:** 32
- **Spacing:** 4 cm
- **Pattern:** Wenner‑alpha (same as used in this code)
- **Infiltration:** Central 30 cm of the soil surface continuously saturated with **0.0885 mol/L NaCl** solution
- **Neural enhancement:** Neural networks increase effective temporal/spatial resolution by approximately **×15** compared with the original ERT sampling

---

## Installation
> Notebooks and large data are excluded. Provide your OpenFOAM-derived conductivity maps under `data/`.

```bash
conda create -n ert-nn python=3.10 -y
conda activate ert-nn
pip install numpy scipy matplotlib pyyaml tqdm pillow
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or CUDA wheel if available
pip install pygimli
```

To export or share the environment:
```bash
conda env export --no-builds > environment.yml
# recreate later
conda env create -f environment.yml
```

---

## Typical pipeline (what each step PRODUCES)
Below is a concise and accurate summary of what each script **does** and **produces**.

1) **01_generateAppRes.py — Forward modeling to triangular matrices**  
   **Purpose:** Compute apparent resistivity from 2D conductivity maps obtained from OpenFOAM using pyGIMLi (Wenner-alpha configuration, 32 electrodes, 4 cm spacing). Optionally adds noise and reshapes each time frame into a triangular form (29, 26, 23, …, 1).  
   **Inputs:** Time-lapse conductivity or resistivity fields `(N, T, H, W)`.  
   **Outputs:** **Triangular apparent resistivity stacks** per sequence and optional **preview images**.

2) **02_uniteTriangular.py — Merge training/test datasets**  
   **Purpose:** Collect all per-sequence triangular matrices, verify shape consistency, and randomly split into **training** and **testing** sets (with fixed seed if desired). Optionally saves file lists.  
   **Inputs:** Individual triangular matrices.  
   **Outputs:** **Unified 4D datasets (train/test)** containing time-series triangular matrices and optional index lists.

3) **03_generateMeasDesign.py — Dynamic measurement design**  
   **Purpose:** At each time step, compute |true − measured| and select locations with the largest difference (top-k=1 by default). Can save progress frames and logs.  
   **Inputs:** Unified dataset (true resistivity maps).  
   **Outputs:** **Measurement position indices over time** and the corresponding **measured stack** reflecting sequential updates; optional **visual frames**.

4) **04_generateTrainingData.py — Reconstruct using a representative sequence**  
   **Purpose:** From multiple candidate measurement sequences, pick one (median or fixed) and regenerate both train and test stacks using the **same measurement locations** for consistency.  
   **Inputs:** True stacks (train/test) and measurement position sequences.  
   **Outputs:** **Measured stacks (train/test)** aligned to the chosen design and **selected sequence ID**.

5) **05_trainingSequence.py — Seq2Seq LSTM for temporal differences**  
   **Purpose:** Train an encoder–decoder LSTM to predict temporal differences (Δ fields) between consecutive steps. Supports k-fold validation, MSE loss, and saves CSV logs and loss-curve plots.  
   **Inputs:** Encoder input sequences and decoder targets (flattened triangular vectors).  
   **Outputs:** **Trained model checkpoints**, **training/validation logs**, and **loss plots**.

6) **06_inferSequence.py — Inference of Δ-series**  
   **Purpose:** Load the trained model (and normalization if available) to generate **autoregressive Δ-series predictions**. Can export PNG visualizations.  
   **Inputs:** Measured stacks or input folder, checkpoint, normalization file.  
   **Outputs:** **Predicted Δ time-series stack** and optional **visualizations**.

7) **07_trainingFirst.py — Regression of the initial frame (value scale)**  
   **Purpose:** Train a model to reconstruct the initial frame (t=0 conductivity, 1000/ρ) from sparse measured sequences. Supports k-fold, normalization, and mean centering.  
   **Inputs:** Preprocessed measured sequences and true initial values (flattened triangular vectors).  
   **Outputs:** **Initial-frame regression model**, **normalization stats**, and **training logs/plots**.

8) **08_inferFirst.py — Inference of the initial frame**  
   **Purpose:** Use the trained model to predict the initial frame (1000/ρ) and optionally visualize or save results.  
   **Inputs:** Measured sequences, checkpoint, optional normalization file.  
   **Outputs:** **Predicted initial frame stack** and **comparison plots**.

9) **09_inferWhole.py — Full time-series reconstruction (values)**  
   **Purpose:** Combine the predicted initial frame (values) with the Δ-series to reconstruct the **complete conductivity sequence**. Optionally computes MAPE and generates visual comparisons.  
   **Inputs:** Predicted Δ-series, predicted initial frame, true values (for validation).  
   **Outputs:** **Reconstructed conductivity time-cube**, **error summaries**, and **comparison figures**.

10) **10_comparePredAndNorm.py — Unified visualization (predicted vs measured)**  
   **Purpose:** Convert predicted and measured stacks to common units, replace invalid values with NaN, and display side-by-side comparisons using a shared or fixed color scale.  
   **Inputs:** Predicted stack `(N, T, H, W)` and measured stack `(N, T, H, W)`.  
   **Outputs:** **Per-sequence comparison panels** and optional **batch exports**.

---

## Data conventions
- **Triangular Wenner-alpha matrix:** Rows sized `[29, 26, 23, …, 2, 1]`, NaN-padded.
- **Shapes:** True/Measured stacks have shape `(N, T, H, W)`.
- **Units:** Conductivity expressed as `1000 / ρ`. Zero and NaN handling are included.

---

---

## Troubleshooting
- **Shape mismatch (e.g., (30,155) vs (29,155))**: Ensure consistency in the number of time steps and whether the first frame is excluded when computing differences.
- **Division by zero / Inf values:** Use safe transforms and proper `nan_fill_value`.
- **Color scale differences:** Use unified color limits in the comparison script.

---

