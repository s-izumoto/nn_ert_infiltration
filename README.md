
---

## 🚀 Pipeline Workflow

1. **Generate resistivity fields**  
   ```bash
   python scripts/01_make_fields.py --config configs/data/make_fields.yml
   ```

2. **Fit PCA and project fields**  
   ```bash
   python scripts/02_fit_pca_and_project.py --config configs/pca/pca_randomized.yml
   ```

3. **Simulate ERT measurements in all the arrays (pyGIMLi physics forward)**  


   ```bash
   python scripts/03_make_surrogate_pairs_all.py --config configs/simulate/make_surrogate_pairs_all.yml
   ```

4. **Simulate ERT measurements in Wenner array (pyGIMLi physics forward)**  
   ```bash
   python scripts/04_make_surrogate_pairs_wenner.py --config configs/simulate/make_surrogate_pairs_wenner.yml
   ```

5. **Inversion of Wenner array**  
   ```bash
   python scripts/05_make_surrogate_wenner_invert.py --config configs/simulate/wenner_invert.yml
   ```

6. **Gaussian process regression (GPR)**  
   ```bash
   python scripts/06_gpr_sequential_design.py --config configs/gpr/gpr_seq_example.yml
   ```

7. **Inversion of GPR results**  
   ```bash
   python scripts/07_invert_from_npz.py --config configs/inversion/inversion.yml
   ```


---
