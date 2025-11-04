
---

## 🚀 Pipeline Workflow

1. **Generate measured apparent resisitivity data from simulated data**  
   ```bash
   python scripts/01_generateAppRes.py --config configs/generateAppRes.yml
   ```

2. **Reshape the data and separate it in trainind+validation data and test data**  
   ```bash
   python scripts/02_unitedTriangular.py --config configs/unitedTriangular.yml
   ```

3. **Decide measurement design**  
   ```bash
   python scripts/03_generateMeasDesign.py --config configs/generateMeasDesign.yml
   ```

4. **Genereate training data**  
   ```bash
   python scripts/04_generateTrainingData.py --config configs/generateTrainingData.yml
   ```

5. **Train NN for predicting sequence**  
   ```bash
   python scripts/05_trainingSequence.py --config configs/trainingSequence.yml
   ```

6. **Infer sequence from test data**  
   ```bash
   python scripts/06_inferSequence.py --config configs/inferSequence.yml
   ```

7. **Train NN for predicting first data**  
   ```bash
   python scripts/07_trainingFirst.py --config configs/trainingFirst.yml
   ```

8. **Infer first data from test data**  
   ```bash
   python scripts/08_inferFirst.py --config configs/inferFirst.yml
   ```
   
9. **Infer whole data from test data**  
   ```bash
   python scripts/09_inferWhole.py --config configs/inferWhole.yml
   ```
---
