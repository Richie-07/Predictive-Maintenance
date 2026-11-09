# AI-Based Predictive Maintenance System
## Hybrid Soft Computing Techniques

**Academic prototype** combining ANN, ANFIS, Mamdani Fuzzy Logic, and Genetic Algorithm for industrial predictive maintenance and scheduling.

---

## Project Structure

```
Predictive_Maintenance/
├── dataset/
│   ├── raw/              predictive_maintenance_v3.csv (24,042 records)
│   └── processed/        train.csv / test.csv (80:20 chronological split)
├── models/               Trained model .joblib files
├── results/              Evaluation JSON & CSV outputs
├── src/                  Source modules
│   ├── data_preprocessing.py
│   ├── ann_model.py
│   ├── ann_optimization.py
│   ├── machine_independent_ann.py
│   ├── ann_fair_comparison.py
│   ├── anfis_model.py
│   ├── anfis_balanced.py
│   ├── fuzzy_logic.py
│   └── genetic_algorithm.py
└── dashboard/
    └── app.py            Streamlit dashboard
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Run each script from the project root in order:

```bash
# 1. Preprocess data
python src/data_preprocessing.py

# 2. Train primary ANN (9 features)
python src/ann_model.py

# 3. Machine-independent ANN evaluation (group-aware CV)
python src/machine_independent_ann.py

# 4. ANN configuration comparison (4 configs, group-aware CV)
python src/ann_optimization.py

# 5. Four-feature ANN for fair comparison with ANFIS
python src/ann_fair_comparison.py

# 6. ANFIS (original, no oversampling)
python src/anfis_model.py

# 7. Balanced ANFIS (with oversampling)
python src/anfis_balanced.py

# 8. Fuzzy Logic test
python src/fuzzy_logic.py

# 9. Genetic Algorithm maintenance scheduling
python src/genetic_algorithm.py

# 10. Launch dashboard
streamlit run dashboard/app.py
```

---

## Dataset

- **24,042 records** from **20 machines** over **14 days**
- **Target:** `failure_within_24h` (binary: 0 = normal, 1 = failure imminent)
- **Class imbalance:** 85.2% normal, 14.8% failure
- **Features:** vibration_rms, temperature_motor, current_phase_avg, pressure_level, rpm, hours_since_maintenance, ambient_temp, machine_type, operating_mode
- **Excluded from prediction:** failure_type, estimated_repair_cost, rul_hours

---

## Evaluation Results (Trustworthy — Machine-Independent)

The **machine-independent ANN** (5-fold group-aware CV, machines never shared between train and test, with early stopping enabled) serves as the primary benchmark:

| Metric | Machine-Independent ANN (Baseline Config) |
|--------|-------------------------------------------|
| Accuracy | 89.77% |
| Precision | 61.71% |
| Recall | 81.43% |
| F1-score | 70.21% |
| ROC-AUC | 0.9428 |
| PR-AUC | 0.7778 |
| False Positives | 1,799 |
| False Negatives | 661 |

> ⚠️ **Methodological Note:** The single-split primary ANN metrics (97.57% accuracy, 87.00% recall) suffer from machine-level leakage because records from the same 20 machines appear in both train and test splits. The machine-independent cross-validation results above are the scientifically valid numbers to cite.

---

## ANN Configuration Comparison (Group-Aware 5-Fold CV with Early Stopping)

All four configurations were trained with early stopping (`early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=20`):

| Configuration | Hidden Layers | Alpha | Learning Rate | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | False Positives | False Negatives |
|---------------|---------------|-------|---------------|----------|-----------|--------|----|---------|--------|-----------------|-----------------|
| **ANN_1_Baseline** | (32, 16, 8) | 0.001 | 0.001 | 89.77% | 61.71% | 81.43% | 70.21% | 0.9428 | 0.7778 | 1,799 | 661 |
| **ANN_2_Deep** | (64, 32, 16) | 0.001 | 0.001 | 89.17% | 60.48% | 77.50% | 67.94% | 0.9260 | 0.6359 | 1,803 | 801 |
| **ANN_3_Regularized** | (64, 32) | 0.010 | 0.001 | 89.38% | 60.82% | 79.52% | 68.92% | 0.9384 | 0.7583 | 1,824 | 729 |
| **ANN_4_Large** | (128, 64, 32) | 0.001 | 0.0005 | 89.63% | 62.29% | 75.87% | 68.41% | 0.9225 | 0.6861 | 1,635 | 859 |

---

## Threshold Selection Methodology

> ⚠️ **Exploratory OOF vs. Independent Validation:**  
> The threshold sweeps in `results/threshold_analysis*.csv` and `results/ann3_threshold_analysis*.csv` are computed on out-of-fold (OOF) cross-validation predictions. This eliminates ordinary train/test data leakage. However, evaluating candidate thresholds on the same OOF predictions represents **exploratory OOF threshold tuning**. Full independent validation requires evaluating the selected threshold on a dedicated unseen factory holdout partition or through nested cross-validation.

---

## Model Comparison (4-feature, same test split)

| Model | Recall | F1 | ROC-AUC | FN |
|-------|--------|-----|--------|-----|
| Four-feature ANN | 72.33% | 49.89% | 92.40% | 83 |
| Original ANFIS (300 iter) | 53.00% | 59.11% | 93.90% | 141 |
| **Balanced ANFIS (300 iter)** | **92.67%** | **56.33%** | **95.65%** | **22** |

Balanced ANFIS achieves the highest recall (278 of 300 failures detected, only 22 missed) but with 409 false alarms.

---

## GA Scheduling Results

| Strategy | Cost | vs FIFO |
|----------|------|---------|
| FIFO | 4.776 | — |
| Priority-based | 3.150 | −34.0% |
| **Genetic Algorithm** | **3.028** | **−36.6%** |

> Note: The GA objective cost is a risk-weighted prototype metric, **not a monetary amount**. Capacity (4 machines/day) and changeover penalty (0.03) are assumed values.

---

## Limitations

- Dataset covers only 14 days and 20 machines — generalization to new machines is unvalidated.
- Class imbalance is severe in the test set (6.2% failures vs 17.0% in training).
- Machine-independent evaluation shows significant precision drop (77% → 61%) when machines are truly unseen.
- GA improvement over greedy priority-ordering is modest (~3.9%) for 20 machines.

---

## Branch Policy

- `main` — Original project state (do not modify without approval)
- `testing-audit` — All audit fixes and experimental changes go here

