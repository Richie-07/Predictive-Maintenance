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

The **machine-independent ANN** (5-fold group-aware CV, machines never shared between train/test) is the primary evaluation:

| Metric | Machine-Independent ANN |
|--------|------------------------|
| Accuracy | 89.59% |
| Precision | 60.88% |
| Recall | 83.17% |
| F1-score | 70.30% |
| ROC-AUC | 94.32% |

> ⚠️ **Note:** The primary ANN single-split metrics (97.57% accuracy, 87% recall) are inflated because machines appear in both train and test partitions. Use machine-independent results for reporting.

---

## Model Comparison (4-feature, same test split)

| Model | Recall | F1 | ROC-AUC | FN |
|-------|--------|-----|--------|-----|
| Four-feature ANN | 72.33% | 49.89% | 92.40% | 83 |
| Original ANFIS | 54.33% | 60.71% | 94.48% | 137 |
| **Balanced ANFIS** | **90.00%** | **56.49%** | **95.08%** | **30** |

Balanced ANFIS achieves the highest recall (fewest missed failures) but at the cost of more false alarms (386 FP).

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

