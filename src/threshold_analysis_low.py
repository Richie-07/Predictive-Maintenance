
# =============================================================================
# LOW THRESHOLD ANALYSIS — MACHINE-INDEPENDENT OOF PREDICTIONS
# FIX (audit C4): Sweeps low thresholds on out-of-fold CV predictions from
# machine_independent_ann.py — not on the held-out test set.
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT / "results" / "machine_independent_ann_predictions.csv"
)

y_true = df["actual"].astype(int).values
y_prob = df["failure_probability"].astype(float).values

thresholds = np.round(np.arange(0.20, 0.5001, 0.025), 3)

results = []

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "cost_5x": fp + (5 * fn),
        "cost_10x": fp + (10 * fn)
    })

result_df = pd.DataFrame(results)

output = ROOT / "results" / "threshold_analysis_low.csv"
result_df.to_csv(output, index=False)

print("\nLOW-THRESHOLD ANALYSIS")
print(result_df.round(4).to_string(index=False))

print("\nSaved:", output)
