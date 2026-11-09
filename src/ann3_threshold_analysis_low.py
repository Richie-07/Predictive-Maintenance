
# =============================================================================
# ANN3 LOW-THRESHOLD ANALYSIS — OOF PREDICTIONS FROM ann_optimization.py
# FIX (audit C4): Sweeps low thresholds on OOF CV predictions for
# ANN_3_Regularized — not on the final held-out test set.
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "results" / "ann_optimization_predictions.csv"
)

df = df[
    df["configuration"] == "ANN_3_Regularized"
].copy()

y_true = df["actual"].values
probabilities = df["failure_probability"].values

thresholds = np.round(
    np.arange(0.05, 0.251, 0.0125), 4
)

results = []

for threshold in thresholds:

    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "cost_2x": fp + 2 * fn,
        "cost_5x": fp + 5 * fn,
        "cost_10x": fp + 10 * fn
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/ann3_threshold_analysis_low.csv",
    index=False
)

print("\nANN 3 LOWER THRESHOLD ANALYSIS")
print("=" * 100)

print(results_df.to_string(
    index=False,
    formatters={
        "accuracy": "{:.4f}".format,
        "precision": "{:.4f}".format,
        "recall": "{:.4f}".format,
        "f1": "{:.4f}".format
    }
))

print("\nLOWEST COST THRESHOLDS")

for column in ["cost_2x", "cost_5x", "cost_10x"]:

    best = results_df.loc[results_df[column].idxmin()]

    print(
        f"{column}: threshold={best['threshold']}, "
        f"FP={best['FP']}, FN={best['FN']}, "
        f"cost={best[column]}"
    )

print("\nSaved to results/ann3_threshold_analysis_low.csv")
