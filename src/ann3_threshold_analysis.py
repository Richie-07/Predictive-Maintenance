
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load machine-independent ANN optimization predictions
df = pd.read_csv(
    "results/ann_optimization_predictions.csv"
)

# Select ANN 3 predictions
df = df[df["configuration"] == "ANN_3_Regularized"].copy()

y_true = df["actual"].values
probabilities = df["failure_probability"].values

# Thresholds: 0.20 to 0.85, step 0.025
thresholds = np.round(
    np.arange(0.20, 0.851, 0.025), 3
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

# Save results
results_df.to_csv(
    "results/ann3_threshold_analysis.csv",
    index=False
)

print("\nANN 3 THRESHOLD ANALYSIS")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format
        }
    )
)

print("\nLowest-cost thresholds:")
for cost_column in ["cost_2x", "cost_5x", "cost_10x"]:
    best = results_df.loc[
        results_df[cost_column].idxmin()
    ]

    print(
        f"{cost_column}: threshold={best['threshold']}, "
        f"FP={best['FP']}, FN={best['FN']}, "
        f"cost={best[cost_column]}"
    )

print("\nResults saved to results/ann3_threshold_analysis.csv")
