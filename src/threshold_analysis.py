
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

file_path = ROOT / "results" / "machine_independent_ann_predictions.csv"

df = pd.read_csv(file_path)

y = df["actual"]
probabilities = df["failure_probability"]

thresholds = [
    0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70
]

results = []

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1": f1_score(
            y, predictions, zero_division=0
        ),
        "False Positives": fp,
        "False Negatives": fn
    })

results_df = pd.DataFrame(results)

print("\nTHRESHOLD ANALYSIS")
print("=" * 100)

print(results_df.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1": "{:.4f}".format
    }
))

results_df.to_csv(
    ROOT / "results" / "threshold_analysis.csv",
    index=False
)

print("\nResults saved to results/threshold_analysis.csv")
