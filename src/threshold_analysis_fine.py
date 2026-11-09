
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Load machine-independent ANN predictions
input_file = ROOT / "results" / "machine_independent_ann_predictions.csv"

df = pd.read_csv(input_file)

y_true = df["actual"].astype(int).values
y_prob = df["failure_probability"].astype(float).values

# Thresholds: 0.500 to 0.850, step 0.025
thresholds = np.round(np.arange(0.50, 0.8501, 0.025), 3)

# Illustrative relative cost scenarios
failure_cost_ratios = [2, 5, 10]

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    row = {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }

    # Cost of one unnecessary alert = 1 unit
    for ratio in failure_cost_ratios:
        total_cost = fp + (fn * ratio)
        row[f"cost_{ratio}x"] = total_cost

    results.append(row)

results_df = pd.DataFrame(results)

# Save detailed results
output_file = ROOT / "results" / "threshold_analysis_fine.csv"
results_df.to_csv(output_file, index=False)

# Display all threshold results
print("\nFINE-GRAINED THRESHOLD ANALYSIS")
print("=" * 90)

print(results_df.round(4).to_string(index=False))

# Find lowest-cost threshold under each scenario
print("\nCOST-SENSITIVE OPTIMAL THRESHOLDS")
print("=" * 60)

for ratio in failure_cost_ratios:

    cost_column = f"cost_{ratio}x"
    best_index = results_df[cost_column].idxmin()
    best = results_df.loc[best_index]

    print(f"\nMissed failure cost: {ratio}x")
    print(f"Threshold: {best['threshold']:.3f}")
    print(f"False Positives: {int(best['FP'])}")
    print(f"False Negatives: {int(best['FN'])}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"F1 Score: {best['f1']:.4f}")
    print(f"Total Relative Cost: {best[cost_column]:.0f}")

print("\nSaved results to:", output_file)
