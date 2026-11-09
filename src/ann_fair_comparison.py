
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "dataset" / "processed" / "train.csv"
TEST_PATH = ROOT / "dataset" / "processed" / "test.csv"

MODEL_PATH = ROOT / "models" / "ann_fair_comparison.joblib"
METRICS_PATH = ROOT / "results" / "ann_fair_metrics.json"
PREDICTIONS_PATH = ROOT / "results" / "ann_fair_predictions.csv"

# Exactly the same four inputs used by ANFIS.
FEATURES = [
    "temperature_motor",
    "vibration_rms",
    "current_phase_avg",
    "hours_since_maintenance",
]

TARGET = "failure_within_24h"


def main():
    print("\n========== FAIR ANN COMPARISON ==========")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    X_test = test_df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Input features: {len(FEATURES)}")

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("oversampler", RandomOverSampler(random_state=42)),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(32, 16, 8),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])

    print("\nTraining ANN with four inputs...")
    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(y_test, predictions)

    print("\n========== FAIR ANN RESULTS ==========")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"ROC-AUC      : {auc:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "training_time_seconds": float(training_time),
        "confusion_matrix": cm.tolist(),
        "features": FEATURES,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    pd.DataFrame({
        "actual": y_test,
        "predicted": predictions,
        "failure_probability": probabilities,
    }).to_csv(PREDICTIONS_PATH, index=False)

    print("\n========== FILES SAVED ==========")
    print(f"Model:       {MODEL_PATH}")
    print(f"Metrics:     {METRICS_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
