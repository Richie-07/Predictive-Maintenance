
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "dataset/raw/predictive_maintenance_v3.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

NUMERIC_FEATURES = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp"
]

CATEGORICAL_FEATURES = [
    "machine_type",
    "operating_mode"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "failure_within_24h"
GROUP = "machine_id"


def build_model():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("sampler", RandomOverSampler(random_state=42)),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(32, 16, 8),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=False,
            random_state=42
        ))
    ])


def main():
    # FIX (audit M2): All code is inside main() — importing this module
    # no longer triggers CSV loading or computation.
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    groups = df[GROUP].copy()

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    all_predictions = []
    all_probabilities = []
    all_actuals = []
    fold_results = []

    print("\nMACHINE-INDEPENDENT ANN EVALUATION")
    print("=" * 55)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_machines = sorted(groups.iloc[train_idx].unique())
        test_machines = sorted(groups.iloc[test_idx].unique())

        # Sanity-check: no machine should appear in both train and test.
        assert set(train_machines).isdisjoint(set(test_machines)), (
            f"Fold {fold}: machine overlap detected between train and test sets!"
        )

        fold_model = clone(build_model())
        fold_model.fit(X_train, y_train)

        predictions = fold_model.predict(X_test)
        probabilities = fold_model.predict_proba(X_test)[:, 1]

        metrics = {
            "fold": fold,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            # FIX (audit M1): PR-AUC added — more informative than ROC-AUC
            # on imbalanced data (only ~15% positives).
            "pr_auc": average_precision_score(y_test, probabilities),
        }

        fold_results.append(metrics)

        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)
        all_actuals.extend(y_test.tolist())

        print(f"\nFold {fold}")
        print("Training machines:", train_machines)
        print("Testing machines:", test_machines)
        print("Training records:", len(train_idx))
        print("Testing records:", len(test_idx))

        for key, value in metrics.items():
            if key != "fold":
                print(f"{key}: {value:.4f}")

    print("\n" + "=" * 55)
    print("OVERALL MACHINE-INDEPENDENT RESULTS")
    print("=" * 55)

    overall = {
        "accuracy": accuracy_score(all_actuals, all_predictions),
        "precision": precision_score(all_actuals, all_predictions, zero_division=0),
        "recall": recall_score(all_actuals, all_predictions, zero_division=0),
        "f1": f1_score(all_actuals, all_predictions, zero_division=0),
        "roc_auc": roc_auc_score(all_actuals, all_probabilities),
        "pr_auc": average_precision_score(all_actuals, all_probabilities),
        "confusion_matrix": confusion_matrix(
            all_actuals, all_predictions
        ).tolist()
    }

    for key, value in overall.items():
        print(f"{key}: {value}")

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(
        RESULTS_DIR / "machine_independent_ann_folds.csv",
        index=False
    )

    with open(
        RESULTS_DIR / "machine_independent_ann_metrics.json", "w"
    ) as f:
        json.dump({
            "fold_results": fold_results,
            "overall": overall
        }, f, indent=4)

    predictions_df = pd.DataFrame({
        "actual": all_actuals,
        "predicted": all_predictions,
        "failure_probability": all_probabilities
    })

    predictions_df.to_csv(
        RESULTS_DIR / "machine_independent_ann_predictions.csv",
        index=False
    )

    print("\nAll five folds completed.")
    print("Results saved in the results folder.")


if __name__ == "__main__":
    main()
