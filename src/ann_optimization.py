
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline


# --------------------------------------------------
# 1. PATHS AND DATA
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "dataset" / "raw" / "predictive_maintenance_v3.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

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


# --------------------------------------------------
# 2. PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            NUMERIC_FEATURES
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            CATEGORICAL_FEATURES
        )
    ]
)


# --------------------------------------------------
# 3. ANN CONFIGURATIONS
# --------------------------------------------------

CONFIGURATIONS = {
    "ANN_1_Baseline": {
        "hidden_layer_sizes": (32, 16, 8),
        "alpha": 0.001,
        "learning_rate_init": 0.001
    },

    "ANN_2_Deep": {
        "hidden_layer_sizes": (64, 32, 16),
        "alpha": 0.001,
        "learning_rate_init": 0.001
    },

    "ANN_3_Regularized": {
        "hidden_layer_sizes": (64, 32),
        "alpha": 0.01,
        "learning_rate_init": 0.001
    },

    "ANN_4_Large": {
        "hidden_layer_sizes": (128, 64, 32),
        "alpha": 0.001,
        "learning_rate_init": 0.0005
    }
}


def main():
    # FIX (audit M2 / H4): all data loading and training is inside main()
    # so importing this module does not trigger any side-effects.
    # FIX (audit H4): early_stopping=True (was False) prevents overfitting
    # within each CV fold.

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET].astype(int)
    groups = df[GROUP]

    # --------------------------------------------------
    # 4. MACHINE-INDEPENDENT CROSS VALIDATION
    # --------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Generate identical folds for every configuration
    folds = list(cv.split(X, y, groups))

    # --------------------------------------------------
    # 5. TRAIN AND EVALUATE
    # --------------------------------------------------

    all_results = []
    all_predictions = []

    for config_name, config in CONFIGURATIONS.items():

        print("\n" + "=" * 60)
        print("Running:", config_name)
        print("=" * 60)

        oof_probabilities = np.zeros(len(X))
        fold_scores = []

        start_time = time.time()

        for fold_number, (train_idx, test_idx) in enumerate(folds, 1):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # FIX (audit H4): early_stopping=True prevents within-fold
            # overfitting. validation_fraction=0.1 carves a small val set
            # from each training fold for the stopping criterion.
            model = ImbPipeline([
                ("preprocessor", clone(preprocessor)),
                ("oversampler", RandomOverSampler(random_state=42)),
                ("ann", MLPClassifier(
                    hidden_layer_sizes=config["hidden_layer_sizes"],
                    activation="relu",
                    solver="adam",
                    alpha=config["alpha"],
                    learning_rate_init=config["learning_rate_init"],
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42
                ))
            ])

            fold_model = clone(model)

            fold_model.fit(X_train, y_train)

            probabilities = fold_model.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)

            oof_probabilities[test_idx] = probabilities

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
            pr_auc = average_precision_score(y_test, probabilities)

            fold_scores.append({
                "fold": fold_number,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc,
                "pr_auc": pr_auc,
            })

            print(
                f"Fold {fold_number}: "
                f"Acc={accuracy:.4f}, "
                f"Prec={precision:.4f}, "
                f"Recall={recall:.4f}, "
                f"F1={f1:.4f}, "
                f"AUC={auc:.4f}, "
                f"PR-AUC={pr_auc:.4f}"
            )

        # Overall out-of-fold evaluation
        final_predictions = (oof_probabilities >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, final_predictions, labels=[0, 1]
        ).ravel()

        metrics = {
            "configuration": config_name,
            "hidden_layer_sizes": str(config["hidden_layer_sizes"]),
            "alpha": config["alpha"],
            "learning_rate_init": config["learning_rate_init"],
            "accuracy": accuracy_score(y, final_predictions),
            "precision": precision_score(
                y, final_predictions, zero_division=0
            ),
            "recall": recall_score(
                y, final_predictions, zero_division=0
            ),
            "f1": f1_score(
                y, final_predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y, oof_probabilities),
            "pr_auc": average_precision_score(y, oof_probabilities),
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "training_time_seconds": round(time.time() - start_time, 2)
        }

        all_results.append(metrics)

        all_predictions.append(pd.DataFrame({
            "configuration": config_name,
            "actual": y.to_numpy(),
            "failure_probability": oof_probabilities,
            "predicted": final_predictions
        }))

        print("\nOverall results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    # --------------------------------------------------
    # 6. SAVE RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        RESULTS_DIR / "ann_optimization_metrics.csv",
        index=False
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_df.to_csv(
        RESULTS_DIR / "ann_optimization_predictions.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("ANN OPTIMIZATION COMPLETED")
    print("=" * 60)

    print(
        results_df[
            [
                "configuration",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "false_positive",
                "false_negative"
            ]
        ].to_string(index=False)
    )

    print("\nResults saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
