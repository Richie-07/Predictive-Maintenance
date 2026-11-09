
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "dataset" / "processed" / "train.csv"
TEST_PATH = BASE_DIR / "dataset" / "processed" / "test.csv"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

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


# ==========================================
# 3. LOAD DATA
# ==========================================

def load_data():

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Training or testing dataset not found. "
            "Run data_preprocessing.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Both datasets must contain both classes.")

    return X_train, X_test, y_train, y_test


# ==========================================
# 4. BUILD ANN PIPELINE
# ==========================================

def build_model():

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),

        # Balance the training classes only
        ("oversampler", RandomOverSampler(random_state=42)),

        # Feed-forward Artificial Neural Network
        ("ann", MLPClassifier(
            hidden_layer_sizes=(32, 16, 8),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        ))
    ])

    return model


# ==========================================
# 5. TRAIN AND EVALUATE
# ==========================================

def main():

    print("\n========== ANN MODEL TRAINING ==========\n")

    X_train, X_test, y_train, y_test = load_data()

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    model = build_model()

    print("\nTraining ANN... Please wait.")

    model.fit(X_train, y_train)

    print("\nANN training completed.")

    # Predictions
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        ),
        "training_iterations": int(
            model.named_steps["ann"].n_iter_
        )
    }

    # Display results
    print("\n========== ANN RESULTS ==========")

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    # Save model
    joblib.dump(
        model,
        MODEL_DIR / "ann_pipeline.joblib"
    )

    # Save metrics
    with open(
        RESULTS_DIR / "ann_metrics.json", "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    # Save test predictions
    predictions = pd.DataFrame({
        "actual_failure": y_test.values,
        "predicted_failure": y_pred,
        "failure_probability": y_prob
    })

    predictions.to_csv(
        RESULTS_DIR / "ann_predictions.csv",
        index=False
    )

    print("\n========== FILES SAVED ==========")
    print("Model: models/ann_pipeline.joblib")
    print("Metrics: results/ann_metrics.json")
    print("Predictions: results/ann_predictions.csv")


if __name__ == "__main__":
    main()
