
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import joblib

from scipy.optimize import minimize
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "dataset" / "processed" / "train.csv"
TEST_PATH = ROOT / "dataset" / "processed" / "test.csv"

MODEL_PATH = ROOT / "models" / "anfis_balanced_model.joblib"
METRICS_PATH = ROOT / "results" / "anfis_balanced_metrics.json"
PREDICTIONS_PATH = ROOT / "results" / "anfis_balanced_predictions.csv"

FEATURES = [
    "temperature_motor",
    "vibration_rms",
    "current_phase_avg",
    "hours_since_maintenance",
]

TARGET = "failure_within_24h"

NUMBER_OF_MFS = 2
NUMBER_OF_RULES = NUMBER_OF_MFS ** len(FEATURES)

RIDGE_ALPHA = 0.01
OPTIMIZATION_SAMPLE_SIZE = 2500
MAX_OPTIMIZATION_ITERATIONS = 300
RANDOM_SEED = 42


def create_rule_combinations():
    return list(product(range(NUMBER_OF_MFS), repeat=len(FEATURES)))


RULES = create_rule_combinations()


def initialize_premise_parameters(X):
    """Initialize Gaussian membership functions from training data."""

    centers = np.zeros((len(FEATURES), NUMBER_OF_MFS))
    widths = np.zeros((len(FEATURES), NUMBER_OF_MFS))

    for j in range(len(FEATURES)):
        centers[j] = np.quantile(X[:, j], [0.25, 0.75])

        spread = max(float(np.std(X[:, j])), 0.1)
        widths[j] = [spread * 0.75, spread * 0.75]

    return centers, widths


def unpack_parameters(parameters):
    centers = parameters[:8].reshape(4, 2)
    log_widths = parameters[8:].reshape(4, 2)

    widths = np.exp(log_widths)
    widths = np.maximum(widths, 0.05)

    return centers, widths


def calculate_normalized_firing(X, centers, widths):
    """Calculate normalized firing strength for every fuzzy rule."""

    memberships = np.zeros(
        (len(X), len(FEATURES), NUMBER_OF_MFS)
    )

    for j in range(len(FEATURES)):
        for m in range(NUMBER_OF_MFS):
            memberships[:, j, m] = np.exp(
                -0.5 * (
                    (X[:, j] - centers[j, m]) / widths[j, m]
                ) ** 2
            )

    firing_strengths = np.ones((len(X), NUMBER_OF_RULES))

    for rule_index, rule in enumerate(RULES):
        for feature_index, mf_index in enumerate(rule):
            firing_strengths[:, rule_index] *= (
                memberships[:, feature_index, mf_index]
            )

    denominator = firing_strengths.sum(axis=1, keepdims=True)

    normalized = firing_strengths / np.maximum(
        denominator, 1e-12
    )

    return normalized


def create_consequent_matrix(X, normalized_firing):
    """Build the first-order Sugeno consequent design matrix."""

    X_with_bias = np.column_stack([
        np.ones(len(X)),
        X
    ])

    weighted_blocks = [
        normalized_firing[:, rule:rule + 1] * X_with_bias
        for rule in range(NUMBER_OF_RULES)
    ]

    return np.concatenate(weighted_blocks, axis=1)


def fit_consequents(X, y, centers, widths):
    normalized = calculate_normalized_firing(
        X, centers, widths
    )

    design = create_consequent_matrix(X, normalized)

    identity = np.eye(design.shape[1])
    coefficients = np.linalg.solve(
        design.T @ design + RIDGE_ALPHA * identity,
        design.T @ y
    )

    return coefficients


def predict_anfis(X, centers, widths, coefficients):
    normalized = calculate_normalized_firing(
        X, centers, widths
    )

    design = create_consequent_matrix(X, normalized)

    predictions = design @ coefficients

    return np.clip(predictions, 0.0, 1.0)


def premise_objective(parameters, X, y):
    """Optimize Gaussian membership parameters."""

    centers, widths = unpack_parameters(parameters)

    coefficients = fit_consequents(
        X, y, centers, widths
    )

    predictions = predict_anfis(
        X, centers, widths, coefficients
    )

    mse = np.mean((y - predictions) ** 2)

    # Small regularization discourages excessively narrow MFs.
    regularization = 0.0001 * np.mean(
        np.log(widths) ** 2
    )

    return mse + regularization


def main():
    np.random.seed(RANDOM_SEED)

    print("\n========== ANFIS MODEL TRAINING ==========")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train_raw = train_df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    X_test_raw = test_df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    y_train = train_df[TARGET].astype(int).to_numpy()
    y_test = test_df[TARGET].astype(int).to_numpy()

    # Fit preprocessing only on training data.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Balance training data only.
    oversampler = RandomOverSampler(random_state=42)

    X_train, y_train = oversampler.fit_resample(
        X_train, y_train
    )

    print("\nTraining class distribution after oversampling:")
    print(pd.Series(y_train).value_counts())

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Inputs: {len(FEATURES)}")
    print(f"Membership functions per input: {NUMBER_OF_MFS}")
    print(f"Fuzzy rules: {NUMBER_OF_RULES}")

    # Initialize Gaussian membership functions.
    centers, widths = initialize_premise_parameters(X_train)

    initial_parameters = np.concatenate([
        centers.flatten(),
        np.log(widths).flatten()
    ])

    # Use a reproducible subset to make premise optimization practical.
    rng = np.random.default_rng(RANDOM_SEED)

    sample_size = min(
        OPTIMIZATION_SAMPLE_SIZE,
        len(X_train)
    )

    sample_indices = rng.choice(
        len(X_train),
        size=sample_size,
        replace=False
    )

    X_opt = X_train[sample_indices]
    y_opt = y_train[sample_indices]

    print("\nOptimizing membership functions...")

    optimization = minimize(
        premise_objective,
        initial_parameters,
        args=(X_opt, y_opt),
        method="L-BFGS-B",
        options={
            "maxiter": MAX_OPTIMIZATION_ITERATIONS,
            "ftol": 1e-7,
            "gtol": 1e-5,
            "maxls": 30
        }
    )

    centers, widths = unpack_parameters(
        optimization.x
    )

    print("Membership optimization completed.")
    print(f"Optimization success: {optimization.success}")
    print(f"Optimization message: {optimization.message}")

    # Refit consequent parameters using the full training set.
    print("\nLearning Sugeno consequent parameters...")

    coefficients = fit_consequents(
        X_train, y_train, centers, widths
    )

    probabilities = predict_anfis(
        X_test, centers, widths, coefficients
    )

    predictions = (probabilities >= 0.5).astype(int)

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

    print("\n========== ANFIS RESULTS ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Save the complete fitted model and preprocessing objects.
    model = {
        "features": FEATURES,
        "imputer": imputer,
        "scaler": scaler,
        "centers": centers,
        "widths": widths,
        "coefficients": coefficients,
        "rules": RULES,
        "membership_type": "Gaussian",
        "inference_type": "First-order Sugeno",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "number_of_rules": NUMBER_OF_RULES,
        "membership_functions_per_input": NUMBER_OF_MFS,
        "optimization_success": bool(optimization.success),
        "optimization_message": str(optimization.message),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    prediction_output = pd.DataFrame({
        "actual": y_test,
        "predicted": predictions,
        "failure_probability": probabilities,
    })

    prediction_output.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print("\n========== FILES SAVED ==========")
    print(f"Model:       {MODEL_PATH}")
    print(f"Metrics:     {METRICS_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
