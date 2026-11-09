import pandas as pd
import json
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "raw" / "predictive_maintenance_v3.csv"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
RESULTS_DIR = BASE_DIR / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Features selected for ANN
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

TARGET = "failure_within_24h"

REQUIRED_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + [TARGET, "timestamp", "machine_id"]
)


def main():

    print("\n========== DATA PREPROCESSING ==========\n")

    # 1. Load dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # 2. Validate required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    # 3. Check duplicate records
    duplicate_count = int(df.duplicated().sum())

    print("Duplicate records:", duplicate_count)

    df = df.drop_duplicates().copy()

    # 4. Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    invalid_timestamps = int(df["timestamp"].isna().sum())

    print("Invalid timestamps:", invalid_timestamps)

    df = df.dropna(subset=["timestamp", TARGET]).copy()

    # 5. Validate target
    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )

    df = df.dropna(subset=[TARGET]).copy()

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    df[TARGET] = df[TARGET].astype(int)

    # 6. Convert numerical features
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # 7. Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 8. Missing-value analysis
    missing_report = df.isnull().sum()

    print("\nMissing values:")
    print(missing_report)

    # 9. Target distribution
    class_counts = df[TARGET].value_counts().sort_index()

    print("\nTarget distribution:")
    print(class_counts)

    class_percentages = (
        df[TARGET].value_counts(normalize=True) * 100
    )

    # 10. Chronological 80:20 split
    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    if train_df[TARGET].nunique() < 2:
        raise ValueError("Training set contains only one class.")

    if test_df[TARGET].nunique() < 2:
        raise ValueError("Testing set contains only one class.")

    # 11. Save processed datasets
    train_df.to_csv(
        PROCESSED_DIR / "train.csv", index=False
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv", index=False
    )

    # 12. Save preprocessing summary
    summary = {
        "original_rows": int(len(pd.read_csv(DATA_PATH))),
        "processed_rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicates_removed": duplicate_count,
        "invalid_timestamps_removed": invalid_timestamps,
        "missing_values": {
            key: int(value)
            for key, value in missing_report.items()
        },
        "target_distribution": {
            str(key): int(value)
            for key, value in class_counts.items()
        },
        "target_percentages": {
            str(key): round(float(value), 2)
            for key, value in class_percentages.items()
        },
        "training_records": int(len(train_df)),
        "testing_records": int(len(test_df)),
        "training_failure_records": int(
            train_df[TARGET].sum()
        ),
        "testing_failure_records": int(
            test_df[TARGET].sum()
        )
    }

    with open(
        RESULTS_DIR / "preprocessing_summary.json",
        "w"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n========== PREPROCESSING COMPLETE ==========")
    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))
    print("Training data saved:", PROCESSED_DIR / "train.csv")
    print("Testing data saved:", PROCESSED_DIR / "test.csv")
    print("Summary saved:", RESULTS_DIR / "preprocessing_summary.json")


if __name__ == "__main__":
    main()
