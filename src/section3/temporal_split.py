from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")

INPUT_FILE = (
    PROCESSED_DIR / "xauusd_mtf_target.parquet"
)

TRAIN_FILE = (
    PROCESSED_DIR / "xauusd_mtf_train.parquet"
)

VALIDATION_FILE = (
    PROCESSED_DIR / "xauusd_mtf_validation.parquet"
)

TEST_FILE = (
    PROCESSED_DIR / "xauusd_mtf_test.parquet"
)

MANIFEST_FILE = (
    PROCESSED_DIR / "temporal_split_manifest.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")

VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2024-12-31 23:59:59")

TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31 23:59:59")


TARGET_COLUMN = "target"
FUTURE_RETURN_COLUMN = "future_return"


# ============================================================
# UTILITIES
# ============================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_dataset():
    print_header("LOADING TARGET DATASET")

    print("Input:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print("Shape:", df.shape)

    return df


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(df):
    print_header("INPUT VALIDATION")

    required_columns = [
        "timestamp",
        TARGET_COLUMN,
        FUTURE_RETURN_COLUMN,
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    print("PASS: Required columns exist.")

    # Timestamp
    if df["timestamp"].isna().any():
        raise RuntimeError(
            "Timestamp contains NaN values."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    ):
        raise RuntimeError(
            "Timestamp column is not datetime."
        )

    print("PASS: Timestamp is valid.")

    # Sort
    if not df["timestamp"].is_monotonic_increasing:
        raise RuntimeError(
            "Dataset is not sorted by timestamp."
        )

    print("PASS: Timestamp is monotonic increasing.")

    # Duplicate timestamps
    duplicates = df["timestamp"].duplicated().sum()

    print("Duplicate timestamps:", duplicates)

    if duplicates != 0:
        raise RuntimeError(
            "Duplicate timestamps detected."
        )

    print("PASS: No duplicate timestamps.")

    # Target
    valid_targets = {-1, 0, 1}

    actual_targets = set(
        df[TARGET_COLUMN].dropna().unique()
    )

    if not actual_targets.issubset(valid_targets):
        raise RuntimeError(
            f"Invalid target labels: {actual_targets}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise RuntimeError(
            "Target contains NaN values."
        )

    print("PASS: Target labels are valid.")

    # Future return
    if df[FUTURE_RETURN_COLUMN].isna().any():
        raise RuntimeError(
            "future_return contains NaN values."
        )

    if np.isinf(
        df[FUTURE_RETURN_COLUMN].to_numpy()
    ).any():
        raise RuntimeError(
            "future_return contains infinite values."
        )

    print("PASS: Future return is valid.")


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def split_temporally(df):
    print_header("TEMPORAL SPLIT")

    train_mask = (
        (df["timestamp"] >= TRAIN_START)
        & (df["timestamp"] <= TRAIN_END)
    )

    validation_mask = (
        (df["timestamp"] >= VALIDATION_START)
        & (df["timestamp"] <= VALIDATION_END)
    )

    test_mask = (
        (df["timestamp"] >= TEST_START)
        & (df["timestamp"] <= TEST_END)
    )

    train = df.loc[train_mask].copy()
    validation = df.loc[validation_mask].copy()
    test = df.loc[test_mask].copy()

    print("TRAIN rows     :", len(train))
    print("VALIDATION rows:", len(validation))
    print("TEST rows      :", len(test))

    return train, validation, test


# ============================================================
# SPLIT VALIDATION
# ============================================================

def validate_split(
    original,
    train,
    validation,
    test,
):
    print_header("SPLIT VALIDATION")

    # --------------------------------------------------------
    # Row preservation
    # --------------------------------------------------------

    total_split_rows = (
        len(train)
        + len(validation)
        + len(test)
    )

    print("Original rows :", len(original))
    print("Split rows    :", total_split_rows)

    if total_split_rows != len(original):
        raise RuntimeError(
            "Row count mismatch after temporal split."
        )

    print("PASS: Row count preserved.")

    # --------------------------------------------------------
    # Timestamp boundaries
    # --------------------------------------------------------

    if len(train) > 0:
        print(
            "TRAIN:",
            train["timestamp"].min(),
            "->",
            train["timestamp"].max(),
        )

    if len(validation) > 0:
        print(
            "VALIDATION:",
            validation["timestamp"].min(),
            "->",
            validation["timestamp"].max(),
        )

    if len(test) > 0:
        print(
            "TEST:",
            test["timestamp"].min(),
            "->",
            test["timestamp"].max(),
        )

    # --------------------------------------------------------
    # Temporal ordering
    # --------------------------------------------------------

    if len(train) > 0 and len(validation) > 0:
        if train["timestamp"].max() >= validation["timestamp"].min():
            raise RuntimeError(
                "TRAIN/VALIDATION temporal overlap detected."
            )

    if len(validation) > 0 and len(test) > 0:
        if validation["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError(
                "VALIDATION/TEST temporal overlap detected."
            )

    if len(train) > 0 and len(test) > 0:
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError(
                "TRAIN/TEST temporal overlap detected."
            )

    print("PASS: No temporal overlap.")

    # --------------------------------------------------------
    # Timestamp uniqueness across splits
    # --------------------------------------------------------

    train_ts = set(train["timestamp"])
    validation_ts = set(validation["timestamp"])
    test_ts = set(test["timestamp"])

    if train_ts.intersection(validation_ts):
        raise RuntimeError(
            "Timestamp overlap between TRAIN and VALIDATION."
        )

    if validation_ts.intersection(test_ts):
        raise RuntimeError(
            "Timestamp overlap between VALIDATION and TEST."
        )

    if train_ts.intersection(test_ts):
        raise RuntimeError(
            "Timestamp overlap between TRAIN and TEST."
        )

    print("PASS: No timestamp duplication across splits.")

    # --------------------------------------------------------
    # Target availability
    # --------------------------------------------------------

    for name, data in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        if data[TARGET_COLUMN].isna().any():
            raise RuntimeError(
                f"{name} contains NaN targets."
            )

    print("PASS: All splits have valid targets.")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_target_distribution(
    name,
    df,
):
    print(f"\n{name} TARGET DISTRIBUTION")

    counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    labels = {
        -1: "DOWN",
        0: "NEUTRAL",
        1: "UP",
    }

    for target in [-1, 0, 1]:
        count = int(counts.get(target, 0))

        percentage = (
            100.0 * count / total
            if total > 0
            else 0.0
        )

        print(
            f"{target:2d} "
            f"{labels[target]:8s} "
            f"{count:7d} "
            f"({percentage:7.3f}%)"
        )


def validate_target_distributions(
    train,
    validation,
    test,
):
    print_header("TARGET DISTRIBUTION")

    print_target_distribution(
        "TRAIN",
        train,
    )

    print_target_distribution(
        "VALIDATION",
        validation,
    )

    print_target_distribution(
        "TEST",
        test,
    )


# ============================================================
# MODEL-READY COLUMN INFORMATION
# ============================================================

def identify_columns(df):
    print_header("MODEL-READY COLUMN ANALYSIS")

    excluded_columns = {
        "timestamp",
        TARGET_COLUMN,
        FUTURE_RETURN_COLUMN,
    }

    feature_columns = [
        col
        for col in df.columns
        if col not in excluded_columns
    ]

    print("Total columns:", len(df.columns))

    print(
        "Feature columns:",
        len(feature_columns),
    )

    print(
        "Target column:",
        TARGET_COLUMN,
    )

    print(
        "Future return column:",
        FUTURE_RETURN_COLUMN,
    )

    print("\nFirst feature columns:")

    for col in feature_columns[:20]:
        print(" -", col)

    if len(feature_columns) > 20:
        print(
            f" ... and {len(feature_columns) - 20} more"
        )

    return feature_columns


# ============================================================
# SAVE DATASETS
# ============================================================

def save_splits(
    train,
    validation,
    test,
):
    print_header("SAVING TEMPORAL DATASETS")

    train.to_parquet(
        TRAIN_FILE,
        index=False,
    )

    validation.to_parquet(
        VALIDATION_FILE,
        index=False,
    )

    test.to_parquet(
        TEST_FILE,
        index=False,
    )

    print("Saved:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)


# ============================================================
# MANIFEST
# ============================================================

def create_manifest(
    original,
    train,
    validation,
    test,
    feature_columns,
):
    print_header("CREATING TEMPORAL SPLIT MANIFEST")

    manifest = {
        "dataset": "XAUUSD",
        "source_file": str(INPUT_FILE),

        "total_rows": int(len(original)),

        "features": {
            "count": int(len(feature_columns)),
            "columns": feature_columns,
        },

        "target": {
            "column": TARGET_COLUMN,
            "labels": {
                "-1": "DOWN",
                "0": "NEUTRAL",
                "1": "UP",
            },
        },

        "splits": {
            "train": {
                "rows": int(len(train)),
                "start": (
                    str(train["timestamp"].min())
                    if len(train) > 0
                    else None
                ),
                "end": (
                    str(train["timestamp"].max())
                    if len(train) > 0
                    else None
                ),
            },

            "validation": {
                "rows": int(len(validation)),
                "start": (
                    str(validation["timestamp"].min())
                    if len(validation) > 0
                    else None
                ),
                "end": (
                    str(validation["timestamp"].max())
                    if len(validation) > 0
                    else None
                ),
            },

            "test": {
                "rows": int(len(test)),
                "start": (
                    str(test["timestamp"].min())
                    if len(test) > 0
                    else None
                ),
                "end": (
                    str(test["timestamp"].max())
                    if len(test) > 0
                    else None
                ),
            },
        },

        "split_policy": {
            "method": "strict_temporal_split",
            "shuffle": False,
            "train_end": str(TRAIN_END),
            "validation_start": str(VALIDATION_START),
            "validation_end": str(VALIDATION_END),
            "test_start": str(TEST_START),
            "test_end": str(TEST_END),
        },

        "leakage_policy": {
            "test_used_for_training": False,
            "test_used_for_feature_selection": False,
            "test_used_for_hyperparameter_optimization": False,
            "test_used_for_evolutionary_optimization": False,
        },
    }

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("Saved:")
    print(MANIFEST_FILE)


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    train,
    validation,
    test,
):
    print_header("FINAL VALIDATION")

    checks = {
        "TRAIN non-empty": len(train) > 0,
        "VALIDATION non-empty": len(validation) > 0,
        "TEST non-empty": len(test) > 0,

        "TRAIN sorted": train["timestamp"].is_monotonic_increasing,
        "VALIDATION sorted": validation["timestamp"].is_monotonic_increasing,
        "TEST sorted": test["timestamp"].is_monotonic_increasing,
    }

    for name, passed in checks.items():
        if passed:
            print(f"PASS: {name}")
        else:
            raise RuntimeError(
                f"FAIL: {name}"
            )

    print("\nALL TEMPORAL SPLIT CHECKS PASSED.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CAUSAL TEMPORAL DATASET SPLITTING")
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_input(df)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, validation, test = split_temporally(df)

    # --------------------------------------------------------
    # Validate split
    # --------------------------------------------------------

    validate_split(
        df,
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    validate_target_distributions(
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = identify_columns(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_splits(
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    create_manifest(
        df,
        train,
        validation,
        test,
        feature_columns,
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    final_validation(
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_header("TEMPORAL SPLITTING COMPLETE")

    print("TRAIN:")
    print(
        f"  Rows: {len(train)}"
    )

    print("VALIDATION:")
    print(
        f"  Rows: {len(validation)}"
    )

    print("TEST:")
    print(
        f"  Rows: {len(test)}"
    )

    print("\nOutput files:")
    print(f" - {TRAIN_FILE}")
    print(f" - {VALIDATION_FILE}")
    print(f" - {TEST_FILE}")
    print(f" - {MANIFEST_FILE}")


if __name__ == "__main__":
    main()