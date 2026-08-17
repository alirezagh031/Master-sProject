from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "xauusd_mtf_train.parquet"
VALIDATION_FILE = DATA_DIR / "xauusd_mtf_validation.parquet"
TEST_FILE = DATA_DIR / "xauusd_mtf_test.parquet"

OUTPUT_FILE = DATA_DIR / "leakage_audit_report.json"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TIMESTAMP_COL = "timestamp"

TARGET_CANDIDATES = {
    "target",
    "future_return",
}

HORIZON_BARS = 1

# Current target is based on the next M15 candle.
HORIZON_MINUTES = 15

# Names which are explicitly allowed to contain "future".
# These are target-generation artifacts and must never be model features.
ALLOWED_TARGET_COLUMNS = {
    "future_return",
    "target",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def check(condition: bool, message: str) -> dict[str, Any]:
    return {
        "status": "PASS" if condition else "FAIL",
        "message": message,
    }


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Required timestamp column '{TIMESTAMP_COL}' "
            f"not found in {path.name}"
        )

    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL],
        errors="raise",
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        TIMESTAMP_COL,
        "target",
        "future_return",
    }

    return [
        col
        for col in df.columns
        if col not in excluded
    ]


# ---------------------------------------------------------------------
# Dataset-level checks
# ---------------------------------------------------------------------

def audit_timestamp_structure(
    df: pd.DataFrame,
    name: str,
) -> dict[str, Any]:

    timestamps = df[TIMESTAMP_COL]

    return {
        "name": name,
        "rows": int(len(df)),
        "start": str(timestamps.min()),
        "end": str(timestamps.max()),
        "monotonic_increasing": bool(timestamps.is_monotonic_increasing),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
    }


def audit_numeric_integrity(
    df: pd.DataFrame,
    name: str,
) -> dict[str, Any]:

    numeric_df = df.select_dtypes(include=[np.number])

    inf_count = int(
        np.isinf(numeric_df.to_numpy()).sum()
    )

    nan_count = int(
        numeric_df.isna().sum().sum()
    )

    return {
        "name": name,
        "numeric_columns": int(len(numeric_df.columns)),
        "nan_values": nan_count,
        "inf_values": inf_count,
    }


# ---------------------------------------------------------------------
# Split leakage
# ---------------------------------------------------------------------

def audit_split_boundaries(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:

    train_ts = set(train[TIMESTAMP_COL])
    validation_ts = set(validation[TIMESTAMP_COL])
    test_ts = set(test[TIMESTAMP_COL])

    train_validation_overlap = len(
        train_ts.intersection(validation_ts)
    )

    train_test_overlap = len(
        train_ts.intersection(test_ts)
    )

    validation_test_overlap = len(
        validation_ts.intersection(test_ts)
    )

    chronological = (
        train[TIMESTAMP_COL].max()
        < validation[TIMESTAMP_COL].min()
        < validation[TIMESTAMP_COL].max()
        < test[TIMESTAMP_COL].min()
    )

    return {
        "train_validation_overlap": train_validation_overlap,
        "train_test_overlap": train_test_overlap,
        "validation_test_overlap": validation_test_overlap,
        "strict_chronological_order": bool(chronological),
        "status": (
            "PASS"
            if (
                train_validation_overlap == 0
                and train_test_overlap == 0
                and validation_test_overlap == 0
                and chronological
            )
            else "FAIL"
        ),
    }


# ---------------------------------------------------------------------
# Target leakage
# ---------------------------------------------------------------------

def audit_target_columns(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:

    all_columns = set(train.columns)

    target_columns = sorted(
        all_columns.intersection(TARGET_CANDIDATES)
    )

    missing_target = "target" not in all_columns

    future_return_present = "future_return" in all_columns

    return {
        "target_present": not missing_target,
        "target_columns": target_columns,
        "future_return_present": future_return_present,
        "target_dtype": (
            str(train["target"].dtype)
            if "target" in train.columns
            else None
        ),
        "status": (
            "PASS"
            if not missing_target
            else "FAIL"
        ),
    }


# ---------------------------------------------------------------------
# Feature-name audit
# ---------------------------------------------------------------------

def audit_suspicious_feature_names(
    df: pd.DataFrame,
) -> dict[str, Any]:

    feature_columns = get_feature_columns(df)

    suspicious_tokens = (
        "future",
        "target",
        "label",
        "next_",
        "next",
        "lead",
    )

    suspicious = []

    for col in feature_columns:

        name = col.lower()

        if any(token in name for token in suspicious_tokens):

            suspicious.append(col)

    # A suspicious name is not automatically leakage.
    # It only means that the feature needs manual review.

    return {
        "suspicious_feature_names": suspicious,
        "count": len(suspicious),
        "status": (
            "REVIEW"
            if suspicious
            else "PASS"
        ),
    }


# ---------------------------------------------------------------------
# Target horizon audit
# ---------------------------------------------------------------------

def audit_target_horizon(
    df: pd.DataFrame,
) -> dict[str, Any]:

    if "future_return" not in df.columns:
        return {
            "status": "SKIP",
            "message": "future_return column not available.",
        }

    timestamps = df[TIMESTAMP_COL].reset_index(drop=True)
    future_return = df["future_return"].reset_index(drop=True)

    # The target uses the next available M15 observation.
    next_timestamp = timestamps.shift(-HORIZON_BARS)

    expected_delta = pd.Timedelta(
        minutes=HORIZON_MINUTES
    )

    valid = (
        next_timestamp.notna()
        & future_return.notna()
    )

    observed_delta = (
        next_timestamp[valid]
        - timestamps[valid]
    )

    exact_horizon_matches = (
        observed_delta == expected_delta
    ).sum()

    checked = int(valid.sum())

    return {
        "expected_horizon_bars": HORIZON_BARS,
        "expected_horizon_minutes": HORIZON_MINUTES,
        "checked_rows": checked,
        "exact_horizon_matches": int(exact_horizon_matches),
        "non_matching_horizon_rows": int(
            checked - exact_horizon_matches
        ),
        "status": (
            "PASS"
            if exact_horizon_matches == checked
            else "REVIEW"
        ),
    }


# ---------------------------------------------------------------------
# Future timestamp / causal ordering audit
# ---------------------------------------------------------------------

def audit_future_timestamp_patterns(
    df: pd.DataFrame,
) -> dict[str, Any]:

    timestamps = df[TIMESTAMP_COL]

    invalid_rows = int(
        timestamps.isna().sum()
    )

    negative_intervals = int(
        (
            timestamps.diff()
            < pd.Timedelta(0)
        ).sum()
    )

    return {
        "missing_timestamps": invalid_rows,
        "negative_time_intervals": negative_intervals,
        "status": (
            "PASS"
            if invalid_rows == 0
            and negative_intervals == 0
            else "FAIL"
        ),
    }


# ---------------------------------------------------------------------
# Cross-split target boundary audit
# ---------------------------------------------------------------------

def audit_label_boundary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:

    train_end = train[TIMESTAMP_COL].max()
    validation_start = validation[TIMESTAMP_COL].min()

    validation_end = validation[TIMESTAMP_COL].max()
    test_start = test[TIMESTAMP_COL].min()

    train_to_validation_gap = (
        validation_start - train_end
    )

    validation_to_test_gap = (
        test_start - validation_end
    )

    return {
        "train_end": str(train_end),
        "validation_start": str(validation_start),
        "train_validation_gap": str(
            train_to_validation_gap
        ),
        "validation_end": str(validation_end),
        "test_start": str(test_start),
        "validation_test_gap": str(
            validation_to_test_gap
        ),
        "status": "PASS",
    }


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("SECTION 4.1 — LEAKAGE AUDIT")
    print("=" * 72)

    train = load_dataset(TRAIN_FILE)
    validation = load_dataset(VALIDATION_FILE)
    test = load_dataset(TEST_FILE)

    print("\nDatasets loaded:")
    print(f"  TRAIN      : {train.shape}")
    print(f"  VALIDATION : {validation.shape}")
    print(f"  TEST       : {test.shape}")

    report: dict[str, Any] = {
        "stage": "4.1",
        "title": "Leakage Audit",
        "configuration": {
            "timestamp_column": TIMESTAMP_COL,
            "target_columns": sorted(TARGET_CANDIDATES),
            "horizon_bars": HORIZON_BARS,
            "horizon_minutes": HORIZON_MINUTES,
        },
        "datasets": {},
        "checks": {},
    }

    # Dataset checks
    for name, df in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        report["datasets"][name] = {
            "timestamp": audit_timestamp_structure(df, name),
            "numeric_integrity": audit_numeric_integrity(df, name),
        }

    # Split checks
    report["checks"]["split_boundaries"] = (
        audit_split_boundaries(
            train,
            validation,
            test,
        )
    )

    # Target checks
    report["checks"]["target_columns"] = (
        audit_target_columns(
            train,
            validation,
            test,
        )
    )

    # Feature name audit
    report["checks"]["suspicious_feature_names"] = (
        audit_suspicious_feature_names(train)
    )

    # Target horizon
    report["checks"]["target_horizon"] = (
        audit_target_horizon(train)
    )

    # Timestamp causality
    report["checks"]["train_timestamp_causality"] = (
        audit_future_timestamp_patterns(train)
    )

    report["checks"]["validation_timestamp_causality"] = (
        audit_future_timestamp_patterns(validation)
    )

    report["checks"]["test_timestamp_causality"] = (
        audit_future_timestamp_patterns(test)
    )

    # Label boundary
    report["checks"]["label_boundaries"] = (
        audit_label_boundary(
            train,
            validation,
            test,
        )
    )

    # Overall status
    mandatory_checks = [
        report["checks"]["split_boundaries"]["status"],
        report["checks"]["target_columns"]["status"],
        report["checks"]["train_timestamp_causality"]["status"],
        report["checks"]["validation_timestamp_causality"]["status"],
        report["checks"]["test_timestamp_causality"]["status"],
    ]

    report["overall_status"] = (
        "PASS"
        if all(status == "PASS" for status in mandatory_checks)
        else "FAIL"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("\n" + "=" * 72)
    print(
        f"OVERALL STATUS: {report['overall_status']}"
    )
    print("=" * 72)

    print(
        f"\nReport saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()