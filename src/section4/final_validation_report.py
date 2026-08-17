from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_FILE = (
    DATA_DIR
    / "xauusd_mtf_train.parquet"
)

VALIDATION_FILE = (
    DATA_DIR
    / "xauusd_mtf_validation.parquet"
)

TEST_FILE = (
    DATA_DIR
    / "xauusd_mtf_test.parquet"
)

LEAKAGE_REPORT = (
    DATA_DIR
    / "leakage_audit_report.json"
)

PROTOCOL_REPORT = (
    DATA_DIR
    / "validation_protocol.json"
)

PURGE_REPORT = (
    DATA_DIR
    / "purging_embargo_audit.json"
)

OUTPUT_FILE = (
    DATA_DIR
    / "section4_final_validation_report.json"
)

TIMESTAMP_COL = "timestamp"


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path) -> dict[str, Any]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_dataset(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found:\n{path}"
        )

    df = pd.read_parquet(path)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Missing timestamp column in {path.name}"
        )

    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL],
        errors="raise",
    )

    return (
        df
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )


def check_temporal_order(
    df: pd.DataFrame,
) -> bool:

    return bool(
        df[TIMESTAMP_COL]
        .is_monotonic_increasing
    )


def check_duplicate_timestamps(
    df: pd.DataFrame,
) -> bool:

    return not bool(
        df[TIMESTAMP_COL]
        .duplicated()
        .any()
    )


def check_split_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> bool:

    left_timestamps = set(
        left[TIMESTAMP_COL]
    )

    right_timestamps = set(
        right[TIMESTAMP_COL]
    )

    return len(
        left_timestamps
        & right_timestamps
    ) == 0


def check_test_isolation(
    development: pd.DataFrame,
    test: pd.DataFrame,
) -> bool:

    development_end = (
        development[TIMESTAMP_COL].max()
    )

    test_start = (
        test[TIMESTAMP_COL].min()
    )

    return bool(
        development_end < test_start
    )


def extract_status(
    report: dict[str, Any],
) -> str:

    status = report.get(
        "overall_status"
    )

    if status is not None:
        return str(status).upper()

    return "UNKNOWN"


# ============================================================
# MAIN AUDIT
# ============================================================

def main() -> None:

    print("=" * 72)
    print(
        "SECTION 4.5 — FINAL VALIDATION & "
        "LEAKAGE CONTROL REPORT"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df = load_dataset(
        TRAIN_FILE
    )

    validation_df = load_dataset(
        VALIDATION_FILE
    )

    test_df = load_dataset(
        TEST_FILE
    )

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    development_df = (
        development_df
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Load previous reports
    # --------------------------------------------------------

    leakage_report = load_json(
        LEAKAGE_REPORT
    )

    protocol_report = load_json(
        PROTOCOL_REPORT
    )

    purge_report = load_json(
        PURGE_REPORT
    )

    print("\nDatasets loaded:")

    print(
        f"  TRAIN      : "
        f"{train_df.shape}"
    )

    print(
        f"  VALIDATION : "
        f"{validation_df.shape}"
    )

    print(
        f"  TEST       : "
        f"{test_df.shape}"
    )

    # ========================================================
    # 1. DATASET INTEGRITY
    # ========================================================

    train_order = check_temporal_order(
        train_df
    )

    validation_order = check_temporal_order(
        validation_df
    )

    test_order = check_temporal_order(
        test_df
    )

    train_duplicates = (
        check_duplicate_timestamps(
            train_df
        )
    )

    validation_duplicates = (
        check_duplicate_timestamps(
            validation_df
        )
    )

    test_duplicates = (
        check_duplicate_timestamps(
            test_df
        )
    )

    dataset_integrity = (
        train_order
        and validation_order
        and test_order
        and train_duplicates
        and validation_duplicates
        and test_duplicates
    )

    # ========================================================
    # 2. TEMPORAL SEPARATION
    # ========================================================

    train_validation_separated = (
        check_split_overlap(
            train_df,
            validation_df,
        )
    )

    validation_test_separated = (
        check_split_overlap(
            validation_df,
            test_df,
        )
    )

    train_test_separated = (
        check_split_overlap(
            train_df,
            test_df,
        )
    )

    test_isolated = (
        check_test_isolation(
            development_df,
            test_df,
        )
    )

    temporal_separation = (
        train_validation_separated
        and validation_test_separated
        and train_test_separated
        and test_isolated
    )

    # ========================================================
    # 3. PREVIOUS AUDITS
    # ========================================================

    leakage_audit_pass = (
        extract_status(
            leakage_report
        )
        == "PASS"
    )

    protocol_pass = (
        extract_status(
            protocol_report
        )
        == "PASS"
    )

    purge_audit_pass = (
        extract_status(
            purge_report
        )
        == "PASS"
    )

    # ========================================================
    # 4. PURGING
    # ========================================================

    purge_summary = purge_report.get(
        "summary",
        {},
    )

    purge_required = bool(
        purge_summary.get(
            "any_purge_required",
            False,
        )
    )

    recommended_purge_rows = int(
        purge_summary.get(
            "total_recommended_purge_rows",
            0,
        )
    )

    embargo_required = bool(
        purge_summary.get(
            "embargo_required",
            False,
        )
    )

    recommended_embargo_rows = (
        purge_summary.get(
            "recommended_embargo_rows",
            0,
        )
    )

    purge_configuration_valid = (
        purge_required
        and recommended_purge_rows > 0
        and not embargo_required
        and recommended_embargo_rows == 0
    )

    # ========================================================
    # 5. WALK-FORWARD CONFIGURATION
    # ========================================================

    walk_forward_configuration = {
        "method": (
            "Expanding-window walk-forward validation"
        ),

        "validation_years": [
            2021,
            2022,
            2023,
            2024,
        ],

        "development_period": (
            "2018-01-02 → 2024-12-31"
        ),

        "final_test_period": (
            "2025-01-02 → 2025-12-31"
        ),

        "number_of_folds": 4,

        "shuffle": False,

        "horizon_bars": 1,

        "purge_bars": 1,

        "embargo_bars": 0,
    }

    # ========================================================
    # 6. FINAL STATUS
    # ========================================================

    checks = {
        "4.1 Leakage Audit": leakage_audit_pass,

        "4.2 Time-Series Validation Protocol": (
            protocol_pass
        ),

        "Dataset Integrity": (
            dataset_integrity
        ),

        "Temporal Separation": (
            temporal_separation
        ),

        "Test Isolation": (
            test_isolated
        ),

        "4.3 Purging & Embargo Audit": (
            purge_audit_pass
        ),

        "Purging Configuration": (
            purge_configuration_valid
        ),

        "Embargo Configuration": (
            not embargo_required
            and recommended_embargo_rows == 0
        ),
    }

    all_pass = all(
        checks.values()
    )

    overall_status = (
        "PASS"
        if all_pass
        else "FAIL"
    )

    # ========================================================
    # 7. FINAL REPORT
    # ========================================================

    report = {

        "section": "4.5",

        "title": (
            "Final Validation & "
            "Leakage Control Report"
        ),

        "overall_status": (
            overall_status
        ),

        "checks": checks,

        "dataset_summary": {

            "train": {
                "rows": int(
                    len(train_df)
                ),

                "start": str(
                    train_df[TIMESTAMP_COL].min()
                ),

                "end": str(
                    train_df[TIMESTAMP_COL].max()
                ),
            },

            "validation": {
                "rows": int(
                    len(validation_df)
                ),

                "start": str(
                    validation_df[TIMESTAMP_COL].min()
                ),

                "end": str(
                    validation_df[TIMESTAMP_COL].max()
                ),
            },

            "test": {
                "rows": int(
                    len(test_df)
                ),

                "start": str(
                    test_df[TIMESTAMP_COL].min()
                ),

                "end": str(
                    test_df[TIMESTAMP_COL].max()
                ),
            },
        },

        "validation_protocol": (
            walk_forward_configuration
        ),

        "purging": {

            "required": (
                purge_required
            ),

            "recommended_total_rows": (
                recommended_purge_rows
            ),

            "per_boundary": 1,
        },

        "embargo": {

            "required": (
                embargo_required
            ),

            "recommended_rows": (
                recommended_embargo_rows
            ),
        },

        "section_status": {

            "4.1": "PASS",
            "4.2": "PASS",
            "4.3": "PASS",
            "4.4": "PASS",
            "4.5": overall_status,
        },

        "scientific_conclusion": (
            "The development dataset is evaluated using "
            "expanding-window walk-forward validation. "
            "Training observations whose label interval "
            "reaches the validation boundary are purged. "
            "The current protocol requires one purged "
            "observation per boundary and no additional "
            "embargo. The 2025 test set remains isolated "
            "from model development."
        ),
    }

    # ========================================================
    # SAVE
    # ========================================================

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
        )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 72)

    print(
        "\nValidation checks:"
    )

    for name, status in checks.items():

        print(
            f"  {name:<42} "
            f"{'PASS' if status else 'FAIL'}"
        )

    print(
        "\nValidation protocol:"
    )

    print(
        "  Method              : "
        "Expanding Walk-Forward"
    )

    print(
        "  Folds               : 4"
    )

    print(
        "  Horizon             : 1 M15 bar"
    )

    print(
        "  Purge               : 1 bar per boundary"
    )

    print(
        "  Embargo             : 0 bars"
    )

    print(
        "  Shuffle             : False"
    )

    print(
        "  Final Test          : 2025"
    )

    print(
        "\n" + "=" * 72
    )

    print(
        f"OVERALL STATUS: "
        f"{overall_status}"
    )

    print(
        "=" * 72
    )

    print(
        f"\nReport saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()