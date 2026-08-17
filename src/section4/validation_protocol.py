from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "xauusd_mtf_train.parquet"
VALIDATION_FILE = DATA_DIR / "xauusd_mtf_validation.parquet"
TEST_FILE = DATA_DIR / "xauusd_mtf_test.parquet"

OUTPUT_FILE = DATA_DIR / "validation_protocol.json"

TIMESTAMP_COL = "timestamp"

# Minimum number of complete calendar years used
# for the first training window.
MIN_TRAIN_YEARS = 3

# Annual walk-forward validation.
VALIDATION_YEARS = [2021, 2022, 2023, 2024]

FINAL_TEST_YEAR = 2025


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_parquet(path)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Missing timestamp column: {TIMESTAMP_COL}"
        )

    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL],
        errors="raise",
    )

    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    return df


def year_range(start_year: int, end_year: int) -> list[int]:
    return list(range(start_year, end_year + 1))


def get_years(df: pd.DataFrame) -> list[int]:
    return sorted(
        df[TIMESTAMP_COL].dt.year.unique().tolist()
    )


def make_period(
    year: int,
) -> dict[str, str]:
    return {
        "start": f"{year}-01-01 00:00:00",
        "end": f"{year}-12-31 23:59:59",
    }


def create_walk_forward_folds(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:

    available_years = get_years(df)

    first_validation_year = VALIDATION_YEARS[0]

    first_train_year = (
        first_validation_year - MIN_TRAIN_YEARS
    )

    folds = []

    for fold_number, validation_year in enumerate(
        VALIDATION_YEARS,
        start=1,
    ):

        train_start_year = first_train_year
        train_end_year = validation_year - 1

        train_years = year_range(
            train_start_year,
            train_end_year,
        )

        validation_years = [validation_year]

        train_mask = (
            df[TIMESTAMP_COL].dt.year.isin(train_years)
        )

        validation_mask = (
            df[TIMESTAMP_COL].dt.year.isin(
                validation_years
            )
        )

        train_df = df.loc[train_mask]
        validation_df = df.loc[validation_mask]

        if train_df.empty:
            raise ValueError(
                f"Fold {fold_number}: empty training set."
            )

        if validation_df.empty:
            raise ValueError(
                f"Fold {fold_number}: empty validation set."
            )

        train_start = train_df[TIMESTAMP_COL].min()
        train_end = train_df[TIMESTAMP_COL].max()

        validation_start = (
            validation_df[TIMESTAMP_COL].min()
        )

        validation_end = (
            validation_df[TIMESTAMP_COL].max()
        )

        # Basic chronological guarantee.
        if not train_end < validation_start:
            raise ValueError(
                f"Fold {fold_number}: "
                "training period is not strictly "
                "before validation period."
            )

        folds.append(
            {
                "fold": fold_number,

                "type": "expanding_window",

                "train": {
                    "years": train_years,
                    "start": str(train_start),
                    "end": str(train_end),
                    "rows": int(len(train_df)),
                },

                "validation": {
                    "years": validation_years,
                    "start": str(validation_start),
                    "end": str(validation_end),
                    "rows": int(len(validation_df)),
                },

                "chronological_order": bool(
                    train_end < validation_start
                ),

                # Purging/embargo deliberately handled
                # in Stage 4.3.
                "purge_applied": False,
                "embargo_applied": False,
            }
        )

    return folds


def validate_final_test(
    test_df: pd.DataFrame,
) -> dict[str, Any]:

    years = get_years(test_df)

    if years != [FINAL_TEST_YEAR]:
        raise ValueError(
            "Final test dataset does not contain "
            f"only {FINAL_TEST_YEAR}."
        )

    return {
        "year": FINAL_TEST_YEAR,
        "start": str(
            test_df[TIMESTAMP_COL].min()
        ),
        "end": str(
            test_df[TIMESTAMP_COL].max()
        ),
        "rows": int(len(test_df)),
        "used_in_walk_forward": False,
        "reserved_for_final_evaluation": True,
    }


def validate_fold_independence(
    folds: list[dict[str, Any]],
) -> dict[str, Any]:

    errors = []

    previous_validation_end = None

    for fold in folds:

        train_start = pd.Timestamp(
            fold["train"]["start"]
        )

        train_end = pd.Timestamp(
            fold["train"]["end"]
        )

        validation_start = pd.Timestamp(
            fold["validation"]["start"]
        )

        validation_end = pd.Timestamp(
            fold["validation"]["end"]
        )

        if not train_start < train_end:
            errors.append(
                f"Fold {fold['fold']}: "
                "invalid training range."
            )

        if not validation_start <= validation_end:
            errors.append(
                f"Fold {fold['fold']}: "
                "invalid validation range."
            )

        if not train_end < validation_start:
            errors.append(
                f"Fold {fold['fold']}: "
                "train/validation overlap."
            )

        if previous_validation_end is not None:
            if not previous_validation_end < validation_start:
                errors.append(
                    f"Fold {fold['fold']}: "
                    "validation periods are not ordered."
                )

        previous_validation_end = validation_end

    return {
        "status": (
            "PASS"
            if not errors
            else "FAIL"
        ),
        "errors": errors,
    }


def main() -> None:

    print("=" * 72)
    print("SECTION 4.2 — TIME-SERIES VALIDATION PROTOCOL")
    print("=" * 72)

    train_df = load_dataset(TRAIN_FILE)
    validation_df = load_dataset(VALIDATION_FILE)
    test_df = load_dataset(TEST_FILE)

    print("\nDatasets loaded:")
    print(f"  TRAIN      : {train_df.shape}")
    print(f"  VALIDATION : {validation_df.shape}")
    print(f"  TEST       : {test_df.shape}")

    # Combine Train + Validation because the walk-forward
    # protocol needs to access all years 2018–2024.
    development_df = pd.concat(
        [train_df, validation_df],
        axis=0,
        ignore_index=True,
    )

    development_df = (
        development_df
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )

    available_years = get_years(development_df)

    required_years = year_range(
        2018,
        2024,
    )

    missing_years = [
        year
        for year in required_years
        if year not in available_years
    ]

    if missing_years:
        raise ValueError(
            f"Missing development years: {missing_years}"
        )

    folds = create_walk_forward_folds(
        development_df
    )

    fold_validation = validate_fold_independence(
        folds
    )

    final_test = validate_final_test(
        test_df
    )

    report = {
        "stage": "4.2",

        "title": (
            "Time-Series / Walk-Forward "
            "Validation Protocol"
        ),

        "method": {
            "name": "Expanding Window",
            "random_k_fold": False,
            "shuffle": False,
            "temporal_order_preserved": True,
        },

        "development_period": {
            "start_year": 2018,
            "end_year": 2024,
            "rows": int(len(development_df)),
        },

        "configuration": {
            "minimum_training_years": MIN_TRAIN_YEARS,
            "validation_years": VALIDATION_YEARS,
            "final_test_year": FINAL_TEST_YEAR,
            "purging": {
                "applied": False,
                "stage": "4.3",
            },
            "embargo": {
                "applied": False,
                "stage": "4.3",
            },
        },

        "folds": folds,

        "fold_validation": fold_validation,

        "final_test": final_test,

        "overall_status": (
            "PASS"
            if fold_validation["status"] == "PASS"
            else "FAIL"
        ),
    }

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

    print("\nWalk-Forward folds:")

    for fold in folds:
        print(
            f"\nFold {fold['fold']}"
        )

        print(
            f"  Train      : "
            f"{fold['train']['start']} "
            f"→ "
            f"{fold['train']['end']}"
        )

        print(
            f"  Train rows : "
            f"{fold['train']['rows']:,}"
        )

        print(
            f"  Validation : "
            f"{fold['validation']['start']} "
            f"→ "
            f"{fold['validation']['end']}"
        )

        print(
            f"  Valid rows : "
            f"{fold['validation']['rows']:,}"
        )

    print("\nFinal Test:")
    print(
        f"  {final_test['start']} "
        f"→ "
        f"{final_test['end']}"
    )
    print(
        f"  Rows: {final_test['rows']:,}"
    )

    print("\n" + "=" * 72)
    print(
        f"OVERALL STATUS: "
        f"{report['overall_status']}"
    )
    print("=" * 72)

    print(
        f"\nReport saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()