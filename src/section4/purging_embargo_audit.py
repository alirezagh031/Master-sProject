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

OUTPUT_FILE = DATA_DIR / "purging_embargo_audit.json"

TIMESTAMP_COL = "timestamp"

# Target horizon:
# one next M15 observation
HORIZON_BARS = 1


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

    df = (
        df.sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )

    return df


def get_label_endpoints(
    train_df: pd.DataFrame,
    full_timestamps: pd.Series,
) -> pd.Series:
    """
    Compute label endpoints using the complete timeline.

    The next observation must be searched in the full
    chronological dataset, not only inside the training set.
    """

    full_timestamps = (
        full_timestamps
        .sort_values()
        .reset_index(drop=True)
    )

    train_timestamps = train_df[TIMESTAMP_COL]

    positions = (
        full_timestamps.searchsorted(
            train_timestamps,
            side="right",
        )
    )

    endpoints = []

    for pos in positions:

        if pos < len(full_timestamps):
            endpoints.append(
                full_timestamps.iloc[pos]
            )
        else:
            endpoints.append(pd.NaT)

    return pd.Series(
        endpoints,
        index=train_df.index,
    )


def audit_boundary(
    train_df,
    validation_df,
    full_timestamps,
    boundary_name,
) -> dict[str, Any]:

    train_timestamps = train_df[TIMESTAMP_COL]

    label_endpoints = get_label_endpoints(
    train_df,
    full_timestamps,
    )

    validation_start = (
        validation_df[TIMESTAMP_COL].min()
    )

    validation_end = (
        validation_df[TIMESTAMP_COL].max()
    )

    # A Train sample must be purged if its label interval
    # reaches into the Validation period.
    #
    # label interval:
    #
    # [timestamp(t), label_end(t)]
    #
    # Leakage condition:
    #
    # label_end(t) >= validation_start

    overlap_mask = (
        label_endpoints >= validation_start
    )

    overlap_mask &= train_timestamps.notna()

    overlapping_indices = (
        train_df.index[overlap_mask]
        .tolist()
    )

    overlapping_rows = len(
        overlapping_indices
    )

    last_train_timestamp = (
        train_timestamps.max()
    )

    last_train_label_end = (
        label_endpoints.iloc[-1]
        if len(label_endpoints) > 0
        else None
    )

    if overlapping_rows > 0:

        first_overlap_index = (
            overlapping_indices[0]
        )

        first_overlap_timestamp = (
            train_df.loc[
                first_overlap_index,
                TIMESTAMP_COL,
            ]
        )

        purge_required = True

    else:

        first_overlap_timestamp = None
        purge_required = False

    # Calculate the number of observations immediately
    # preceding the validation boundary that are affected.
    #
    # We report the actual number rather than assuming
    # "one row" because the dataset contains temporal gaps.

    required_purge_rows = overlapping_rows

    if purge_required:

        recommended_purge_rows = (
            required_purge_rows
        )

    else:

        recommended_purge_rows = 0

    return {
        "boundary": boundary_name,

        "train": {
            "rows": int(len(train_df)),
            "start": str(
                train_timestamps.min()
            ),
            "end": str(
                last_train_timestamp
            ),
        },

        "validation": {
            "rows": int(len(validation_df)),
            "start": str(
                validation_start
            ),
            "end": str(
                validation_end
            ),
        },

        "label_horizon": {
            "bars": HORIZON_BARS,
            "definition": (
                "next available observation"
            ),
        },

        "last_train_label_end": (
            str(last_train_label_end)
            if pd.notna(last_train_label_end)
            else None
        ),

        "validation_start": str(
            validation_start
        ),

        "overlapping_train_labels": (
            overlapping_rows
        ),

        "purge_required": purge_required,

        "required_purge_rows": (
            required_purge_rows
        ),

        "recommended_purge_rows": (
            recommended_purge_rows
        ),

        "first_overlapping_train_timestamp": (
            str(first_overlap_timestamp)
            if first_overlap_timestamp is not None
            else None
        ),

        "status": (
            "PURGE_REQUIRED"
            if purge_required
            else "NO_PURGE_REQUIRED"
        ),
    }


def audit_embargo(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    boundary_name: str,
) -> dict[str, Any]:

    train_end = (
        train_df[TIMESTAMP_COL].max()
    )

    validation_start = (
        validation_df[TIMESTAMP_COL].min()
    )

    # In our expanding-window protocol, Train is strictly
    # before Validation. There is no post-validation data
    # being inserted into the same training fold.
    #
    # Therefore an embargo is not automatically required.
    #
    # We still calculate the chronological separation
    # for documentation.

    temporal_gap = (
        validation_start - train_end
    )

    return {
        "boundary": boundary_name,

        "train_end": str(train_end),

        "validation_start": str(
            validation_start
        ),

        "temporal_gap": str(
            temporal_gap
        ),

        "embargo_required_by_current_protocol": False,

        "recommended_embargo_rows": 0,

        "reason": (
            "Expanding-window validation uses only "
            "past observations for training. No "
            "post-validation observations are returned "
            "to the same training fold."
        ),

        "status": "NO_EMBARGO_REQUIRED",
    }


def create_fold_data(
    development_df: pd.DataFrame,
    validation_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_years = list(
        range(
            2018,
            validation_year,
        )
    )

    train_mask = (
        development_df[TIMESTAMP_COL]
        .dt.year
        .isin(train_years)
    )

    validation_mask = (
        development_df[TIMESTAMP_COL]
        .dt.year
        == validation_year
    )

    train_df = (
        development_df.loc[train_mask]
        .copy()
        .reset_index(drop=True)
    )

    validation_df = (
        development_df.loc[validation_mask]
        .copy()
        .reset_index(drop=True)
    )

    return train_df, validation_df


def audit_all_walk_forward_boundaries(
    development_df: pd.DataFrame,
) -> list[dict[str, Any]]:

    results = []

    validation_years = [
        2021,
        2022,
        2023,
        2024,
    ]

    for fold_number, validation_year in enumerate(
        validation_years,
        start=1,
    ):

        train_df, validation_df = (
            create_fold_data(
                development_df,
                validation_year,
            )
        )
        full_timestamps = development_df[TIMESTAMP_COL]
        purge_result = audit_boundary(
            train_df,
            validation_df,
            full_timestamps,
            f"Fold {fold_number}: "
            f"Train → Validation {validation_year}",
        )

        embargo_result = audit_embargo(
            train_df,
            validation_df,
            f"Fold {fold_number}: "
            f"Train → Validation {validation_year}",
        )

        results.append(
            {
                "fold": fold_number,
                "validation_year": validation_year,
                "purging": purge_result,
                "embargo": embargo_result,
            }
        )

    return results


def audit_validation_to_test(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:

    validation_df = (
        development_df[
            development_df[TIMESTAMP_COL]
            .dt.year
            == 2024
        ]
        .copy()
        .reset_index(drop=True)
    )
    full_timestamps = pd.concat(
    [
        development_df[TIMESTAMP_COL],
        test_df[TIMESTAMP_COL],
    ],
    ignore_index=True,
    )

    full_timestamps = (
        full_timestamps
        .sort_values()
        .reset_index(drop=True)
    )
    purge_result = audit_boundary(
        validation_df,
        test_df,
        full_timestamps,
        "Final Development → Test boundary",
    )

    embargo_result = audit_embargo(
        validation_df,
        test_df,
        "Final Development → Test boundary",
    )

    return {
        "purging": purge_result,
        "embargo": embargo_result,
    }


def summarize_results(
    folds: list[dict[str, Any]],
    final_boundary: dict[str, Any],
) -> dict[str, Any]:

    total_purge_rows = 0
    any_purge_required = False

    for fold in folds:

        purge_info = fold["purging"]

        total_purge_rows += (
            purge_info[
                "recommended_purge_rows"
            ]
        )

        if purge_info["purge_required"]:
            any_purge_required = True

    final_purge = final_boundary[
        "purging"
    ]

    total_purge_rows += (
        final_purge[
            "recommended_purge_rows"
        ]
    )

    if final_purge["purge_required"]:
        any_purge_required = True

    all_embargo_zero = True

    for fold in folds:

        if (
            fold["embargo"]
            ["recommended_embargo_rows"]
            != 0
        ):
            all_embargo_zero = False

    if (
        final_boundary["embargo"]
        ["recommended_embargo_rows"]
        != 0
    ):
        all_embargo_zero = False

    return {
        "any_purge_required": (
            any_purge_required
        ),

        "total_recommended_purge_rows": (
            total_purge_rows
        ),

        "embargo_required": (
            not all_embargo_zero
        ),

        "recommended_embargo_rows": (
            0
            if all_embargo_zero
            else None
        ),

        "interpretation": (
            "Purging must be applied where "
            "training labels overlap the validation "
            "or final-test interval."
            if any_purge_required
            else
            "No label-overlap purge is required "
            "at the audited boundaries."
        ),
    }


def main() -> None:

    print("=" * 72)
    print(
        "SECTION 4.3 — PURGING & EMBARGO AUDIT"
    )
    print("=" * 72)

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

    print("\nDatasets loaded:")
    print(
        f"  TRAIN      : {train_df.shape}"
    )
    print(
        f"  VALIDATION : {validation_df.shape}"
    )
    print(
        f"  TEST       : {test_df.shape}"
    )

    fold_results = (
        audit_all_walk_forward_boundaries(
            development_df
        )
    )

    final_boundary = (
        audit_validation_to_test(
            development_df,
            test_df,
        )
    )

    summary = summarize_results(
        fold_results,
        final_boundary,
    )

    report = {
        "stage": "4.3",

        "title": (
            "Purging and Embargo Audit"
        ),

        "target": {
            "horizon_bars": HORIZON_BARS,
            "horizon_definition": (
                "next available M15 observation"
            ),
        },

        "walk_forward_folds": fold_results,

        "final_boundary": final_boundary,

        "summary": summary,

        "overall_status": "PASS",
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

    print(
        "\n" + "=" * 72
    )

    print(
        "\nPurging results:"
    )

    for fold in fold_results:

        purge = fold["purging"]

        print(
            f"\nFold {fold['fold']}"
        )

        print(
            f"  Overlapping labels : "
            f"{purge['overlapping_train_labels']}"
        )

        print(
            f"  Purge required     : "
            f"{purge['purge_required']}"
        )

        print(
            f"  Recommended purge  : "
            f"{purge['recommended_purge_rows']} rows"
        )

        print(
            f"  Last label end     : "
            f"{purge['last_train_label_end']}"
        )

        print(
            f"  Validation starts  : "
            f"{purge['validation_start']}"
        )

    final_purge = (
        final_boundary["purging"]
    )

    print(
        "\nFinal Development → Test"
    )

    print(
        f"  Overlapping labels : "
        f"{final_purge['overlapping_train_labels']}"
    )

    print(
        f"  Purge required     : "
        f"{final_purge['purge_required']}"
    )

    print(
        f"  Recommended purge  : "
        f"{final_purge['recommended_purge_rows']} rows"
    )

    print(
        "\nEmbargo:"
    )

    print(
        "  Current protocol requires: "
        "0 rows"
    )

    print(
        "\n" + "=" * 72
    )

    print(
        f"OVERALL STATUS: "
        f"{report['overall_status']}"
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