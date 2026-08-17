from pathlib import Path

import pandas as pd

from purged_walk_forward import (
    PurgedWalkForwardSplitter,
)


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


def main() -> None:

    print("=" * 72)
    print(
        "SECTION 4.4 — PURGED WALK-FORWARD SPLITTER TEST"
    )
    print("=" * 72)

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    development_df["timestamp"] = pd.to_datetime(
        development_df["timestamp"]
    )

    development_df = (
        development_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    splitter = PurgedWalkForwardSplitter(
        validation_years=[
            2021,
            2022,
            2023,
            2024,
        ],
        first_train_year=2018,
        horizon_bars=1,
        embargo_bars=0,
    )

    total_folds = 0

    for (
        train_indices,
        validation_indices,
        metadata,
    ) in splitter.split(
        development_df
    ):

        total_folds += 1

        train_fold = development_df.loc[
            train_indices
        ]

        validation_fold = development_df.loc[
            validation_indices
        ]

        print(
            f"\nFold {metadata.fold}"
        )

        print(
            f"  Train before purge : "
            f"{metadata.train_rows_before_purge:,}"
        )

        print(
            f"  Purged rows        : "
            f"{metadata.purged_rows:,}"
        )

        print(
            f"  Train after purge  : "
            f"{metadata.train_rows_after_purge:,}"
        )

        print(
            f"  Validation rows    : "
            f"{metadata.validation_rows:,}"
        )

        print(
            f"  Train end before   : "
            f"{metadata.train_end_before_purge}"
        )

        print(
            f"  Train end after    : "
            f"{metadata.train_end_after_purge}"
        )

        print(
            f"  Validation start   : "
            f"{metadata.validation_start}"
        )

        print(
            f"  Purge bars         : "
            f"{metadata.purge_bars}"
        )

        print(
            f"  Embargo bars       : "
            f"{metadata.embargo_bars}"
        )

        # --------------------------------------------------
        # ASSERTIONS
        # --------------------------------------------------

        assert (
            metadata.purged_rows == 1
        ), (
            f"Expected exactly 1 purged row "
            f"in Fold {metadata.fold}, "
            f"got {metadata.purged_rows}"
        )

        assert (
            metadata.train_rows_after_purge
            == metadata.train_rows_before_purge - 1
        )

        assert (
            metadata.train_end_after_purge
            < metadata.validation_start
        )

        assert (
            train_fold["timestamp"].max()
            < validation_fold["timestamp"].min()
        )

        assert not (
            set(train_indices)
            & set(validation_indices)
        )

    assert total_folds == 4

    print(
        "\n" + "=" * 72
    )

    print(
        "ALL PURGED WALK-FORWARD TESTS PASSED"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()