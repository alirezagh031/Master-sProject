from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


TIMESTAMP_COL = "timestamp"


@dataclass(frozen=True)
class WalkForwardFold:
    """
    Immutable description of one purged walk-forward fold.
    """

    fold: int

    train_start: pd.Timestamp
    train_end_before_purge: pd.Timestamp
    train_end_after_purge: pd.Timestamp

    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    train_rows_before_purge: int
    purged_rows: int
    train_rows_after_purge: int
    validation_rows: int

    purge_bars: int
    embargo_bars: int


class PurgedWalkForwardSplitter:
    """
    Expanding-window walk-forward splitter with label-overlap purging.

    Expected data:
        A single chronological development dataset
        containing 2018–2024.

    Validation years:
        2021, 2022, 2023, 2024.

    Final test data:
        2025 is NOT accepted by this splitter.

    Purging:
        A training observation is removed when its label endpoint
        reaches or crosses the validation start.

    Embargo:
        0 bars, because the current expanding-window protocol
        never places post-validation observations into the same
        training fold.
    """

    def __init__(
        self,
        validation_years: list[int] | tuple[int, ...] = (
            2021,
            2022,
            2023,
            2024,
        ),
        first_train_year: int = 2018,
        horizon_bars: int = 1,
        embargo_bars: int = 0,
        timestamp_col: str = TIMESTAMP_COL,
    ) -> None:

        if horizon_bars < 1:
            raise ValueError(
                "horizon_bars must be >= 1."
            )

        if embargo_bars != 0:
            raise ValueError(
                "For the current thesis protocol, "
                "embargo_bars must be 0."
            )

        if not validation_years:
            raise ValueError(
                "validation_years cannot be empty."
            )

        self.validation_years = tuple(
            validation_years
        )

        self.first_train_year = (
            first_train_year
        )

        self.horizon_bars = horizon_bars
        self.embargo_bars = embargo_bars
        self.timestamp_col = timestamp_col

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> None:

        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"Missing timestamp column: "
                f"{self.timestamp_col}"
            )

        if df.empty:
            raise ValueError(
                "Input dataframe is empty."
            )

        if not pd.api.types.is_datetime64_any_dtype(
            df[self.timestamp_col]
        ):
            raise TypeError(
                f"{self.timestamp_col} must be "
                "datetime-like."
            )

        if not df[self.timestamp_col].is_monotonic_increasing:
            raise ValueError(
                "Input dataframe must be sorted "
                "chronologically."
            )

        if df[self.timestamp_col].duplicated().any():
            raise ValueError(
                "Duplicate timestamps detected."
            )

    def _label_endpoints(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate label endpoint for every observation
        using the complete development timeline.
        """

        timestamps = (
            df[self.timestamp_col]
            .reset_index(drop=True)
        )

        return timestamps.shift(
            -self.horizon_bars
        )

    def split(
        self,
        df: pd.DataFrame,
    ) -> Iterator[
        tuple[pd.Index, pd.Index, WalkForwardFold]
    ]:
        """
        Yield:
            train_indices,
            validation_indices,
            fold_metadata
        """

        self._validate_dataframe(df)

        working_df = (
            df
            .reset_index(drop=False)
            .copy()
        )

        original_indices = (
            working_df["index"]
        )

        timestamps = (
            working_df[self.timestamp_col]
        )

        years = timestamps.dt.year

        label_endpoints = (
            self._label_endpoints(
                working_df
            )
        )

        for fold_number, validation_year in enumerate(
            self.validation_years,
            start=1,
        ):

            train_years = range(
                self.first_train_year,
                validation_year,
            )

            train_mask = years.isin(
                train_years
            )

            validation_mask = (
                years == validation_year
            )

            train_positions = (
                working_df.index[
                    train_mask
                ]
            )

            validation_positions = (
                working_df.index[
                    validation_mask
                ]
            )

            if len(train_positions) == 0:
                raise ValueError(
                    f"Fold {fold_number}: "
                    "empty training set."
                )

            if len(validation_positions) == 0:
                raise ValueError(
                    f"Fold {fold_number}: "
                    "empty validation set."
                )

            train_end_before_purge = (
                timestamps.iloc[
                    train_positions[-1]
                ]
            )

            validation_start = (
                timestamps.iloc[
                    validation_positions[0]
                ]
            )

            validation_end = (
                timestamps.iloc[
                    validation_positions[-1]
                ]
            )

            # --------------------------------------------------
            # PURGING
            # --------------------------------------------------
            #
            # Remove every training observation whose label
            # endpoint reaches the validation start.
            #
            # Leakage condition:
            #
            # label_end >= validation_start
            #
            purge_mask = (
                label_endpoints
                >= validation_start
            )

            purge_mask &= (
                train_mask
            )

            purged_positions = (
                working_df.index[
                    purge_mask
                ]
            )

            purged_rows = len(
                purged_positions
            )

            if purged_rows > 0:

                purged_train_positions = (
                    train_positions[
                        ~train_positions.isin(
                            purged_positions
                        )
                    ]
                )

            else:

                purged_train_positions = (
                    train_positions
                )

            if len(
                purged_train_positions
            ) == 0:
                raise ValueError(
                    f"Fold {fold_number}: "
                    "purging removed the entire "
                    "training set."
                )

            train_end_after_purge = (
                timestamps.iloc[
                    purged_train_positions[-1]
                ]
            )

            # --------------------------------------------------
            # TEMPORAL SAFETY CHECKS
            # --------------------------------------------------

            if not (
                train_end_after_purge
                < validation_start
            ):
                raise RuntimeError(
                    f"Fold {fold_number}: "
                    "purged training set still "
                    "touches validation."
                )

            if validation_start > validation_end:
                raise RuntimeError(
                    f"Fold {fold_number}: "
                    "invalid validation range."
                )

            # Final indices must refer to the original
            # dataframe supplied by the caller.
            train_indices = (
                original_indices.iloc[
                    purged_train_positions
                ]
                .tolist()
            )

            validation_indices = (
                original_indices.iloc[
                    validation_positions
                ]
                .tolist()
            )

            metadata = WalkForwardFold(
                fold=fold_number,

                train_start=timestamps.iloc[
                    train_positions[0]
                ],

                train_end_before_purge=(
                    train_end_before_purge
                ),

                train_end_after_purge=(
                    train_end_after_purge
                ),

                validation_start=(
                    validation_start
                ),

                validation_end=(
                    validation_end
                ),

                train_rows_before_purge=(
                    len(train_positions)
                ),

                purged_rows=(
                    purged_rows
                ),

                train_rows_after_purge=(
                    len(purged_train_positions)
                ),

                validation_rows=(
                    len(validation_positions)
                ),

                purge_bars=(
                    self.horizon_bars
                ),

                embargo_bars=(
                    self.embargo_bars
                ),
            )

            yield (
                pd.Index(train_indices),
                pd.Index(validation_indices),
                metadata,
            )