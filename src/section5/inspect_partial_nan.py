"""
Read-only diagnostic for the H1/D1 *_partial_* NaN pattern.

Does NOT modify any file, does NOT train anything, does NOT touch the
Stage 5 pipeline or config. Answers questions 1-6 and 8 from a purely
empirical inspection of the two parquet files. Question 7 (Stage 3
alignment code) and the final "likely cause" (question 9) still need the
actual Stage 3 source and/or this script's real output to confirm.

Usage (from the project root, with the .venv active):

    python inspect_partial_nan.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PARTIAL_COLS = {
    "h1": ["h1_partial_open", "h1_partial_high", "h1_partial_low", "h1_partial_close", "h1_partial_tickvol"],
    "d1": ["d1_partial_open", "d1_partial_high", "d1_partial_low", "d1_partial_close", "d1_partial_tickvol"],
}
COMPLETED_COLS = {
    "h1": ["h1_completed_open", "h1_completed_high", "h1_completed_low", "h1_completed_close"],
    "d1": ["d1_completed_open", "d1_completed_high", "d1_completed_low", "d1_completed_close"],
}
PATHS = {
    "h1": Path("data/processed/section5/train_h1.parquet"),
    "d1": Path("data/processed/section5/train_d1.parquet"),
}
# "h1" bars roll over every hour; "d1" bars roll over every calendar day.
# This defines what "the first M15 bar of a new {H1 hour, D1 day}" means,
# without assuming trading starts exactly at :00 or at midnight UTC.
PERIOD_FLOOR = {"h1": "h", "d1": "D"}


def inspect(timeframe: str) -> None:
    print(f"\n{'=' * 78}\n{timeframe.upper()}\n{'=' * 78}")
    path = PATHS[timeframe]
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"loaded {path}: {df.shape[0]} rows, "
          f"{df['timestamp'].min()} -> {df['timestamp'].max()}")

    partial_cols = PARTIAL_COLS[timeframe]
    completed_cols = COMPLETED_COLS[timeframe]

    nan_mask = df[partial_cols].isna().any(axis=1)
    nan_rows = df.loc[nan_mask]
    n_nan = len(nan_rows)
    print(f"\nrows with NaN/inf in {partial_cols}: {n_nan} / {len(df)} ({n_nan / len(df):.4%})")

    if n_nan == 0:
        print("no NaNs found for this timeframe -- nothing further to report.")
        return

    # [1] exact timestamp range
    print(f"\n[1] timestamp range of NaN rows: "
          f"{nan_rows['timestamp'].min()}  ->  {nan_rows['timestamp'].max()}")

    # [2] contiguous vs scattered (run-length of consecutive M15 steps)
    step = df["timestamp"].diff().mode().iloc[0]
    gaps = nan_rows["timestamp"].diff()
    run_lengths, current = [], 1
    for g in gaps.iloc[1:]:
        if g == step:
            current += 1
        else:
            run_lengths.append(current)
            current = 1
    run_lengths.append(current)
    print(f"\n[2] contiguous-block lengths (in M15 steps): "
          f"min={min(run_lengths)}, max={max(run_lengths)}, "
          f"mean={np.mean(run_lengths):.2f}, n_blocks={len(run_lengths)}")
    block_dist = pd.Series(run_lengths).value_counts().sort_index()
    print(f"    block-length distribution (length: count): {block_dist.to_dict()}")

    # [3] unique dates
    unique_nan_dates = nan_rows["timestamp"].dt.date.nunique()
    unique_all_dates = df["timestamp"].dt.date.nunique()
    print(f"\n[3] unique dates containing >=1 NaN row: {unique_nan_dates} / {unique_all_dates} total dates")

    # [4a] weekday distribution (0=Mon .. 6=Sun) -- weekend/holiday check
    print(f"\n[4] weekday distribution of NaN rows (0=Mon..6=Sun): "
          f"{nan_rows['timestamp'].dt.dayofweek.value_counts().sort_index().to_dict()}")
    print(f"    for comparison, ALL rows' weekday distribution:      "
          f"{df['timestamp'].dt.dayofweek.value_counts().sort_index().to_dict()}")

    # [4b] the decisive check: is this "first M15 bar of a new {hour,day}"?
    period = df["timestamp"].dt.floor(PERIOD_FLOOR[timeframe])
    first_in_period_ts = df.groupby(period)["timestamp"].transform("min")
    is_first_of_period = (df["timestamp"] == first_in_period_ts).to_numpy()

    n_first = int(is_first_of_period.sum())
    n_first_and_nan = int((is_first_of_period & nan_mask.to_numpy()).sum())
    n_rest = len(df) - n_first
    n_rest_and_nan = int(n_nan - n_first_and_nan)
    print(f"\n[4b] 'first M15 bar of a new "
          f"{'H1 hour' if timeframe == 'h1' else 'D1 day'}' rows: {n_first}")
    print(f"     of those, NaN in partial_*: {n_first_and_nan} "
          f"({n_first_and_nan / n_first:.4%})")
    print(f"     of all OTHER rows, NaN in partial_*: {n_rest_and_nan} "
          f"({(n_rest_and_nan / n_rest if n_rest else float('nan')):.4%})")

    # [5] does completed_* stay valid on the same rows where partial_* is NaN?
    completed_also_nan = nan_rows[completed_cols].isna().any(axis=1).sum()
    print(f"\n[5] of the {n_nan} NaN-partial rows, {completed_also_nan} "
          f"also have NaN in completed_* ({completed_also_nan / n_nan:.4%}); "
          f"the other {n_nan - completed_also_nan} have a fully valid "
          f"completed_* alongside a NaN partial_*.")


if __name__ == "__main__":
    for tf in ("h1", "d1"):
        inspect(tf)
    print("\nDone -- no files were modified.")
