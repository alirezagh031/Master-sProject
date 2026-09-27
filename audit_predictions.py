"""
Lightweight, read-only audit of an already-produced Stage 5 prediction file.

Does NOT modify anything, does NOT retrain anything. Reuses the exact
verification logic from train_base_learners.py (already proven during the
real Stage 5 run -- that's where "[verify/validation] OK" came from) and
adds the couple of checks that function does not already cover: printing
the exact column names for visual confirmation, and STRICT timestamp
uniqueness (is_monotonic_increasing alone allows adjacent duplicates).

Usage (from the project root):

    python audit_predictions.py
    python audit_predictions.py --file data/processed/section5/predictions/test_base_predictions.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.section5 import config as cfg  # noqa: E402
from src.section5.train_base_learners import (  # noqa: E402
    expected_prediction_columns,
    verify_predictions_output,
)


def audit(path: Path, expected_rows: Optional[int] = None) -> None:
    print(f"Auditing {path} ...")
    df = pd.read_parquet(path)
    print(f"loaded {df.shape[0]} rows x {df.shape[1]} columns")

    expected_cols = expected_prediction_columns()
    print(f"\nexact column names (in order):\n  {list(df.columns)}")
    print(f"\nexpected column names (in order):\n  {expected_cols}")
    print(f"\ncolumns match expected exactly (names AND order): {list(df.columns) == expected_cols}")

    n_prob_cols = len([c for c in expected_cols if c not in (cfg.TIMESTAMP_COLUMN, cfg.TARGET_COLUMN)])
    print(f"probability columns present: {n_prob_cols} (expected 18)")

    # --- reuse the exact, already-proven Stage 5 verification suite ---
    # (column presence/order, row count, target encoding, probability sums
    #  ~= 1 and non-negative for all 6 models, timestamps sorted)
    verify_predictions_output(df, expected_rows, path.stem)

    # --- supplementary check: STRICT timestamp uniqueness ---
    # is_monotonic_increasing (used above) is non-decreasing in pandas, so
    # it would NOT by itself catch two adjacent rows sharing a timestamp.
    n_dupes = int(df[cfg.TIMESTAMP_COLUMN].duplicated().sum())
    print(f"\nduplicate timestamps: {n_dupes} (expected 0)")
    assert n_dupes == 0, (
        f"{n_dupes} duplicate timestamp(s) found -- rows are not uniquely "
        f"keyed by timestamp."
    )

    print(f"\nAUDIT PASSED: {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=cfg.VALIDATION_PREDICTIONS_PATH)
    parser.add_argument("--expected-rows", type=int, default=None)
    args = parser.parse_args()
    audit(args.file, args.expected_rows)
