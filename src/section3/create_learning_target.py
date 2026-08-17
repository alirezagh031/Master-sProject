from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = Path("data/processed")

INPUT_FILE = (
    PROCESSED_DIR / "xauusd_mtf_price_features.parquet"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "xauusd_mtf_target.parquet"
)

# ------------------------------------------------------------
# Target configuration
# ------------------------------------------------------------

# Prediction horizon in M15 candles.
#
# horizon = 1 means:
# predict the NEXT M15 candle return.
#
# Later we can experiment with 4, 16, etc.,
# without changing the feature dataset.
HORIZON = 1

# Classification threshold.
#
# 0.025% = 0.00025
#
# Example:
# future_return > +0.00025 -> UP
# future_return < -0.00025 -> DOWN
# otherwise -> NEUTRAL
THRESHOLD = 0.00025


# ============================================================
# TARGET LABELS
# ============================================================

DOWN = -1
NEUTRAL = 0
UP = 1


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 80)
    print("LOADING FEATURE DATASET")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print("Input file:")
    print(INPUT_FILE)

    print("\nInput shape:", df.shape)

    required_columns = {
        "timestamp",
        "close_x",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

def validate_input(df):
    print("\n" + "=" * 80)
    print("INPUT VALIDATION")
    print("=" * 80)

    if df["timestamp"].isna().any():
        raise ValueError(
            "Null timestamps detected."
        )

    if df["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Timestamp is not monotonic increasing."
        )

    if df["close_x"].isna().any():
        raise ValueError(
            "NaN values detected in close_x."
        )

    if not np.isfinite(
        df["close_x"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values detected in close_x."
        )

    print("PASS: Timestamp validation.")
    print("PASS: Close price validation.")


# ============================================================
# CREATE FUTURE RETURN
# ============================================================

def create_future_return(df):
    """
    Create the future return used as the prediction target.

    At time t:

        future_close = Close[t + HORIZON]

        future_return =
            (future_close - current_close)
            / current_close

    IMPORTANT:
    The future value is used ONLY as TARGET.

    It must never be used to construct any feature.
    """

    result = df.copy()

    current_close = result["close_x"]

    future_close = (
        result["close_x"].shift(-HORIZON)
    )

    result["future_close"] = future_close

    result["future_return"] = (
        future_close - current_close
    ) / current_close

    return result


# ============================================================
# CREATE CLASSIFICATION TARGET
# ============================================================

def create_target_labels(df):
    """
    Convert future return into a 3-class classification target.

        future_return > +threshold -> +1 UP

        future_return < -threshold -> -1 DOWN

        otherwise                  -> 0 NEUTRAL
    """

    result = df.copy()

    result["target"] = np.select(
        [
            result["future_return"] > THRESHOLD,
            result["future_return"] < -THRESHOLD,
        ],
        [
            UP,
            DOWN,
        ],
        default=NEUTRAL,
    ).astype("int8")

    result["target_label"] = (
        result["target"]
        .map(
            {
                DOWN: "DOWN",
                NEUTRAL: "NEUTRAL",
                UP: "UP",
            }
        )
        .astype("string")
    )

    return result


# ============================================================
# REMOVE UNUSABLE TERMINAL ROWS
# ============================================================

def remove_terminal_rows(df):
    """
    The final HORIZON rows do not have a future observation.

    They cannot be used for supervised learning.

    We remove ONLY rows where the target itself is unavailable.
    """

    before = len(df)

    result = df.dropna(
        subset=[
            "future_close",
            "future_return",
        ]
    ).copy()

    result.reset_index(drop=True, inplace=True)

    removed = before - len(result)

    print("\n" + "=" * 80)
    print("TARGET AVAILABILITY")
    print("=" * 80)

    print("Rows before:", before)
    print("Rows after :", len(result))
    print("Removed    :", removed)

    expected_removed = HORIZON

    if removed != expected_removed:
        raise RuntimeError(
            "Unexpected number of rows removed."
        )

    print(
        f"PASS: Exactly {HORIZON} terminal "
        "rows removed."
    )

    return result


# ============================================================
# TARGET VALIDATION
# ============================================================

def validate_target(df):
    print("\n" + "=" * 80)
    print("TARGET VALIDATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Valid labels
    # --------------------------------------------------------

    valid_labels = {-1, 0, 1}

    actual_labels = set(
        df["target"].dropna().astype(int).unique()
    )

    invalid_labels = actual_labels - valid_labels

    print("Target labels:", sorted(actual_labels))

    if invalid_labels:
        raise RuntimeError(
            f"Invalid target labels: {invalid_labels}"
        )

    print("PASS: Target labels are valid.")

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    future_return_nan = int(
        df["future_return"].isna().sum()
    )

    target_nan = int(
        df["target"].isna().sum()
    )

    print(
        "future_return NaN:",
        future_return_nan,
    )

    print(
        "target NaN:",
        target_nan,
    )

    if future_return_nan != 0:
        raise RuntimeError(
            "NaN future_return detected."
        )

    if target_nan != 0:
        raise RuntimeError(
            "NaN target detected."
        )

    print("PASS: Target has no NaN values.")

    # --------------------------------------------------------
    # Inf
    # --------------------------------------------------------

    future_return_inf = int(
        np.isinf(
            df["future_return"].to_numpy(
                dtype=float
            )
        ).sum()
    )

    print(
        "future_return Inf:",
        future_return_inf,
    )

    if future_return_inf != 0:
        raise RuntimeError(
            "Infinite future_return detected."
        )

    print("PASS: No infinite target values.")

    # --------------------------------------------------------
    # Mathematical consistency
    # --------------------------------------------------------

    expected_return = (
        df["future_close"] - df["close_x"]
    ) / df["close_x"]

    max_error = float(
        np.max(
            np.abs(
                expected_return
                - df["future_return"]
            )
        )
    )

    print(
        "Maximum return calculation error:",
        max_error,
    )

    if max_error > 1e-12:
        raise RuntimeError(
            "future_return calculation inconsistency."
        )

    print(
        "PASS: future_return calculation is correct."
    )

    # --------------------------------------------------------
    # Label consistency
    # --------------------------------------------------------

    expected_target = np.select(
        [
            df["future_return"] > THRESHOLD,
            df["future_return"] < -THRESHOLD,
        ],
        [
            UP,
            DOWN,
        ],
        default=NEUTRAL,
    )

    label_mismatch = int(
        (
            df["target"].astype(int).to_numpy()
            != expected_target
        ).sum()
    )

    print(
        "Target label mismatches:",
        label_mismatch,
    )

    if label_mismatch != 0:
        raise RuntimeError(
            "Target label calculation mismatch."
        )

    print(
        "PASS: Target labels are mathematically consistent."
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_target_distribution(df):
    print("\n" + "=" * 80)
    print("TARGET DISTRIBUTION")
    print("=" * 80)

    counts = (
        df["target"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    labels = {
        DOWN: "DOWN",
        NEUTRAL: "NEUTRAL",
        UP: "UP",
    }

    for value in [DOWN, NEUTRAL, UP]:

        count = int(counts.get(value, 0))

        percentage = (
            count / total * 100
        )

        print(
            f"{value:>2} "
            f"{labels[value]:<8} "
            f"{count:>8} "
            f"({percentage:>7.3f}%)"
        )

    print("\nTotal labeled rows:", total)


# ============================================================
# TEMPORAL TARGET SANITY CHECK
# ============================================================

def validate_target_temporal_order(df):
    """
    Verify that future_close actually occurs after
    the current timestamp.

    This is an important causal sanity check.
    """

    print("\n" + "=" * 80)
    print("TEMPORAL TARGET VALIDATION")
    print("=" * 80)

    # The source dataset has M15 timestamps.
    # We verify that future_close belongs to a later row.
    future_timestamp = (
        df["timestamp"].shift(-HORIZON)
    )

    # Because terminal rows were removed,
    # all rows should have a future timestamp.
    invalid = (
        future_timestamp <= df["timestamp"]
    )

    invalid_count = int(invalid.sum())

    print(
        "Future timestamp violations:",
        invalid_count,
    )

    if invalid_count != 0:
        raise RuntimeError(
            "Target temporal ordering violation detected."
        )

    print(
        "PASS: Target always refers to a future observation."
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df):
    print("\n" + "=" * 80)
    print("SAVING TARGET DATASET")
    print("=" * 80)

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("Saved:")
    print(OUTPUT_FILE)

    print("\nFinal shape:", df.shape)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CAUSAL MULTI-TIMEFRAME TARGET CONSTRUCTION")
    print("=" * 80)

    print("\nConfiguration:")
    print("Horizon:", HORIZON, "M15 candle(s)")
    print(
        "Threshold:",
        THRESHOLD,
        f"({THRESHOLD * 100:.4f}%)",
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_input(df)

    # --------------------------------------------------------
    # Create future return
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 1/4 — FUTURE RETURN")
    print("=" * 80)

    df = create_future_return(df)

    print(
        "Future return created with horizon:",
        HORIZON,
    )

    # --------------------------------------------------------
    # Create labels
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 2/4 — TARGET LABELS")
    print("=" * 80)

    df = create_target_labels(df)

    print(
        "Target created:",
        "DOWN / NEUTRAL / UP",
    )

    # --------------------------------------------------------
    # Remove rows without target
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 3/4 — TARGET AVAILABILITY")
    print("=" * 80)

    df = remove_terminal_rows(df)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 4/4 — VALIDATION")
    print("=" * 80)

    validate_target(df)

    validate_target_temporal_order(df)

    print_target_distribution(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(df)

    print("\n" + "=" * 80)
    print("TARGET CONSTRUCTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()