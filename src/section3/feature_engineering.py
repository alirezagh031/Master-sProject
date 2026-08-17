from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")

INPUT_FILE = (
    PROCESSED_DIR /
    "xauusd_m15_mtf_aligned.parquet"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "xauusd_mtf_price_features.parquet"
)


EPS = 1e-12


# ============================================================
# Utility
# ============================================================

def safe_divide(a, b):
    """
    Numerically safe division.
    """
    return a / (b.abs() + EPS)


# ============================================================
# M15 FEATURES
# ============================================================

def build_m15_price_features(df):
    """
    Build causal price/return features for M15.

    All features at timestamp t use only information
    contained in the current M15 candle or earlier candles.
    """

    result = df.copy()

    close = result["open_x"].astype(float)
    high = result["high_x"].astype(float)
    low = result["low_x"].astype(float)
    open_ = result["open_x"].astype(float)

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    result["m15_return_1"] = (
        result["close_x"] /
        result["close_x"].shift(1)
        - 1.0
    )

    result["m15_return_4"] = (
        result["close_x"] /
        result["close_x"].shift(4)
        - 1.0
    )

    result["m15_return_16"] = (
        result["close_x"] /
        result["close_x"].shift(16)
        - 1.0
    )

    result["m15_log_return_1"] = np.log(
        result["close_x"] /
        result["close_x"].shift(1)
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    result["m15_range"] = high - low

    result["m15_body"] = (
        result["close_x"] - open_
    )

    result["m15_body_abs"] = (
        result["m15_body"].abs()
    )

    result["m15_upper_wick"] = (
        high -
        np.maximum(open_, result["close_x"])
    )

    result["m15_lower_wick"] = (
        np.minimum(open_, result["close_x"]) -
        low
    )

    result["m15_body_range_ratio"] = safe_divide(
        result["m15_body_abs"],
        result["m15_range"],
    )

    result["m15_upper_wick_ratio"] = safe_divide(
        result["m15_upper_wick"],
        result["m15_range"],
    )

    result["m15_lower_wick_ratio"] = safe_divide(
        result["m15_lower_wick"],
        result["m15_range"],
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    result["m15_momentum_4"] = (
        result["close_x"] -
        result["close_x"].shift(4)
    )

    result["m15_momentum_16"] = (
        result["close_x"] -
        result["close_x"].shift(16)
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    result["m15_range_mean_4"] = (
        result["m15_range"]
        .rolling(4, min_periods=4)
        .mean()
    )

    result["m15_range_mean_16"] = (
        result["m15_range"]
        .rolling(16, min_periods=16)
        .mean()
    )

    result["m15_return_std_16"] = (
        result["m15_return_1"]
        .rolling(16, min_periods=16)
        .std()
    )

    return result


# ============================================================
# H1 FEATURES
# ============================================================

def build_h1_price_features(df):
    """
    Build causal features from the H1 state available
    at each M15 decision time.

    Important:
    H1 completed values are historical and safe.

    H1 partial values represent only information available
    before the current M15 timestamp.
    """

    result = df.copy()

    close = result["h1_partial_close"]
    open_ = result["h1_partial_open"]
    high = result["h1_partial_high"]
    low = result["h1_partial_low"]

    # --------------------------------------------------------
    # Current partial H1 candle structure
    # --------------------------------------------------------

    result["h1_partial_range"] = high - low

    result["h1_partial_body"] = (
        close - open_
    )

    result["h1_partial_body_abs"] = (
        result["h1_partial_body"].abs()
    )

    result["h1_partial_body_range_ratio"] = safe_divide(
        result["h1_partial_body_abs"],
        result["h1_partial_range"],
    )

    result["h1_partial_upper_wick"] = (
        high -
        np.maximum(open_, close)
    )

    result["h1_partial_lower_wick"] = (
        np.minimum(open_, close) -
        low
    )

    # --------------------------------------------------------
    # Completed H1 returns
    # --------------------------------------------------------

    h1_close = result["h1_completed_close"]

    result["h1_return_1"] = (
        h1_close /
        h1_close.shift(1)
        - 1.0
    )

    # The above shift operates on M15 rows and is therefore
    # not a valid H1 lag. We therefore use the completed H1
    # timestamp to construct actual H1 changes below.
    # --------------------------------------------------------

    result["h1_completed_return"] = (
        result["h1_completed_close"] /
        result["h1_completed_open"]
        - 1.0
    )

    # --------------------------------------------------------
    # H1 price relative to completed candle
    # --------------------------------------------------------

    result["m15_to_h1_close_distance"] = safe_divide(
        result["close_x"] -
        result["h1_completed_close"],
        result["h1_completed_close"],
    )

    return result


# ============================================================
# D1 FEATURES
# ============================================================

def build_d1_price_features(df):
    """
    Build causal daily features.

    Only completed D1 information and the causal partial
    daily state are used.
    """

    result = df.copy()

    close = result["d1_partial_close"]
    open_ = result["d1_partial_open"]
    high = result["d1_partial_high"]
    low = result["d1_partial_low"]

    # --------------------------------------------------------
    # Current partial D1 candle
    # --------------------------------------------------------

    result["d1_partial_range"] = high - low

    result["d1_partial_body"] = (
        close - open_
    )

    result["d1_partial_body_abs"] = (
        result["d1_partial_body"].abs()
    )

    result["d1_partial_body_range_ratio"] = safe_divide(
        result["d1_partial_body_abs"],
        result["d1_partial_range"],
    )

    result["d1_partial_upper_wick"] = (
        high -
        np.maximum(open_, close)
    )

    result["d1_partial_lower_wick"] = (
        np.minimum(open_, close) -
        low
    )

    # --------------------------------------------------------
    # Completed daily candle
    # --------------------------------------------------------

    d1_close = result["d1_completed_close"]

    result["d1_completed_return"] = (
        result["d1_completed_close"] /
        result["d1_completed_open"]
        - 1.0
    )

    # --------------------------------------------------------
    # M15 relative to daily context
    # --------------------------------------------------------

    result["m15_to_d1_close_distance"] = safe_divide(
        result["close_x"] -
        result["d1_completed_close"],
        result["d1_completed_close"],
    )

    return result


# ============================================================
# CROSS TIMEFRAME FEATURES
# ============================================================

def build_cross_timeframe_features(df):
    """
    Build simple causal relationships between M15, H1 and D1.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Directional alignment
    # --------------------------------------------------------

    m15_direction = np.sign(
        result["m15_return_1"]
    )

    h1_direction = np.sign(
        result["h1_completed_return"]
    )

    d1_direction = np.sign(
        result["d1_completed_return"]
    )

    result["mtf_m15_h1_alignment"] = (
        m15_direction * h1_direction
    )

    result["mtf_m15_d1_alignment"] = (
        m15_direction * d1_direction
    )

    result["mtf_h1_d1_alignment"] = (
        h1_direction * d1_direction
    )

    # --------------------------------------------------------
    # Multi-timeframe agreement
    # --------------------------------------------------------

    result["mtf_all_direction_agreement"] = (
        (
            (m15_direction == h1_direction)
            &
            (h1_direction == d1_direction)
            &
            (m15_direction != 0)
        )
        .astype("int8")
    )

    # --------------------------------------------------------
    # Price location inside H1/D1 context
    # --------------------------------------------------------

    result["m15_position_in_h1_range"] = safe_divide(
        result["close_x"] -
        result["h1_partial_low"],
        result["h1_partial_range"],
    )

    result["m15_position_in_d1_range"] = safe_divide(
        result["close_x"] -
        result["d1_partial_low"],
        result["d1_partial_range"],
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_features(df):
    print("\n" + "=" * 80)
    print("FEATURE VALIDATION")
    print("=" * 80)

    feature_columns = [
        c for c in df.columns
        if c not in {
            "timestamp",
        }
    ]

    numeric = df[feature_columns].select_dtypes(
        include=[np.number]
    )

    nan_count = int(numeric.isna().sum().sum())

    inf_count = int(
        np.isinf(numeric.to_numpy()).sum()
    )

    print("Total columns:", len(df.columns))
    print("Feature columns:", len(numeric.columns))
    print("NaN values:", nan_count)
    print("Inf values:", inf_count)

    constant_features = []

    for column in numeric.columns:
        if numeric[column].nunique(dropna=True) <= 1:
            constant_features.append(column)

    print(
        "Constant features:",
        len(constant_features)
    )

    if constant_features:
        for column in constant_features:
            print(" -", column)

    if inf_count > 0:
        raise RuntimeError(
            "Feature validation failed: Inf values detected."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CAUSAL MULTI-TIMEFRAME FEATURE ENGINEERING")
    print("=" * 80)

    print("\nLoading:")
    print(INPUT_FILE)

    df = pd.read_parquet(INPUT_FILE)

    df.sort_values(
        "timestamp",
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    print("Input shape:", df.shape)

    # --------------------------------------------------------
    # M15
    # --------------------------------------------------------

    print("\n[1/4] M15 price features...")

    df = build_m15_price_features(df)

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    print("[2/4] H1 causal price features...")

    df = build_h1_price_features(df)

    # --------------------------------------------------------
    # D1
    # --------------------------------------------------------

    print("[3/4] D1 causal price features...")

    df = build_d1_price_features(df)

    # --------------------------------------------------------
    # Cross timeframe
    # --------------------------------------------------------

    print("[4/4] Cross-timeframe features...")

    df = build_cross_timeframe_features(df)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    df.sort_values(
        "timestamp",
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    validate_features(df)

    print("\n" + "=" * 80)
    print("FINAL FEATURE DATASET")
    print("=" * 80)

    print("Shape:", df.shape)

    feature_columns = [
        c for c in df.columns
        if c not in {"timestamp"}
    ]

    print(
        "Number of feature columns:",
        len(feature_columns)
    )

    print("\nNew features:")

    original_columns = pd.read_parquet(
        INPUT_FILE,
        columns=None,
    ).columns

    new_features = [
        c for c in df.columns
        if c not in original_columns
    ]

    for column in new_features:
        print(" -", column)

    print("\nSaving:")

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(OUTPUT_FILE)

    print("\nDONE.")


if __name__ == "__main__":
    main()