from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")

M15_FILE = PROCESSED_DIR / "xauusd_m15_standardized.parquet"
H1_FILE = PROCESSED_DIR / "xauusd_h1_standardized.parquet"
D1_FILE = PROCESSED_DIR / "xauusd_d1_standardized.parquet"

OUTPUT_FILE = PROCESSED_DIR / "xauusd_m15_mtf_aligned.parquet"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    m15 = pd.read_parquet(M15_FILE)
    h1 = pd.read_parquet(H1_FILE)
    d1 = pd.read_parquet(D1_FILE)

    for df in (m15, h1, d1):
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return m15, h1, d1


# ============================================================
# COMPLETED H1 CONTEXT
# ============================================================

def build_completed_h1_context(m15, h1):
    """
    Attach the latest COMPLETED H1 candle.

    Rule:

        H1 timestamp < M15 timestamp

    Therefore the currently forming H1 candle is never used
    as completed context.
    """

    h1_context = h1[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "tickvol",
            "vol",
            "spread",
        ]
    ].copy()

    h1_context.rename(
        columns={
            "timestamp": "h1_completed_timestamp",
            "open": "h1_completed_open",
            "high": "h1_completed_high",
            "low": "h1_completed_low",
            "close": "h1_completed_close",
            "tickvol": "h1_completed_tickvol",
            "vol": "h1_completed_vol",
            "spread": "h1_completed_spread",
        },
        inplace=True,
    )

    result = pd.merge_asof(
        m15.sort_values("timestamp"),
        h1_context.sort_values("h1_completed_timestamp"),
        left_on="timestamp",
        right_on="h1_completed_timestamp",
        direction="backward",
        allow_exact_matches=False,
    )

    return result


# ============================================================
# CAUSAL PARTIAL H1 STATE
# ============================================================

def build_partial_h1_state(m15):
    """
    Build the causal partial H1 state available at each M15 timestamp.

    At time t, ONLY M15 observations with timestamp < t are used.

    Therefore:
        current M15 candle -> never included
        future M15 candles -> never included
    """

    temp = m15[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "tickvol",
        ]
    ].copy()

    temp["h1_period"] = temp["timestamp"].dt.floor("h")

    group = temp.groupby("h1_period", sort=False)

    # Cumulative values INCLUDING current candle
    temp["partial_open_including_current"] = group["open"].transform("first")
    temp["partial_high_including_current"] = group["high"].cummax()
    temp["partial_low_including_current"] = group["low"].cummin()
    temp["partial_close_including_current"] = temp["close"]
    temp["partial_tickvol_including_current"] = group["tickvol"].cumsum()

    # Shift everything by one observation.
    #
    # Therefore row t can only see observations before t.
    temp["h1_partial_open"] = (
        temp["partial_open_including_current"].shift(1)
    )

    temp["h1_partial_high"] = (
        temp["partial_high_including_current"].shift(1)
    )

    temp["h1_partial_low"] = (
        temp["partial_low_including_current"].shift(1)
    )

    temp["h1_partial_close"] = (
        temp["partial_close_including_current"].shift(1)
    )

    temp["h1_partial_tickvol"] = (
        temp["partial_tickvol_including_current"].shift(1)
    )

    temp["h1_partial_last_m15_timestamp"] = (
        temp["timestamp"].shift(1)
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # A shifted value from the PREVIOUS H1 period must never
    # become the partial state of the NEW H1 period.
    #
    # Clear values whenever the previous candle belongs to
    # another H1 period.
    # ---------------------------------------------------------

    previous_same_h1 = (
        temp["h1_partial_last_m15_timestamp"].dt.floor("h")
        == temp["h1_period"]
    )

    partial_columns = [
        "h1_partial_open",
        "h1_partial_high",
        "h1_partial_low",
        "h1_partial_close",
        "h1_partial_tickvol",
        "h1_partial_last_m15_timestamp",
    ]

    temp.loc[~previous_same_h1, partial_columns] = pd.NA

    temp["h1_partial_period"] = temp["h1_period"]
    result = temp[
        [
            "timestamp",
            "h1_partial_period",
            "h1_partial_open",
            "h1_partial_high",
            "h1_partial_low",
            "h1_partial_close",
            "h1_partial_tickvol",
            "h1_partial_last_m15_timestamp",
        ]
    ].copy()

    return result


# ============================================================
# COMPLETED D1 CONTEXT
# ============================================================

def build_completed_d1_context(m15, d1):
    """
    Attach the latest COMPLETED Daily candle.

    Rule:

        D1 date < current M15 calendar date

    Therefore the current day's unfinished Daily candle
    cannot appear as completed context.
    """

    m15_temp = m15.copy()
    d1_temp = d1.copy()

    m15_temp["date_key"] = m15_temp["timestamp"].dt.normalize()
    d1_temp["date_key"] = d1_temp["timestamp"].dt.normalize()

    d1_context = d1_temp[
        [
            "date_key",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "tickvol",
            "vol",
            "spread",
        ]
    ].copy()

    d1_context.rename(
        columns={
            "date_key": "d1_completed_date",
            "timestamp": "d1_completed_timestamp",
            "open": "d1_completed_open",
            "high": "d1_completed_high",
            "low": "d1_completed_low",
            "close": "d1_completed_close",
            "tickvol": "d1_completed_tickvol",
            "vol": "d1_completed_vol",
            "spread": "d1_completed_spread",
        },
        inplace=True,
    )

    result = pd.merge_asof(
        m15_temp.sort_values("date_key"),
        d1_context.sort_values("d1_completed_date"),
        left_on="date_key",
        right_on="d1_completed_date",
        direction="backward",
        allow_exact_matches=False,
    )

    return result


# ============================================================
# CAUSAL PARTIAL D1 STATE
# ============================================================

def build_partial_d1_state(m15):
    """
    Build the causal partial D1 state available at each M15 timestamp.

    At time t, ONLY M15 observations with timestamp < t are used.

    Therefore no future information from the current day can
    enter the partial daily state.
    """

    temp = m15[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "tickvol",
        ]
    ].copy()

    temp["d1_period"] = temp["timestamp"].dt.normalize()

    group = temp.groupby("d1_period", sort=False)

    # Cumulative values INCLUDING current candle
    temp["partial_open_including_current"] = (
        group["open"].transform("first")
    )

    temp["partial_high_including_current"] = (
        group["high"].cummax()
    )

    temp["partial_low_including_current"] = (
        group["low"].cummin()
    )

    temp["partial_close_including_current"] = temp["close"]

    temp["partial_tickvol_including_current"] = (
        group["tickvol"].cumsum()
    )

    # Shift by one M15 candle.
    temp["d1_partial_open"] = (
        temp["partial_open_including_current"].shift(1)
    )

    temp["d1_partial_high"] = (
        temp["partial_high_including_current"].shift(1)
    )

    temp["d1_partial_low"] = (
        temp["partial_low_including_current"].shift(1)
    )

    temp["d1_partial_close"] = (
        temp["partial_close_including_current"].shift(1)
    )

    temp["d1_partial_tickvol"] = (
        temp["partial_tickvol_including_current"].shift(1)
    )

    temp["d1_partial_last_m15_timestamp"] = (
        temp["timestamp"].shift(1)
    )

    # ---------------------------------------------------------
    # Prevent previous day's state from entering the new day.
    # ---------------------------------------------------------

    previous_same_day = (
        temp["d1_partial_last_m15_timestamp"].dt.normalize()
        == temp["d1_period"]
    )

    partial_columns = [
        "d1_partial_open",
        "d1_partial_high",
        "d1_partial_low",
        "d1_partial_close",
        "d1_partial_tickvol",
        "d1_partial_last_m15_timestamp",
    ]

    temp.loc[~previous_same_day, partial_columns] = pd.NA

    temp["d1_partial_period"] = temp["d1_period"]

    result = temp[
        [
            "timestamp",
            "d1_partial_period",
            "d1_partial_open",
            "d1_partial_high",
            "d1_partial_low",
            "d1_partial_close",
            "d1_partial_tickvol",
            "d1_partial_last_m15_timestamp",
        ]
    ].copy()

    return result

# ============================================================
# LEAKAGE VALIDATION
# ============================================================

def validate_no_future_information(result):
    """
    Explicit leakage checks.

    The last M15 timestamp used to construct a partial state
    must ALWAYS be strictly earlier than the current timestamp.
    """

    print("\n" + "=" * 80)
    print("CAUSALITY / LEAKAGE VALIDATION")
    print("=" * 80)

    checks = {}

    # --------------------------------------------------------
    # H1 partial state
    # --------------------------------------------------------

    h1_mask = result["h1_partial_last_m15_timestamp"].notna()

    h1_future = (
        result.loc[h1_mask, "h1_partial_last_m15_timestamp"]
        >= result.loc[h1_mask, "timestamp"]
    ).sum()

    checks["H1 partial future violations"] = int(h1_future)

    # --------------------------------------------------------
    # D1 partial state
    # --------------------------------------------------------

    d1_mask = result["d1_partial_last_m15_timestamp"].notna()

    d1_future = (
        result.loc[d1_mask, "d1_partial_last_m15_timestamp"]
        >= result.loc[d1_mask, "timestamp"]
    ).sum()

    checks["D1 partial future violations"] = int(d1_future)

    # --------------------------------------------------------
    # H1 completed context
    # --------------------------------------------------------

    h1_completed_mask = result["h1_completed_timestamp"].notna()

    h1_completed_future = (
        result.loc[h1_completed_mask, "h1_completed_timestamp"]
        >= result.loc[h1_completed_mask, "timestamp"]
    ).sum()

    checks["H1 completed future violations"] = int(
        h1_completed_future
    )

    # --------------------------------------------------------
    # D1 completed context
    # --------------------------------------------------------

    d1_completed_mask = result["d1_completed_timestamp"].notna()

    d1_completed_future = (
        result.loc[d1_completed_mask, "d1_completed_timestamp"]
        >= result.loc[d1_completed_mask, "timestamp"]
    ).sum()

    checks["D1 completed future violations"] = int(
        d1_completed_future
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    for name, value in checks.items():
        status = "PASS" if value == 0 else "FAIL"
        print(f"{status}: {name}: {value}")

    if any(value != 0 for value in checks.values()):
        raise RuntimeError(
            "CAUSALITY VALIDATION FAILED: "
            "Future information detected."
        )

    print("\nALL CAUSALITY CHECKS PASSED.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CAUSAL MULTI-TIMEFRAME ALIGNMENT")
    print("=" * 80)

    m15, h1, d1 = load_data()

    print("\nInput datasets:")
    print("M15:", len(m15))
    print("H1 :", len(h1))
    print("D1 :", len(d1))

    # --------------------------------------------------------
    # H1 completed context
    # --------------------------------------------------------

    print("\n[1/4] Building completed H1 context...")

    result = build_completed_h1_context(
        m15,
        h1,
    )

    # --------------------------------------------------------
    # H1 partial state
    # --------------------------------------------------------

    print("[2/4] Building causal H1 partial state...")

    h1_partial = build_partial_h1_state(m15)

    h1_partial = h1_partial.drop(
        columns=["h1_period"],
        errors="ignore",
    )

    result = result.merge(
        h1_partial,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # D1 completed context
    # --------------------------------------------------------

    print("[3/4] Building completed D1 context...")

    d1_completed = build_completed_d1_context(
        m15,
        d1,
    )

    d1_completed = d1_completed.drop(
        columns=["date_key"],
        errors="ignore",
    )

    result = result.merge(
        d1_completed,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # D1 partial state
    # --------------------------------------------------------

    print("[4/4] Building causal D1 partial state...")

    d1_partial = build_partial_d1_state(m15)

    d1_partial = d1_partial.drop(
        columns=["d1_period"],
        errors="ignore",
    )

    result = result.merge(
        d1_partial,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    result.sort_values("timestamp", inplace=True)
    result.reset_index(drop=True, inplace=True)

    # --------------------------------------------------------
    # Leakage validation
    # --------------------------------------------------------

    validate_no_future_information(result)

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL DATASET")
    print("=" * 80)

    print("Shape:", result.shape)

    print("\nFirst 5 rows:")
    print(result.head().to_string())

    print("\nLast 5 rows:")
    print(result.tail().to_string())

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()