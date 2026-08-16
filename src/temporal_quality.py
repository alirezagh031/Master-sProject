from pathlib import Path
import pandas as pd


PROCESSED_DIR = Path("data/processed")

FILES = {
    "M15": "xauusd_m15_standardized.parquet",
    "H1": "xauusd_h1_standardized.parquet",
    "D1": "xauusd_d1_standardized.parquet",
}

EXPECTED_INTERVALS = {
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "D1": pd.Timedelta(days=1),
}


def analyze_timeframe(name: str, filename: str) -> None:
    print("\n" + "=" * 90)
    print(f"TIMEFRAME: {name}")
    print("=" * 90)

    path = PROCESSED_DIR / filename
    df = pd.read_parquet(path)

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---------------------------------------------------------
    # Basic information
    # ---------------------------------------------------------
    print("\n--- BASIC ---")
    print("Rows:", len(df))
    print("Start:", df["timestamp"].iloc[0])
    print("End:", df["timestamp"].iloc[-1])

    # ---------------------------------------------------------
    # Timestamp quality
    # ---------------------------------------------------------
    print("\n--- TIMESTAMP QUALITY ---")

    duplicate_timestamps = df["timestamp"].duplicated().sum()
    monotonic = df["timestamp"].is_monotonic_increasing
    null_timestamps = df["timestamp"].isna().sum()

    print("Null timestamps:", null_timestamps)
    print("Duplicate timestamps:", duplicate_timestamps)
    print("Monotonic increasing:", monotonic)

    # ---------------------------------------------------------
    # Time intervals
    # ---------------------------------------------------------
    df["delta"] = df["timestamp"].diff()

    expected = EXPECTED_INTERVALS[name]

    print("\n--- INTERVAL ANALYSIS ---")
    print("Expected interval:", expected)

    print("\nMost common intervals:")
    print(df["delta"].value_counts().head(15))

    unexpected = df.loc[
        df["delta"].notna() & (df["delta"] != expected),
        ["timestamp", "delta"]
    ]

    print("\nUnexpected intervals:", len(unexpected))

    if len(unexpected) > 0:
        print("\nLargest unexpected intervals:")
        print(
            unexpected
            .sort_values("delta", ascending=False)
            .head(20)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # Gap analysis
    # ---------------------------------------------------------
    # A gap is any interval larger than the expected interval.
    gaps = df.loc[
        df["delta"].notna() & (df["delta"] > expected),
        ["timestamp", "delta"]
    ].copy()

    print("\n--- GAP ANALYSIS ---")
    print("Total gaps:", len(gaps))

    if len(gaps) > 0:
        gaps["missing_intervals_est"] = (
            gaps["delta"] / expected
        ).astype("int64") - 1

        print("\nTop 20 gaps:")
        print(
            gaps
            .sort_values("delta", ascending=False)
            .head(20)
            .to_string(index=False)
        )

        print("\nEstimated missing intervals:")
        print(
            "Total:",
            gaps["missing_intervals_est"].sum()
        )

    # ---------------------------------------------------------
    # OHLC validity
    # ---------------------------------------------------------
    print("\n--- OHLC VALIDITY ---")

    invalid_ohlc = (
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["high"] < df["low"])
    )

    print("Invalid OHLC rows:", invalid_ohlc.sum())

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------
    print("\n--- VOLUME ---")

    print("Tick volume <= 0:", (df["tickvol"] <= 0).sum())
    print("Real volume < 0:", (df["vol"] < 0).sum())
    print("Real volume == 0:", (df["vol"] == 0).sum())

    # ---------------------------------------------------------
    # Spread
    # ---------------------------------------------------------
    print("\n--- SPREAD ---")

    print("Spread < 0:", (df["spread"] < 0).sum())
    print("Spread == 0:", (df["spread"] == 0).sum())

    # ---------------------------------------------------------
    # Final memory usage
    # ---------------------------------------------------------
    print("\n--- MEMORY ---")
    print(
        f"Memory usage: "
        f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB"
    )


def main() -> None:
    for name, filename in FILES.items():
        analyze_timeframe(name, filename)


if __name__ == "__main__":
    main()