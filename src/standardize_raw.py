from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

FILE_MAP = {
    "M15": "XAUUSD_M15_201801020900_202512311915.csv",
    "H1": "XAUUSD_H1_201801020900_202512311900.csv",
    "D1": "XAUUSD_Daily_201801020000_202512310000.csv",
}


def load_and_standardize(path: Path, timeframe: str) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"STANDARDIZING: {timeframe}")
    print("=" * 80)

    df = pd.read_csv(
        path,
        sep="\t",
        low_memory=False
    )

    # Normalize column names.
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
        .str.lower()
    )

    # Build a single canonical timestamp.
    if timeframe == "D1":
        df["timestamp"] = pd.to_datetime(
            df["date"],
            format="%Y.%m.%d",
            errors="raise"
        )
    else:
        df["timestamp"] = pd.to_datetime(
            df["date"] + " " + df["time"],
            format="%Y.%m.%d %H:%M:%S",
            errors="raise"
        )

    # Numeric columns.
    float_cols = ["open", "high", "low", "close"]
    int_cols = ["tickvol", "vol", "spread"]

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float32")

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Use smaller integer types where safe.
    df["tickvol"] = df["tickvol"].astype("int64")
    df["vol"] = df["vol"].astype("int64")
    df["spread"] = df["spread"].astype("int32")

    # Keep only canonical columns.
    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "tickvol",
        "vol",
        "spread",
    ]

    df = df[columns]

    # Sort chronologically.
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic structural checks.
    assert df["timestamp"].notna().all()
    assert not df["timestamp"].duplicated().any()
    assert df["timestamp"].is_monotonic_increasing

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Timestamp dtype:", df["timestamp"].dtype)
    print("Start:", df["timestamp"].iloc[0])
    print("End:", df["timestamp"].iloc[-1])
    print("Duplicate timestamps:", df["timestamp"].duplicated().sum())
    print("Monotonic:", df["timestamp"].is_monotonic_increasing)

    print("\nDtypes:")
    print(df.dtypes)

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nLast 3 rows:")
    print(df.tail(3).to_string(index=False))

    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for timeframe, filename in FILE_MAP.items():
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = load_and_standardize(path, timeframe)

        output_path = PROCESSED_DIR / f"xauusd_{timeframe.lower()}_standardized.parquet"

        df.to_parquet(
            output_path,
            index=False,
            engine="pyarrow"
        )

        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()