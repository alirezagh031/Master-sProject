from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")


def inspect_file(path: Path) -> None:
    print("\n" + "=" * 80)
    print(f"FILE: {path.name}")
    print("=" * 80)

    print(f"Size: {path.stat().st_size / (1024 ** 2):.2f} MB")

    # Read a small sample first to identify the structure.
    sample = pd.read_csv(
        path,
        sep="\t",
        nrows=5,
        low_memory=False
    )

    print("\nColumns:")
    print(list(sample.columns))

    print("\nSample:")
    print(sample.to_string(index=False))

    # Full read for audit only.
    df = pd.read_csv(
        path,
        sep="\t",
        low_memory=False
    )

    print("\nShape:")
    print(df.shape)

    print("\nDtypes:")
    print(df.dtypes)

    print("\nFirst rows:")
    print(df.head(3).to_string(index=False))

    print("\nLast rows:")
    print(df.tail(3).to_string(index=False))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # Try to identify date/time columns.
    date_candidates = [
        c for c in df.columns
        if str(c).strip().upper() in {"<DATE>", "DATE"}
    ]

    time_candidates = [
        c for c in df.columns
        if str(c).strip().upper() in {"<TIME>", "TIME"}
    ]

    print("\nDate columns:", date_candidates)
    print("Time columns:", time_candidates)

    if date_candidates:
        date_col = date_candidates[0]
        print(
            f"\nDate range in {date_col}: "
            f"{df[date_col].iloc[0]} -> {df[date_col].iloc[-1]}"
        )

    print("\n")


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files were found in data/raw/"
        )

    print(f"Found {len(files)} CSV file(s).")

    for path in files:
        inspect_file(path)


if __name__ == "__main__":
    main()