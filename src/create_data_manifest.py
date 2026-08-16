from pathlib import Path
import json

import pandas as pd


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/processed")


FILES = {
    "M15": "xauusd_m15_standardized.parquet",
    "H1": "xauusd_h1_standardized.parquet",
    "D1": "xauusd_d1_standardized.parquet",
}


def audit_dataframe(df: pd.DataFrame, timeframe: str) -> dict:
    df = df.sort_values("timestamp").reset_index(drop=True)

    result = {
        "timeframe": timeframe,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "timestamp": {
            "null": int(df["timestamp"].isna().sum()),
            "duplicates": int(df["timestamp"].duplicated().sum()),
            "monotonic_increasing": bool(
                df["timestamp"].is_monotonic_increasing
            ),
        },
        "ohlc": {
            "invalid_rows": int(
                (
                    (df["open"] <= 0)
                    | (df["high"] <= 0)
                    | (df["low"] <= 0)
                    | (df["close"] <= 0)
                    | (df["high"] < df["open"])
                    | (df["high"] < df["close"])
                    | (df["low"] > df["open"])
                    | (df["low"] > df["close"])
                    | (df["high"] < df["low"])
                ).sum()
            )
        },
        "volume": {
            "tickvol_non_positive": int(
                (df["tickvol"] <= 0).sum()
            ),
            "real_volume_zero": int(
                (df["vol"] == 0).sum()
            ),
            "real_volume_negative": int(
                (df["vol"] < 0).sum()
            ),
        },
        "spread": {
            "negative": int(
                (df["spread"] < 0).sum()
            ),
            "zero": int(
                (df["spread"] == 0).sum()
            ),
        },
    }

    return result


def main():
    manifest = {
        "project": "Evolutionary Ensemble Learning Based on Multiple Time Frames for Explainable Financial Markets Analysis",
        "instrument": "XAUUSD",
        "primary_timeline": "M15",
        "context_timeframes": ["H1", "D1"],
        "raw_data_policy": {
            "raw_files_modified": False,
            "synthetic_candles_created": False,
            "gap_filling_performed": False,
            "rows_deleted_during_standardization": False,
        },
        "quality_interpretation": {
            "timestamp": "Valid, unique and chronologically ordered.",
            "ohlc": "No invalid OHLC rows detected.",
            "gaps": (
                "Observed temporal gaps are retained as structural market/session/holiday gaps. "
                "They are not automatically treated as missing observations."
            ),
            "real_volume": (
                "Real volume is zero for most observations and should not be assumed "
                "to represent reliable traded volume."
            ),
            "tick_volume": (
                "Tick volume is positive across the audited datasets and may be considered "
                "as a market-activity variable subject to later feature validation."
            ),
            "spread": (
                "Spread contains valid non-negative values, including some zero values. "
                "Its treatment in transaction-cost modeling will be defined later."
            ),
        },
        "datasets": {},
    }

    for timeframe, filename in FILES.items():
        path = PROCESSED_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {path}"
            )

        df = pd.read_parquet(path)

        manifest["datasets"][timeframe] = audit_dataframe(
            df,
            timeframe
        )

    output_path = REPORT_DIR / "data_manifest.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 80)
    print("DATA MANIFEST CREATED")
    print("=" * 80)

    for timeframe, info in manifest["datasets"].items():
        print(f"\n{timeframe}")
        print("-" * 40)
        print("Rows:", info["rows"])
        print("Start:", info["start"])
        print("End:", info["end"])
        print(
            "Duplicate timestamps:",
            info["timestamp"]["duplicates"]
        )
        print(
            "Invalid OHLC rows:",
            info["ohlc"]["invalid_rows"]
        )
        print(
            "Zero real-volume rows:",
            info["volume"]["real_volume_zero"]
        )
        print(
            "Zero spread rows:",
            info["spread"]["zero"]
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()