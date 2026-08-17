from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROCESSED_DIR = Path("data/processed")

INPUT_FILE = (
    PROCESSED_DIR / "xauusd_mtf_price_features.parquet"
)

OUTPUT_REPORT = (
    PROCESSED_DIR / "mtf_feature_validation_report.json"
)


# ============================================================
# EXPECTED FEATURE GROUPS
# ============================================================

EXPECTED_FEATURE_PREFIXES = (
    "m15_",
    "h1_",
    "d1_",
    "mtf_",
)

NON_FEATURE_COLUMNS = {
    "timestamp",
    "open_x",
    "high_x",
    "low_x",
    "close_x",
    "tickvol_x",
    "vol_x",
    "spread_x",
    "h1_completed_timestamp",
    "h1_completed_open",
    "h1_completed_high",
    "h1_completed_low",
    "h1_completed_close",
    "h1_completed_tickvol",
    "h1_completed_vol",
    "h1_completed_spread",
    "h1_partial_period",
    "h1_partial_open",
    "h1_partial_high",
    "h1_partial_low",
    "h1_partial_close",
    "h1_partial_tickvol",
    "h1_partial_last_m15_timestamp",
    "open_y",
    "high_y",
    "low_y",
    "close_y",
    "tickvol_y",
    "vol_y",
    "spread_y",
    "d1_completed_date",
    "d1_completed_timestamp",
    "d1_completed_open",
    "d1_completed_high",
    "d1_completed_low",
    "d1_completed_close",
    "d1_completed_tickvol",
    "d1_completed_vol",
    "d1_completed_spread",
    "d1_partial_period",
    "d1_partial_open",
    "d1_partial_high",
    "d1_partial_low",
    "d1_partial_close",
    "d1_partial_tickvol",
    "d1_partial_last_m15_timestamp",
}


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def is_feature_column(column):
    """
    A feature must explicitly belong to one of the
    approved feature namespaces.
    """

    return (
        column not in NON_FEATURE_COLUMNS
        and column.startswith(EXPECTED_FEATURE_PREFIXES)
    )


def get_feature_columns(df):
    return [
        column
        for column in df.columns
        if is_feature_column(column)
    ]


def check_duplicate_columns(df):
    duplicated = df.columns[df.columns.duplicated()].tolist()

    if duplicated:
        print("FAIL: Duplicate column names found:")
        for column in duplicated:
            print(" -", column)
        return False

    print("PASS: No duplicate column names.")
    return True


def check_timestamp(df):
    print_section("1. TIMESTAMP VALIDATION")

    passed = True

    if "timestamp" not in df.columns:
        print("FAIL: timestamp column is missing.")
        return False

    null_count = int(df["timestamp"].isna().sum())

    duplicate_count = int(
        df["timestamp"].duplicated().sum()
    )

    monotonic = bool(
        df["timestamp"].is_monotonic_increasing
    )

    print("Null timestamps:", null_count)
    print("Duplicate timestamps:", duplicate_count)
    print("Monotonic increasing:", monotonic)

    if null_count != 0:
        print("FAIL: Null timestamps detected.")
        passed = False

    if duplicate_count != 0:
        print("FAIL: Duplicate timestamps detected.")
        passed = False

    if not monotonic:
        print("FAIL: Timestamp is not monotonic increasing.")
        passed = False

    if passed:
        print("PASS: Timestamp validation.")

    return passed


def check_feature_count(df):
    print_section("2. FEATURE COUNT VALIDATION")

    feature_columns = get_feature_columns(df)

    total_columns = len(df.columns)
    non_feature_columns = [
        column
        for column in df.columns
        if column not in feature_columns
    ]

    print("Total columns:", total_columns)
    print("Non-feature columns:", len(non_feature_columns))
    print("Feature columns:", len(feature_columns))

    print("\nFeature groups:")

    groups = {
        "M15": [
            c for c in feature_columns
            if c.startswith("m15_")
        ],
        "H1": [
            c for c in feature_columns
            if c.startswith("h1_")
        ],
        "D1": [
            c for c in feature_columns
            if c.startswith("d1_")
        ],
        "MTF": [
            c for c in feature_columns
            if c.startswith("mtf_")
        ],
    }

    for name, columns in groups.items():
        print(f" - {name}: {len(columns)}")

    return True


def check_numeric_features(df, feature_columns):
    print_section("3. NUMERIC FEATURE VALIDATION")

    non_numeric = []

    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            non_numeric.append(column)

    if non_numeric:
        print("FAIL: Non-numeric feature columns:")
        for column in non_numeric:
            print(" -", column)
        return False

    print("PASS: All feature columns are numeric.")

    return True


def check_inf(df, feature_columns):
    print_section("4. INFINITY VALIDATION")

    inf_counts = {}

    for column in feature_columns:
        count = int(
            np.isinf(
                df[column].to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
            ).sum()
        )

        if count > 0:
            inf_counts[column] = count

    total_inf = sum(inf_counts.values())

    print("Total Inf values:", total_inf)

    if inf_counts:
        print("FAIL: Infinite values detected:")

        for column, count in sorted(
            inf_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f" - {column}: {count}")

        return False

    print("PASS: No Inf values.")

    return True


def check_nan(df, feature_columns):
    print_section("5. NaN VALIDATION")

    nan_counts = df[feature_columns].isna().sum()

    total_nan = int(nan_counts.sum())

    affected_features = (
        nan_counts[nan_counts > 0]
        .sort_values(ascending=False)
    )

    print("Total NaN values:", total_nan)
    print(
        "Features containing NaN:",
        len(affected_features),
    )

    if len(affected_features) > 0:

        print("\nTop NaN features:")

        for column, count in affected_features.head(20).items():
            percentage = (
                count / len(df) * 100
            )

            print(
                f" - {column}: "
                f"{count} ({percentage:.4f}%)"
            )

    print(
        "\nNOTE:"
        "\nNaN values at the beginning of rolling/causal"
        "\nfeatures can be expected."
    )

    return True


def check_constant_features(df, feature_columns):
    print_section("6. CONSTANT FEATURE VALIDATION")

    constant_features = []

    for column in feature_columns:

        if df[column].nunique(
            dropna=True
        ) <= 1:
            constant_features.append(column)

    print(
        "Constant features:",
        len(constant_features),
    )

    if constant_features:

        print("FAIL: Constant features detected:")

        for column in constant_features:
            print(" -", column)

        return False

    print("PASS: No constant features.")

    return True


def check_duplicate_features(df, feature_columns):
    print_section("7. DUPLICATE FEATURE VALIDATION")

    duplicated_features = []

    # Compare only feature columns.
    # This is O(n_features^2), but the feature count
    # is small enough for this project.

    for i, column_a in enumerate(feature_columns):

        for column_b in feature_columns[i + 1:]:

            if df[column_a].equals(df[column_b]):
                duplicated_features.append(
                    (column_a, column_b)
                )

    print(
        "Duplicate feature pairs:",
        len(duplicated_features),
    )

    if duplicated_features:

        print("WARNING: Identical feature pairs found:")

        for column_a, column_b in duplicated_features:
            print(
                f" - {column_a} == {column_b}"
            )

        # This is reported as WARNING rather than FAIL
        # because identical features can sometimes be
        # intentional.
        return True

    print("PASS: No identical feature pairs.")

    return True


def check_feature_prefixes(df, feature_columns):
    print_section("8. FEATURE NAMESPACE VALIDATION")

    invalid = []

    for column in feature_columns:

        if not column.startswith(
            EXPECTED_FEATURE_PREFIXES
        ):
            invalid.append(column)

    if invalid:

        print("FAIL: Invalid feature namespaces:")

        for column in invalid:
            print(" -", column)

        return False

    print(
        "PASS: All feature columns belong to approved namespaces."
    )

    return True


def check_required_features(df):
    print_section("9. REQUIRED FEATURE VALIDATION")

    required_features = [

        # M15
        "m15_return_1",
        "m15_return_4",
        "m15_return_16",
        "m15_log_return_1",
        "m15_range",
        "m15_body",
        "m15_body_abs",

        # H1
        "h1_partial_range",
        "h1_partial_body",
        "h1_return_1",
        "h1_completed_return",

        # D1
        "d1_partial_range",
        "d1_partial_body",
        "d1_completed_return",

        # Cross timeframe
        "m15_to_h1_close_distance",
        "m15_to_d1_close_distance",
        "mtf_m15_h1_alignment",
        "mtf_m15_d1_alignment",
        "mtf_h1_d1_alignment",
        "mtf_all_direction_agreement",
    ]

    missing = [
        column
        for column in required_features
        if column not in df.columns
    ]

    if missing:

        print("FAIL: Required features missing:")

        for column in missing:
            print(" -", column)

        return False

    print(
        f"PASS: All {len(required_features)} required "
        "features exist."
    )

    return True


def check_row_count(df):
    print_section("10. ROW COUNT VALIDATION")

    expected_rows = 188633
    actual_rows = len(df)

    print("Expected rows:", expected_rows)
    print("Actual rows:", actual_rows)

    if actual_rows != expected_rows:
        print("FAIL: Row count changed unexpectedly.")
        return False

    print("PASS: Row count preserved.")

    return True


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 80)
    print("CAUSAL MULTI-TIMEFRAME FEATURE VALIDATION")
    print("=" * 80)

    print("\nLoading:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print("Input shape:", df.shape)

    feature_columns = get_feature_columns(df)

    results = {}

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    results["duplicate_columns"] = (
        check_duplicate_columns(df)
    )

    results["timestamp"] = check_timestamp(df)

    results["row_count"] = check_row_count(df)

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    check_feature_count(df)

    results["numeric"] = check_numeric_features(
        df,
        feature_columns,
    )

    results["feature_namespace"] = (
        check_feature_prefixes(
            df,
            feature_columns,
        )
    )

    results["required_features"] = (
        check_required_features(df)
    )

    results["inf"] = check_inf(
        df,
        feature_columns,
    )

    results["nan"] = check_nan(
        df,
        feature_columns,
    )

    results["constant"] = check_constant_features(
        df,
        feature_columns,
    )

    results["duplicate_features"] = (
        check_duplicate_features(
            df,
            feature_columns,
        )
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print_section("FINAL VALIDATION RESULT")

    failed = [
        name
        for name, passed in results.items()
        if not passed
    ]

    if failed:

        print("FAILED VALIDATIONS:")

        for name in failed:
            print(" -", name)

        print("\nFEATURE VALIDATION FAILED.")

        raise RuntimeError(
            "Multi-timeframe feature validation failed."
        )

    print("PASS: All mandatory validations passed.")

    print("\nDataset:")
    print("Rows:", len(df))
    print("Total columns:", len(df.columns))
    print("Feature columns:", len(feature_columns))

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {
        "input_file": str(INPUT_FILE),
        "rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "feature_columns": int(len(feature_columns)),
        "feature_names": feature_columns,
        "validation_results": results,
    }

    import json

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nValidation report saved:")
    print(OUTPUT_REPORT)

    print("\nDONE.")


if __name__ == "__main__":
    main()