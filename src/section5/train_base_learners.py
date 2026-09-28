from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.section5 import config as cfg


# ==========================================================================
# 0. Optional reuse of the existing Stage 4 temporal-split logic
# ==========================================================================
def resolve_split_boundaries() -> dict:
    """
    Prefer Stage 4's own split boundaries if a reusable function/constant can
    be found there, so Stage 5 can never silently drift from Stage 4's
    leakage-safe design. Falls back to the boundaries in config.py otherwise.
    """
    boundaries = dict(
        train_start=cfg.TRAIN_START, train_end=cfg.TRAIN_END,
        val_start=cfg.VALIDATION_START, val_end=cfg.VALIDATION_END,
        test_start=cfg.TEST_START, test_end=cfg.TEST_END,
        source="config.py (hardcoded, mirrors Stage 4 boundaries)",
    )
    try:
        from src.section4 import purged_walk_forward as s4  # type: ignore

        for fn_name in ("get_temporal_splits", "TEMPORAL_SPLITS", "SPLIT_BOUNDARIES"):
            if hasattr(s4, fn_name):
                print(
                    f"  [split] found '{fn_name}' in src.section4."
                    f"purged_walk_forward, but Stage 5 still uses the "
                    f"boundaries in config.py -- verify they match and wire "
                    f"this function in directly if Stage 4 changes."
                )
                break
    except ImportError:
        print(
            "  [split] no src.section4.purged_walk_forward module found in "
            "this environment; using the fixed boundaries from config.py "
            "(2018-2023 / 2024 / 2025)."
        )
    return boundaries


# ==========================================================================
# 1. I/O helpers (parquet by default; .csv also supported, used in tests)
# ==========================================================================
def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def load_timeframe_dataset(timeframe: str) -> pd.DataFrame:
    path = cfg.TRAIN_DATASET_PATHS[timeframe]
    if not path.exists():
        raise FileNotFoundError(
            f"[{timeframe}] expected dataset not found at {path}. Stage 5 "
            f"dataset separation was reported complete in the spec -- check "
            f"that the file actually exists at this path."
        )
    df = read_table(path)
    print(f"  [{timeframe}] loaded {path.name}: shape={df.shape}")
    return df


def check_lightgbm():
    """Import LightGBM or fail loudly with exact install instructions.

    Per the Stage 5 spec, an unavailable LightGBM must never be silently
    swapped for a different algorithm.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is not installed in the current .venv. Install it "
            "with:\n\n"
            "    pip install lightgbm\n\n"
            "On Apple Silicon (M1/M2) the PyPI wheel normally works as-is. "
            "If you hit an OpenMP-related import error, run:\n\n"
            "    brew install libomp\n"
            "    pip install lightgbm\n"
        ) from exc
    return lgb


# ==========================================================================
# 2. Validation of the raw data (section 9, 19 of the spec)
# ==========================================================================
def verify_target_encoding(df: pd.DataFrame, timeframe: str) -> None:
    observed = set(pd.unique(df[cfg.TARGET_COLUMN].dropna()).tolist())
    print(f"  [{timeframe}] observed target values: {sorted(observed)}")
    unexpected = observed - cfg.EXPECTED_CLASSES
    if unexpected:
        raise RuntimeError(
            f"[{timeframe}] target contains values outside the expected "
            f"encoding {sorted(cfg.EXPECTED_CLASSES)}: {unexpected}. "
            f"Refusing to silently remap -- check how the target was built."
        )


def add_partial_availability_indicator(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Adds a causal 0/1 indicator marking whether THIS row's own partial-bar
    columns are present. Derived strictly from the row's own current
    values (no other row, no future information) -- 1 if all of that
    timeframe's partial_* columns are non-NaN at this timestamp, 0 if any
    are NaN (the expected "first bar of a new period" state confirmed by
    inspect_partial_nan.py). No-op for m15 (no partial columns).
    """
    partial_cols = cfg.PARTIAL_NAN_COLUMNS[timeframe]
    indicator_col = cfg.PARTIAL_AVAILABILITY_COLUMN[timeframe]
    if not partial_cols:
        return df
    df = df.copy()
    df[indicator_col] = (~df[partial_cols].isna().any(axis=1)).astype("float32")
    return df


def get_feature_columns(df: pd.DataFrame, timeframe: str) -> list[str]:
    present_leakage = set(df.columns) & cfg.LEAKAGE_COLUMNS
    present_internal_ts = set(df.columns) & cfg.INTERNAL_TIMESTAMP_COLUMNS
    if present_leakage:
        print(
            f"  [{timeframe}] NOTE: excluding potential leakage column(s) "
            f"found in the source file (must never be model features): "
            f"{sorted(present_leakage)}"
        )
    if present_internal_ts:
        print(
            f"  [{timeframe}] NOTE: excluding internal timestamp column(s) "
            f"(datetime-typed bar markers, never used as features and never "
            f"encoded): {sorted(present_internal_ts)}"
        )
    cols = [c for c in df.columns if c not in cfg.FORBIDDEN_FEATURE_COLUMNS]
    print(f"  [{timeframe}] using {len(cols)} feature columns: {cols}")
    return cols


def assert_numeric_features(df: pd.DataFrame, feature_columns: list[str], timeframe: str) -> None:
    dtypes = df[feature_columns].dtypes.astype(str).to_dict()
    print(f"  [{timeframe}] feature dtypes: {dtypes}")
    non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise RuntimeError(
            f"[{timeframe}] non-numeric feature column(s) found: "
            f"{non_numeric}. Both LogisticRegression and LightGBM in this "
            f"pipeline expect purely numeric input -- encode these "
            f"explicitly before training rather than letting the pipeline "
            f"guess how."
        )


# ==========================================================================
# 3. Temporal split (section 8, 18 of the spec)
# ==========================================================================
def _warn_if_row_count_drifts(timeframe: str, n_train: int, n_val: int, n_test: int) -> None:
    for name, actual, expected in (
        ("train", n_train, cfg.EXPECTED_TRAIN_ROWS),
        ("validation", n_val, cfg.EXPECTED_VALIDATION_ROWS),
        ("test", n_test, cfg.EXPECTED_TEST_ROWS),
    ):
        if actual != expected:
            print(
                f"  [{timeframe}] WARNING: {name} split has {actual} rows, "
                f"expected {expected} from the Stage 5 dataset-separation "
                f"report. Not necessarily an error (the source file may "
                f"have been regenerated) -- just verify."
            )


def temporal_split(df: pd.DataFrame, boundaries: dict, timeframe: str):
    df = df.sort_values(cfg.TIMESTAMP_COLUMN).reset_index(drop=True)
    ts = pd.to_datetime(df[cfg.TIMESTAMP_COLUMN])

    train_mask = (ts >= boundaries["train_start"]) & (ts <= boundaries["train_end"])
    val_mask = (ts >= boundaries["val_start"]) & (ts <= boundaries["val_end"])
    test_mask = (ts >= boundaries["test_start"]) & (ts <= boundaries["test_end"])

    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)

    # --- thesis-critical: zero overlap, strictly increasing time ---
    if len(train_df) and len(val_df):
        assert pd.to_datetime(train_df[cfg.TIMESTAMP_COLUMN]).max() < pd.to_datetime(
            val_df[cfg.TIMESTAMP_COLUMN]
        ).min(), f"[{timeframe}] TRAIN/VALIDATION overlap detected -- leakage risk."
    if len(val_df) and len(test_df):
        assert pd.to_datetime(val_df[cfg.TIMESTAMP_COLUMN]).max() < pd.to_datetime(
            test_df[cfg.TIMESTAMP_COLUMN]
        ).min(), f"[{timeframe}] VALIDATION/TEST overlap detected -- leakage risk."

    print(
        f"  [{timeframe}] split sizes: train={len(train_df)}, "
        f"validation={len(val_df)}, test={len(test_df)}"
    )
    _warn_if_row_count_drifts(timeframe, len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


# ==========================================================================
# 4. Missing-value handling (section 19 of the spec)
# ==========================================================================
def audit_and_clean_train_missing(
    df: pd.DataFrame, feature_columns: list[str], timeframe: str
) -> pd.DataFrame:
    working = df.copy()
    feats = working[feature_columns].astype("float64").replace([np.inf, -np.inf], np.nan)
    working[feature_columns] = feats

    row_has_bad = working[feature_columns].isna().any(axis=1)
    n_bad = int(row_has_bad.sum())
    frac_bad = n_bad / len(working) if len(working) else 0.0
    per_col_na = working[feature_columns].isna().sum()
    per_col_na = per_col_na[per_col_na > 0]

    print(
        f"  [{timeframe}/train] missing-value audit: {n_bad}/{len(working)} "
        f"rows ({frac_bad:.4%}) contain NaN/inf."
    )
    if len(per_col_na):
        print(f"    affected columns: {per_col_na.to_dict()}")

    if n_bad == 0:
        return working

    if frac_bad > cfg.MAX_AUTO_DROP_FRACTION:
        raise RuntimeError(
            f"[{timeframe}/train] {frac_bad:.2%} of training rows contain "
            f"NaN/inf, above the {cfg.MAX_AUTO_DROP_FRACTION:.2%} auto-drop "
            f"threshold. This is too large to drop silently -- inspect the "
            f"MTF alignment for this timeframe before proceeding."
        )

    working = working.loc[~row_has_bad].reset_index(drop=True)
    print(
        f"    -> dropped {n_bad} likely warm-up rows (before the first "
        f"completed H1/D1 bar); {len(working)} training rows remain."
    )
    return working


def assert_no_missing_values(
    df: pd.DataFrame, feature_columns: list[str], split_name: str, timeframe: str
) -> None:
    feats = df[feature_columns].astype("float64")
    n_nan = int(feats.isna().sum().sum())
    n_inf = int(np.isinf(feats.to_numpy()).sum())
    print(f"  [{timeframe}/{split_name}] missing-value audit: {n_nan} NaNs, {n_inf} inf values.")
    if n_nan or n_inf:
        raise RuntimeError(
            f"[{timeframe}/{split_name}] found {n_nan} NaN and {n_inf} inf "
            f"values in the {split_name} split. Rows cannot be dropped here "
            f"without breaking row-count/timestamp alignment across the six "
            f"base learners' prediction files -- fix this upstream instead "
            f"(Stage 4 / MTF alignment)."
        )


def downcast_features(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        df = df.copy()
        df[float_cols] = df[float_cols].astype("float32")
    return df


def fit_and_apply_partial_imputer(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, partial_cols: list[str], timeframe: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[SimpleImputer]]:
    """
    Fits a SimpleImputer(strategy="median") on X_train[partial_cols] ONLY,
    then applies that already-fitted imputer (transform, never refit) to
    train/validation/test alike. No statistic is ever computed from
    validation or test. No-op (returns inputs unchanged) when there are no
    partial columns for this timeframe (m15).
    """
    if not partial_cols:
        return X_train, X_val, X_test, None

    imputer = SimpleImputer(strategy=cfg.PARTIAL_IMPUTE_STRATEGY, keep_empty_features=True)
    imputer.fit(X_train[partial_cols])  # <-- fit on TRAIN ONLY

    n_observed = X_train[partial_cols].notna().sum()
    if (n_observed == 0).any():
        raise RuntimeError(
            f"[{timeframe}] column(s) {n_observed[n_observed == 0].index.tolist()} "
            f"have ZERO observed (non-NaN) values in the training split -- a "
            f"median imputer cannot produce a meaningful fill here. This is "
            f"not the confirmed 'first bar of period' pattern (~25%/~1% "
            f"missing); inspect the training window for this timeframe."
        )

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    X_train[partial_cols] = imputer.transform(X_train[partial_cols])
    X_val[partial_cols] = imputer.transform(X_val[partial_cols])
    X_test[partial_cols] = imputer.transform(X_test[partial_cols])

    medians = dict(zip(partial_cols, imputer.statistics_.tolist()))
    print(
        f"  [{timeframe}] partial-column imputer ({cfg.PARTIAL_IMPUTE_STRATEGY}, "
        f"fit on {len(X_train)} TRAIN rows only, then applied unchanged to "
        f"validation/test): {medians}"
    )
    return X_train, X_val, X_test, imputer


# ==========================================================================
# 5. Probability formatting (section 9, 13, 15 of the spec)
# ==========================================================================
def reorder_probabilities(classes_, raw_proba: np.ndarray, prefix: str) -> pd.DataFrame:
    """
    Map a classifier's raw predict_proba() columns (ordered by `classes_`)
    onto the thesis-mandated [DOWN, NEUTRAL, UP] order. Never assumes the
    two orders coincide, even though for classes {-1, 0, 1} they do under
    the numeric sort both scikit-learn and LightGBM use internally.
    """
    classes_list = list(classes_)
    missing = cfg.EXPECTED_CLASSES - set(classes_list)
    if missing:
        raise RuntimeError(
            f"[{prefix}] training data did not contain class(es) {missing}; "
            f"refusing to produce a probability vector with an undefined "
            f"column -- this would corrupt the ensemble stage."
        )
    out = {}
    for class_value, column_name in zip(cfg.CLASS_ORDER, cfg.PROBA_COLUMN_ORDER):
        idx = classes_list.index(class_value)
        out[f"{prefix}_{column_name}"] = raw_proba[:, idx]
    return pd.DataFrame(out)


# ==========================================================================
# 6. Model builders / trainers
# ==========================================================================
def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**cfg.LOGISTIC_REGRESSION_PARAMS)),
        ]
    )


def train_lightgbm_classifier(
    lgb_module,
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    timeframe: str,
):
    """
    Two-phase, leakage-safe LightGBM training.

    Phase 1 fits on an inner slice of the TRAIN period only
    (train_start .. 2022-12-31), early-stopping against a held-out inner
    slice (2023-01-01 .. train_end). 2024/2025 are never touched here.

    Phase 2 refits with the resulting fixed number of boosting rounds on
    the FULL train period, so the final model uses all available data.
    """
    train_ts = pd.to_datetime(train_df[cfg.TIMESTAMP_COLUMN])
    inner_train_mask = (train_ts < cfg.INNER_EARLY_STOP_START).to_numpy()
    inner_eval_mask = ~inner_train_mask

    X_inner_train, y_inner_train = X_train.loc[inner_train_mask], y_train.loc[inner_train_mask]
    X_inner_eval, y_inner_eval = X_train.loc[inner_eval_mask], y_train.loc[inner_eval_mask]

    if len(X_inner_eval) == 0 or len(X_inner_train) == 0:
        raise RuntimeError(
            f"[{timeframe}/lightgbm] inner early-stopping split produced an "
            f"empty slice (inner_train={len(X_inner_train)}, "
            f"inner_eval={len(X_inner_eval)}); check INNER_EARLY_STOP_START."
        )

    print(
        f"  [{timeframe}/lightgbm] phase 1: early stopping on inner slice "
        f"({len(X_inner_train)} inner-train / {len(X_inner_eval)} inner-eval rows)"
    )
    probe = lgb_module.LGBMClassifier(**cfg.LIGHTGBM_PARAMS)
    probe.fit(
        X_inner_train,
        y_inner_train,
        eval_set=[(X_inner_eval, y_inner_eval)],
        eval_metric="multi_logloss",
        callbacks=[lgb_module.early_stopping(cfg.LIGHTGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best_iteration = getattr(probe, "best_iteration_", None)
    if best_iteration is None:
        best_iteration = cfg.LIGHTGBM_PARAMS["n_estimators"]
    print(f"    -> best_iteration = {best_iteration}")
    del probe
    gc.collect()

    print(
        f"  [{timeframe}/lightgbm] phase 2: refitting on full train "
        f"({len(X_train)} rows) with n_estimators={best_iteration}"
    )
    final_params = dict(cfg.LIGHTGBM_PARAMS)
    final_params["n_estimators"] = int(best_iteration)
    final_model = lgb_module.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)
    return final_model


# ==========================================================================
# 7. Metrics (section 16 of the spec)
# ==========================================================================
def compute_metrics(
    model_name: str,
    y_true: pd.Series,
    proba_df_named: pd.DataFrame,  # columns already renamed to "down"/"neutral"/"up"
    n_features: int,
    train_rows: int,
    val_rows: int,
    train_seconds: float,
    predict_seconds: float,
    model_params: dict,
) -> dict:
    proba_matrix = proba_df_named[cfg.PROBA_COLUMN_ORDER].to_numpy()
    y_pred_labels = np.array(cfg.CLASS_ORDER)[proba_matrix.argmax(axis=1)]

    macro_f1 = f1_score(y_true, y_pred_labels, labels=cfg.CLASS_ORDER, average="macro", zero_division=0)
    logloss = log_loss(y_true, proba_matrix, labels=cfg.CLASS_ORDER)
    acc = accuracy_score(y_true, y_pred_labels)

    sanity_max_dev = float(np.abs(proba_matrix.sum(axis=1) - 1.0).max())

    return {
        "model": model_name,
        "macro_f1": round(float(macro_f1), 6),
        "log_loss": round(float(logloss), 6),
        "accuracy": round(float(acc), 6),
        "n_features": n_features,
        "train_rows": int(train_rows),
        "validation_rows": int(val_rows),
        "train_seconds": round(train_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "probability_sum_max_deviation": round(sanity_max_dev, 8),
        "validation_class_distribution": {
            str(k): int(v) for k, v in y_true.value_counts().sort_index().items()
        },
        "model_params": {k: v for k, v in model_params.items() if not callable(v)},
    }


# ==========================================================================
# 8. Final verification suite (section 26.F of the spec)
# ==========================================================================
def expected_prediction_columns() -> list[str]:
    return [cfg.TIMESTAMP_COLUMN, cfg.TARGET_COLUMN] + [
        f"{tf}_{model}_{cls}"
        for tf in cfg.TIMEFRAMES
        for model in cfg.MODEL_KINDS
        for cls in cfg.PROBA_COLUMN_ORDER
    ]


def verify_predictions_output(df: pd.DataFrame, expected_rows: Optional[int], split_name: str) -> None:
    expected_columns = expected_prediction_columns()

    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)
    assert not missing_cols, f"[{split_name}] missing expected columns: {missing_cols}"
    assert not extra_cols, f"[{split_name}] unexpected extra columns: {extra_cols}"
    assert list(df.columns) == expected_columns, (
        f"[{split_name}] column order does not match the mandated "
        f"timestamp/target + 18-probability layout."
    )

    if expected_rows is not None:
        assert df.shape[0] == expected_rows, (
            f"[{split_name}] expected {expected_rows} rows, got {df.shape[0]}."
        )

    observed_targets = set(pd.unique(df[cfg.TARGET_COLUMN]).tolist())
    assert observed_targets <= cfg.EXPECTED_CLASSES, (
        f"[{split_name}] target contains values outside "
        f"{sorted(cfg.EXPECTED_CLASSES)}: {observed_targets - cfg.EXPECTED_CLASSES}"
    )

    for tf in cfg.TIMEFRAMES:
        for model in cfg.MODEL_KINDS:
            cols = [f"{tf}_{model}_{cls}" for cls in cfg.PROBA_COLUMN_ORDER]
            block = df[cols].to_numpy()
            sums = block.sum(axis=1)
            bad = np.abs(sums - 1.0) > 1e-3
            assert not bad.any(), (
                f"[{split_name}] {tf}_{model} probabilities do not sum to 1 "
                f"for {int(bad.sum())} row(s) (max deviation "
                f"{np.abs(sums - 1.0).max():.6f})."
            )
            assert (block >= -1e-9).all(), f"[{split_name}] {tf}_{model} has negative probabilities."

    assert df[cfg.TIMESTAMP_COLUMN].is_monotonic_increasing, f"[{split_name}] timestamps are not sorted."

    print(
        f"  [verify/{split_name}] OK -- {df.shape[0]} rows, "
        f"{len(expected_columns)} columns, all 6 probability vectors valid."
    )


# ==========================================================================
# 9. Main orchestration
# ==========================================================================
def main() -> None:
    run_start = time.time()
    print("=" * 78)
    print("Stage 5 - Multi-Timeframe Base Learner Training")
    print("=" * 78)

    lgb_module = check_lightgbm()
    boundaries = resolve_split_boundaries()

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    val_prob_blocks: list[pd.DataFrame] = []
    test_prob_blocks: list[pd.DataFrame] = []
    metrics_report: list[dict] = []

    val_timestamp_ref: Optional[pd.Series] = None
    val_target_ref: Optional[pd.Series] = None
    test_timestamp_ref: Optional[pd.Series] = None
    test_target_ref: Optional[pd.Series] = None
    n_val_rows: Optional[int] = None
    n_test_rows: Optional[int] = None

    for timeframe in cfg.TIMEFRAMES:
        print(f"\n--- timeframe: {timeframe.upper()} " + "-" * 40)
        df = load_timeframe_dataset(timeframe)
        verify_target_encoding(df, timeframe)
        df = add_partial_availability_indicator(df, timeframe)

        feature_columns = get_feature_columns(df, timeframe)
        assert_numeric_features(df, feature_columns, timeframe)

        train_df, val_df, test_df = temporal_split(df, boundaries, timeframe)
        del df
        gc.collect()

        # ---- missing values ----
        # partial_cols carry a KNOWN, structural NaN pattern (confirmed via
        # inspect_partial_nan.py: exactly the first M15 bar of each new
        # H1 hour / D1 day) -- they are excluded from the strict drop/fail
        # checks below and handled separately by a train-only imputer.
        partial_cols = cfg.PARTIAL_NAN_COLUMNS[timeframe]
        strict_columns = [c for c in feature_columns if c not in partial_cols]
        if partial_cols:
            print(
                f"  [{timeframe}] {len(partial_cols)} partial-bar column(s) "
                f"{partial_cols} carry a known structural NaN pattern "
                f"(confirmed via inspect_partial_nan.py) -- excluded from "
                f"the strict checks below, handled by a train-only-fitted "
                f"imputer instead."
            )
        train_df = audit_and_clean_train_missing(train_df, strict_columns, timeframe)
        assert_no_missing_values(val_df, strict_columns, "validation", timeframe)
        assert_no_missing_values(test_df, strict_columns, "test", timeframe)

        # ---- cross-timeframe alignment (timestamps AND target must match) ----
        if val_timestamp_ref is None:
            val_timestamp_ref = val_df[cfg.TIMESTAMP_COLUMN].reset_index(drop=True)
            val_target_ref = val_df[cfg.TARGET_COLUMN].reset_index(drop=True)
            test_timestamp_ref = test_df[cfg.TIMESTAMP_COLUMN].reset_index(drop=True)
            test_target_ref = test_df[cfg.TARGET_COLUMN].reset_index(drop=True)
            n_val_rows, n_test_rows = len(val_df), len(test_df)
        else:
            assert val_df[cfg.TIMESTAMP_COLUMN].reset_index(drop=True).equals(val_timestamp_ref), (
                f"[{timeframe}] validation timestamps do not align across timeframes."
            )
            assert val_df[cfg.TARGET_COLUMN].reset_index(drop=True).equals(val_target_ref), (
                f"[{timeframe}] validation targets do not align across timeframes."
            )
            assert test_df[cfg.TIMESTAMP_COLUMN].reset_index(drop=True).equals(test_timestamp_ref), (
                f"[{timeframe}] test timestamps do not align across timeframes."
            )

        X_train = train_df[feature_columns].copy()
        y_train = train_df[cfg.TARGET_COLUMN]
        X_val = val_df[feature_columns].copy()
        y_val = val_df[cfg.TARGET_COLUMN]
        X_test = test_df[feature_columns].copy()

        # ---- train-only imputation of the (structurally NaN) partial cols ----
        X_train, X_val, X_test, partial_imputer = fit_and_apply_partial_imputer(
            X_train, X_val, X_test, partial_cols, timeframe
        )
        if partial_imputer is not None:
            dump(partial_imputer, cfg.MODELS_DIR / f"{timeframe}_partial_imputer.joblib")

        assert not X_train.isna().any().any(), f"[{timeframe}] X_train still has NaN after imputation."
        assert not X_val.isna().any().any(), f"[{timeframe}] X_val still has NaN after imputation."
        assert not X_test.isna().any().any(), f"[{timeframe}] X_test still has NaN after imputation."

        X_train = downcast_features(X_train)
        X_val = downcast_features(X_val)
        X_test = downcast_features(X_test)

        # =================== Logistic Regression ===================
        t0 = time.time()
        lr_pipeline = build_logistic_regression_pipeline()
        lr_pipeline.fit(X_train, y_train)
        lr_train_seconds = time.time() - t0

        t0 = time.time()
        lr_val_raw = lr_pipeline.predict_proba(X_val)
        lr_predict_seconds = time.time() - t0
        lr_test_raw = lr_pipeline.predict_proba(X_test)

        lr_classes = lr_pipeline.named_steps["clf"].classes_
        lr_val_proba = reorder_probabilities(lr_classes, lr_val_raw, prefix=f"{timeframe}_lr")
        lr_test_proba = reorder_probabilities(lr_classes, lr_test_raw, prefix=f"{timeframe}_lr")

        metrics_report.append(
            compute_metrics(
                model_name=f"{timeframe}_logistic_regression",
                y_true=y_val,
                proba_df_named=lr_val_proba.rename(columns=lambda c: c.rsplit("_", 1)[-1]),
                n_features=len(feature_columns),
                train_rows=len(X_train),
                val_rows=len(X_val),
                train_seconds=lr_train_seconds,
                predict_seconds=lr_predict_seconds,
                model_params=cfg.LOGISTIC_REGRESSION_PARAMS,
            )
        )
        dump(lr_pipeline, cfg.logistic_regression_model_path(timeframe))
        val_prob_blocks.append(lr_val_proba)
        test_prob_blocks.append(lr_test_proba)
        del lr_pipeline, lr_val_raw, lr_test_raw
        gc.collect()

        # ========================= LightGBM =========================
        t0 = time.time()
        lgbm_model = train_lightgbm_classifier(lgb_module, train_df, X_train, y_train, timeframe)
        lgbm_train_seconds = time.time() - t0

        t0 = time.time()
        lgbm_val_raw = lgbm_model.predict_proba(X_val)
        lgbm_predict_seconds = time.time() - t0
        lgbm_test_raw = lgbm_model.predict_proba(X_test)

        lgbm_classes = lgbm_model.classes_
        lgbm_val_proba = reorder_probabilities(lgbm_classes, lgbm_val_raw, prefix=f"{timeframe}_lgbm")
        lgbm_test_proba = reorder_probabilities(lgbm_classes, lgbm_test_raw, prefix=f"{timeframe}_lgbm")

        metrics_report.append(
            compute_metrics(
                model_name=f"{timeframe}_lightgbm",
                y_true=y_val,
                proba_df_named=lgbm_val_proba.rename(columns=lambda c: c.rsplit("_", 1)[-1]),
                n_features=len(feature_columns),
                train_rows=len(X_train),
                val_rows=len(X_val),
                train_seconds=lgbm_train_seconds,
                predict_seconds=lgbm_predict_seconds,
                model_params={k: v for k, v in cfg.LIGHTGBM_PARAMS.items()},
            )
        )
        lgbm_model.booster_.save_model(str(cfg.lightgbm_model_path(timeframe)))
        val_prob_blocks.append(lgbm_val_proba)
        test_prob_blocks.append(lgbm_test_proba)
        del lgbm_model, lgbm_val_raw, lgbm_test_raw
        gc.collect()

        del train_df, val_df, test_df, X_train, X_val, X_test
        gc.collect()

    # ---------------- assemble validation predictions ----------------
    print("\n--- assembling prediction files " + "-" * 40)
    val_predictions = pd.concat(
        [val_timestamp_ref.rename(cfg.TIMESTAMP_COLUMN), val_target_ref.rename(cfg.TARGET_COLUMN)]
        + [b.reset_index(drop=True) for b in val_prob_blocks],
        axis=1,
    )[expected_prediction_columns()]
    verify_predictions_output(val_predictions, n_val_rows, "validation")
    write_table(val_predictions, cfg.VALIDATION_PREDICTIONS_PATH)
    print(f"  wrote {cfg.VALIDATION_PREDICTIONS_PATH}")

    # ---------------- assemble test predictions (locked) ----------------
    # NOTE: these are stored for the future ensemble/evaluation stage only.
    # No metric on this file has been computed or reported here, and none
    # of it fed back into training, tuning, or model selection above.
    test_predictions = pd.concat(
        [test_timestamp_ref.rename(cfg.TIMESTAMP_COLUMN), test_target_ref.rename(cfg.TARGET_COLUMN)]
        + [b.reset_index(drop=True) for b in test_prob_blocks],
        axis=1,
    )[expected_prediction_columns()]
    verify_predictions_output(test_predictions, n_test_rows, "test")
    write_table(test_predictions, cfg.TEST_PREDICTIONS_PATH)
    print(f"  wrote {cfg.TEST_PREDICTIONS_PATH} (locked -- not evaluated at this stage)")

    # ---------------- metrics report ----------------
    cfg.METRICS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.METRICS_REPORT_PATH, "w") as fh:
        json.dump(metrics_report, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print("Validation-set metrics (2024, computed ONLY on the validation split)")
    print("=" * 78)
    summary = pd.DataFrame(metrics_report)[
        ["model", "macro_f1", "log_loss", "accuracy", "n_features", "train_rows", "train_seconds"]
    ]
    print(summary.to_string(index=False))

    print(f"\nTotal runtime: {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
