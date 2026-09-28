from __future__ import annotations

from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# This file lives at <project_root>/src/section5/config.py, so:
#   parents[0] = src/section5
#   parents[1] = src
#   parents[2] = <project_root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SECTION5_DATA_DIR = DATA_PROCESSED_DIR / "section5"
PREDICTIONS_DIR = SECTION5_DATA_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models" / "section5"

TRAIN_M15_PATH = SECTION5_DATA_DIR / "train_m15.parquet"
TRAIN_H1_PATH = SECTION5_DATA_DIR / "train_h1.parquet"
TRAIN_D1_PATH = SECTION5_DATA_DIR / "train_d1.parquet"

TRAIN_DATASET_PATHS = {
    "m15": TRAIN_M15_PATH,
    "h1": TRAIN_H1_PATH,
    "d1": TRAIN_D1_PATH,
}

VALIDATION_PREDICTIONS_PATH = PREDICTIONS_DIR / "validation_base_predictions.parquet"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_base_predictions.parquet"
METRICS_REPORT_PATH = SECTION5_DATA_DIR / "base_learner_metrics_report.json"

# --------------------------------------------------------------------------
# Temporal split (fixed - must match the Stage 4 leakage-control design)
# --------------------------------------------------------------------------
# NOTE: train_base_learners.py will try to import an existing split utility
# from src.section4 first (see `resolve_split_boundaries()`), and only falls
# back to these hardcoded boundaries if no such utility can be found. Do not
# change these values without re-checking Stage 4's temporal split logic -
# they must describe the exact same boundaries.
TRAIN_START = "2018-01-01"
TRAIN_END = "2023-12-31 23:59:59"
VALIDATION_START = "2024-01-01"
VALIDATION_END = "2024-12-31 23:59:59"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31 23:59:59"

# Expected row counts, from the already-completed Stage 5 dataset-separation
# step (188,632 aligned rows total). Used only as a soft sanity check right
# after splitting -- a mismatch prints a warning, it does not stop the run,
# since it may just mean the underlying parquet files were regenerated.
EXPECTED_TRAIN_ROWS = 141_366
EXPECTED_VALIDATION_ROWS = 23_738
EXPECTED_TEST_ROWS = 23_528

# An inner slice of the TRAIN period ONLY, used exclusively to early-stop
# LightGBM. This never touches 2024 (validation) or 2025 (test) data, so
# model-selection stays leakage-safe.
INNER_EARLY_STOP_START = "2023-01-01"

# --------------------------------------------------------------------------
# Target encoding (verify against the real data before trusting this)
# --------------------------------------------------------------------------
CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP = -1, 0, 1
EXPECTED_CLASSES = {CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP}
CLASS_ORDER = [CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP]          # DOWN, NEUTRAL, UP
CLASS_NAMES = {CLASS_DOWN: "down", CLASS_NEUTRAL: "neutral", CLASS_UP: "up"}
PROBA_COLUMN_ORDER = ["down", "neutral", "up"]

TARGET_COLUMN = "target"
TIMESTAMP_COLUMN = "timestamp"

# Columns that would leak future information into a feature matrix.
LEAKAGE_COLUMNS = {
    "target_label",
    "future_close",
    "future_return",
}

# Internal per-bar timestamp columns discovered in the real H1/D1 datasets
# (confirmed dtype: datetime64). These describe *when* a partial H1/D1 bar
# started, not a market observation -- they must be excluded from the
# feature matrix exactly like TIMESTAMP_COLUMN itself, and NEVER encoded
# (not even as a Unix timestamp / ordinal). h1_partial_period was confirmed
# datetime64[us] from a real Stage 5 run; d1_partial_period is excluded
# pre-emptively by the same construction even though it has not been
# independently confirmed against the real d1 parquet file.
INTERNAL_TIMESTAMP_COLUMNS = {
    "h1_partial_period",
    "d1_partial_period",
}

# Columns that must NEVER be used as model features, for any reason.
FORBIDDEN_FEATURE_COLUMNS = (
    {TARGET_COLUMN, TIMESTAMP_COLUMN} | LEAKAGE_COLUMNS | INTERNAL_TIMESTAMP_COLUMNS
)

TIMEFRAMES = ("m15", "h1", "d1")
MODEL_KINDS = ("lr", "lgbm")

# --------------------------------------------------------------------------
# Partial-bar structural NaN (confirmed via inspect_partial_nan.py against
# the real data on 2026-09-27): *_partial_* is NaN at exactly, and only,
# the first M15 bar of a new H1 hour / D1 day -- 100% coincidence with that
# boundary, 0% elsewhere, completed_* stays valid throughout. This is an
# expected "no partial bar observed yet" state, not corrupted data, so it
# is imputed (train-only fit) rather than dropped or hard-failed on. Empty
# for m15, which has no missing values.
# --------------------------------------------------------------------------
PARTIAL_NAN_COLUMNS: dict[str, list[str]] = {
    "m15": [],
    "h1": ["h1_partial_open", "h1_partial_high", "h1_partial_low", "h1_partial_close", "h1_partial_tickvol"],
    "d1": ["d1_partial_open", "d1_partial_high", "d1_partial_low", "d1_partial_close", "d1_partial_tickvol"],
}
PARTIAL_AVAILABILITY_COLUMN: dict[str, Optional[str]] = {
    "m15": None,
    "h1": "h1_partial_available",
    "d1": "d1_partial_available",
}
PARTIAL_IMPUTE_STRATEGY = "median"

# --------------------------------------------------------------------------
# Reproducibility / resource limits (MacBook Air M1, 8 GB RAM)
# --------------------------------------------------------------------------
RANDOM_STATE = 42
N_JOBS = 2  # conservative - six models train sequentially anyway, so this
            # only bounds how many threads LightGBM itself may use.

# --------------------------------------------------------------------------
# Logistic Regression
# --------------------------------------------------------------------------
LOGISTIC_REGRESSION_PARAMS = dict(
    # penalty is left at its scikit-learn default (L2) deliberately: newer
    # sklearn releases deprecate passing penalty="l2" explicitly in favor of
    # l1_ratio/C, while older releases expect it spelled out -- omitting it
    # keeps this code correct across both.
    C=1.0,
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",   # do not assume balanced DOWN/NEUTRAL/UP classes
    random_state=RANDOM_STATE,
)

# --------------------------------------------------------------------------
# LightGBM (conservative, CPU-only, early-stopping enabled)
# --------------------------------------------------------------------------
LIGHTGBM_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    boosting_type="gbdt",
    n_estimators=600,          # upper bound; early stopping cuts this down
    num_leaves=15,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbosity=-1,
)
LIGHTGBM_EARLY_STOPPING_ROUNDS = 30

# --------------------------------------------------------------------------
# Missing-value handling policy
# --------------------------------------------------------------------------
# If the fraction of TRAIN rows containing NaN/inf is below this threshold,
# they are dropped automatically (documented cause: warm-up period before
# the first completed H1/D1 bar exists). Above this threshold, the run
# stops loudly instead of guessing. VALIDATION/TEST rows are never dropped
# (see assert_no_missing_values in train_base_learners.py) because doing so
# would silently break row-count/timestamp alignment across the six models.
MAX_AUTO_DROP_FRACTION = 0.02


def logistic_regression_model_path(timeframe: str) -> Path:
    return MODELS_DIR / f"{timeframe}_logistic_regression.joblib"


def lightgbm_model_path(timeframe: str) -> Path:
    return MODELS_DIR / f"{timeframe}_lightgbm.txt"
