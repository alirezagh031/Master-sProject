from __future__ import annotations

from src.section5 import config as s5cfg
from src.section6 import config as s6cfg

PROJECT_ROOT = s5cfg.PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "section7"

TEST_PREDICTIONS_PATH = s5cfg.TEST_PREDICTIONS_PATH  # opened here, and ONLY here
TRAIN_M15_PATH = s5cfg.TRAIN_M15_PATH
ENSEMBLE_WEIGHTS_PATH = s6cfg.ENSEMBLE_WEIGHTS_PATH      # loaded, never refit
REGIME_THRESHOLDS_PATH = s6cfg.REGIME_THRESHOLDS_PATH    # loaded, never refit

TEST_ENSEMBLE_PREDICTIONS_PATH = DATA_DIR / "test_ensemble_predictions.parquet"
TRADING_EVALUATION_REPORT_PATH = DATA_DIR / "trading_evaluation_report.json"

TIMESTAMP_COLUMN = s5cfg.TIMESTAMP_COLUMN
TARGET_COLUMN = s5cfg.TARGET_COLUMN
CLASS_ORDER = s5cfg.CLASS_ORDER
PROBA_COLUMN_ORDER = s5cfg.PROBA_COLUMN_ORDER
BASE_LEARNERS = s6cfg.BASE_LEARNERS
REGIME_LABELS = s6cfg.REGIME_LABELS
VOLATILITY_WINDOW = s6cfg.VOLATILITY_WINDOW


def proba_columns_for(base_learner: str) -> list[str]:
    return s6cfg.proba_columns_for(base_learner)
