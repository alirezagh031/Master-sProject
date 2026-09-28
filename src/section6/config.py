
from __future__ import annotations

from pathlib import Path

from src.section5 import config as s5cfg

# --------------------------------------------------------------------------
# Paths (reusing Stage 5's project root and directory layout)
# --------------------------------------------------------------------------
PROJECT_ROOT = s5cfg.PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models" / "section6"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "section6"

# Inputs -- reused directly from Stage 5, never redefined/duplicated.
VALIDATION_PREDICTIONS_PATH = s5cfg.VALIDATION_PREDICTIONS_PATH
TEST_PREDICTIONS_PATH = s5cfg.TEST_PREDICTIONS_PATH  # NEVER opened by Stage 6
TRAIN_M15_PATH = s5cfg.TRAIN_M15_PATH  # source of close_x for regime + forward return

# Outputs
ENSEMBLE_WEIGHTS_PATH = MODELS_DIR / "ensemble_weights.json"
REGIME_THRESHOLDS_PATH = MODELS_DIR / "regime_thresholds.json"
PARETO_FRONTS_PATH = DATA_DIR / "pareto_fronts.json"
VALIDATION_ENSEMBLE_PREDICTIONS_PATH = DATA_DIR / "validation_ensemble_predictions.parquet"
METRICS_REPORT_PATH = DATA_DIR / "ensemble_metrics_report.json"

# --------------------------------------------------------------------------
# Reused from Stage 5 (never redefined)
# --------------------------------------------------------------------------
TIMESTAMP_COLUMN = s5cfg.TIMESTAMP_COLUMN
TARGET_COLUMN = s5cfg.TARGET_COLUMN
CLASS_ORDER = s5cfg.CLASS_ORDER              # [-1, 0, 1] = [DOWN, NEUTRAL, UP]
PROBA_COLUMN_ORDER = s5cfg.PROBA_COLUMN_ORDER  # ["down", "neutral", "up"]
TIMEFRAMES = s5cfg.TIMEFRAMES                # ("m15", "h1", "d1")
MODEL_KINDS = s5cfg.MODEL_KINDS              # ("lr", "lgbm")
VALIDATION_START = s5cfg.VALIDATION_START
VALIDATION_END = s5cfg.VALIDATION_END
RANDOM_STATE = s5cfg.RANDOM_STATE

# The 6 base learners, in a fixed order -- this defines what each of the 6
# ensemble weight genes means. Never re-ordered once optimization starts.
BASE_LEARNERS = [f"{tf}_{model}" for tf in TIMEFRAMES for model in MODEL_KINDS]
N_BASE_LEARNERS = len(BASE_LEARNERS)  # 6


def proba_columns_for(base_learner: str) -> list[str]:
    """e.g. 'h1_lgbm' -> ['h1_lgbm_down', 'h1_lgbm_neutral', 'h1_lgbm_up']"""
    return [f"{base_learner}_{c}" for c in PROBA_COLUMN_ORDER]


# --------------------------------------------------------------------------
# Regime detection (causal, fit on VALIDATION only -- see train_ensemble.py)
# --------------------------------------------------------------------------
VOLATILITY_WINDOW = 16          # bars, trailing rolling std of M15 returns
N_REGIMES = 3
REGIME_LABELS = ["low_vol", "medium_vol", "high_vol"]

# --------------------------------------------------------------------------
# Multi-objective optimization (NSGA-II), kept small for an 8GB M1
# --------------------------------------------------------------------------
POPULATION_SIZE = 40
N_GENERATIONS = 60
CROSSOVER_ETA = 15.0     # SBX distribution index
MUTATION_ETA = 20.0      # polynomial mutation distribution index
MUTATION_PROB = 1.0 / N_BASE_LEARNERS
TOURNAMENT_SIZE = 2

OBJECTIVE_NAMES = ["macro_f1", "sharpe"]
