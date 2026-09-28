from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from src.section6 import ensemble_optimizer as eo
from src.section7 import config as cfg


def load_price_context() -> pd.DataFrame:
    df = pd.read_parquet(cfg.TRAIN_M15_PATH)[[cfg.TIMESTAMP_COLUMN, "close_x"]]
    df[cfg.TIMESTAMP_COLUMN] = pd.to_datetime(df[cfg.TIMESTAMP_COLUMN])
    df = df.sort_values(cfg.TIMESTAMP_COLUMN).reset_index(drop=True)
    df["m15_return"] = df["close_x"].pct_change()
    df["vol_16"] = df["m15_return"].rolling(cfg.VOLATILITY_WINDOW, min_periods=cfg.VOLATILITY_WINDOW).std()
    df["forward_return_1"] = df["close_x"].shift(-1) / df["close_x"] - 1.0
    return df[[cfg.TIMESTAMP_COLUMN, "vol_16", "forward_return_1"]]


def assign_regimes_fixed(vol_16: pd.Series, thresholds: dict) -> pd.Series:
    bins = [-np.inf, thresholds["q33"], thresholds["q67"], np.inf]
    return pd.cut(vol_16, bins=bins, labels=cfg.REGIME_LABELS).astype(str)


def build_proba_stack(df: pd.DataFrame, base_learners: list[str]) -> np.ndarray:
    return np.stack([df[cfg.proba_columns_for(bl)].to_numpy() for bl in base_learners], axis=0)


def equity_curve_metrics(position: np.ndarray, forward_return: np.ndarray) -> dict:
    valid = ~np.isnan(forward_return)
    position, forward_return = position[valid], forward_return[valid]
    strategy_return = position.astype(float) * forward_return

    if len(strategy_return) == 0:
        return {"n_bars_evaluated": 0, "n_trades": 0, "win_rate": 0.0, "total_return": 0.0,
                "max_drawdown": 0.0, "sharpe_raw_per_bar": 0.0}

    equity = np.cumprod(1.0 + strategy_return)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max

    n_trades = int((position != 0).sum())
    wins = int(((position != 0) & (strategy_return > 0)).sum())

    return {
        "n_bars_evaluated": int(len(strategy_return)),
        "n_trades": n_trades,
        "win_rate": (wins / n_trades) if n_trades else 0.0,
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe_raw_per_bar": eo.compute_sharpe(position, forward_return),
    }


def baseline_metrics(forward_return: np.ndarray, position_value: float) -> dict:
    return equity_curve_metrics(np.full(len(forward_return), position_value), forward_return)


def main() -> None:
    run_start = time.time()
    print("=" * 78)
    print("Stage 7 - Locked Test Set Evaluation (Trading Evaluation)")
    print("*** ONLY run in this project that opens the 2025 test set ***")
    print("=" * 78)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(cfg.ENSEMBLE_WEIGHTS_PATH) as fh:
        ensemble_weights = json.load(fh)
    with open(cfg.REGIME_THRESHOLDS_PATH) as fh:
        thresholds = json.load(fh)
    print(f"loaded FIXED artifacts (not refit): {cfg.ENSEMBLE_WEIGHTS_PATH.name}, {cfg.REGIME_THRESHOLDS_PATH.name}")

    test_df = pd.read_parquet(cfg.TEST_PREDICTIONS_PATH)
    test_df[cfg.TIMESTAMP_COLUMN] = pd.to_datetime(test_df[cfg.TIMESTAMP_COLUMN])
    test_df = test_df.sort_values(cfg.TIMESTAMP_COLUMN).reset_index(drop=True)
    print(f"loaded test predictions: {test_df.shape}")

    price_ctx = load_price_context()
    merged = test_df.merge(price_ctx, on=cfg.TIMESTAMP_COLUMN, how="left", validate="one_to_one")
    assert len(merged) == len(test_df), "merge with price context changed row count."
    assert merged["vol_16"].notna().all(), "vol_16 missing for some test row(s) -- unexpected."

    merged["regime"] = assign_regimes_fixed(merged["vol_16"], thresholds)
    print(f"regime distribution (test, Stage-6-fixed cutoffs): {merged['regime'].value_counts().to_dict()}")

    combined_proba = np.zeros((len(merged), 3))
    for regime_name in cfg.REGIME_LABELS:
        mask = (merged["regime"] == regime_name).to_numpy()
        if not mask.any():
            continue
        weights = np.array([ensemble_weights[regime_name][bl] for bl in cfg.BASE_LEARNERS])
        proba_stack = build_proba_stack(merged.loc[mask], cfg.BASE_LEARNERS)
        combined_proba[mask] = eo.combine_probabilities(weights, proba_stack)

    y_pred = eo.predicted_class_from_proba(combined_proba, cfg.CLASS_ORDER)
    y_true = merged[cfg.TARGET_COLUMN].to_numpy()

    macro_f1 = f1_score(y_true, y_pred, labels=cfg.CLASS_ORDER, average="macro", zero_division=0)
    logloss = log_loss(y_true, combined_proba, labels=cfg.CLASS_ORDER)
    trading = equity_curve_metrics(y_pred, merged["forward_return_1"].to_numpy())
    buy_hold = baseline_metrics(merged["forward_return_1"].to_numpy(), 1.0)
    flat = baseline_metrics(merged["forward_return_1"].to_numpy(), 0.0)

    per_regime = []
    for regime_name in cfg.REGIME_LABELS:
        mask = (merged["regime"] == regime_name).to_numpy()
        if not mask.any():
            continue
        r_f1 = f1_score(y_true[mask], y_pred[mask], labels=cfg.CLASS_ORDER, average="macro", zero_division=0)
        r_trading = equity_curve_metrics(y_pred[mask], merged.loc[mask, "forward_return_1"].to_numpy())
        per_regime.append({"regime": regime_name, "n_rows": int(mask.sum()), "macro_f1": float(r_f1), **r_trading})

    print("\n" + "=" * 78)
    print("FINAL LOCKED TEST-SET RESULTS (2025)")
    print("=" * 78)
    print(f"macro_f1={macro_f1:.4f}  log_loss={logloss:.4f}")
    print(f"ensemble trading:  {trading}")
    print(f"buy_and_hold:      {buy_hold}")
    print(f"flat (no trading): {flat}")
    for r in per_regime:
        print(f"  [{r['regime']}] n={r['n_rows']} macro_f1={r['macro_f1']:.4f} sharpe={r['sharpe_raw_per_bar']:.4f}")

    output_df = pd.DataFrame(
        {
            cfg.TIMESTAMP_COLUMN: merged[cfg.TIMESTAMP_COLUMN],
            cfg.TARGET_COLUMN: merged[cfg.TARGET_COLUMN],
            "regime": merged["regime"],
            "ensemble_down": combined_proba[:, 0],
            "ensemble_neutral": combined_proba[:, 1],
            "ensemble_up": combined_proba[:, 2],
            "predicted_class": y_pred,
        }
    )
    output_df.to_parquet(cfg.TEST_ENSEMBLE_PREDICTIONS_PATH, index=False)
    print(f"\nwrote {cfg.TEST_ENSEMBLE_PREDICTIONS_PATH}")

    report = {
        "overall_macro_f1": float(macro_f1),
        "overall_log_loss": float(logloss),
        "overall_trading": trading,
        "buy_and_hold_baseline": buy_hold,
        "flat_baseline": flat,
        "per_regime": per_regime,
    }
    with open(cfg.TRADING_EVALUATION_REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {cfg.TRADING_EVALUATION_REPORT_PATH}")
    print(f"\nTotal runtime: {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
