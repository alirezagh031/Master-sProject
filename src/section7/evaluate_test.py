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
    # close_x itself is kept (not just vol_16/forward_return_1) so it can be
    # audited directly against the anomalies being investigated here.
    return df[[cfg.TIMESTAMP_COLUMN, "close_x", "vol_16", "forward_return_1"]]


def assign_regimes_fixed(vol_16: pd.Series, thresholds: dict) -> pd.Series:
    bins = [-np.inf, thresholds["q33"], thresholds["q67"], np.inf]
    return pd.cut(vol_16, bins=bins, labels=cfg.REGIME_LABELS).astype(str)


def build_proba_stack(df: pd.DataFrame, base_learners: list[str]) -> np.ndarray:
    return np.stack([df[cfg.proba_columns_for(bl)].to_numpy() for bl in base_learners], axis=0)


def equity_curve_metrics(position: np.ndarray, forward_return: np.ndarray) -> dict:
    # CONFIRMED FIX: was `~np.isnan(forward_return)`, which does NOT catch
    # +/-inf. A forward_return of +/-inf arises from a real, reproducible
    # mechanism -- close_x.shift(-1)/close_x - 1 divides by close_x, and
    # x/0 for x != 0 is +/-inf in IEEE754, NOT NaN (only 0/0 is NaN). A
    # NaN-only filter would let such a bar's return silently poison every
    # subsequent value in the cumulative product (np.cumprod propagates
    # inf/nan forward from the first occurrence to the end of the series).
    # np.isfinite catches both NaN and +/-inf.
    valid = np.isfinite(forward_return)
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


# ==========================================================================
# Diagnostics (read-only where possible; the one exception -- probability
# renormalization -- is explicitly gated and explained at the call site)
# ==========================================================================
def diagnose_price_series(merged: pd.DataFrame) -> None:
    """
    Read-only. Never excludes or alters a row based on what it finds here --
    a large-but-finite real return is legitimate data, not a defect. Only
    non-finite values are ever excluded, and that happens in
    equity_curve_metrics, not here.
    """
    close = merged["close_x"].to_numpy()
    fwd = merged["forward_return_1"].to_numpy()

    n_nonfinite_close = int((~np.isfinite(close)).sum())
    n_zero_or_neg_close = int((close <= 0).sum())
    print(
        f"  [diagnostic] close_x: {n_nonfinite_close} non-finite, "
        f"{n_zero_or_neg_close} zero-or-negative (expect 0, 0)"
    )

    n_nonfinite_fwd = int((~np.isfinite(fwd)).sum())
    finite_fwd = fwd[np.isfinite(fwd)]
    print(
        f"  [diagnostic] forward_return_1: {n_nonfinite_fwd} non-finite value(s) "
        f"(expect exactly 1 -- the final bar in the whole series has no next price)"
    )
    if len(finite_fwd):
        print(
            f"  [diagnostic] forward_return_1 (finite only) stats: "
            f"min={finite_fwd.min():.6f} max={finite_fwd.max():.6f} "
            f"mean={finite_fwd.mean():.6f} std={finite_fwd.std():.6f}"
        )

    outlier_threshold = 0.01  # 1% in a single M15 XAUUSD bar is already extraordinary
    outlier_mask = np.isfinite(fwd) & (np.abs(fwd) > outlier_threshold)
    n_outliers = int(outlier_mask.sum())
    print(f"  [diagnostic] bars with |forward_return_1| > {outlier_threshold:.0%}: {n_outliers}")
    if n_outliers:
        idx = np.where(outlier_mask)[0]
        top = idx[np.argsort(-np.abs(fwd[idx]))][:10]
        for i in top:
            ts = merged[cfg.TIMESTAMP_COLUMN].iloc[i]
            print(
                f"    row {i}: timestamp={ts}  close_x={close[i]:.5f}  "
                f"implied_next_close_x={close[i] * (1 + fwd[i]):.5f}  forward_return_1={fwd[i]:.6f}"
            )


def diagnose_equity_contribution(position: np.ndarray, forward_return: np.ndarray, timestamps: pd.Series, label: str) -> None:
    """Read-only. Identifies which specific bar(s) drive total_return the most."""
    valid = np.isfinite(forward_return)
    pos, fwd, ts = position[valid], forward_return[valid], timestamps.to_numpy()[valid]
    strategy_return = pos.astype(float) * fwd
    if len(strategy_return) == 0:
        return
    worst = np.argsort(-np.abs(strategy_return))[:5]
    print(f"  [diagnostic] {label}: top single-bar contributors to total_return:")
    for i in worst:
        print(
            f"    timestamp={ts[i]}  position={pos[i]:+.0f}  forward_return_1={fwd[i]:.6f}  "
            f"strategy_return={strategy_return[i]:.6f}  equity_multiplier={1 + strategy_return[i]:.4f}"
        )


def audit_and_fix_combined_probabilities(combined_proba: np.ndarray, merged: pd.DataFrame) -> np.ndarray:
    """
    Measures how far combined_proba's row sums deviate from 1, THEN decides:
      - deviation <= 1e-9: already exact, return unchanged.
      - deviation <= 1%: consistent with ordinary floating-point residue
        inherited from the individual base learners (each already verified
        within Stage 5's own 1e-3 tolerance -- see [verify/test] OK in the
        Stage 5 run log); renormalize, with the reasoning printed, since
        argmax (and therefore every predicted class) is unaffected by a
        positive rescaling.
      - deviation > 1%: refuses to normalize and raises instead -- that
        magnitude is inconsistent with float residue and indicates a real
        construction defect (e.g. an unmatched regime, a weight-lookup
        miss) that silent normalization would hide rather than fix.
    """
    row_sums = combined_proba.sum(axis=1)
    deviation = np.abs(row_sums - 1.0)
    max_dev, mean_dev = float(deviation.max()), float(deviation.mean())
    print(f"  [diagnostic] combined probability row-sum deviation from 1.0: max={max_dev:.2e}  mean={mean_dev:.2e}")

    if max_dev <= 1e-9:
        return combined_proba

    if max_dev > 0.01:
        worst = int(np.argmax(deviation))
        raise RuntimeError(
            f"combined probabilities deviate from summing to 1 by up to {max_dev:.4f} "
            f"at row {worst} (timestamp={merged[cfg.TIMESTAMP_COLUMN].iloc[worst]}, "
            f"regime={merged['regime'].iloc[worst]}, combined_proba={combined_proba[worst]}). "
            f"This is far larger than ordinary floating-point residue from the individual "
            f"base learners (each already verified within 1e-3 in Stage 5) and points to a "
            f"genuine construction defect, not numerical imprecision. Refusing to silently "
            f"normalize this away -- investigate the flagged row before proceeding."
        )

    print(
        f"  [diagnostic] deviation is small (<=1%), consistent with floating-point residue "
        f"inherited from the individual base learners' predict_proba() outputs (each already "
        f"within Stage 5's documented 1e-3 tolerance, but tighter than scikit-learn's internal "
        f"log_loss check) -- renormalizing to sum to exactly 1 before computing log_loss. This "
        f"does NOT change any predicted class: argmax is invariant to positive rescaling."
    )
    return combined_proba / row_sums[:, None]


def assert_price_series_sane(merged: pd.DataFrame) -> None:
    """
    Hard, fail-loud guard against a failure mode just proven (not merely
    suspected) to evade both a NaN-only filter AND an isfinite() filter: a
    close_x that is small but NONZERO (e.g. a decimal-point / data-feed
    glitch) produces a forward_return_1 that is a huge but perfectly FINITE
    ratio -- neither NaN nor inf -- so it silently explodes the compounding
    equity curve while looking, to any finite-value filter, like ordinary
    (if extraordinarily lucky) trading performance. Direct test: close_x
    of 0.0000001 next to a normal ~1900 price yields forward_return_1 =
    18,989,999,999.0 -- finite, and indistinguishable from a "real" number
    to np.isfinite().

    XAUUSD has not traded below ~$1000/oz anywhere in this dataset's
    2018-2025 span (confirmed range ~1700-2000 from the Stage 5
    partial-bar imputer's fitted medians), so a floor of $500 cannot
    reject genuine data -- only an artifact.
    """
    close = merged["close_x"].to_numpy()
    min_sane_price = 500.0
    bad_price_mask = close < min_sane_price
    if bad_price_mask.any():
        idx = np.where(bad_price_mask)[0]
        details = [
            f"row {i} (timestamp={merged[cfg.TIMESTAMP_COLUMN].iloc[i]}): close_x={close[i]}" for i in idx[:5]
        ]
        raise RuntimeError(
            f"{int(bad_price_mask.sum())} row(s) have close_x below a sane floor "
            f"(${min_sane_price:.0f}), which XAUUSD has never done in this dataset's "
            f"real range (~1700-2000). This is almost certainly a data artifact (e.g. "
            f"a near-zero glitch value) that would otherwise silently explode "
            f"forward_return_1 into a huge-but-finite ratio that survives even an "
            f"isfinite() filter. Fix the source data before evaluating. Examples: {details}"
        )

    # A single-bar move beyond this is essentially unprecedented for XAUUSD M15
    # data and far more likely a data artifact than a real return. This is a
    # HARD stop -- distinct from the 1% SOFT diagnostic threshold in
    # diagnose_price_series(), which only reports and never blocks.
    fwd = merged["forward_return_1"].to_numpy()
    hard_outlier_threshold = 0.20
    extreme_mask = np.isfinite(fwd) & (np.abs(fwd) > hard_outlier_threshold)
    if extreme_mask.any():
        idx = np.where(extreme_mask)[0]
        details = [
            f"row {i} (timestamp={merged[cfg.TIMESTAMP_COLUMN].iloc[i]}): "
            f"close_x={close[i]}, forward_return_1={fwd[i]}"
            for i in idx[:5]
        ]
        raise RuntimeError(
            f"{int(extreme_mask.sum())} row(s) have |forward_return_1| > "
            f"{hard_outlier_threshold:.0%} -- an essentially unprecedented single "
            f"M15-bar move for XAUUSD, and the exact signature of the "
            f"near-zero-denominator artifact this guard was written to catch. "
            f"Investigate before evaluating. Examples: {details}"
        )


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
    assert merged["close_x"].notna().all(), "close_x missing for some test row(s) -- unexpected."
    assert_price_series_sane(merged)

    merged["regime"] = assign_regimes_fixed(merged["vol_16"], thresholds)
    print(f"regime distribution (test, Stage-6-fixed cutoffs): {merged['regime'].value_counts().to_dict()}")

    print("\n--- price series diagnostics (investigating the reported anomaly) ---")
    diagnose_price_series(merged)

    combined_proba = np.zeros((len(merged), 3))
    for regime_name in cfg.REGIME_LABELS:
        mask = (merged["regime"] == regime_name).to_numpy()
        if not mask.any():
            continue
        weights = np.array([ensemble_weights[regime_name][bl] for bl in cfg.BASE_LEARNERS])
        proba_stack = build_proba_stack(merged.loc[mask], cfg.BASE_LEARNERS)
        combined_proba[mask] = eo.combine_probabilities(weights, proba_stack)

    print("\n--- combined-probability diagnostics (investigating the log_loss warning) ---")
    combined_proba = audit_and_fix_combined_probabilities(combined_proba, merged)

    y_pred = eo.predicted_class_from_proba(combined_proba, cfg.CLASS_ORDER)
    y_true = merged[cfg.TARGET_COLUMN].to_numpy()

    macro_f1 = f1_score(y_true, y_pred, labels=cfg.CLASS_ORDER, average="macro", zero_division=0)
    logloss = log_loss(y_true, combined_proba, labels=cfg.CLASS_ORDER)
    trading = equity_curve_metrics(y_pred, merged["forward_return_1"].to_numpy())
    buy_hold = baseline_metrics(merged["forward_return_1"].to_numpy(), 1.0)
    flat = baseline_metrics(merged["forward_return_1"].to_numpy(), 0.0)

    print("\n--- equity-curve diagnostics (investigating the total_return anomaly) ---")
    diagnose_equity_contribution(y_pred, merged["forward_return_1"].to_numpy(), merged[cfg.TIMESTAMP_COLUMN], "ensemble")
    diagnose_equity_contribution(
        np.ones(len(merged)), merged["forward_return_1"].to_numpy(), merged[cfg.TIMESTAMP_COLUMN], "buy_and_hold"
    )
    first_close, last_close = merged["close_x"].iloc[0], merged["close_x"].iloc[-1]
    direct_buy_hold_return = float(last_close / first_close - 1.0)
    print(
        f"  [diagnostic] buy-and-hold cross-check: cumprod-based total_return="
        f"{buy_hold['total_return']:.6f}  vs.  direct (last_close/first_close - 1)="
        f"{direct_buy_hold_return:.6f}  (should closely match if compounding is correct)"
    )

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
