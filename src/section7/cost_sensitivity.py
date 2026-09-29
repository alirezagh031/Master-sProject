from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.section6 import ensemble_optimizer as eo
from src.section7 import config as cfg
from src.section7.evaluate_test import assert_price_series_sane, load_price_context

# --------------------------------------------------------------------------
# Illustrative cost scenarios (decimal, per unit of |position change|)
# --------------------------------------------------------------------------
COST_SCENARIOS = {
    "zero_cost": 0.0000,  # reproduces the original baseline exactly
    "low_cost": 0.0001,  # 1 bp   -- illustrative: tight ECN-style spread
    "moderate_cost": 0.0005,  # 5 bps  -- illustrative: typical retail CFD spread+commission
    "high_cost": 0.0020,  # 20 bps -- illustrative: wide spread / adverse slippage
}
# These four values are ILLUSTRATIVE ASSUMPTIONS chosen to span a plausible
# range for a retail XAUUSD CFD/forex account. None are derived from any
# specific broker's published spread or commission schedule -- no such data
# was available to this analysis.

OUTPUT_PATH = cfg.DATA_DIR / "cost_sensitivity_report.json"


def compute_position_changes(position: np.ndarray) -> np.ndarray:
    """|position_t - position_{t-1}|, with position_{-1} := 0 (start flat)."""
    prev = np.concatenate([[0.0], position[:-1].astype(float)])
    return np.abs(position.astype(float) - prev)


def equity_curve_with_costs(position: np.ndarray, forward_return: np.ndarray, cost_rate: float) -> dict:
    """
    Same filtering and compounding convention as evaluate_test's
    equity_curve_metrics (np.isfinite, geometric cumprod, equity starts at
    1.0), extended with a cost deducted from the bar in which the exposure
    change occurs. At cost_rate=0.0 this is mathematically identical to the
    original baseline computation -- verified in main() against the saved
    trading_evaluation_report.json, not merely assumed.
    """
    valid = np.isfinite(forward_return)
    position = position[valid]
    forward_return = forward_return[valid]

    if len(position) == 0:
        return {
            "n_bars_evaluated": 0, "n_directional_bars": 0, "n_position_changes": 0,
            "gross_total_return": 0.0, "total_cost_charged": 0.0, "net_total_return": 0.0,
            "net_ending_equity": 1.0, "max_drawdown_net": 0.0, "win_rate_net": 0.0,
            "sharpe_raw_per_bar_net": 0.0,
        }

    changes = compute_position_changes(position)
    cost = cost_rate * changes

    gross_return = position.astype(float) * forward_return
    net_return = gross_return - cost

    gross_equity = np.cumprod(1.0 + gross_return)
    net_equity = np.cumprod(1.0 + net_return)
    running_max = np.maximum.accumulate(net_equity)
    drawdown = (net_equity - running_max) / running_max

    n_directional_bars = int((position != 0).sum())  # old "n_trades" definition, kept for comparability
    n_position_changes = int((changes > 0).sum())  # genuine entries/exits/reversals
    wins_net = int(((position != 0) & (net_return > 0)).sum())

    return {
        "n_bars_evaluated": int(len(net_return)),
        "n_directional_bars": n_directional_bars,
        "n_position_changes": n_position_changes,
        "gross_total_return": float(gross_equity[-1] - 1.0),
        "total_cost_charged": float(cost.sum()),  # uncompounded sum of per-bar charges
        "net_total_return": float(net_equity[-1] - 1.0),
        "net_ending_equity": float(net_equity[-1]),
        "max_drawdown_net": float(drawdown.min()),
        "win_rate_net": (wins_net / n_directional_bars) if n_directional_bars else 0.0,
        # reuse eo.compute_sharpe unmodified: feeding position=1 makes its
        # internal "position*forward_return" reduce to exactly net_return.
        "sharpe_raw_per_bar_net": eo.compute_sharpe(np.ones(len(net_return)), net_return),
    }


def annualized_sharpe(raw_per_bar_sharpe: float, n_bars: int, span_days: float) -> float:
    """
    raw_per_bar_sharpe * sqrt(bars_per_year), with bars_per_year computed
    EMPIRICALLY from the actual evaluated span rather than assumed from a
    traditional-market trading-day convention (inappropriate for near-24/5
    XAUUSD data). Explicit so the convention is auditable, not hidden.
    """
    if span_days <= 0:
        return 0.0
    bars_per_year = n_bars / (span_days / 365.25)
    return float(raw_per_bar_sharpe * np.sqrt(bars_per_year))


def run_strategy_across_scenarios(position: np.ndarray, forward_return: np.ndarray) -> dict:
    results = {}
    for name, rate in COST_SCENARIOS.items():
        results[name] = equity_curve_with_costs(position, forward_return, rate)
    # SAFEGUARD: a positive cost can never IMPROVE net return relative to
    # zero cost. If it ever does, that is a sign/alignment bug, not a
    # legitimate result -- fail loudly rather than report it.
    zero = results["zero_cost"]["net_total_return"]
    for name, rate in COST_SCENARIOS.items():
        if rate > 0:
            assert results[name]["net_total_return"] <= zero + 1e-9, (
                f"COST SIGN BUG: '{name}' (rate={rate}) produced a net_total_return "
                f"({results[name]['net_total_return']}) higher than zero-cost ({zero}) -- "
                f"a positive transaction cost must never improve returns."
            )
    return results


def main() -> None:
    run_start = time.time()
    print("=" * 78)
    print("Stage 7 - Transaction-Cost Sensitivity Analysis")
    print("(separate from, and does not modify, the original baseline)")
    print("=" * 78)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- reuse the already-saved ensemble predictions (no re-derivation) ----
    ens_df = pd.read_parquet(cfg.TEST_ENSEMBLE_PREDICTIONS_PATH)
    ens_df[cfg.TIMESTAMP_COLUMN] = pd.to_datetime(ens_df[cfg.TIMESTAMP_COLUMN])
    ens_df = ens_df.sort_values(cfg.TIMESTAMP_COLUMN).reset_index(drop=True)
    print(f"loaded {cfg.TEST_ENSEMBLE_PREDICTIONS_PATH.name}: {ens_df.shape}")

    price_ctx = load_price_context()  # reused verbatim from evaluate_test.py
    merged = ens_df.merge(price_ctx, on=cfg.TIMESTAMP_COLUMN, how="left", validate="one_to_one")
    assert len(merged) == len(ens_df), "merge with price context changed row count."
    assert merged["close_x"].notna().all(), "close_x missing for some row(s) -- unexpected."
    assert_price_series_sane(merged)  # reused verbatim -- no reason to relax it here

    position = merged["predicted_class"].to_numpy()
    forward_return = merged["forward_return_1"].to_numpy()

    print(f"\nevaluating {len(COST_SCENARIOS)} cost scenarios: {COST_SCENARIOS}")
    ensemble_results = run_strategy_across_scenarios(position, forward_return)
    buy_hold_results = run_strategy_across_scenarios(np.ones(len(merged)), forward_return)
    flat_results = run_strategy_across_scenarios(np.zeros(len(merged)), forward_return)

    # ---- validate: zero-cost scenario must reproduce the original baseline ----
    print("\n--- reproduction check against the ORIGINAL saved baseline ---")
    try:
        with open(cfg.TRADING_EVALUATION_REPORT_PATH) as fh:
            original = json.load(fh)["overall_trading"]
        zero = ensemble_results["zero_cost"]
        checks = [
            ("total_return", original["total_return"], zero["net_total_return"]),
            ("max_drawdown", original["max_drawdown"], zero["max_drawdown_net"]),
            ("sharpe_raw_per_bar", original["sharpe_raw_per_bar"], zero["sharpe_raw_per_bar_net"]),
        ]
        all_match = True
        for name, orig_val, new_val in checks:
            match = abs(orig_val - new_val) < 1e-6
            all_match &= match
            print(f"  {name}: original={orig_val:.8f}  zero_cost={new_val:.8f}  match={match}")
        print(f"BASELINE REPRODUCED: {all_match}")
        if not all_match:
            print(
                "  WARNING: zero-cost scenario does NOT exactly reproduce the saved "
                "baseline -- do not trust the cost scenarios below until this is resolved."
            )
    except FileNotFoundError:
        print(
            f"  Could not find {cfg.TRADING_EVALUATION_REPORT_PATH} -- run "
            f"evaluate_test.py first to generate the baseline to compare against."
        )
        all_match = None

    # ---- span (days) for the annualization convention, computed empirically ----
    span_days = (merged[cfg.TIMESTAMP_COLUMN].iloc[-1] - merged[cfg.TIMESTAMP_COLUMN].iloc[0]).total_seconds() / 86400.0

    print("\n" + "=" * 78)
    print("GROSS vs NET performance by cost scenario (ensemble)")
    print("=" * 78)
    for name, rate in COST_SCENARIOS.items():
        r = ensemble_results[name]
        ann_sharpe = annualized_sharpe(r["sharpe_raw_per_bar_net"], r["n_bars_evaluated"], span_days)
        print(
            f"[{name:14s}] rate={rate:.4%}  gross_return={r['gross_total_return']:.4f}  "
            f"total_cost={r['total_cost_charged']:.4f}  net_return={r['net_total_return']:.4f}  "
            f"net_equity={r['net_ending_equity']:.4f}  max_dd_net={r['max_drawdown_net']:.4f}  "
            f"sharpe_net(raw/ann)={r['sharpe_raw_per_bar_net']:.4f}/{ann_sharpe:.2f}  "
            f"n_dir_bars={r['n_directional_bars']}  n_pos_changes={r['n_position_changes']}"
        )

    print("\n" + "=" * 78)
    print("Buy-and-hold under the SAME cost scenarios (for comparison)")
    print("=" * 78)
    for name, rate in COST_SCENARIOS.items():
        r = buy_hold_results[name]
        print(
            f"[{name:14s}] rate={rate:.4%}  gross_return={r['gross_total_return']:.4f}  "
            f"total_cost={r['total_cost_charged']:.6f}  net_return={r['net_total_return']:.4f}  "
            f"n_pos_changes={r['n_position_changes']}  (buy-and-hold trades ONCE, at entry)"
        )

    report = {
        "cost_scenarios_decimal": COST_SCENARIOS,
        "cost_model": "cost_t = cost_rate * |position_t - position_{t-1}|, position_{-1}=0; "
                      "net_return_t = position_t * forward_return_1_t - cost_t",
        "cost_model_note": "cost_rate is a single combined spread+slippage+commission rate; "
                            "all scenario values are illustrative, not observed broker data",
        "baseline_reproduced": all_match,
        "span_days_evaluated": span_days,
        "ensemble": ensemble_results,
        "buy_and_hold": buy_hold_results,
        "flat": flat_results,
    }
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {OUTPUT_PATH}")
    print(f"Total runtime: {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
