from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.section6 import config as cfg
from src.section6 import ensemble_optimizer as eo


# ==========================================================================
# Price context: causal volatility regime + forward return
# ==========================================================================
def load_price_context() -> pd.DataFrame:
    """
    Loads the FULL train_m15.parquet close price series (2018-2025) to
    compute a causal rolling volatility and a 1-bar-forward return. The
    full series is needed only so the rolling window has correct trailing
    context right at the 2024 boundary (late-2023 bars) -- no value derived
    from a 2025 (test) row is ever used except where explicitly excluded
    below.
    """
    df = pd.read_parquet(cfg.TRAIN_M15_PATH)[[cfg.TIMESTAMP_COLUMN, "close_x"]]
    df[cfg.TIMESTAMP_COLUMN] = pd.to_datetime(df[cfg.TIMESTAMP_COLUMN])
    df = df.sort_values(cfg.TIMESTAMP_COLUMN).reset_index(drop=True)
    df["m15_return"] = df["close_x"].pct_change()
    df["vol_16"] = df["m15_return"].rolling(cfg.VOLATILITY_WINDOW, min_periods=cfg.VOLATILITY_WINDOW).std()
    df["next_timestamp"] = df[cfg.TIMESTAMP_COLUMN].shift(-1)
    df["forward_return_1"] = df["close_x"].shift(-1) / df["close_x"] - 1.0
    return df[[cfg.TIMESTAMP_COLUMN, "vol_16", "forward_return_1", "next_timestamp"]]


def attach_price_context(val_df: pd.DataFrame) -> pd.DataFrame:
    price_ctx = load_price_context()
    merged = val_df.merge(price_ctx, on=cfg.TIMESTAMP_COLUMN, how="left", validate="one_to_one")
    assert len(merged) == len(val_df), (
        "merge with price context changed row count -- validation timestamps "
        "are not fully covered by train_m15.parquet."
    )
    assert merged["vol_16"].notna().all(), (
        "vol_16 has NaN for some validation row(s) -- unexpected, since "
        f"validation (2024+) has {cfg.VOLATILITY_WINDOW}+ bars of trailing "
        "history from 2018-2023 available."
    )

    # exclude, from the SHARPE objective only, any row whose "next bar" for
    # forward_return_1 would fall in the locked test period.
    validation_end_ts = pd.Timestamp(cfg.VALIDATION_END)
    crosses_into_test = merged["next_timestamp"] > validation_end_ts
    n_excluded = int(crosses_into_test.sum())
    if n_excluded:
        print(
            f"  excluding {n_excluded} row(s) from the Sharpe objective only "
            f"(their next M15 bar falls outside the validation period, i.e. "
            f"in the locked test set) -- still included normally in macro-F1."
        )
        merged.loc[crosses_into_test, "forward_return_1"] = np.nan

    # structural guarantee: this function never reads or needs anything
    # beyond VALIDATION_END for any row actually used.
    assert merged[cfg.TIMESTAMP_COLUMN].max() <= validation_end_ts, (
        "a row past VALIDATION_END slipped into the ensemble input -- this "
        "must never happen; the 2025 test set must stay locked."
    )
    return merged


def assign_regimes(vol_16: pd.Series) -> tuple[pd.Series, dict]:
    """Tertile cutoffs fit ONLY on these (validation) vol_16 values."""
    q1, q2 = vol_16.quantile([1 / 3, 2 / 3])
    thresholds = {"q33": float(q1), "q67": float(q2)}
    regime = pd.cut(vol_16, bins=[-np.inf, q1, q2, np.inf], labels=cfg.REGIME_LABELS)
    return regime.astype(str), thresholds


# ==========================================================================
# Per-regime optimization
# ==========================================================================
def build_proba_stack(df: pd.DataFrame, base_learners: list[str]) -> np.ndarray:
    return np.stack([df[cfg.proba_columns_for(bl)].to_numpy() for bl in base_learners], axis=0)


def optimize_regime(regime_name: str, regime_df: pd.DataFrame, seed: int) -> dict:
    proba_stack = build_proba_stack(regime_df, cfg.BASE_LEARNERS)
    y_true = regime_df[cfg.TARGET_COLUMN].to_numpy()
    forward_return = regime_df["forward_return_1"].to_numpy()
    eval_fn = eo.make_evaluator(proba_stack, y_true, forward_return, cfg.CLASS_ORDER)

    print(f"\n  [{regime_name}] optimizing on {len(regime_df)} rows ...")
    t0 = time.time()
    pareto_genes, pareto_objs = eo.run_optimization(
        eval_fn,
        n_genes=cfg.N_BASE_LEARNERS,
        pop_size=cfg.POPULATION_SIZE,
        n_gen=cfg.N_GENERATIONS,
        seed=seed,
    )
    elapsed = time.time() - t0
    selected_weights, selected_objs = eo.select_compromise(pareto_genes, pareto_objs)
    print(
        f"  [{regime_name}] done in {elapsed:.1f}s -- Pareto front size={len(pareto_genes)}, "
        f"selected macro_f1={selected_objs[0]:.4f}, sharpe={selected_objs[1]:.4f}"
    )
    print(f"  [{regime_name}] selected weights: {dict(zip(cfg.BASE_LEARNERS, selected_weights.round(4)))}")

    return {
        "regime": regime_name,
        "n_rows": int(len(regime_df)),
        "pareto_front": [
            {
                "weights": dict(zip(cfg.BASE_LEARNERS, eo.genes_to_weights(g).tolist())),
                "macro_f1": float(o[0]),
                "sharpe": float(o[1]),
            }
            for g, o in zip(pareto_genes, pareto_objs)
        ],
        "selected_weights": dict(zip(cfg.BASE_LEARNERS, selected_weights.tolist())),
        "selected_macro_f1": float(selected_objs[0]),
        "selected_sharpe": float(selected_objs[1]),
    }


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    run_start = time.time()
    print("=" * 78)
    print("Stage 6 - Evolutionary Multi-Objective Ensemble")
    print("=" * 78)
    print(f"reading ONLY: {cfg.VALIDATION_PREDICTIONS_PATH.name}, {cfg.TRAIN_M15_PATH.name}")
    print(f"NEVER opened in this module: {cfg.TEST_PREDICTIONS_PATH.name} (2025, locked)")

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_parquet(cfg.VALIDATION_PREDICTIONS_PATH)
    val_df[cfg.TIMESTAMP_COLUMN] = pd.to_datetime(val_df[cfg.TIMESTAMP_COLUMN])
    val_df = val_df.sort_values(cfg.TIMESTAMP_COLUMN).reset_index(drop=True)
    print(f"\nloaded validation predictions: {val_df.shape}")

    merged = attach_price_context(val_df)

    regime_series, thresholds = assign_regimes(merged["vol_16"])
    merged["regime"] = regime_series
    print(f"regime thresholds (vol_16 tertiles, fit on validation only): {thresholds}")
    print(f"regime distribution: {merged['regime'].value_counts().to_dict()}")

    regime_results = []
    for i, regime_name in enumerate(cfg.REGIME_LABELS):
        regime_df = merged.loc[merged["regime"] == regime_name].reset_index(drop=True)
        assert len(regime_df) > cfg.N_BASE_LEARNERS, f"[{regime_name}] too few rows to optimize."
        regime_results.append(optimize_regime(regime_name, regime_df, seed=cfg.RANDOM_STATE + i))

    # ---- apply each regime's selected weights back across the full validation set ----
    combined_proba_full = np.zeros((len(merged), 3))
    for result in regime_results:
        mask = (merged["regime"] == result["regime"]).to_numpy()
        weights = np.array([result["selected_weights"][bl] for bl in cfg.BASE_LEARNERS])
        proba_stack = build_proba_stack(merged.loc[mask], cfg.BASE_LEARNERS)
        combined_proba_full[mask] = eo.combine_probabilities(weights, proba_stack)

    y_pred_full = eo.predicted_class_from_proba(combined_proba_full, cfg.CLASS_ORDER)
    overall_macro_f1 = f1_score(
        merged[cfg.TARGET_COLUMN], y_pred_full, labels=cfg.CLASS_ORDER, average="macro", zero_division=0
    )
    overall_sharpe = eo.compute_sharpe(y_pred_full, merged["forward_return_1"].to_numpy())

    print("\n" + "=" * 78)
    print("OVERALL validation performance (regime-routed ensemble, 2024 only)")
    print("=" * 78)
    print(f"macro_f1 = {overall_macro_f1:.4f}   sharpe (raw, per-bar) = {overall_sharpe:.4f}")

    # ---- save artifacts ----
    ensemble_weights = {r["regime"]: r["selected_weights"] for r in regime_results}
    with open(cfg.ENSEMBLE_WEIGHTS_PATH, "w") as fh:
        json.dump(ensemble_weights, fh, indent=2)
    print(f"\nwrote {cfg.ENSEMBLE_WEIGHTS_PATH}")

    with open(cfg.REGIME_THRESHOLDS_PATH, "w") as fh:
        json.dump(thresholds, fh, indent=2)
    print(f"wrote {cfg.REGIME_THRESHOLDS_PATH}")

    pareto_fronts = {r["regime"]: r["pareto_front"] for r in regime_results}
    with open(cfg.PARETO_FRONTS_PATH, "w") as fh:
        json.dump(pareto_fronts, fh, indent=2)
    print(f"wrote {cfg.PARETO_FRONTS_PATH}")

    output_df = pd.DataFrame(
        {
            cfg.TIMESTAMP_COLUMN: merged[cfg.TIMESTAMP_COLUMN],
            cfg.TARGET_COLUMN: merged[cfg.TARGET_COLUMN],
            "regime": merged["regime"],
            "ensemble_down": combined_proba_full[:, 0],
            "ensemble_neutral": combined_proba_full[:, 1],
            "ensemble_up": combined_proba_full[:, 2],
            "predicted_class": y_pred_full,
        }
    )
    output_df.to_parquet(cfg.VALIDATION_ENSEMBLE_PREDICTIONS_PATH, index=False)
    print(f"wrote {cfg.VALIDATION_ENSEMBLE_PREDICTIONS_PATH}")

    metrics_report = {
        "overall_macro_f1": float(overall_macro_f1),
        "overall_sharpe_raw_per_bar": float(overall_sharpe),
        "per_regime": [
            {
                "regime": r["regime"],
                "n_rows": r["n_rows"],
                "selected_macro_f1": r["selected_macro_f1"],
                "selected_sharpe": r["selected_sharpe"],
                "pareto_front_size": len(r["pareto_front"]),
            }
            for r in regime_results
        ],
    }
    with open(cfg.METRICS_REPORT_PATH, "w") as fh:
        json.dump(metrics_report, fh, indent=2)
    print(f"wrote {cfg.METRICS_REPORT_PATH}")

    print(f"\nTotal runtime: {time.time() - run_start:.1f}s")
    print(f"2025 test set: NOT loaded, NOT referenced, NOT used anywhere in this run.")


if __name__ == "__main__":
    main()
