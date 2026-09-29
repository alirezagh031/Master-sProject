import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.section7 import cost_sensitivity as cs  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


# ---------------------------------------------------------------------
# Hand-computed toy example:
#   position       = [ 1,    1,   -1,    0]   long, long(hold), short(reverse), flat(exit)
#   forward_return = [0.01,-0.02, 0.01, 0.005]
# Hand math:
#   gross_return = [0.01, -0.02, -0.01, 0.0]
#   changes      = [1, 0, 2, 1]     (enter=1, hold=0, reversal=2, exit=1)
#   cost @0.001  = [0.001, 0, 0.002, 0.001]  ->  total = 0.004
#   net_return   = [0.009, -0.02, -0.012, -0.001]
# ---------------------------------------------------------------------
position = np.array([1.0, 1.0, -1.0, 0.0])
forward_return = np.array([0.01, -0.02, 0.01, 0.005])

print("=== position-change counting (enter / hold / reverse / exit) ===")
changes = cs.compute_position_changes(position)
assert list(changes) == [1.0, 0.0, 2.0, 1.0], f"FAIL: got {changes}"
print("OK:", changes.tolist())

print("\n=== cost deduction + compounding, cost_rate=0.001 ===")
result = cs.equity_curve_with_costs(position, forward_return, 0.001)
expected_gross = np.prod(1 + np.array([0.01, -0.02, -0.01, 0.0])) - 1
expected_cost = 0.001 * 4
expected_net = np.prod(1 + np.array([0.009, -0.02, -0.012, -0.001])) - 1

assert close(result["gross_total_return"], expected_gross), result["gross_total_return"]
assert close(result["total_cost_charged"], expected_cost), result["total_cost_charged"]
assert close(result["net_total_return"], expected_net), result["net_total_return"]
assert result["n_directional_bars"] == 3
assert result["n_position_changes"] == 3
print(f"OK: gross={result['gross_total_return']:.6f} net={result['net_total_return']:.6f} "
      f"cost={result['total_cost_charged']:.6f}")
print("OK: n_directional_bars=3 (nonzero exposure) vs n_position_changes=3 (enter/reverse/exit) -- "
      "correctly distinct; the 'hold' bar is directional but not a change")

print("\n=== long/short/neutral sign behavior ===")
up = np.array([0.02])
assert cs.equity_curve_with_costs(np.array([1.0]), up, 0.0)["gross_total_return"] > 0
assert cs.equity_curve_with_costs(np.array([-1.0]), up, 0.0)["gross_total_return"] < 0
assert cs.equity_curve_with_costs(np.array([0.0]), up, 0.0)["gross_total_return"] == 0
print("OK: long profits / short loses on a price rise; flat is always exactly zero")

print("\n=== zero cost_rate is an exact no-op ===")
r0 = cs.equity_curve_with_costs(position, forward_return, 0.0)
assert close(r0["net_total_return"], r0["gross_total_return"])
assert r0["total_cost_charged"] == 0.0
print("OK")

print("\n=== safeguard: a positive cost rate can never improve net return ===")
results = cs.run_strategy_across_scenarios(position, forward_return)  # raises internally if violated
vals = [results[name]["net_total_return"] for name in cs.COST_SCENARIOS]
assert vals == sorted(vals, reverse=True), "net return should be non-increasing as cost rate rises"
print("OK:", {k: round(v, 6) for k, v in zip(cs.COST_SCENARIOS, vals)})

print("\nALL COST-MODEL TESTS PASSED")
