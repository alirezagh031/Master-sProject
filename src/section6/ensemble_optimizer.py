from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.metrics import f1_score


# ==========================================================================
# Weight <-> gene mapping
# ==========================================================================
def genes_to_weights(genes: np.ndarray) -> np.ndarray:
    """Non-negative genes in [0,1]^6 -> a probability-simplex weight vector."""
    genes = np.clip(genes, 0.0, None)
    total = genes.sum()
    if total < 1e-9:
        return np.ones_like(genes) / len(genes)
    return genes / total


# ==========================================================================
# Ensemble mechanics
# ==========================================================================
def combine_probabilities(weights: np.ndarray, proba_stack: np.ndarray) -> np.ndarray:
    """
    weights: (n_models,) non-negative, sums to 1
    proba_stack: (n_models, n_rows, 3)
    -> combined: (n_rows, 3). A convex combination of simplex vectors is
       itself on the simplex, so combined.sum(axis=1) == 1 automatically --
       no renormalization needed.
    """
    return np.tensordot(weights, proba_stack, axes=(0, 0))


def predicted_class_from_proba(combined_proba: np.ndarray, class_order: list[int]) -> np.ndarray:
    idx = combined_proba.argmax(axis=1)
    return np.asarray(class_order)[idx]


def compute_sharpe(position: np.ndarray, forward_return: np.ndarray) -> float:
    """
    Raw (non-annualized) per-bar Sharpe of position * forward_return. Rows
    with a NaN forward_return (e.g. the validation row whose next bar would
    fall in the locked test period) are excluded from this computation --
    they still count normally for macro_f1 elsewhere, just not here.
    """
    strategy_return = position.astype(float) * forward_return
    strategy_return = strategy_return[~np.isnan(strategy_return)]
    if len(strategy_return) == 0:
        return 0.0
    std = strategy_return.std()
    if std < 1e-12:
        return 0.0
    return float(strategy_return.mean() / std)


def make_evaluator(
    proba_stack: np.ndarray, y_true: np.ndarray, forward_return: np.ndarray, class_order: list[int]
) -> Callable[[np.ndarray], np.ndarray]:
    """Returns eval(genes) -> np.array([macro_f1, sharpe]), both maximized."""

    def evaluate(genes: np.ndarray) -> np.ndarray:
        weights = genes_to_weights(genes)
        combined = combine_probabilities(weights, proba_stack)
        y_pred = predicted_class_from_proba(combined, class_order)
        f1 = f1_score(y_true, y_pred, labels=class_order, average="macro", zero_division=0)
        sharpe = compute_sharpe(y_pred, forward_return)
        return np.array([f1, sharpe])

    return evaluate


# ==========================================================================
# NSGA-II core: non-dominated sorting + crowding distance
# ==========================================================================
def _dominates(p: np.ndarray, q: np.ndarray) -> bool:
    """p dominates q under MAXIMIZATION of all objectives."""
    return bool(np.all(p >= q) and np.any(p > q))


def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    n = len(objectives)
    dominated_by = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominated_by[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # trailing empty front
    return fronts


def crowding_distance(front_objectives: np.ndarray) -> np.ndarray:
    n = len(front_objectives)
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    n_obj = front_objectives.shape[1]
    for m in range(n_obj):
        order = np.argsort(front_objectives[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        obj_min = front_objectives[order[0], m]
        obj_max = front_objectives[order[-1], m]
        if obj_max - obj_min < 1e-12:
            continue
        for k in range(1, n - 1):
            distances[order[k]] += (
                front_objectives[order[k + 1], m] - front_objectives[order[k - 1], m]
            ) / (obj_max - obj_min)
    return distances


def _ranks_and_crowding(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fronts = fast_non_dominated_sort(objectives)
    n = len(objectives)
    ranks = np.zeros(n, dtype=int)
    crowding = np.zeros(n)
    for rank, front in enumerate(fronts):
        if not front:
            continue
        ranks_arr = np.asarray(front)
        ranks[ranks_arr] = rank
        cd = crowding_distance(objectives[ranks_arr])
        crowding[ranks_arr] = cd
    return ranks, crowding


def _environmental_selection(
    pop: np.ndarray, objectives: np.ndarray, target_size: int
) -> tuple[np.ndarray, np.ndarray]:
    fronts = fast_non_dominated_sort(objectives)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= target_size:
            selected.extend(front)
        else:
            remaining = target_size - len(selected)
            cd = crowding_distance(objectives[front])
            order = np.argsort(-cd)
            selected.extend([front[i] for i in order[:remaining]])
            break
    return pop[selected], objectives[selected]


# ==========================================================================
# Variation operators: blend crossover + Gaussian mutation, bounded [0, 1]
# ==========================================================================
def _tournament_select(ranks: np.ndarray, crowding: np.ndarray, rng: np.random.Generator, k: int) -> int:
    candidates = rng.integers(0, len(ranks), size=k)
    best = candidates[0]
    for c in candidates[1:]:
        if ranks[c] < ranks[best] or (ranks[c] == ranks[best] and crowding[c] > crowding[best]):
            best = c
    return int(best)


def _blend_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, alpha: float = 0.25):
    r = rng.uniform(-alpha, 1 + alpha, size=len(p1))
    child1 = np.clip(r * p1 + (1 - r) * p2, 0.0, 1.0)
    child2 = np.clip(r * p2 + (1 - r) * p1, 0.0, 1.0)
    return child1, child2


def _gaussian_mutation(ind: np.ndarray, rng: np.random.Generator, prob: float, sigma: float = 0.1) -> np.ndarray:
    ind = ind.copy()
    mask = rng.random(len(ind)) < prob
    if mask.any():
        ind[mask] = ind[mask] + rng.normal(0, sigma, size=int(mask.sum()))
    return np.clip(ind, 0.0, 1.0)


def nsga2_optimize(
    eval_fn: Callable[[np.ndarray], np.ndarray],
    n_genes: int,
    pop_size: int,
    n_gen: int,
    seed: int,
    tournament_size: int = 2,
    mutation_prob: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pareto_genes, pareto_objectives), both maximized objectives."""
    rng = np.random.default_rng(seed)
    mutation_prob = mutation_prob if mutation_prob is not None else 1.0 / n_genes

    pop = rng.random((pop_size, n_genes))
    objectives = np.array([eval_fn(ind) for ind in pop])

    for _ in range(n_gen):
        ranks, crowding = _ranks_and_crowding(objectives)
        offspring = []
        while len(offspring) < pop_size:
            i1 = _tournament_select(ranks, crowding, rng, tournament_size)
            i2 = _tournament_select(ranks, crowding, rng, tournament_size)
            c1, c2 = _blend_crossover(pop[i1], pop[i2], rng)
            c1 = _gaussian_mutation(c1, rng, mutation_prob)
            c2 = _gaussian_mutation(c2, rng, mutation_prob)
            offspring.extend([c1, c2])
        offspring = np.array(offspring[:pop_size])
        offspring_obj = np.array([eval_fn(ind) for ind in offspring])

        combined_pop = np.vstack([pop, offspring])
        combined_obj = np.vstack([objectives, offspring_obj])
        pop, objectives = _environmental_selection(combined_pop, combined_obj, pop_size)

    fronts = fast_non_dominated_sort(objectives)
    pareto_idx = fronts[0]
    return pop[pareto_idx], objectives[pareto_idx]


# ==========================================================================
# Optional pymoo backend (used automatically if importable; safe fallback)
# ==========================================================================
def _try_pymoo_optimize(eval_fn, n_genes, pop_size, n_gen, seed):
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize as pymoo_minimize

        class _EnsembleProblem(Problem):
            def __init__(self):
                super().__init__(n_var=n_genes, n_obj=2, xl=0.0, xu=1.0)

            def _evaluate(self, X, out, *args, **kwargs):
                out["F"] = np.array([-eval_fn(x) for x in X])  # pymoo minimizes

        algorithm = NSGA2(pop_size=pop_size)
        res = pymoo_minimize(_EnsembleProblem(), algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
        pareto_genes, pareto_objectives = np.atleast_2d(res.X), -np.atleast_2d(res.F)
        return pareto_genes, pareto_objectives
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure -> fallback
        print(
            f"  [nsga2] pymoo backend unavailable or failed "
            f"({type(exc).__name__}: {exc}); using the built-in numpy NSGA-II instead."
        )
        return None


def run_optimization(
    eval_fn: Callable[[np.ndarray], np.ndarray],
    n_genes: int,
    pop_size: int,
    n_gen: int,
    seed: int,
    prefer_pymoo: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if prefer_pymoo:
        result = _try_pymoo_optimize(eval_fn, n_genes, pop_size, n_gen, seed)
        if result is not None:
            print("  [nsga2] using pymoo backend")
            return result
    print("  [nsga2] using built-in numpy NSGA-II backend")
    return nsga2_optimize(eval_fn, n_genes, pop_size, n_gen, seed)


# ==========================================================================
# Pareto-front compromise selection (for a single recommended weight vector)
# ==========================================================================
def select_compromise(pareto_genes: np.ndarray, pareto_objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Picks one point off the Pareto front: min-max normalize each objective
    across the front, then take the point maximizing their (equal-weight)
    sum. The FULL front is saved separately -- this is just one documented,
    reasonable compromise, not the only valid choice.
    """
    obj_min = pareto_objectives.min(axis=0)
    obj_max = pareto_objectives.max(axis=0)
    span = np.where(obj_max - obj_min < 1e-12, 1.0, obj_max - obj_min)
    normalized = (pareto_objectives - obj_min) / span
    best_idx = int(np.argmax(normalized.sum(axis=1)))
    return genes_to_weights(pareto_genes[best_idx]), pareto_objectives[best_idx]
