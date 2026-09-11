"""Branch-and-bound quasi-MLE search for early-walking tCNOT leakage."""

from __future__ import annotations

import heapq
import itertools
import time

import numpy as np


class _BudgetExhausted(Exception):
    pass


class _Budget:
    def __init__(self, max_nodes: int | None, deadline: float | None):
        self.max_nodes = max_nodes
        self.deadline = deadline
        self.n_nodes = 0

    def spend(self):
        self.n_nodes += 1
        if self.max_nodes is not None and self.n_nodes > self.max_nodes:
            raise _BudgetExhausted
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise _BudgetExhausted


def _explains(hyperedges, fired_detectors: frozenset[int]) -> bool:
    reproduced: set[int] = set()
    for hyperedge in hyperedges:
        reproduced ^= hyperedge.detectors
    return reproduced == fired_detectors


def _require_exact_backend(hyperedge_decoder):
    """Reject beam-limited Tesseract configurations used as search bounds."""

    config_options = getattr(hyperedge_decoder, "config_options", None)
    if config_options is None:
        return
    tesseract = getattr(hyperedge_decoder, "_tesseract", None)
    if tesseract is None:
        return
    if config_options.get("det_beam", 0) < tesseract.INF_DET_BEAM:
        raise ValueError(
            "Quasi-MLE requires exact Tesseract bounds. Construct the backend with "
            "det_beam=tesseract.INF_DET_BEAM."
        )


def find_valid_solution(
    decoder,
    syndrome,
    erasure_checks,
    max_nodes: int | None = None,
    mle_time_limit: float | None = None,
):
    """Find a lowest-cost decoding using one leakage location per erasure check."""

    _require_exact_backend(decoder.hyperedge_decoder)
    checks = decoder.flatten_erasure_checks(erasure_checks)
    if not checks:
        raise ValueError("quasi-MLE search requires at least one erasure check.")
    candidates = [
        decoder.get_leakage_locations_from_erasure_check(*check)
        for check in checks
    ]
    if any(not row for row in candidates):
        return None

    order = sorted(range(len(checks)), key=lambda index: len(candidates[index]))
    checks = [checks[index] for index in order]
    candidates = [candidates[index] for index in order]

    dive_cost = sum(len(row) for row in candidates)
    if max_nodes is not None and max_nodes < dive_cost:
        raise ValueError(
            f"max_nodes={max_nodes} is below the {dive_cost} node solves needed "
            "to reach the first complete assignment."
        )
    if mle_time_limit is not None and mle_time_limit < 0:
        raise ValueError("mle_time_limit must be non-negative or None.")

    budget = _Budget(
        max_nodes,
        None if mle_time_limit is None else time.monotonic() + mle_time_limit,
    )
    syndrome = np.asarray(syndrome)
    fired_detectors = frozenset(np.flatnonzero(syndrome).tolist())

    def evaluate(fixed):
        budget.spend()
        model = decoder.calculate_search_reweights(
            fixed_leakage_locations=fixed,
            relaxed_erasure_checks=checks[len(fixed) :],
        )
        prediction, cost, hyperedges = decoder.solve_model(model, syndrome)
        if not _explains(hyperedges, fired_detectors):
            return float("inf"), None
        return cost, prediction

    incumbent_cost = float("inf")
    incumbent_solution = None
    try:
        fixed = ()
        while len(fixed) < len(checks):
            cost, prediction, fixed = min(
                (
                    evaluate(fixed + (location,))
                    + (fixed + (location,),)
                    for location in candidates[len(fixed)]
                ),
                key=lambda scored: scored[0],
            )
        incumbent_cost, incumbent_solution = cost, prediction
    except _BudgetExhausted:
        return None

    counter = itertools.count()
    frontier = []
    try:
        heapq.heappush(frontier, (evaluate(())[0], next(counter), (), None))
        while frontier:
            _, _, fixed, solution = heapq.heappop(frontier)
            if len(fixed) == len(checks):
                return solution
            for location in candidates[len(fixed)]:
                child = fixed + (location,)
                child_cost, child_solution = evaluate(child)
                if child_cost >= incumbent_cost:
                    continue
                if len(child) == len(checks):
                    incumbent_cost = child_cost
                    incumbent_solution = child_solution
                heapq.heappush(
                    frontier,
                    (child_cost, next(counter), child, child_solution),
                )
    except _BudgetExhausted:
        return incumbent_solution
    return incumbent_solution

