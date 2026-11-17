"""Nearest Neighbor TSP heuristic — classical greedy baseline."""

from __future__ import annotations

import time

import numpy as np

from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm, TSPResult
from bioinspired.core.tsp_problem import InspectionTSPProblem


class NNTSPSolver(BaseTSPAlgorithm):
    """Greedy nearest-neighbor TSP heuristic.

    Starting from the depot, repeatedly visits the closest unvisited target.
    Deterministic (seed ignored).  No iterations — snapshot_every ignored.
    """

    name = "nn_tsp"

    def solve(self, problem: InspectionTSPProblem, seed: int = 0) -> TSPResult:
        n = problem.n_targets
        dist = problem.distance_matrix  # (n, n) target-to-target
        d2t = problem.depot_to_targets  # (n,) depot→each target
        t2e = problem.targets_to_end  # (n,) each target→end

        t0 = time.perf_counter()
        unvisited = list(range(n))
        tour: list[int] = []

        # Start: pick nearest target from depot
        first = int(np.argmin(d2t))
        tour.append(first)
        unvisited.remove(first)

        while unvisited:
            last = tour[-1]
            nearest = min(unvisited, key=lambda j: dist[last, j])
            tour.append(nearest)
            unvisited.remove(nearest)

        # Compute cost: depot→first + tour legs + last→end
        cost = d2t[tour[0]]
        for k in range(len(tour) - 1):
            cost += dist[tour[k], tour[k + 1]]
        cost += t2e[tour[-1]]

        runtime = time.perf_counter() - t0
        return TSPResult(
            best_tour=tour,
            best_cost=float(cost),
            convergence_history=np.array([cost]),
            runtime_seconds=runtime,
            metadata={"algorithm": "nn_tsp", "seed": seed},
        )
