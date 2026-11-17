"""ACO for TSP — Ant System on complete tour graph (Dorigo et al. 1996).

Classic Ant System applied to the TSP distance matrix. Each ant constructs a
tour greedily using tau^alpha * eta^beta (eta = 1/d[i][j]). After all ants
build their tours, pheromones evaporate and each ant deposits Q/L on its tour
edges. The matrix diagonal (d[i,i]=0) is excluded by the visited-set mask.

Deterministic with np.random.default_rng(seed).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import ClassVar

import numpy as np

from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm, TSPResult
from bioinspired.core.tsp_problem import InspectionTSPProblem

INITIAL_PHEROMONE = 1.0


class ACOTSPSolver(BaseTSPAlgorithm):
    """Ant System for TSP on precomputed distance matrix.

    Attributes:
        n_ants:         number of ants per iteration.
        alpha:          pheromone influence exponent.
        beta:           heuristic influence exponent.
        rho:            evaporation rate in [0, 1].
        Q:              deposit constant; ant deposits Q/tour_length per edge.
        max_iterations: number of iterations.
    """

    name: ClassVar[str] = "aco_tsp"

    def __init__(
        self,
        n_ants: int = 30,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.5,
        Q: float = 100.0,  # noqa: N803
        max_iterations: int = 300,
        snapshot_every: int | None = None,
        local_search: Callable[[list[int], InspectionTSPProblem], tuple[list[int], int]]
        | None = None,
        memetic_elites: int = 1,
    ) -> None:
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.max_iterations = max_iterations
        self.snapshot_every = snapshot_every
        self.local_search = local_search
        self.memetic_elites = memetic_elites

    def solve(self, problem: InspectionTSPProblem, seed: int) -> TSPResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        n = problem.n_targets
        d = problem.distance_matrix  # (n, n) A* distances

        # Heuristic matrix: 1/d[i,j], with 0 on diagonal
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(d > 0, 1.0 / d, 0.0)

        tau = np.full((n, n), INITIAL_PHEROMONE, dtype=np.float64)

        best_tour: list[int] = list(rng.permutation(n))
        best_cost = problem.tour_cost(best_tour)
        n_evals = 1  # contador real de evaluaciones (llamadas a tour_cost)
        history: list[float] = [best_cost]
        snapshots: list[dict] = []
        if self.snapshot_every is not None:
            snapshots.append({"iteration": 0, "tour": best_tour[:], "cost": best_cost})

        for it in range(self.max_iterations):
            tours: list[list[int]] = []
            tour_costs: list[float] = []

            for _ant in range(self.n_ants):
                start = int(rng.integers(n))
                tour = self._build_tour(start, tau, eta, n, rng)
                cost = problem.tour_cost(tour)
                n_evals += 1
                tours.append(tour)
                tour_costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour

            # Pheromone update: evaporate + deposit
            tau *= 1.0 - self.rho
            for tour, cost in zip(tours, tour_costs, strict=True):
                if cost <= 0 or not np.isfinite(cost):
                    continue
                deposit = self.Q / cost
                for i in range(len(tour) - 1):
                    tau[tour[i], tour[i + 1]] += deposit
                    tau[tour[i + 1], tour[i]] += deposit  # symmetric

            if self.local_search is not None:
                improved_tour, ev = self.local_search(list(best_tour), problem)
                n_evals += ev
                c = problem.tour_cost(improved_tour)
                n_evals += 1
                if c < best_cost:
                    best_tour = improved_tour
                    best_cost = c
                    deposit = self.Q / c
                    for i in range(len(improved_tour) - 1):
                        tau[improved_tour[i], improved_tour[i + 1]] += deposit
                        tau[improved_tour[i + 1], improved_tour[i]] += deposit

            history.append(best_cost)
            if self.snapshot_every is not None and (it + 1) % self.snapshot_every == 0:
                snapshots.append({"iteration": it + 1, "tour": best_tour[:], "cost": best_cost})

        if tour_costs:
            order = np.argsort(tour_costs)
            final_pop = [list(tours[i]) for i in order]
            final_costs = [float(tour_costs[i]) for i in order]
        else:
            final_pop = [list(best_tour)]
            final_costs = [float(best_cost)]

        return TSPResult(
            best_tour=best_tour,
            best_cost=best_cost,
            convergence_history=np.array(history, dtype=np.float64),
            runtime_seconds=time.perf_counter() - t0,
            metadata={
                "algorithm": self.name,
                "seed": seed,
                "n_targets": n,
                "n_evaluations": n_evals,
                "snapshots": snapshots,
                "final_pop": final_pop,
                "final_costs": final_costs,
            },
        )

    def _build_tour(
        self,
        start: int,
        tau: np.ndarray,
        eta: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """Build one ant tour from `start` visiting all n cities."""
        visited = [False] * n
        visited[start] = True
        tour = [start]
        current = start

        for _ in range(n - 1):
            weights = np.array(
                [
                    (tau[current, j] ** self.alpha) * (eta[current, j] ** self.beta)
                    if not visited[j]
                    else 0.0
                    for j in range(n)
                ]
            )
            total = float(weights.sum())
            if total <= 0.0:
                # Fallback: pick first unvisited city
                unvisited = [j for j in range(n) if not visited[j]]
                next_city = unvisited[0]
            else:
                probs = weights / total
                next_city = int(rng.choice(n, p=probs))

            visited[next_city] = True
            tour.append(next_city)
            current = next_city

        return tour
