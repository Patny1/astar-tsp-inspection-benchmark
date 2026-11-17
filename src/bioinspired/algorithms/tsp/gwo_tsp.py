"""GWO for TSP — random-key encoding (Mirjalili et al. 2014, adapted).

Grey Wolf Optimizer uses real-valued random-key encoding (Bean 1994): each
wolf is a vector in [0,1]^n and the tour is decoded via argsort. The standard
GWO position update is applied in the continuous domain; no discrete operators
needed.

Deterministic with np.random.default_rng(seed).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import ClassVar

import numpy as np

from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm, TSPResult
from bioinspired.core.tsp_problem import InspectionTSPProblem


class GWOTSPSolver(BaseTSPAlgorithm):
    """Grey Wolf Optimizer with random-key encoding for TSP.

    Attributes:
        n_wolves:       pack size.
        max_iterations: number of iterations (a decreases 2→0 linearly).
    """

    name: ClassVar[str] = "gwo_tsp"

    def __init__(
        self,
        n_wolves: int = 60,
        max_iterations: int = 300,
        snapshot_every: int | None = None,
        local_search: Callable[[list[int], InspectionTSPProblem], tuple[list[int], int]]
        | None = None,
        memetic_elites: int = 1,
    ) -> None:
        self.n_wolves = n_wolves
        self.max_iterations = max_iterations
        self.snapshot_every = snapshot_every
        self.local_search = local_search
        self.memetic_elites = memetic_elites

    def solve(self, problem: InspectionTSPProblem, seed: int) -> TSPResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        n = problem.n_targets

        pos = rng.random((self.n_wolves, n))

        n_evals = 0  # contador real de evaluaciones (llamadas a tour_cost)

        def _cost(p: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            return problem.tour_cost(np.argsort(p))

        costs = np.array([_cost(p) for p in pos])

        sorted_idxs = np.argsort(costs)
        alpha_pos, alpha_cost = pos[sorted_idxs[0]].copy(), float(costs[sorted_idxs[0]])
        beta_pos = pos[sorted_idxs[1]].copy()
        delta_pos = pos[sorted_idxs[2]].copy()

        history: list[float] = [alpha_cost]
        snapshots: list[dict] = []
        if self.snapshot_every is not None:
            snapshots.append(
                {"iteration": 0, "tour": np.argsort(alpha_pos).tolist(), "cost": alpha_cost}
            )

        for t in range(self.max_iterations):
            a = 2.0 - 2.0 * t / self.max_iterations  # decreases from 2 to 0

            r1 = rng.random((self.n_wolves, n))
            r2 = rng.random((self.n_wolves, n))
            A1 = 2 * a * r1 - a  # noqa: N806
            C1 = 2 * r2  # noqa: N806
            D_alpha = np.abs(C1 * alpha_pos - pos)  # noqa: N806
            X1 = alpha_pos - A1 * D_alpha  # noqa: N806

            r1 = rng.random((self.n_wolves, n))
            r2 = rng.random((self.n_wolves, n))
            A2 = 2 * a * r1 - a  # noqa: N806
            C2 = 2 * r2  # noqa: N806
            D_beta = np.abs(C2 * beta_pos - pos)  # noqa: N806
            X2 = beta_pos - A2 * D_beta  # noqa: N806

            r1 = rng.random((self.n_wolves, n))
            r2 = rng.random((self.n_wolves, n))
            A3 = 2 * a * r1 - a  # noqa: N806
            C3 = 2 * r2  # noqa: N806
            D_delta = np.abs(C3 * delta_pos - pos)  # noqa: N806
            X3 = delta_pos - A3 * D_delta  # noqa: N806

            pos = np.clip((X1 + X2 + X3) / 3.0, 0.0, 1.0)
            costs = np.array([_cost(p) for p in pos])

            if self.local_search is not None:
                for r in np.argsort(costs)[: self.memetic_elites]:
                    improved_tour, ev = self.local_search(list(np.argsort(pos[r])), problem)
                    n_evals += ev
                    c = problem.tour_cost(improved_tour)
                    n_evals += 1
                    if c < costs[r]:
                        pos[r] = np.argsort(improved_tour).astype(np.float64) / max(n - 1, 1)
                        costs[r] = c

            sorted_idxs = np.argsort(costs)
            if costs[sorted_idxs[0]] < alpha_cost:
                alpha_cost = float(costs[sorted_idxs[0]])
                alpha_pos = pos[sorted_idxs[0]].copy()
            beta_pos = pos[sorted_idxs[1]].copy()
            delta_pos = pos[sorted_idxs[2]].copy()
            history.append(alpha_cost)
            if self.snapshot_every is not None and (t + 1) % self.snapshot_every == 0:
                snapshots.append(
                    {"iteration": t + 1, "tour": np.argsort(alpha_pos).tolist(), "cost": alpha_cost}
                )

        best_tour = np.argsort(alpha_pos).tolist()
        final_order = np.argsort(costs)
        final_pop = [list(np.argsort(pos[i])) for i in final_order]
        final_costs = [float(costs[i]) for i in final_order]
        return TSPResult(
            best_tour=best_tour,
            best_cost=alpha_cost,
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
