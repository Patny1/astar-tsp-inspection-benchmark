"""PSO for TSP — random-key encoding (Bean 1994).

Each particle is a real-valued vector in [0,1]^n. The tour is decoded as
argsort(position). Standard PSO velocity/position update is applied in the
real domain; the permutation is recovered only for fitness evaluation.
This avoids the complexity of discrete velocity operators.

Deterministic with np.random.default_rng(seed).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import ClassVar

import numpy as np

from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm, TSPResult
from bioinspired.core.tsp_problem import InspectionTSPProblem


class PSOTSPSolver(BaseTSPAlgorithm):
    """PSO with random-key encoding for TSP.

    Attributes:
        n_particles:    swarm size.
        w:              inertia weight.
        c1:             cognitive coefficient (personal best).
        c2:             social coefficient (global best).
        v_max:          maximum velocity per dimension (clipped to [-v_max, v_max]).
        max_iterations: number of iterations.
    """

    name: ClassVar[str] = "pso_tsp"

    def __init__(
        self,
        n_particles: int = 60,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: float = 0.4,
        max_iterations: int = 300,
        snapshot_every: int | None = None,
        local_search: Callable[[list[int], InspectionTSPProblem], tuple[list[int], int]]
        | None = None,
        memetic_elites: int = 1,
    ) -> None:
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.max_iterations = max_iterations
        self.snapshot_every = snapshot_every
        self.local_search = local_search
        self.memetic_elites = memetic_elites

    def solve(self, problem: InspectionTSPProblem, seed: int) -> TSPResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        n = problem.n_targets

        pos = rng.random((self.n_particles, n))
        vel = rng.uniform(-self.v_max, self.v_max, (self.n_particles, n))

        n_evals = 0  # contador real de evaluaciones (llamadas a tour_cost)

        def _cost(p: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            return problem.tour_cost(np.argsort(p))

        costs = np.array([_cost(p) for p in pos])
        pbest_pos = pos.copy()
        pbest_cost = costs.copy()

        gbest_idx = int(np.argmin(costs))
        gbest_pos = pos[gbest_idx].copy()
        gbest_cost = float(costs[gbest_idx])
        history: list[float] = [gbest_cost]
        snapshots: list[dict] = []
        if self.snapshot_every is not None:
            snapshots.append(
                {"iteration": 0, "tour": np.argsort(gbest_pos).tolist(), "cost": gbest_cost}
            )

        for it in range(self.max_iterations):
            r1 = rng.random((self.n_particles, n))
            r2 = rng.random((self.n_particles, n))
            vel = self.w * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            vel = np.clip(vel, -self.v_max, self.v_max)
            pos = np.clip(pos + vel, 0.0, 1.0)

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

            improved = costs < pbest_cost
            pbest_pos = np.where(improved[:, None], pos, pbest_pos)
            pbest_cost = np.where(improved, costs, pbest_cost)

            gen_best_idx = int(np.argmin(pbest_cost))
            if pbest_cost[gen_best_idx] < gbest_cost:
                gbest_cost = float(pbest_cost[gen_best_idx])
                gbest_pos = pbest_pos[gen_best_idx].copy()
            history.append(gbest_cost)
            if self.snapshot_every is not None and (it + 1) % self.snapshot_every == 0:
                snapshots.append(
                    {
                        "iteration": it + 1,
                        "tour": np.argsort(gbest_pos).tolist(),
                        "cost": gbest_cost,
                    }
                )

        best_tour = np.argsort(gbest_pos).tolist()
        final_order = np.argsort(costs)
        final_pop = [list(np.argsort(pos[i])) for i in final_order]
        final_costs = [float(costs[i]) for i in final_order]
        return TSPResult(
            best_tour=best_tour,
            best_cost=gbest_cost,
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
