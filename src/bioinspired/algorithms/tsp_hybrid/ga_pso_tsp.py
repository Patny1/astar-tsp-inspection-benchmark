"""GA+PSO hybrid for TSP.

Phase A — GA with Order Crossover builds a diverse population of good tours.

Transfer — GA permutations are converted to PSO random-key positions:
for permutation π, position[π[i]] = (i + 0.5) / n.
This encodes the tour ordering into the continuous PSO space so that
argsort(position) recovers the original permutation exactly.

Phase B — PSO is initialized from the GA population (converted to positions)
and refines further in the continuous random-key space.

Deterministic with np.random.default_rng(seed).
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm, TSPResult
from bioinspired.algorithms.tsp.ga_tsp import GATSPSolver
from bioinspired.core.tsp_problem import InspectionTSPProblem


def _perm_to_rk(perm: np.ndarray) -> np.ndarray:
    """Convert a permutation to a random-key vector.

    argsort(result) == perm (exact recovery guaranteed).
    """
    n = len(perm)
    rk = np.empty(n, dtype=np.float64)
    for rank, city in enumerate(perm):
        rk[city] = (rank + 0.5) / n
    return rk


class GAPSOTSPSolver(BaseTSPAlgorithm):
    """GA → PSO hybrid TSP solver.

    Attributes:
        ga_iters:       iterations for the GA phase.
        pso_iters:      iterations for the PSO phase.
        pop_size:       shared population / swarm size.
        w:              PSO inertia weight.
        c1:             PSO cognitive coefficient.
        c2:             PSO social coefficient.
        v_max:          PSO max velocity per dimension.
        snapshot_every: if set, saves intermediate tours every N iterations.
    """

    name: ClassVar[str] = "ga_pso_tsp"

    def __init__(
        self,
        ga_iters: int = 150,
        pso_iters: int = 150,
        pop_size: int = 60,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: float = 0.4,
        snapshot_every: int | None = None,
    ) -> None:
        self.ga_iters = ga_iters
        self.pso_iters = pso_iters
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.snapshot_every = snapshot_every

    def solve(self, problem: InspectionTSPProblem, seed: int) -> TSPResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        n = problem.n_targets

        # ── Phase A: GA ───────────────────────────────────────────────────────
        ga = GATSPSolver(
            pop_size=self.pop_size,
            max_iterations=self.ga_iters,
            snapshot_every=self.snapshot_every,
        )
        ga_result = ga.solve(problem, seed=seed)
        history: list[float] = list(ga_result.convergence_history)
        snapshots: list[dict] = list(ga_result.metadata.get("snapshots", []))

        # ── Transfer: GA EVOLVED population → PSO initial positions ────────────
        # Usa la poblacion evolucionada de la fase A (no el init random).
        final_pop = ga_result.metadata["final_pop"]
        pos = np.array([_perm_to_rk(np.array(tour)) for tour in final_pop])
        vel = rng.uniform(-self.v_max, self.v_max, (self.pop_size, n))

        n_evals = ga_result.metadata["n_evaluations"]

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

        # Also track the GA best
        ga_rk = _perm_to_rk(np.array(ga_result.best_tour))
        if ga_result.best_cost < gbest_cost:
            gbest_cost = ga_result.best_cost
            gbest_pos = ga_rk.copy()

        # ── Phase B: PSO from GA-seeded positions ─────────────────────────────
        for it in range(self.pso_iters):
            r1 = rng.random((self.pop_size, n))
            r2 = rng.random((self.pop_size, n))
            vel = self.w * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            vel = np.clip(vel, -self.v_max, self.v_max)
            pos = np.clip(pos + vel, 0.0, 1.0)

            costs = np.array([_cost(p) for p in pos])
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
                        "iteration": it + 1 + self.ga_iters,
                        "tour": np.argsort(gbest_pos).tolist(),
                        "cost": gbest_cost,
                    }
                )

        best_tour = np.argsort(gbest_pos).tolist()
        best_cost = gbest_cost

        # Keep GA best if PSO didn't improve
        if ga_result.best_cost < best_cost:
            best_tour = ga_result.best_tour
            best_cost = ga_result.best_cost

        return TSPResult(
            best_tour=best_tour,
            best_cost=best_cost,
            convergence_history=np.array(history, dtype=np.float64),
            runtime_seconds=time.perf_counter() - t0,
            metadata={
                "algorithm": self.name,
                "seed": seed,
                "n_targets": n,
                "snapshots": snapshots,
                "ga_best": ga_result.best_cost,
                "pso_best": gbest_cost,
                "transition_iter": self.ga_iters,
                "n_evaluations": n_evals,
            },
        )
