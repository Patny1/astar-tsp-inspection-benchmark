"""PSO+ACO hybrid for TSP (random-key encoding).

Phase A — PSO explores the random-key space; the top-K particles are decoded
as tours and their edges seed the ACO pheromone matrix.

Phase B — ACO runs with the pre-seeded matrix and refines.

Deterministic with np.random.default_rng(seed).
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from bioinspired.algorithms.tsp.aco_tsp import ACOTSPSolver
from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm, TSPResult
from bioinspired.algorithms.tsp.pso_tsp import PSOTSPSolver
from bioinspired.algorithms.tsp_hybrid.ga_aco_tsp import _run_aco_with_seeded_tau
from bioinspired.core.tsp_problem import InspectionTSPProblem

INITIAL_PHEROMONE = 1.0


class PSOACOTSPSolver(BaseTSPAlgorithm):
    """PSO → ACO hybrid TSP solver.

    Attributes:
        pso_iters:      iterations for the PSO phase.
        aco_iters:      iterations for the ACO phase.
        pso_n_particles: PSO swarm size.
        n_seed_tours:   top-K PSO particles used to seed ACO pheromones.
        pheromone_boost: deposit multiplier for seeded edges.
        aco_n_ants:     number of ACO ants per iteration.
        snapshot_every: if set, saves intermediate tours every N iterations.
    """

    name: ClassVar[str] = "pso_aco_tsp"

    def __init__(
        self,
        pso_iters: int = 150,
        aco_iters: int = 150,
        pso_n_particles: int = 60,
        n_seed_tours: int = 5,
        pheromone_boost: float = 3.0,
        aco_n_ants: int = 60,
        snapshot_every: int | None = None,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: float = 0.4,
        aco_alpha: float = 1.0,
        aco_beta: float = 2.0,
        aco_rho: float = 0.5,
        aco_q: float = 100.0,
    ) -> None:
        self.pso_iters = pso_iters
        self.aco_iters = aco_iters
        self.pso_n_particles = pso_n_particles
        self.n_seed_tours = n_seed_tours
        self.pheromone_boost = pheromone_boost
        self.aco_n_ants = aco_n_ants
        # Se exponen para que las dos etapas no corran con los defaults de clase
        # y queden al alcance de los presets y de la sintonizacion.
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.aco_alpha = aco_alpha
        self.aco_beta = aco_beta
        self.aco_rho = aco_rho
        self.aco_q = aco_q
        self.snapshot_every = snapshot_every

    def solve(self, problem: InspectionTSPProblem, seed: int) -> TSPResult:
        t0 = time.perf_counter()
        n = problem.n_targets

        # ── Phase A: PSO ─────────────────────────────────────────────────────
        pso = PSOTSPSolver(
            n_particles=self.pso_n_particles,
            max_iterations=self.pso_iters,
            snapshot_every=self.snapshot_every,
            w=self.w,
            c1=self.c1,
            c2=self.c2,
            v_max=self.v_max,
        )
        pso_result = pso.solve(problem, seed=seed)
        history: list[float] = list(pso_result.convergence_history)
        snapshots: list[dict] = list(pso_result.metadata.get("snapshots", []))

        # ── Transfer: seed ACO from the EVOLVED top-K PSO tours ───────────────
        final_pop = pso_result.metadata["final_pop"]
        final_costs = pso_result.metadata["final_costs"]
        tau = np.full((n, n), INITIAL_PHEROMONE, dtype=np.float64)
        for tour, cost in zip(
            final_pop[: self.n_seed_tours], final_costs[: self.n_seed_tours], strict=True
        ):
            if cost <= 0 or not np.isfinite(cost):
                continue
            deposit = self.pheromone_boost / cost
            for i in range(len(tour) - 1):
                tau[tour[i], tour[i + 1]] += deposit
                tau[tour[i + 1], tour[i]] += deposit

        # ── Phase B: ACO ─────────────────────────────────────────────────────
        aco = ACOTSPSolver(
            n_ants=self.aco_n_ants,
            max_iterations=self.aco_iters,
            alpha=self.aco_alpha,
            beta=self.aco_beta,
            rho=self.aco_rho,
            Q=self.aco_q,
        )
        aco_result = _run_aco_with_seeded_tau(
            aco,
            problem,
            seed=seed + 1,
            tau=tau,
            best_from_ga=pso_result.best_tour,
            best_cost_from_ga=pso_result.best_cost,
            snapshot_every=self.snapshot_every,
        )

        history.extend(aco_result.convergence_history.tolist())
        if self.snapshot_every is not None:
            for s in aco_result.metadata.get("snapshots", []):
                snapshots.append(
                    {
                        "iteration": s["iteration"] + self.pso_iters,
                        "tour": s["tour"],
                        "cost": s["cost"],
                    }
                )

        best_tour = aco_result.best_tour
        best_cost = aco_result.best_cost
        if pso_result.best_cost < best_cost:
            best_tour = pso_result.best_tour
            best_cost = pso_result.best_cost

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
                "pso_best": pso_result.best_cost,
                "aco_best": aco_result.best_cost,
                "transition_iter": self.pso_iters,
                "n_evaluations": pso_result.metadata["n_evaluations"]
                + aco_result.metadata["n_evaluations"],
            },
        )
