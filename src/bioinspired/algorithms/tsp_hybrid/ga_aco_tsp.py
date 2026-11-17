"""GA+ACO hybrid for TSP (permutation encoding).

Phase A — GA with Order Crossover explores the permutation space and builds
a diverse set of good tours.

Transfer — the top-K solutions deposit pheromones on their edges, seeding
the ACO pheromone matrix with a bias toward promising arcs.

Phase B — ACO runs with the pre-seeded matrix and refines further.

Deterministic with np.random.default_rng(seed).
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from bioinspired.algorithms.tsp.aco_tsp import ACOTSPSolver
from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm, TSPResult
from bioinspired.algorithms.tsp.ga_tsp import GATSPSolver
from bioinspired.core.tsp_problem import InspectionTSPProblem

INITIAL_PHEROMONE = 1.0


class GACOTSPSolver(BaseTSPAlgorithm):
    """GA → ACO hybrid TSP solver.

    Attributes:
        ga_iters:       iterations for the GA phase.
        aco_iters:      iterations for the ACO phase.
        ga_pop_size:    GA population size.
        n_seed_tours:   top-K GA tours used to seed ACO pheromones.
        pheromone_boost: deposit multiplier for seeded edges (added on top of
                         baseline INITIAL_PHEROMONE).
        aco_n_ants:     number of ACO ants per iteration.
        snapshot_every: if set, saves intermediate tours every N iterations.
    """

    name: ClassVar[str] = "ga_aco_tsp"

    def __init__(
        self,
        ga_iters: int = 150,
        aco_iters: int = 150,
        ga_pop_size: int = 60,
        n_seed_tours: int = 5,
        pheromone_boost: float = 3.0,
        aco_n_ants: int = 60,
        snapshot_every: int | None = None,
        aco_alpha: float = 1.0,
        aco_beta: float = 2.0,
        aco_rho: float = 0.5,
        aco_q: float = 100.0,
    ) -> None:
        self.ga_iters = ga_iters
        self.aco_iters = aco_iters
        self.ga_pop_size = ga_pop_size
        self.n_seed_tours = n_seed_tours
        self.pheromone_boost = pheromone_boost
        self.aco_n_ants = aco_n_ants
        # Se exponen para que la etapa ACO no corra con los defaults de clase y
        # quede al alcance de los presets y de la sintonizacion.
        self.aco_alpha = aco_alpha
        self.aco_beta = aco_beta
        self.aco_rho = aco_rho
        self.aco_q = aco_q
        self.snapshot_every = snapshot_every

    def solve(self, problem: InspectionTSPProblem, seed: int) -> TSPResult:
        t0 = time.perf_counter()
        n = problem.n_targets

        # ── Phase A: GA ───────────────────────────────────────────────────────
        ga = GATSPSolver(
            pop_size=self.ga_pop_size,
            max_iterations=self.ga_iters,
            snapshot_every=self.snapshot_every,
        )
        ga_result = ga.solve(problem, seed=seed)
        history: list[float] = list(ga_result.convergence_history)
        snapshots: list[dict] = list(ga_result.metadata.get("snapshots", []))

        # ── Transfer: seed ACO pheromones from the EVOLVED top-K GA tours ──────
        # Usa la poblacion evolucionada de la fase A (no el init random).
        final_pop = ga_result.metadata["final_pop"]
        final_costs = ga_result.metadata["final_costs"]
        tau = np.full((n, n), INITIAL_PHEROMONE, dtype=np.float64)
        for tour, cost in zip(
            final_pop[: self.n_seed_tours], final_costs[: self.n_seed_tours], strict=True
        ):
            if cost <= 0 or not np.isfinite(cost):
                continue
            deposit = self.pheromone_boost / cost
            for i in range(len(tour) - 1):
                tau[tour[i], tour[i + 1]] += deposit
                tau[tour[i + 1], tour[i]] += deposit  # symmetric

        # ── Phase B: ACO with seeded pheromones ───────────────────────────────
        aco = ACOTSPSolver(
            n_ants=self.aco_n_ants,
            max_iterations=self.aco_iters,
            snapshot_every=self.snapshot_every,
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
            best_from_ga=ga_result.best_tour,
            best_cost_from_ga=ga_result.best_cost,
            snapshot_every=self.snapshot_every,
        )

        # Merge histories and snapshots
        history.extend(aco_result.convergence_history.tolist())
        if self.snapshot_every is not None:
            for s in aco_result.metadata.get("snapshots", []):
                snapshots.append(
                    {
                        "iteration": s["iteration"] + self.ga_iters,
                        "tour": s["tour"],
                        "cost": s["cost"],
                    }
                )

        best_tour = aco_result.best_tour
        best_cost = aco_result.best_cost
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
                "aco_best": aco_result.best_cost,
                "transition_iter": self.ga_iters,
                "n_evaluations": ga_result.metadata["n_evaluations"]
                + aco_result.metadata["n_evaluations"],
            },
        )


def _run_aco_with_seeded_tau(
    aco: ACOTSPSolver,
    problem: InspectionTSPProblem,
    seed: int,
    tau: np.ndarray,
    best_from_ga: list[int],
    best_cost_from_ga: float,
    snapshot_every: int | None,
) -> TSPResult:
    """Run ACO with a pre-built pheromone matrix as starting point."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = problem.n_targets
    d = problem.distance_matrix

    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(d > 0, 1.0 / d, 0.0)

    best_tour = best_from_ga[:]
    best_cost = best_cost_from_ga
    history: list[float] = [best_cost]
    snapshots: list[dict] = []
    n_evals = 0  # best_cost_from_ga ya viene contado en la fase GA (no re-evaluar)
    if snapshot_every is not None:
        snapshots.append({"iteration": 0, "tour": best_tour[:], "cost": best_cost})

    for it in range(aco.max_iterations):
        tours: list[list[int]] = []
        tour_costs: list[float] = []
        for _ant in range(aco.n_ants):
            start = int(rng.integers(n))
            tour = aco._build_tour(start, tau, eta, n, rng)
            cost = problem.tour_cost(tour)
            n_evals += 1
            tours.append(tour)
            tour_costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour

        tau *= 1.0 - aco.rho
        for tour, cost in zip(tours, tour_costs, strict=True):
            if cost <= 0 or not np.isfinite(cost):
                continue
            deposit = aco.Q / cost
            for i in range(len(tour) - 1):
                tau[tour[i], tour[i + 1]] += deposit
                tau[tour[i + 1], tour[i]] += deposit

        history.append(best_cost)
        if snapshot_every is not None and (it + 1) % snapshot_every == 0:
            snapshots.append({"iteration": it + 1, "tour": best_tour[:], "cost": best_cost})

    return TSPResult(
        best_tour=best_tour,
        best_cost=best_cost,
        convergence_history=np.array(history, dtype=np.float64),
        runtime_seconds=time.perf_counter() - t0,
        metadata={"algorithm": "aco_phase", "snapshots": snapshots, "n_evaluations": n_evals},
    )
