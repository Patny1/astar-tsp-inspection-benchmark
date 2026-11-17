"""GA for TSP — Order Crossover (OX) + swap mutation.

Permutation-encoding GA using the Order Crossover operator (Davis 1985), which
preserves relative order of cities. Swap mutation randomly transposes two
positions. Tournament selection with elitism.

Deterministic with np.random.default_rng(seed).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import ClassVar

import numpy as np

from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm, TSPResult
from bioinspired.core.tsp_problem import InspectionTSPProblem


class GATSPSolver(BaseTSPAlgorithm):
    """Genetic Algorithm with Order Crossover for TSP.

    Attributes:
        pop_size:       population size.
        prob_crossover: probability of applying OX per pair.
        prob_mutation:  probability of swap mutation per individual.
        tournament_k:   tournament size for selection.
        n_elite:        number of elite individuals preserved per generation.
        max_iterations: number of generations.
        local_search:   hook opcional de búsqueda local Lamarckiana; recibe
                        (tour, problem) y devuelve (tour_mejorado, n_evals).
        memetic_elites: número de élites a refinar por generación cuando
                        local_search no es None.
    """

    name: ClassVar[str] = "ga_tsp"

    def __init__(
        self,
        pop_size: int = 60,
        prob_crossover: float = 0.9,
        prob_mutation: float = 0.2,
        tournament_k: int = 3,
        n_elite: int = 1,
        max_iterations: int = 300,
        snapshot_every: int | None = None,
        local_search: Callable[[list[int], InspectionTSPProblem], tuple[list[int], int]]
        | None = None,
        memetic_elites: int = 1,
    ) -> None:
        self.pop_size = pop_size
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.tournament_k = tournament_k
        self.n_elite = n_elite
        self.max_iterations = max_iterations
        self.snapshot_every = snapshot_every
        self.local_search = local_search
        self.memetic_elites = memetic_elites

    def solve(self, problem: InspectionTSPProblem, seed: int) -> TSPResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        n = problem.n_targets

        # Random initial population (permutations of range(n))
        pop = np.array([rng.permutation(n) for _ in range(self.pop_size)])
        costs = np.array([problem.tour_cost(ind) for ind in pop])
        n_evals = self.pop_size  # contador real de evaluaciones (llamadas a tour_cost)

        best_idx = int(np.argmin(costs))
        best_tour = pop[best_idx].tolist()
        best_cost = float(costs[best_idx])
        history: list[float] = [best_cost]
        snapshots: list[dict] = []
        if self.snapshot_every is not None:
            snapshots.append({"iteration": 0, "tour": best_tour[:], "cost": best_cost})

        for it in range(self.max_iterations):
            # Elite carry-over
            elite_idxs = np.argsort(costs)[: self.n_elite]
            new_pop = list(pop[elite_idxs])
            new_costs = list(costs[elite_idxs])

            while len(new_pop) < self.pop_size:
                # Tournament selection for two parents
                p1 = pop[self._tournament(costs, rng)]
                p2 = pop[self._tournament(costs, rng)]

                if rng.random() < self.prob_crossover:
                    child = self._ox_crossover(p1, p2, rng)
                else:
                    child = p1.copy()

                if rng.random() < self.prob_mutation:
                    child = self._swap_mutation(child, rng)

                new_pop.append(child)
                new_costs.append(problem.tour_cost(child))
                n_evals += 1

            pop = np.array(new_pop)
            costs = np.array(new_costs)

            # Refinamiento meméticos Lamarckiano: mejora las memetic_elites mejores
            if self.local_search is not None:
                order = np.argsort(costs)
                for r in order[: self.memetic_elites]:
                    improved, ev = self.local_search(pop[r].tolist(), problem)
                    n_evals += ev
                    pop[r] = np.array(improved)
                    costs[r] = problem.tour_cost(improved)
                    n_evals += 1

            gen_best_idx = int(np.argmin(costs))
            gen_best = float(costs[gen_best_idx])
            if gen_best < best_cost:
                best_cost = gen_best
                best_tour = pop[gen_best_idx].tolist()
            history.append(best_cost)
            if self.snapshot_every is not None and (it + 1) % self.snapshot_every == 0:
                snapshots.append({"iteration": it + 1, "tour": best_tour[:], "cost": best_cost})

        # Exponer población final ordenada por costo asc para híbridos fase B
        final_order = np.argsort(costs)
        final_pop = [pop[i].tolist() for i in final_order]
        final_costs = [float(costs[i]) for i in final_order]

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

    # ── operators ──────────────────────────────────────────────────────────

    def _tournament(self, costs: np.ndarray, rng: np.random.Generator) -> int:
        idxs = rng.choice(len(costs), size=self.tournament_k, replace=False)
        return int(idxs[np.argmin(costs[idxs])])

    @staticmethod
    def _ox_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Order Crossover (OX): copy a segment from p1, fill gaps in p2 order."""
        n = len(p1)
        a, b = sorted(rng.choice(n, 2, replace=False))
        child = np.full(n, -1, dtype=p1.dtype)
        child[a : b + 1] = p1[a : b + 1]
        segment_set = set(child[a : b + 1].tolist())
        fill = [x for x in p2 if x not in segment_set]
        fill_iter = iter(fill)
        for i in range(n):
            if child[i] == -1:
                child[i] = next(fill_iter)
        return child

    @staticmethod
    def _swap_mutation(ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Swap two random positions."""
        out = ind.copy()
        i, j = rng.choice(len(ind), 2, replace=False)
        out[i], out[j] = out[j], out[i]
        return out
