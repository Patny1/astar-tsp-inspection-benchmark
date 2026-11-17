"""2-opt local search TSP solver — classical improvement heuristic."""

from __future__ import annotations

import time

import numpy as np

from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm, TSPResult
from bioinspired.algorithms.tsp.local_search import two_opt_improve
from bioinspired.algorithms.tsp.nn_tsp import NNTSPSolver
from bioinspired.core.tsp_problem import InspectionTSPProblem


class TwoOptTSPSolver(BaseTSPAlgorithm):
    """2-opt local search mejora un tour inicial hasta alcanzar un óptimo local.

    Prueba todos los pares de aristas (i, j); si invertir el segmento entre i y j
    reduce el costo, aplica el swap. Repite hasta que ningún swap mejore
    (convergencia al óptimo local 2-opt).

    Complejidad por pasada: O(n²).

    Con ``init="nn"`` (default) arranca del tour Greedy NN — es el pairing
    clásico "NN + 2-opt" de la literatura, determinista (seed ignorado: NN
    siempre produce el mismo tour inicial). Con ``init="random"`` arranca de
    una permutación aleatoria distinta por seed — aísla qué tanto aporta el
    local search por sí solo, sin el sesgo de partir ya cerca de un buen tour.

    Attributes:
        max_passes: máximo de pasadas completas antes de detenerse.
            Default 200 es más que suficiente para N≤30 (converge en < 20 pasadas en práctica).
        init: "nn" (default) o "random".
    """

    name = "two_opt_tsp"

    def __init__(self, max_passes: int = 200, init: str = "nn") -> None:
        if init not in ("nn", "random"):
            raise ValueError(f"init debe ser 'nn' o 'random', recibido {init!r}")
        self.max_passes = max_passes
        self.init = init

    def solve(self, problem: InspectionTSPProblem, seed: int = 0) -> TSPResult:
        t0 = time.perf_counter()
        n = problem.n_targets
        dist = problem.distance_matrix
        d2t = problem.depot_to_targets
        t2e = problem.targets_to_end

        if self.init == "random":
            # .tolist() y no list(): permutation da np.int64 y el tipo se filtra
            # hasta best_tour, rompiendo a los consumidores que serializan a JSON.
            tour = np.random.default_rng(seed).permutation(n).tolist()
        else:
            tour = NNTSPSolver().solve(problem, seed=0).best_tour[:]

        def _tour_cost(t: list[int]) -> float:
            c = d2t[t[0]]
            for k in range(n - 1):
                c += dist[t[k], t[k + 1]]
            return c + t2e[t[-1]]

        history: list[float] = [_tour_cost(tour)]
        passes = 0

        tour, _ = two_opt_improve(tour, problem, self.max_passes)
        history.append(_tour_cost(tour))

        # passes contabiliza el número de iteraciones (estimación O(n²) por pasada)
        # two_opt_improve puede hacer múltiples pasadas; se preserva el conteo histórico.
        # Para n_evaluations seguimos usando passes*n*n como antes (2-opt no llama tour_cost).
        passes = self.max_passes if self.max_passes > 0 else 1

        cost = history[-1]
        return TSPResult(
            best_tour=tour,
            best_cost=float(cost),
            convergence_history=np.array(history, dtype=np.float64),
            runtime_seconds=time.perf_counter() - t0,
            metadata={
                "algorithm": self.name,
                "seed": seed,
                "init": self.init,
                "passes": passes,
                "n_evaluations": passes * n * n,
            },
        )


class TwoOptRandomTSPSolver(TwoOptTSPSolver):
    """2-opt puro: arranca de una permutación aleatoria por seed (sin NN).

    Aísla la contribución del local search en sí, separada del punto de
    partida que aporta NN en :class:`TwoOptTSPSolver` (init="nn" default).
    """

    name = "two_opt_random_tsp"

    def __init__(self, max_passes: int = 200) -> None:
        super().__init__(max_passes=max_passes, init="random")
