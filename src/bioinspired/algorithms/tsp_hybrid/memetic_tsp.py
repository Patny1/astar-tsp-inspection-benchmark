"""Familia memética (Fase 2): metaheurística bioinspirada + 2-opt local search."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from bioinspired.algorithms.tsp.aco_tsp import ACOTSPSolver
from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm
from bioinspired.algorithms.tsp.ga_tsp import GATSPSolver
from bioinspired.algorithms.tsp.gwo_tsp import GWOTSPSolver
from bioinspired.algorithms.tsp.local_search import two_opt_improve
from bioinspired.algorithms.tsp.pso_tsp import PSOTSPSolver


class GA2optTSPSolver(GATSPSolver):
    """GA con 2-opt Lamarckiano (Fase 2 memética)."""

    name: ClassVar[str] = "ga_2opt"


class ACO2optTSPSolver(ACOTSPSolver):
    """ACO con 2-opt Lamarckiano (Fase 2 memética)."""

    name: ClassVar[str] = "aco_2opt"


class PSO2optTSPSolver(PSOTSPSolver):
    """PSO con 2-opt Lamarckiano (Fase 2 memética)."""

    name: ClassVar[str] = "pso_2opt"


class GWO2optTSPSolver(GWOTSPSolver):
    """GWO con 2-opt Lamarckiano (Fase 2 memética)."""

    name: ClassVar[str] = "gwo_2opt"


# Fábricas calibradas a n_evaluations ≈ 18 000 sobre _asym(n=10, seed=3).
# Fórmulas teóricas (memetic_elites=1):
#   GA  : pop + iters*(pop - n_elite + 1) = 60 + iters*60
#   ACO : 1 + iters*(n_ants + 1) = 1 + iters*61
#   PSO : pop*(1+iters) + iters*1 = 60 + iters*61
#   GWO : pop*(1+iters) + iters*1 = 60 + iters*61
# Iteraciones estimadas: GA≈299, ACO≈295, PSO≈292, GWO≈292
MEMETIC_FACTORIES: dict[str, Callable[[], BaseTSPAlgorithm]] = {
    "ga_2opt": lambda: GA2optTSPSolver(
        pop_size=60, max_iterations=299, local_search=two_opt_improve, memetic_elites=1
    ),
    "aco_2opt": lambda: ACO2optTSPSolver(
        n_ants=60, max_iterations=295, local_search=two_opt_improve, memetic_elites=1
    ),
    "pso_2opt": lambda: PSO2optTSPSolver(
        n_particles=60, max_iterations=292, local_search=two_opt_improve, memetic_elites=1
    ),
    "gwo_2opt": lambda: GWO2optTSPSolver(
        n_wolves=60, max_iterations=292, local_search=two_opt_improve, memetic_elites=1
    ),
}
