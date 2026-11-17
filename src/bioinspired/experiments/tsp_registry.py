"""Registry compartido de escenarios y solvers TSP (factories picklables)."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

from bioinspired.algorithms.tsp import (
    ACOTSPSolver,
    GATSPSolver,
    GWOTSPSolver,
    NNTSPSolver,
    PSOTSPSolver,
    TwoOptRandomTSPSolver,
    TwoOptTSPSolver,
)
from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm
from bioinspired.algorithms.tsp.local_search import two_opt_improve
from bioinspired.algorithms.tsp_hybrid import GACOTSPSolver, GAPSOTSPSolver, PSOACOTSPSolver
from bioinspired.algorithms.tsp_hybrid.memetic_tsp import (
    ACO2optTSPSolver,
    GA2optTSPSolver,
    GWO2optTSPSolver,
    PSO2optTSPSolver,
)
from bioinspired.core.problem import PathPlanningProblem
from bioinspired.environment.scenarios import (
    generate_building_tsp_scenario,
    generate_city_tsp_scenario,
    generate_forest_tsp_scenario,
    generate_solar_farm_tsp_scenario,
    generate_tiny_tsp_scenario,
    generate_urban_tsp_scenario,
)
from bioinspired.experiments.tsp_presets import PRESET_FASE1

POP_SIZE = 60
MAX_ITER = 300  # evals estocásticos = POP_SIZE * MAX_ITER = 18_000


def _tiny(seed: int, n_targets: int) -> PathPlanningProblem:
    """Wrapper top-level (seed, n_targets) → tiny TSP scenario."""
    return generate_tiny_tsp_scenario(seed=seed, n_targets=n_targets)


def _urban(seed: int, n_targets: int) -> PathPlanningProblem:
    """Wrapper top-level (seed, n_targets) → urban TSP scenario."""
    return generate_urban_tsp_scenario(seed=seed, n_targets=n_targets)


def _forest(seed: int, n_targets: int) -> PathPlanningProblem:
    """Wrapper top-level (seed, n_targets) → forest TSP scenario."""
    return generate_forest_tsp_scenario(seed=seed, n_targets=n_targets)


def _building(seed: int, n_targets: int) -> PathPlanningProblem:
    """Wrapper top-level (seed, n_targets) → building inspection TSP scenario."""
    return generate_building_tsp_scenario(seed=seed, n_targets=n_targets)


def _solar_farm(seed: int, n_targets: int) -> PathPlanningProblem:
    """Wrapper top-level (seed, n_targets) → solar farm inspection TSP scenario."""
    return generate_solar_farm_tsp_scenario(seed=seed, n_targets=n_targets)


def _city(seed: int, n_targets: int) -> PathPlanningProblem:
    """Wrapper top-level (seed, n_targets) → city block inspection TSP scenario."""
    return generate_city_tsp_scenario(seed=seed, n_targets=n_targets)


# Wrappers (seed, n_targets) → llaman a las factories reales por keyword. Top-level
# (no lambdas) para firma uniforme y picklabilidad.
SCENARIO_FACTORIES: dict[str, Callable[[int, int], PathPlanningProblem]] = {
    "tiny": _tiny,
    "urban": _urban,
    "forest": _forest,
    "building": _building,
    "solar_farm": _solar_farm,
    "city": _city,
}

DETERMINISTIC_ORDERERS = frozenset({"nn", "two_opt"})


# Clase de cada ordenador. Separado del preset: el preset dice CON QUÉ valores,
# esto dice QUÉ se instancia.
_CLASES: dict[str, type[BaseTSPAlgorithm]] = {
    "nn": NNTSPSolver,
    "two_opt": TwoOptTSPSolver,
    "two_opt_random": TwoOptRandomTSPSolver,
    "ga": GATSPSolver,
    "pso": PSOTSPSolver,
    "gwo": GWOTSPSolver,
    "aco": ACOTSPSolver,
    "ga_aco": GACOTSPSolver,
    "pso_aco": PSOACOTSPSolver,
    "ga_pso": GAPSOTSPSolver,
    "ga_2opt": GA2optTSPSolver,
    "aco_2opt": ACO2optTSPSolver,
    "pso_2opt": PSO2optTSPSolver,
    "gwo_2opt": GWO2optTSPSolver,
}

# Los meméticos necesitan el hook de búsqueda local, que no es un valor y por eso
# no vive en el preset.
_EXTRA = {
    n: {"local_search": two_opt_improve} for n in ("ga_2opt", "aco_2opt", "pso_2opt", "gwo_2opt")
}


def _construir(clase: type[BaseTSPAlgorithm], valores: dict[str, Any]) -> BaseTSPAlgorithm:
    """Instancia pasando sólo lo que el constructor acepta.

    Los presets registran los parámetros **efectivos**, y algunas subclases fijan
    uno en vez de exponerlo: `TwoOptRandomTSPSolver` hereda `init` pero lo clava
    en "random" y no lo recibe. Filtrar por la firma evita tener que mantener una
    lista de excepciones — y no puede ocultar un error, porque
    `test_tsp_presets.py` compara los atributos resultantes contra el preset.
    """
    acepta = inspect.signature(clase.__init__).parameters
    return clase(**{k: v for k, v in valores.items() if k in acepta})


def make_orderers(
    preset: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Callable[[], BaseTSPAlgorithm]]:
    """Dict nombre -> factory zero-arg (un solver fresco por corrida).

    Args:
        preset: valores de hiperparámetros. ``None`` usa ``PRESET_FASE1``, que es
            con lo que corrieron Fase 1 y la primera Fase 2 — el default se
            mantiene ahí para que nada cambie sin pedirlo. Fase 2 en adelante
            pasa ``PRESET_CANONICO``; ver `tsp_presets.py`.
    """
    p = PRESET_FASE1 if preset is None else preset
    faltantes = set(_CLASES) - set(p)
    if faltantes:
        raise ValueError(f"el preset no cubre {sorted(faltantes)}")
    return {
        nombre: partial(_construir, clase, {**p[nombre], **_EXTRA.get(nombre, {})})
        for nombre, clase in _CLASES.items()
    }
