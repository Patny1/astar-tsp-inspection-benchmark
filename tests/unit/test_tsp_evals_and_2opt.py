"""TDD: 2-opt correcto en matrices ASIMÉTRICAS + n_evaluations real por solver.

- Las matrices A*/RRT* reales son asimétricas; el delta de 2-opt debe contar las
  aristas internas que se revierten, no solo las de borde.
- Cada solver estocástico debe reportar metadata['n_evaluations'] = nº real de
  llamadas a tour_cost (hoy caen al fallback POP*ITER en tsp_runner).
"""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.algorithms.tsp.aco_tsp import ACOTSPSolver
from bioinspired.algorithms.tsp.ga_tsp import GATSPSolver
from bioinspired.algorithms.tsp.gwo_tsp import GWOTSPSolver
from bioinspired.algorithms.tsp.pso_tsp import PSOTSPSolver
from bioinspired.algorithms.tsp.two_opt_tsp import TwoOptTSPSolver
from bioinspired.core.tsp_problem import InspectionTSPProblem
from bioinspired.environment.drone_model import DroneModel


def _make_asym_problem(n: int = 8, seed: int = 0) -> InspectionTSPProblem:
    rng = np.random.default_rng(seed)
    dist = rng.uniform(1.0, 10.0, size=(n, n))
    np.fill_diagonal(dist, 0.0)
    return InspectionTSPProblem(
        targets_arr=rng.uniform(0.0, 50.0, size=(n, 3)),
        distance_matrix=dist,
        depot_to_targets=rng.uniform(1.0, 5.0, size=n),
        targets_to_end=rng.uniform(1.0, 5.0, size=n),
        depot_arr=np.zeros(3, dtype=np.float64),
        end_arr=np.full(3, 50.0, dtype=np.float64),
        scenario_id="test_asym",
        drone_model=DroneModel(),
    )


def test_two_opt_monotone_on_asymmetric_matrix() -> None:
    """Sobre una matriz asimétrica, 2-opt no debe aceptar movimientos que empeoren:
    su convergence_history debe ser monótona no-creciente."""
    problem = _make_asym_problem(n=8, seed=4)
    res = TwoOptTSPSolver().solve(problem)
    h = res.convergence_history
    assert np.all(np.diff(h) <= 1e-9), f"2-opt aceptó un movimiento que empeoró: {h.tolist()}"


class _CountingProblem:
    """Delega todo a un InspectionTSPProblem real, pero cuenta las llamadas a tour_cost."""

    def __init__(self, inner: InspectionTSPProblem) -> None:
        self.inner = inner
        self.calls = 0

    def tour_cost(self, tour: list[int] | np.ndarray) -> float:
        self.calls += 1
        return self.inner.tour_cost(tour)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


@pytest.mark.parametrize(
    "solver",
    [
        GATSPSolver(pop_size=10, max_iterations=5),
        ACOTSPSolver(n_ants=10, max_iterations=5),
        PSOTSPSolver(n_particles=10, max_iterations=5),
        GWOTSPSolver(n_wolves=10, max_iterations=5),
    ],
    ids=["ga", "aco", "pso", "gwo"],
)
def test_n_evaluations_matches_actual_tour_cost_calls(solver: object) -> None:
    """metadata['n_evaluations'] debe igualar el nº real de llamadas a tour_cost."""
    counting = _CountingProblem(_make_asym_problem(n=6, seed=1))
    res = solver.solve(counting, seed=0)  # type: ignore[attr-defined]
    assert "n_evaluations" in res.metadata, "el solver no reporta n_evaluations"
    assert res.metadata["n_evaluations"] == counting.calls
