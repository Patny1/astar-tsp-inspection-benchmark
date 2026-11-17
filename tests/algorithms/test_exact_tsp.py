"""Tests para ExactTSPSolver — óptimo por fuerza bruta."""

import numpy as np
import pytest

from bioinspired.algorithms.tsp.exact_tsp import ExactTSPSolver
from bioinspired.environment.scenarios import generate_tiny_tsp_scenario
from bioinspired.experiments.tsp_matrix_cache import build_or_load_matrix


def _tiny_problem(tmp_path):
    from bioinspired.core.tsp_problem import InspectionTSPProblem

    prob = generate_tiny_tsp_scenario(seed=0, n_targets=6)
    (dist, d2t, t2e, *_), _, _ = build_or_load_matrix(
        "astar", prob, "tiny", 6, 0, 5.0, tmp_path, compute_energy=False
    )
    return InspectionTSPProblem.from_path_problem(prob, dist, d2t, t2e)


def test_exact_matches_brute_force(tmp_path) -> None:
    from itertools import permutations

    problem = _tiny_problem(tmp_path)
    solver = ExactTSPSolver(max_n=10)
    result = solver.solve(problem, seed=0)
    opt = min(problem.tour_cost(list(p)) for p in permutations(range(problem.n_targets)))
    assert np.isclose(result.best_cost, opt)


def test_exact_raises_above_max_n(tmp_path) -> None:
    problem = _tiny_problem(tmp_path)
    solver = ExactTSPSolver(max_n=4)
    with pytest.raises(ValueError):
        solver.solve(problem, seed=0)


def test_exact_resuelve_matriz_asimetrica() -> None:
    """El caso que importa: invertir un segmento cambia su costo interno.

    Con una matriz simetrica cualquier implementacion razonable acierta, asi
    que el test se hace sobre una asimetrica construida a proposito.
    """
    from itertools import permutations

    from bioinspired.core.tsp_problem import InspectionTSPProblem
    from bioinspired.environment.drone_model import DroneModel

    rng = np.random.default_rng(0)
    n = 8
    m = rng.uniform(1.0, 20.0, size=(n, n))
    np.fill_diagonal(m, 0.0)
    assert not np.allclose(m, m.T), "la matriz debe ser asimetrica"

    problem = InspectionTSPProblem(
        targets_arr=rng.uniform(0, 10, size=(n, 3)),
        distance_matrix=m,
        depot_to_targets=rng.uniform(1.0, 20.0, size=n),
        targets_to_end=rng.uniform(1.0, 20.0, size=n),
        depot_arr=np.zeros(3),
        end_arr=np.zeros(3),
        scenario_id="asimetrica_sintetica",
        drone_model=DroneModel(),
    )
    resultado = ExactTSPSolver(max_n=10).solve(problem, seed=0)
    esperado = min(problem.tour_cost(list(p)) for p in permutations(range(n)))
    assert np.isclose(resultado.best_cost, esperado)
    # el tour devuelto debe reproducir el costo, no solo empatarlo por casualidad
    assert np.isclose(problem.tour_cost(resultado.best_tour), resultado.best_cost)
    assert sorted(resultado.best_tour) == list(range(n))


def test_exact_escala_mas_alla_de_fuerza_bruta(tmp_path) -> None:
    """n=12 es 479M permutaciones; Held-Karp lo resuelve igual."""
    from bioinspired.core.tsp_problem import InspectionTSPProblem

    prob = generate_tiny_tsp_scenario(seed=1, n_targets=12)
    (dist, d2t, t2e, *_), _, _ = build_or_load_matrix(
        "astar", prob, "tiny", 12, 1, 5.0, tmp_path, compute_energy=False
    )
    problem = InspectionTSPProblem.from_path_problem(prob, dist, d2t, t2e)
    resultado = ExactTSPSolver(max_n=16).solve(problem, seed=0)
    assert np.isclose(problem.tour_cost(resultado.best_tour), resultado.best_cost)
    assert sorted(resultado.best_tour) == list(range(12))
