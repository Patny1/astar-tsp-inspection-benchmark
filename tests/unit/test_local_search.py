"""TDD: two_opt_improve como función reutilizable."""

import numpy as np

from bioinspired.algorithms.tsp.local_search import two_opt_improve
from bioinspired.algorithms.tsp.two_opt_tsp import TwoOptRandomTSPSolver, TwoOptTSPSolver
from bioinspired.core.tsp_problem import InspectionTSPProblem
from bioinspired.environment.drone_model import DroneModel


def _asym(n=8, seed=4):
    rng = np.random.default_rng(seed)
    d = rng.uniform(1.0, 10.0, size=(n, n))
    np.fill_diagonal(d, 0.0)
    return InspectionTSPProblem(
        targets_arr=rng.uniform(0, 50, (n, 3)),
        distance_matrix=d,
        depot_to_targets=rng.uniform(1, 5, n),
        targets_to_end=rng.uniform(1, 5, n),
        depot_arr=np.zeros(3),
        end_arr=np.full(3, 50.0),
        scenario_id="t",
        drone_model=DroneModel(),
    )


def test_two_opt_improve_never_worsens_on_asymmetric():
    p = _asym()
    start = list(range(p.n_targets))
    improved, _ = two_opt_improve(start, p)
    assert p.tour_cost(improved) <= p.tour_cost(start) + 1e-9
    assert sorted(improved) == list(range(p.n_targets))  # permutación válida


def test_two_opt_random_varia_entre_seeds():
    """El ablation ``init="random"`` depende de que el seed cambie el arranque.

    Si esto colapsa (p.ej. un refactor que hardcodea ``default_rng(0)`` como en
    la rama NN de al lado), las 30 corridas del paper devuelven el mismo tour,
    la varianza cae a cero y Wilcoxon reporta un p **falsamente** significativo
    sin emitir warning. Falla silenciosa que llega al paper: por eso se fija.
    """
    p = _asym(n=10)
    solver = TwoOptRandomTSPSolver()
    assert solver.solve(p, seed=0).best_tour != solver.solve(p, seed=1).best_tour


def test_two_opt_random_es_reproducible_por_seed():
    p = _asym(n=10)
    solver = TwoOptRandomTSPSolver()
    a, b = solver.solve(p, seed=7), solver.solve(p, seed=7)
    assert a.best_tour == b.best_tour
    assert sorted(a.best_tour) == list(range(p.n_targets))  # permutación válida
    assert a.metadata["init"] == "random"


def test_two_opt_random_devuelve_ints_de_python():
    """El tour debe ser ``list[int]``, no ``list[np.int64]``.

    ``np.random.permutation`` devuelve int64 y el tipo se filtra hasta el tour.
    Rompe cualquier consumidor que serialice a JSON — el runner del benchmark
    cachea los tours así y falla con TypeError.
    """
    import json

    p = _asym(n=10)
    tour = TwoOptRandomTSPSolver().solve(p, seed=0).best_tour
    assert all(type(i) is int for i in tour)
    json.dumps(tour)  # no debe levantar


def test_two_opt_nn_ignora_el_seed():
    """La variante NN es determinista por diseño: el runner del paper la trata
    como tal (``is_deterministic``) y le asigna un solo algo_seed."""
    p = _asym(n=10)
    solver = TwoOptTSPSolver()
    assert solver.solve(p, seed=0).best_tour == solver.solve(p, seed=99).best_tour
