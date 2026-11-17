"""Tests Fase 2 — Híbridos TSP: hook de local search + población final en GA, PSO, GWO y ACO."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.algorithms.tsp.aco_tsp import ACOTSPSolver
from bioinspired.algorithms.tsp.ga_tsp import GATSPSolver
from bioinspired.algorithms.tsp.gwo_tsp import GWOTSPSolver
from bioinspired.algorithms.tsp.local_search import two_opt_improve
from bioinspired.algorithms.tsp.pso_tsp import PSOTSPSolver
from bioinspired.algorithms.tsp_hybrid.memetic_tsp import MEMETIC_FACTORIES
from bioinspired.core.tsp_problem import InspectionTSPProblem
from bioinspired.environment.drone_model import DroneModel


def _asym(n: int = 8, seed: int = 2) -> InspectionTSPProblem:
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


def test_ga_exposes_final_pop() -> None:
    res = GATSPSolver(pop_size=10, max_iterations=5).solve(_asym(), seed=0)
    fp = res.metadata["final_pop"]
    assert len(fp) == 10
    assert all(sorted(t) == list(range(8)) for t in fp)
    # ordenada por costo asc
    fc = res.metadata["final_costs"]
    assert list(fc) == sorted(fc)


def test_ga_memetic_beats_or_ties_pure_ga() -> None:
    p = _asym()
    pure = GATSPSolver(pop_size=10, max_iterations=10).solve(p, seed=0).best_cost
    mem = (
        GATSPSolver(pop_size=10, max_iterations=10, local_search=two_opt_improve, memetic_elites=2)
        .solve(p, seed=0)
        .best_cost
    )
    assert mem <= pure + 1e-9


def test_pso_exposes_final_pop() -> None:
    res = PSOTSPSolver(n_particles=10, max_iterations=5).solve(_asym(), seed=0)
    fp = res.metadata["final_pop"]
    assert len(fp) == 10
    assert all(sorted(t) == list(range(8)) for t in fp)
    # ordenada por costo asc
    fc = res.metadata["final_costs"]
    assert list(fc) == sorted(fc)


def test_pso_memetic_beats_or_ties_pure() -> None:
    p = _asym()
    pure = PSOTSPSolver(n_particles=10, max_iterations=10).solve(p, seed=0).best_cost
    mem = (
        PSOTSPSolver(
            n_particles=10, max_iterations=10, local_search=two_opt_improve, memetic_elites=2
        )
        .solve(p, seed=0)
        .best_cost
    )
    assert mem <= pure + 1e-9


def test_gwo_exposes_final_pop() -> None:
    res = GWOTSPSolver(n_wolves=10, max_iterations=5).solve(_asym(), seed=0)
    fp = res.metadata["final_pop"]
    assert len(fp) == 10
    assert all(sorted(t) == list(range(8)) for t in fp)
    # ordenada por costo asc
    fc = res.metadata["final_costs"]
    assert list(fc) == sorted(fc)


def test_gwo_memetic_beats_or_ties_pure() -> None:
    p = _asym()
    pure = GWOTSPSolver(n_wolves=10, max_iterations=10).solve(p, seed=0).best_cost
    mem = (
        GWOTSPSolver(n_wolves=10, max_iterations=10, local_search=two_opt_improve, memetic_elites=2)
        .solve(p, seed=0)
        .best_cost
    )
    assert mem <= pure + 1e-9


def test_aco_exposes_final_pop() -> None:
    res = ACOTSPSolver(n_ants=10, max_iterations=5).solve(_asym(), seed=0)
    fp = res.metadata["final_pop"]
    assert len(fp) == 10
    assert all(sorted(t) == list(range(8)) for t in fp)
    # ordenada por costo asc
    fc = res.metadata["final_costs"]
    assert list(fc) == sorted(fc)


def test_aco_memetic_beats_or_ties_pure() -> None:
    p = _asym()
    pure = ACOTSPSolver(n_ants=10, max_iterations=10).solve(p, seed=0).best_cost
    mem = (
        ACOTSPSolver(n_ants=10, max_iterations=10, local_search=two_opt_improve, memetic_elites=1)
        .solve(p, seed=0)
        .best_cost
    )
    assert mem <= pure + 1e-9


@pytest.mark.parametrize("factory_name", ["ga_2opt", "aco_2opt", "pso_2opt", "gwo_2opt"])
def test_memetic_budget_and_determinism(factory_name: str) -> None:
    p = _asym(n=10, seed=3)
    s1 = MEMETIC_FACTORIES[factory_name]()
    s2 = MEMETIC_FACTORIES[factory_name]()
    r1 = s1.solve(p, seed=0)
    r2 = s2.solve(p, seed=0)
    assert r1.best_tour == r2.best_tour  # determinismo
    assert sorted(r1.best_tour) == list(range(10))  # permutación válida
    assert 17000 <= r1.metadata["n_evaluations"] <= 19000  # presupuesto ~18k


def test_gaco_seeds_from_evolved_not_random() -> None:
    """GACO siembra ACO desde la población evolucionada de GA (no el init random)."""
    from bioinspired.algorithms.tsp_hybrid.ga_aco_tsp import GACOTSPSolver

    p = _asym(n=12, seed=5)
    res = GACOTSPSolver().solve(p, seed=0)
    assert "transition_iter" in res.metadata
    # El seeding ya no recrea el init random: el mejor de la fase A debe ser mucho
    # mejor que el mejor de una población random fresca del mismo tamaño.
    rng = np.random.default_rng(0)
    rand_best = min(p.tour_cost(rng.permutation(12).tolist()) for _ in range(60))
    assert res.metadata["ga_best"] < rand_best


def test_gaco_budget() -> None:
    """GA ~8910 + ACO 1+150*60=9001 → total ≈ 18k evaluaciones."""
    from bioinspired.algorithms.tsp_hybrid.ga_aco_tsp import GACOTSPSolver

    p = _asym(n=12, seed=5)
    res = GACOTSPSolver().solve(p, seed=0)
    assert 17000 <= res.metadata["n_evaluations"] <= 19000


def test_psoaco_seeds_from_evolved_not_random() -> None:
    """PSOACO siembra ACO desde la población evolucionada de PSO (no el init random)."""
    from bioinspired.algorithms.tsp_hybrid.pso_aco_tsp import PSOACOTSPSolver

    p = _asym(n=12, seed=5)
    res = PSOACOTSPSolver().solve(p, seed=0)
    assert "transition_iter" in res.metadata
    # La fase A (PSO evolucionado) debe superar ampliamente un swarm aleatorio fresco.
    rng = np.random.default_rng(0)
    rand_best = min(p.tour_cost(rng.permutation(12).tolist()) for _ in range(60))
    assert res.metadata["pso_best"] < rand_best


def test_psoaco_budget() -> None:
    """PSO 150*60=9000 + ACO 1+150*60=9001 → total ≈ 18k evaluaciones."""
    from bioinspired.algorithms.tsp_hybrid.pso_aco_tsp import PSOACOTSPSolver

    p = _asym(n=12, seed=5)
    res = PSOACOTSPSolver().solve(p, seed=0)
    assert 17000 <= res.metadata["n_evaluations"] <= 19000


def test_gapso_seeds_from_evolved_not_random() -> None:
    """GAPSO siembra PSO desde la población evolucionada de GA (no el init random)."""
    from bioinspired.algorithms.tsp_hybrid.ga_pso_tsp import GAPSOTSPSolver

    p = _asym(n=12, seed=5)
    res = GAPSOTSPSolver().solve(p, seed=0)
    assert "transition_iter" in res.metadata
    # El seeding ya no recrea el init random: el mejor de la fase A debe ser mucho
    # mejor que el mejor de una población random fresca del mismo tamaño.
    rng = np.random.default_rng(0)
    rand_best = min(p.tour_cost(rng.permutation(12).tolist()) for _ in range(60))
    assert res.metadata["ga_best"] < rand_best


def test_gapso_budget() -> None:
    """GA ~8910 + PSO phase 150*60=9000 → total ≈ 18k evaluaciones."""
    from bioinspired.algorithms.tsp_hybrid.ga_pso_tsp import GAPSOTSPSolver

    p = _asym(n=12, seed=5)
    res = GAPSOTSPSolver().solve(p, seed=0)
    assert 17000 <= res.metadata["n_evaluations"] <= 19000
