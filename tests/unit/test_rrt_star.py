"""Tests para RRTStarPlanner."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.algorithms.soo.rrt_star import RRTStarPlanner
from bioinspired.environment.scenarios import generate_tiny_scenario
from bioinspired.sim_backends.numpy_backend import NumpyBackend


class TestRRTStarValidation:
    def test_rechaza_max_iterations_no_positivo(self) -> None:
        with pytest.raises(ValueError):
            RRTStarPlanner(max_iterations=0)

    def test_rechaza_goal_bias_fuera_de_rango(self) -> None:
        with pytest.raises(ValueError):
            RRTStarPlanner(goal_bias=1.5)

    def test_rechaza_step_size_no_positivo(self) -> None:
        with pytest.raises(ValueError):
            RRTStarPlanner(step_size_m=0.0)


class TestRRTStarBasicTreeGrowth:
    def test_solve_devuelve_optimization_result(self) -> None:
        planner = RRTStarPlanner(max_iterations=200)
        problem = generate_tiny_scenario(seed=42)
        result = planner.solve(problem, seed=42, backend=NumpyBackend())
        assert result.trajectory.waypoints_arr.shape == (problem.n_waypoints + 2, 3)
        assert np.allclose(result.trajectory.waypoints_arr[0], problem.start_arr)
        assert (
            np.allclose(result.trajectory.waypoints_arr[-1], problem.goal_arr)
            or result.metadata["reached_goal"] is False
        )
        assert result.metadata["algorithm"] == "rrt_star"

    def test_determinismo_bit_a_bit(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        a = RRTStarPlanner(max_iterations=200).solve(problem, seed=7, backend=NumpyBackend())
        b = RRTStarPlanner(max_iterations=200).solve(problem, seed=7, backend=NumpyBackend())
        np.testing.assert_array_equal(a.trajectory.waypoints_arr, b.trajectory.waypoints_arr)


class TestRRTStarOptimality:
    def test_costo_decrece_con_mas_iteraciones(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        backend = NumpyBackend()
        short = RRTStarPlanner(max_iterations=100).solve(problem, seed=42, backend=backend)
        long = RRTStarPlanner(max_iterations=2000).solve(problem, seed=42, backend=backend)
        # Más iteraciones → energía igual o menor (asintóticamente óptimo)
        assert long.metadata["best_energy"] <= short.metadata["best_energy"] * 1.05

    def test_alcanza_goal_en_tiny(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        result = RRTStarPlanner(max_iterations=2000).solve(problem, seed=42, backend=NumpyBackend())
        assert result.metadata["reached_goal"] is True

    def test_metadata_tiene_3_campos_de_energia(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        result = RRTStarPlanner().solve(problem, seed=42, backend=NumpyBackend())
        md = result.metadata
        assert "best_energy_raw" in md
        assert "best_fitness" in md
        assert "best_energy" in md  # legacy
        # En A*/RRT* sin colisiones: raw == fitness
        assert md["best_energy"] == md["best_fitness"]
        assert md["best_energy_raw"] >= 0.0


class TestRRTStarSnapshotEvery:
    def test_snapshots_capturados_cada_n_iters(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        # max_iterations=300, snapshot_every=100 → 3 snaps en iters 100, 200, 300.
        # 300 iters dan tiempo a alcanzar el goal en tiny.
        planner = RRTStarPlanner(max_iterations=300, snapshot_every=100)
        result = planner.solve(problem, seed=42, backend=NumpyBackend())
        snaps = result.metadata["snapshots"]
        assert len(snaps) == 3
        assert snaps[0]["iteration"] == 100
        assert snaps[1]["iteration"] == 200
        assert snaps[2]["iteration"] == 300
        # Cada snapshot trae waypoints con shape canónico tras resample.
        for s in snaps:
            assert s["best_waypoints_arr"].shape == (problem.n_waypoints + 2, 3)
            assert isinstance(s["best_fitness"], float)

    def test_sin_snapshot_every_lista_vacia(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        planner = RRTStarPlanner(max_iterations=100)
        result = planner.solve(problem, seed=42, backend=NumpyBackend())
        assert result.metadata["snapshots"] == []

    def test_snapshot_every_cero_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            RRTStarPlanner(snapshot_every=0)

    def test_snapshot_every_negativo_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            RRTStarPlanner(snapshot_every=-3)
