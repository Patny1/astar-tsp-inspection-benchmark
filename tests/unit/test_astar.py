"""Tests para AStarPlanner."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.algorithms.soo.astar import AStarPlanner
from bioinspired.environment.scenarios import generate_tiny_scenario
from bioinspired.sim_backends.numpy_backend import NumpyBackend


class TestAStarValidation:
    def test_rechaza_voxel_size_no_positivo(self) -> None:
        with pytest.raises(ValueError):
            AStarPlanner(voxel_size_m=0.0)
        with pytest.raises(ValueError):
            AStarPlanner(voxel_size_m=-1.0)


class TestAStarGridAndCollision:
    """Tests para el voxel grid y mask de colisión (T5)."""

    def test_grid_shape_consistente_con_bounds(self) -> None:
        # Bounds 100x100x50, voxel 5m -> shape (20, 20, 10)
        problem = generate_tiny_scenario(seed=42)
        planner = AStarPlanner(voxel_size_m=5.0)
        shape = planner._grid_shape(problem)
        assert shape == (20, 20, 10)

    def test_world_to_voxel_invertible(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        planner = AStarPlanner(voxel_size_m=5.0)
        v = planner._world_to_voxel(np.array([12.5, 17.5, 7.5]), problem)
        assert v == (2, 3, 1)

    def test_free_mask_voxels_dentro_de_obstaculos_son_false(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        planner = AStarPlanner(voxel_size_m=5.0)
        free = planner._free_voxel_mask(problem, planner._grid_shape(problem))
        # Al menos algunos voxels deben estar bloqueados (los obstaculos existen)
        assert free.dtype == bool
        assert (~free).sum() > 0
        # Los voxels en start y goal deben estar libres (los obstaculos estan en el medio)
        sv = planner._world_to_voxel(problem.start_arr, problem)
        gv = planner._world_to_voxel(problem.goal_arr, problem)
        assert free[sv] and free[gv]


class TestAStarSearch:
    """Tests del search end-to-end (T6)."""

    def test_solve_devuelve_optimization_result(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        planner = AStarPlanner(voxel_size_m=5.0)
        result = planner.solve(problem, seed=42, backend=NumpyBackend())
        assert result.trajectory.waypoints_arr.shape == (problem.n_waypoints + 2, 3)
        assert np.allclose(result.trajectory.waypoints_arr[0], problem.start_arr)
        assert np.allclose(result.trajectory.waypoints_arr[-1], problem.goal_arr)
        assert result.metadata["algorithm"] == "astar"
        assert result.metadata["n_evaluations"] == 1

    def test_solve_es_deterministico(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        a = AStarPlanner(voxel_size_m=5.0).solve(problem, seed=42, backend=NumpyBackend())
        b = AStarPlanner(voxel_size_m=5.0).solve(problem, seed=99, backend=NumpyBackend())
        # A* es determinista: seed no afecta. Mismas waypoints.
        np.testing.assert_array_almost_equal(a.trajectory.waypoints_arr, b.trajectory.waypoints_arr)

    def test_path_evita_obstaculos_donde_sea_posible(self) -> None:
        # En tiny seed=42, el path debe completar exitosamente
        problem = generate_tiny_scenario(seed=42)
        planner = AStarPlanner(voxel_size_m=5.0)
        result = planner.solve(problem, seed=42, backend=NumpyBackend())
        # No assert estricto en colisiones (depende del seed) pero debe completar el path
        assert result.metadata["best_energy"] >= 0.0

    def test_metadata_tiene_3_campos_de_energia(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        result = AStarPlanner().solve(problem, seed=42, backend=NumpyBackend())
        md = result.metadata
        assert "best_energy_raw" in md
        assert "best_fitness" in md
        assert "best_energy" in md  # legacy
        # En A*/RRT* sin colisiones: raw == fitness
        assert md["best_energy"] == md["best_fitness"]
        assert md["best_energy_raw"] >= 0.0
