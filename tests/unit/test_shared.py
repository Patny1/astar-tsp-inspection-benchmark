"""Tests para algorithms/soo/_shared.py."""

from __future__ import annotations

import numpy as np

from bioinspired.algorithms.soo._shared import (
    COLLISION_PENALTY,
    N_NEIGHBORS_26,
    VoxelGrid,
    evaluate_with_penalty,
    genome_to_trajectory,
    neighbors_offsets,
    resample_arclength,
    segment_free,
)
from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.scenarios import generate_tiny_scenario
from bioinspired.sim_backends.numpy_backend import NumpyBackend


class TestNeighborsOffsets:
    def test_devuelve_26_vecinos(self) -> None:
        offs = neighbors_offsets()
        assert offs.shape == (26, 3)
        assert offs.dtype == np.int32
        # No incluye centro (0,0,0)
        assert not np.any((offs == 0).all(axis=1))

    def test_constante_n_neighbors_consistente(self) -> None:
        assert N_NEIGHBORS_26 == 26 == neighbors_offsets().shape[0]


class TestVoxelGrid:
    def test_from_problem_shape_correcta(self) -> None:
        problem = generate_tiny_scenario(seed=42)  # 100x100x50
        grid = VoxelGrid.from_problem(problem, voxel_size_m=5.0)
        assert grid.shape == (20, 20, 10)
        assert grid.voxel_size_m == 5.0

    def test_world_voxel_invertible_aprox(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        grid = VoxelGrid.from_problem(problem, voxel_size_m=5.0)
        # Centro de voxel (1,2,3) en mundo
        v_in = (1, 2, 3)
        world = grid.voxels_to_world([v_in])[0]
        v_out = grid.world_to_voxel(world)
        assert v_out == v_in

    def test_world_to_voxel_clipea_a_bounds(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        grid = VoxelGrid.from_problem(problem, voxel_size_m=5.0)
        # Punto muy fuera de bounds
        v = grid.world_to_voxel(np.array([1000.0, 1000.0, 1000.0]))
        assert v == (grid.shape[0] - 1, grid.shape[1] - 1, grid.shape[2] - 1)

    def test_free_mask_dtype_y_shape(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        grid = VoxelGrid.from_problem(problem, voxel_size_m=5.0)
        mask = grid.free_voxel_mask(problem.obstacles_arr)
        assert mask.dtype == bool
        assert mask.shape == grid.shape
        assert (~mask).sum() > 0  # algunos voxels bloqueados


class TestResampleArclength:
    def test_n_iguales_devuelve_mismo(self) -> None:
        path = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=np.float64)
        out = resample_arclength(path, 3)
        np.testing.assert_array_almost_equal(out, path)

    def test_n_mayor_interpola(self) -> None:
        path = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64)
        out = resample_arclength(path, 5)
        assert out.shape == (5, 3)
        np.testing.assert_array_almost_equal(out[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(out[-1], [10, 0, 0])

    def test_path_degenerado(self) -> None:
        path = np.array([[5, 5, 5]], dtype=np.float64)
        out = resample_arclength(path, 4)
        assert out.shape == (4, 3)
        # Deberia repetir el unico punto
        for row in out:
            np.testing.assert_array_almost_equal(row, [5, 5, 5])


class TestSegmentFree:
    def test_sin_obstaculos_es_true(self) -> None:
        assert segment_free(
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            np.empty((0, 4)),
        )

    def test_atraviesa_esfera(self) -> None:
        # Esfera en (5,5,5) radio 3; segmento (0,5,5)->(10,5,5) la atraviesa
        obs = np.array([[5.0, 5.0, 5.0, 3.0]], dtype=np.float64)
        assert not segment_free(np.array([0.0, 5.0, 5.0]), np.array([10.0, 5.0, 5.0]), obs)

    def test_esquiva_esfera(self) -> None:
        obs = np.array([[5.0, 5.0, 5.0, 1.0]], dtype=np.float64)
        # Segmento muy alejado
        assert segment_free(np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), obs)


class TestEvaluateWithPenalty:
    def test_retorna_tupla_raw_y_fitness(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        traj = Trajectory(
            waypoints_arr=np.linspace(problem.start_arr, problem.goal_arr, problem.n_waypoints + 2)
        )
        e_raw, fit = evaluate_with_penalty(traj, problem, NumpyBackend())
        assert isinstance(e_raw, float)
        assert isinstance(fit, float)
        assert fit >= e_raw  # fitness incluye penalizacion (>=0)

    def test_collision_penalty_constante(self) -> None:
        assert COLLISION_PENALTY == 1000.0


class TestGenomeToTrajectory:
    def test_construye_shape_correcta(self) -> None:
        problem = generate_tiny_scenario(seed=42)
        # n_waypoints intermedios = 5; genome flat = 15
        genome = np.arange(problem.n_waypoints * 3, dtype=np.float64)
        traj = genome_to_trajectory(genome, problem)
        assert traj.waypoints_arr.shape == (problem.n_waypoints + 2, 3)
        np.testing.assert_array_equal(traj.waypoints_arr[0], problem.start_arr)
        np.testing.assert_array_equal(traj.waypoints_arr[-1], problem.goal_arr)
