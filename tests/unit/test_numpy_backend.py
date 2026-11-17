"""Tests para NumpyBackend."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import NullEnergyModel
from bioinspired.sim_backends.base import SimResult
from bioinspired.sim_backends.numpy_backend import NumpyBackend


def _make_problem(obstacles_arr: np.ndarray | None = None) -> PathPlanningProblem:
    return PathPlanningProblem(
        start_arr=np.array([0.0, 0.0, 5.0]),
        goal_arr=np.array([10.0, 10.0, 5.0]),
        bounds_arr=np.array([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]),
        obstacles_arr=obstacles_arr if obstacles_arr is not None else np.zeros((0, 4)),
        drone_model=DroneModel(),
        energy_model=NullEnergyModel(),
        n_waypoints=2,
        scenario_id="t",
        seed=42,
    )


class TestNumpyBackend:
    def test_evaluate_devuelve_simresult(self) -> None:
        backend = NumpyBackend()
        problem = _make_problem()
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 5.0], [10.0, 10.0, 5.0]]))
        result = backend.evaluate(traj, problem)
        assert isinstance(result, SimResult)

    def test_completion_true_si_llega_al_goal(self) -> None:
        backend = NumpyBackend()
        problem = _make_problem()
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 5.0], [10.0, 10.0, 5.0]]))
        result = backend.evaluate(traj, problem)
        assert result.completion is True

    def test_completion_false_si_no_llega(self) -> None:
        backend = NumpyBackend()
        problem = _make_problem()
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 5.0], [5.0, 5.0, 5.0]]))
        result = backend.evaluate(traj, problem)
        assert result.completion is False

    def test_energy_es_distancia_con_null_model(self) -> None:
        backend = NumpyBackend()
        problem = _make_problem()
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
        result = backend.evaluate(traj, problem)
        assert result.energy == pytest.approx(5.0)

    def test_collisions_cuenta_total(self) -> None:
        backend = NumpyBackend()
        problem = _make_problem(obstacles_arr=np.array([[5.0, 5.0, 5.0, 2.0]]))
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 5.0], [10.0, 10.0, 5.0]]))
        result = backend.evaluate(traj, problem)
        # el segmento atraviesa la esfera
        assert result.collisions >= 1

    def test_metadata_marca_backend(self) -> None:
        backend = NumpyBackend()
        problem = _make_problem()
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 5.0], [10.0, 10.0, 5.0]]))
        result = backend.evaluate(traj, problem)
        assert result.metadata.get("backend") == "numpy"
