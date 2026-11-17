"""Tests para PathPlanningProblem."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import NullEnergyModel


def _make_problem(**overrides: object) -> PathPlanningProblem:
    defaults: dict[str, object] = {
        "start_arr": np.array([0.0, 0.0, 5.0]),
        "goal_arr": np.array([100.0, 100.0, 5.0]),
        "bounds_arr": np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 50.0]]),
        "obstacles_arr": np.array([[50.0, 50.0, 5.0, 10.0]]),
        "drone_model": DroneModel(),
        "energy_model": NullEnergyModel(),
        "n_waypoints": 5,
        "scenario_id": "test",
        "seed": 42,
    }
    defaults.update(overrides)
    return PathPlanningProblem(**defaults)  # type: ignore[arg-type]


class TestPathPlanningProblem:
    def test_construye(self) -> None:
        p = _make_problem()
        assert p.scenario_id == "test"
        assert p.n_waypoints == 5
        assert p.n_obstacles == 1

    def test_start_y_goal_dentro_de_bounds(self) -> None:
        with pytest.raises(ValueError, match="start_arr"):
            _make_problem(start_arr=np.array([-10.0, 0.0, 5.0]))
        with pytest.raises(ValueError, match="goal_arr"):
            _make_problem(goal_arr=np.array([200.0, 0.0, 5.0]))

    def test_bounds_consistentes(self) -> None:
        with pytest.raises(ValueError, match="bounds"):
            _make_problem(bounds_arr=np.array([[100.0, 100.0, 50.0], [0.0, 0.0, 0.0]]))

    def test_obstacles_arr_shape(self) -> None:
        with pytest.raises(ValueError, match="obstacles_arr"):
            _make_problem(obstacles_arr=np.array([[1.0, 2.0, 3.0]]))  # falta radio

    def test_n_waypoints_positivo(self) -> None:
        with pytest.raises(ValueError, match="n_waypoints"):
            _make_problem(n_waypoints=0)

    def test_inmutable(self) -> None:
        p = _make_problem()
        with pytest.raises((AttributeError, TypeError)):
            p.scenario_id = "otro"  # type: ignore[misc]
