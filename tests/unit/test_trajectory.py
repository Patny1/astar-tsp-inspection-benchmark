"""Tests para la clase Trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.core.trajectory import Trajectory


class TestTrajectory:
    def test_construye_con_array_valido(self) -> None:
        waypoints_arr = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        traj = Trajectory(waypoints_arr=waypoints_arr)
        assert traj.waypoints_arr.shape == (2, 3)

    def test_rechaza_array_2d_de_dimension_incorrecta(self) -> None:
        waypoints_arr = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="shape"):
            Trajectory(waypoints_arr=waypoints_arr)

    def test_rechaza_menos_de_dos_puntos(self) -> None:
        waypoints_arr = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="al menos 2 puntos"):
            Trajectory(waypoints_arr=waypoints_arr)

    def test_inmutable(self) -> None:
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        with pytest.raises((AttributeError, TypeError)):
            traj.waypoints_arr = np.zeros((2, 3))  # type: ignore[misc]

    def test_n_puntos(self) -> None:
        traj = Trajectory(waypoints_arr=np.zeros((5, 3)))
        assert traj.n_points == 5

    def test_start_y_goal(self) -> None:
        waypoints_arr = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]])
        traj = Trajectory(waypoints_arr=waypoints_arr)
        np.testing.assert_array_equal(traj.start_arr, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(traj.goal_arr, [10.0, 10.0, 10.0])
