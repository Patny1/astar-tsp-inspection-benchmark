"""Tests para metrics.path."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.drone_model import DroneModel
from bioinspired.metrics.path import (
    count_collisions,
    kinematic_violations,
    length,
    smoothness,
)


class TestLength:
    def test_dos_puntos(self) -> None:
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
        assert length(traj) == pytest.approx(5.0)

    def test_segmentos_acumulados(self) -> None:
        traj = Trajectory(
            waypoints_arr=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]])
        )
        assert length(traj) == pytest.approx(17.0)


class TestCountCollisions:
    def test_sin_obstaculos(self) -> None:
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        result = count_collisions(traj, np.zeros((0, 4)))
        assert result == {"waypoints_inside": 0, "segments_crossing": 0, "total": 0}

    def test_waypoint_dentro_de_esfera(self) -> None:
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
        # esfera de radio 1 en (5,0,0) -> el waypoint final esta dentro
        obstacles = np.array([[5.0, 0.0, 0.0, 1.0]])
        result = count_collisions(traj, obstacles)
        assert result["waypoints_inside"] == 1
        assert result["total"] >= 1

    def test_segmento_atraviesa_esfera(self) -> None:
        # Linea de (0,0,0) a (10,0,0) atraviesa la esfera (5,0,0,1)
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        obstacles = np.array([[5.0, 0.0, 0.0, 1.0]])
        result = count_collisions(traj, obstacles)
        assert result["segments_crossing"] >= 1

    def test_segmento_no_toca_esfera_lejana(self) -> None:
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        obstacles = np.array([[5.0, 100.0, 0.0, 1.0]])
        result = count_collisions(traj, obstacles)
        assert result["total"] == 0

    def test_multiples_obstaculos(self) -> None:
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        obstacles = np.array(
            [
                [3.0, 0.0, 0.0, 0.5],  # atravesado por el segmento
                [7.0, 0.0, 0.0, 0.5],  # tambien atravesado
            ]
        )
        result = count_collisions(traj, obstacles)
        assert result["segments_crossing"] == 2


class TestSmoothness:
    def test_linea_recta_es_cero(self) -> None:
        traj = Trajectory(
            waypoints_arr=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        )
        assert smoothness(traj) == pytest.approx(0.0, abs=1e-6)

    def test_zigzag_es_alto(self) -> None:
        traj = Trajectory(
            waypoints_arr=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            )
        )
        # 3 vueltas a 90 grados = pi/2 rad cada una. Promedio = pi/2.
        assert smoothness(traj) == pytest.approx(np.pi / 2, abs=1e-3)

    def test_dos_puntos_o_menos_retorna_cero(self) -> None:
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]))
        assert smoothness(traj) == 0.0


class TestKinematicViolations:
    def test_enable_constraints_false_retorna_ceros(self) -> None:
        drone = DroneModel(enable_constraints=False)
        traj = Trajectory(
            waypoints_arr=np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 200.0]]  # vuelo vertical extremo
            )
        )
        v = kinematic_violations(traj, drone)
        assert v == {
            "turn_violations": 0,
            "climb_violations": 0,
            "altitude_violations": 0,
            "range_violations": 0,
            "endurance_violations": 0,
            "total": 0,
        }

    def test_climb_violation_contada(self) -> None:
        # max_climb_angle_deg=30 default; subir 90m horizontalmente 1m -> ~89 grados -> viola
        drone = DroneModel()
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 100.0]]))
        v = kinematic_violations(traj, drone)
        assert v["climb_violations"] == 1
        assert v["total"] >= 1

    def test_turn_violation_contada(self) -> None:
        # max_turn_angle_deg=45 default; girar 90 grados viola
        drone = DroneModel()
        traj = Trajectory(
            waypoints_arr=np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0]])
        )
        v = kinematic_violations(traj, drone)
        assert v["turn_violations"] == 1

    def test_altitude_violation_contada(self) -> None:
        # min_altitude=2, max_altitude=120 default
        drone = DroneModel()
        traj = Trajectory(
            waypoints_arr=np.array(
                [
                    [0.0, 0.0, 0.5],  # 0.5 < 2
                    [10.0, 10.0, 50.0],
                    [20.0, 20.0, 200.0],  # 200 > 120
                ]
            )
        )
        v = kinematic_violations(traj, drone)
        assert v["altitude_violations"] == 2

    def test_sin_violaciones(self) -> None:
        drone = DroneModel()
        traj = Trajectory(
            waypoints_arr=np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 12.0], [20.0, 5.0, 14.0]])
        )
        v = kinematic_violations(traj, drone)
        assert v["total"] == 0

    def test_total_es_suma(self) -> None:
        drone = DroneModel()
        traj = Trajectory(
            waypoints_arr=np.array(
                [
                    [0.0, 0.0, 0.5],  # altitude_violation
                    [10.0, 0.0, 10.0],
                    [10.0, 10.0, 10.0],  # turn_violation (90 grados)
                    [11.0, 10.0, 100.0],  # climb_violation
                ]
            )
        )
        v = kinematic_violations(traj, drone)
        assert v["total"] == v["turn_violations"] + v["climb_violations"] + v["altitude_violations"]
