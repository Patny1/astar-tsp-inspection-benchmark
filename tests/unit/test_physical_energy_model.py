"""Tests para PhysicalEnergyModel."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import PhysicalEnergyModel


class TestPhysicalEnergyModel:
    def test_horizontal_solo_consume_hover_y_cinetica(self) -> None:
        model = PhysicalEnergyModel()
        drone = DroneModel(mass_kg=1.5, cruise_velocity_mps=5.0, hover_power_w=150.0)
        # 5m horizontal, sin Δh
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0]]))
        e = model.compute(traj, drone)
        assert e > 0.0

    def test_subir_consume_mas_que_bajar(self) -> None:
        model = PhysicalEnergyModel()
        drone = DroneModel()
        up = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 50.0]]))
        down = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 5.0]]))
        assert model.compute(up, drone) > model.compute(down, drone)

    def test_acumula_segmentos(self) -> None:
        model = PhysicalEnergyModel()
        drone = DroneModel()
        traj = Trajectory(
            waypoints_arr=np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [10.0, 0.0, 50.0]])
        )
        seg1 = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]]))
        seg2 = Trajectory(waypoints_arr=np.array([[10.0, 0.0, 5.0], [10.0, 0.0, 50.0]]))
        assert model.compute(traj, drone) == pytest.approx(
            model.compute(seg1, drone) + model.compute(seg2, drone)
        )

    def test_proxy_kinetic_term_escala_lineal_con_distancia(self) -> None:
        """Documenta el comportamiento proxy del término cinético.

        En una trayectoria horizontal (Δh = 0), tanto el término cinético
        ``0.5·m·v²·Δt`` como el hover ``hover_power·Δt`` son lineales en
        ``Δt = distancia/velocidad``. Por tanto el coste total escala
        **linealmente con distancia** — NO es energía cinética física
        (que dependería sólo de v² única). Este test ancla la semántica
        documentada en la docstring de PhysicalEnergyModel: el término
        cinético es un **proxy heurístico** del coste sostenido, no
        energía cinética en sentido estricto.
        """
        model = PhysicalEnergyModel()
        drone = DroneModel(mass_kg=1.0, cruise_velocity_mps=1.0, hover_power_w=1.0)
        short = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]]))
        long = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0]]))
        e_short = model.compute(short, drone)
        e_long = model.compute(long, drone)
        # Distancia 10x → coste 10x (cinético y hover ambos escalan con Δt linealmente)
        assert e_long == pytest.approx(10 * e_short)
