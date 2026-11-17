"""Tests para los modelos de energía."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import BaseEnergyModel, NullEnergyModel


class TestNullEnergyModel:
    def test_es_subclase_de_base(self) -> None:
        assert issubclass(NullEnergyModel, BaseEnergyModel)

    def test_compute_devuelve_distancia_euclidiana(self) -> None:
        model = NullEnergyModel()
        drone = DroneModel()
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
        # distancia = 5.0
        assert model.compute(traj, drone) == pytest.approx(5.0)

    def test_compute_acumula_segmentos(self) -> None:
        model = NullEnergyModel()
        drone = DroneModel()
        traj = Trajectory(
            waypoints_arr=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]])
        )
        # 5 + 12 = 17
        assert model.compute(traj, drone) == pytest.approx(17.0)


class TestBaseEnergyModelEsAbstracta:
    def test_no_se_puede_instanciar_directamente(self) -> None:
        with pytest.raises(TypeError):
            BaseEnergyModel()  # type: ignore[abstract]
