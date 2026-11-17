"""Tests para DroneModel."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.environment.drone_model import DroneModel


class TestDroneModel:
    def test_construye_con_defaults(self) -> None:
        d = DroneModel()
        assert d.mass_kg == 1.5
        assert d.max_velocity_mps == 10.0
        assert d.cruise_velocity_mps == 5.0
        assert d.max_acceleration_mps2 == 3.0
        assert d.max_climb_angle_deg == 30.0
        assert d.max_turn_angle_deg == 45.0
        assert d.min_altitude_m == 2.0
        assert d.max_altitude_m == 120.0
        assert d.battery_capacity_wh == 100.0
        assert d.hover_power_w == 150.0
        assert d.enable_constraints is True
        assert d.body_radius_m == 0.0
        assert d.payload_kg == 0.0
        assert d.wind_speed_mps == 0.0
        assert d.wind_direction_deg == 0.0
        assert d.max_range_m == 0.0
        assert d.endurance_min == 0.0

    def test_overrides(self) -> None:
        d = DroneModel(mass_kg=3.0, enable_constraints=False)
        assert d.mass_kg == 3.0
        assert d.enable_constraints is False

    def test_total_mass_incluye_payload(self) -> None:
        d = DroneModel(mass_kg=1.5, payload_kg=0.5)
        assert d.total_mass_kg == pytest.approx(2.0)

    def test_total_mass_sin_payload(self) -> None:
        d = DroneModel(mass_kg=1.5)
        assert d.total_mass_kg == pytest.approx(1.5)

    def test_wind_vector_norte(self) -> None:
        d = DroneModel(wind_speed_mps=5.0, wind_direction_deg=0.0)
        wv = d.wind_vector
        assert wv[0] == pytest.approx(0.0, abs=1e-9)  # sin(0) = 0 → X=0
        assert wv[1] == pytest.approx(5.0, abs=1e-9)  # cos(0) = 1 → Y=5
        assert wv[2] == pytest.approx(0.0)

    def test_wind_vector_este(self) -> None:
        d = DroneModel(wind_speed_mps=3.0, wind_direction_deg=90.0)
        wv = d.wind_vector
        assert wv[0] == pytest.approx(3.0, abs=1e-9)  # sin(90) = 1 → X=3
        assert wv[1] == pytest.approx(0.0, abs=1e-9)  # cos(90) = 0 → Y=0

    def test_wind_vector_sin_viento(self) -> None:
        d = DroneModel()
        assert np.allclose(d.wind_vector, [0.0, 0.0, 0.0])

    def test_masa_positiva(self) -> None:
        with pytest.raises(ValueError, match="mass_kg"):
            DroneModel(mass_kg=0.0)

    def test_payload_no_negativo(self) -> None:
        with pytest.raises(ValueError, match="payload_kg"):
            DroneModel(payload_kg=-1.0)

    def test_wind_speed_no_negativa(self) -> None:
        with pytest.raises(ValueError, match="wind_speed_mps"):
            DroneModel(wind_speed_mps=-2.0)

    def test_max_range_no_negativo(self) -> None:
        with pytest.raises(ValueError, match="max_range_m"):
            DroneModel(max_range_m=-10.0)

    def test_endurance_no_negativa(self) -> None:
        with pytest.raises(ValueError, match="endurance_min"):
            DroneModel(endurance_min=-5.0)

    def test_velocidades_consistentes(self) -> None:
        with pytest.raises(ValueError, match="cruise_velocity"):
            DroneModel(cruise_velocity_mps=15.0, max_velocity_mps=10.0)

    def test_altitudes_consistentes(self) -> None:
        with pytest.raises(ValueError, match="altitude"):
            DroneModel(min_altitude_m=120.0, max_altitude_m=100.0)

    def test_angulos_en_rango(self) -> None:
        with pytest.raises(ValueError, match="climb"):
            DroneModel(max_climb_angle_deg=95.0)
        with pytest.raises(ValueError, match="turn"):
            DroneModel(max_turn_angle_deg=180.0)
