"""Tiempo de vuelo por tramo: la misma descomposicion en piernas que la energia.

`leg_energy_j` ya calcula el tiempo por dentro (potencia x tiempo por pierna) y
lo descarta. `leg_time_s` lo expone: horizontal a crucero, ascenso y descenso a
sus tasas. Es un objetivo de Fase 3 en adelante; Fase 1/2 no lo piden nunca.
"""

from __future__ import annotations

import pytest

from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import leg_time_s
from bioinspired.experiments.tsp_harness import make_energy_model


def test_tres_piernas_a_velocidad_constante() -> None:
    assert leg_time_s(100.0, 20.0, 10.0, 5.0, 2.5, 2.5) == pytest.approx(20.0 + 8.0 + 4.0)


def test_tramo_nulo_no_tarda() -> None:
    assert leg_time_s(0.0, 0.0, 0.0, 5.0, 2.5, 2.5) == 0.0


def test_el_modelo_congelado_usa_sus_tasas_verticales() -> None:
    """Con el M210 y gong_coherent: crucero 5 m/s, vertical 2,5 m/s (5*sin 30)."""
    modelo = make_energy_model("gong_coherent")
    m210 = DroneModel.dji_m210()
    assert modelo.vertical_rates_mps(m210) == pytest.approx((2.5, 2.5))
    assert modelo.leg_time_s(100.0, 20.0, 10.0, m210) == pytest.approx(32.0)


def test_la_energia_es_potencia_por_este_tiempo() -> None:
    """Identidad pierna a pierna: E = P(v) * t. Si divergen, tiempo y energia
    dejaron de ser el mismo modelo."""
    modelo = make_energy_model("gong_coherent")
    m210 = DroneModel.dji_m210()
    w = m210.total_mass_kg * modelo.GRAVITY_MPS2
    v_climb, v_descent = modelo.vertical_rates_mps(m210)
    assert modelo.leg_energy_j(100.0, 0.0, 0.0, m210) == pytest.approx(
        modelo.power_forward_w(5.0, w) * modelo.leg_time_s(100.0, 0.0, 0.0, m210)
    )
    assert modelo.leg_energy_j(0.0, 20.0, 0.0, m210) == pytest.approx(
        modelo.power_vertical_w(v_climb, w) * modelo.leg_time_s(0.0, 20.0, 0.0, m210)
    )
    assert modelo.leg_energy_j(0.0, 0.0, 10.0, m210) == pytest.approx(
        modelo.power_vertical_w(-v_descent, w) * modelo.leg_time_s(0.0, 0.0, 10.0, m210)
    )
