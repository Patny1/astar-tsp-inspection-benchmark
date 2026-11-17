"""El rotor coherente del M210 y el factor de batería.

Cubre la parametrización congelada de Fase 2 en adelante: `RotorParameters`
con el rotor real del aparato, y `powertrain_efficiency` como escalar que
mueve el nivel sin tocar la estructura que consume el optimizador.
"""

from __future__ import annotations

import math

import pytest

from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import (
    POWERTRAIN_EFFICIENCY_M210,
    GongEnergyModel,
    RotorParameters,
)
from bioinspired.experiments.tsp_harness import ENERGY_MODELS, make_energy_model

G = 9.81
W_M210 = 4.8 * G


def _tip_speed_mps(params: RotorParameters, weight_n: float) -> float:
    """Punta de pala en hover que implica la relación C_T del propio paper."""
    radius_m = math.sqrt(params.rotor_disc_area_m2 / math.pi)
    thrust_rotor_n = weight_n / params.n_rotors
    omega = math.sqrt(
        thrust_rotor_n
        / (params.thrust_coefficient * params.air_density_kgpm3 * params.rotor_disc_area_m2)
        / radius_m**2
    )
    return omega * radius_m


class TestRotorCoherente:
    def test_geometria_es_la_helice_real(self) -> None:
        """A sale de la 1760S (17"), no del rotor de 20,5" de la Tabla I."""
        p = RotorParameters.dji_m210()
        radius_m = math.sqrt(p.rotor_disc_area_m2 / math.pi)
        assert radius_m == pytest.approx(17.0 * 0.0254 / 2.0, rel=1e-12)
        assert p.rotor_disc_area_m2 == pytest.approx(0.146438, rel=1e-4)
        assert p.rotor_disc_area_m2 < RotorParameters().rotor_disc_area_m2

    def test_ct_ancla_la_punta_de_pala(self) -> None:
        """El C_T derivado devuelve exactamente la punta de pala pedida."""
        p = RotorParameters.dji_m210()
        assert p.thrust_coefficient == pytest.approx(0.004073, rel=1e-3)
        assert _tip_speed_mps(p, W_M210) == pytest.approx(130.0, rel=1e-9)

        lenta = RotorParameters.dji_m210(tip_speed_hover_mps=110.0)
        assert _tip_speed_mps(lenta, W_M210) == pytest.approx(110.0, rel=1e-9)

    def test_la_tabla_i_sin_corregir_implica_un_rotor_imposible(self) -> None:
        """El contraste que motiva esta parametrización, fijado como regresión."""
        assert _tip_speed_mps(RotorParameters(), W_M210) == pytest.approx(198.5, rel=1e-2)
        assert _tip_speed_mps(RotorParameters.dji_m210(), W_M210) == pytest.approx(130.0)

    def test_la_inducida_domina_el_hover(self) -> None:
        """Con el rotor real la proporción se da vuelta: es un multirrotor."""
        m = GongEnergyModel(params=RotorParameters.dji_m210(), continuous_at_hover=True)
        blade = m._blade_power_w(W_M210)
        induced = m._induced_power_w(W_M210)
        assert blade < induced, "la potencia de perfil no puede dominar el hover"
        assert blade / induced == pytest.approx(0.30, abs=0.02)

        viejo = GongEnergyModel(continuous_at_hover=True)
        assert viejo._blade_power_w(W_M210) > viejo._induced_power_w(W_M210)

    def test_figura_de_merito_plausible(self) -> None:
        m = GongEnergyModel(params=RotorParameters.dji_m210(), continuous_at_hover=True)
        p = m.params
        ideal = W_M210**1.5 / math.sqrt(
            2.0 * p.n_rotors * p.air_density_kgpm3 * p.rotor_disc_area_m2
        )
        assert 0.6 <= ideal / m.power_hover_w(W_M210) <= 0.8

    def test_el_tope_de_descenso_sigue_dentro_del_dominio(self) -> None:
        """A cambia, y con ella SFP_perp/A: verificar que 3 m/s sigue siendo válido."""
        m = GongEnergyModel(params=RotorParameters.dji_m210(), continuous_at_hover=True)
        assert m.power_vertical_w(-3.0, W_M210) > 0.0
        with pytest.raises(ValueError, match="fuera del dominio"):
            m.power_vertical_w(-3.5, W_M210)

    def test_argumentos_invalidos(self) -> None:
        with pytest.raises(ValueError, match="weight_n"):
            RotorParameters.dji_m210(weight_n=0.0)
        with pytest.raises(ValueError, match="tip_speed"):
            RotorParameters.dji_m210(tip_speed_hover_mps=-1.0)


class TestEficienciaDeCadena:
    def test_default_es_el_paper_tal_cual(self) -> None:
        """Sin el factor, la salida es aerodinámica y nada cambia."""
        aero = GongEnergyModel(params=RotorParameters.dji_m210(), continuous_at_hover=True)
        assert aero.powertrain_efficiency == 1.0
        assert aero.power_hover_w(W_M210) == pytest.approx(399.6, rel=1e-3)

    def test_lleva_el_hover_a_la_autonomia_declarada(self) -> None:
        """696 W de batería -> 30 min con los 348 Wh de 2x TB55."""
        m = GongEnergyModel(
            params=RotorParameters.dji_m210(),
            continuous_at_hover=True,
            powertrain_efficiency=POWERTRAIN_EFFICIENCY_M210,
        )
        hover_w = m.power_hover_w(W_M210)
        assert hover_w == pytest.approx(696.0, rel=5e-3)
        drone = DroneModel.dji_m210()
        autonomia_min = drone.battery_capacity_wh / hover_w * 60.0
        assert autonomia_min == pytest.approx(30.0, abs=0.5)

    def test_es_un_escalar_puro(self) -> None:
        """No altera ningún ratio — por eso no puede cambiar el orden de un tour."""
        params = RotorParameters.dji_m210()
        aero = GongEnergyModel(params=params, continuous_at_hover=True)
        bat = GongEnergyModel(
            params=params,
            continuous_at_hover=True,
            powertrain_efficiency=POWERTRAIN_EFFICIENCY_M210,
        )
        for v_par, v_perp in ((5.0, 0.0), (0.0, 2.5), (0.0, -2.5), (10.0, 1.5)):
            a = aero.power_total_w(v_par, v_perp, W_M210)
            b = bat.power_total_w(v_par, v_perp, W_M210)
            assert b == pytest.approx(a / POWERTRAIN_EFFICIENCY_M210, rel=1e-12)

        ratio_aero = aero.power_vertical_w(2.5, W_M210) / aero.power_forward_w(5.0, W_M210)
        ratio_bat = bat.power_vertical_w(2.5, W_M210) / bat.power_forward_w(5.0, W_M210)
        assert ratio_aero == pytest.approx(ratio_bat, rel=1e-12)

    def test_no_se_aplica_dos_veces_al_componer(self) -> None:
        """`power_total_w` compone métodos públicos: el factor entra una sola vez."""
        m = GongEnergyModel(
            params=RotorParameters.dji_m210(),
            continuous_at_hover=True,
            powertrain_efficiency=0.5,
        )
        crudo = GongEnergyModel(params=RotorParameters.dji_m210(), continuous_at_hover=True)
        assert m.power_total_w(5.0, 2.5, W_M210) == pytest.approx(
            crudo.power_total_w(5.0, 2.5, W_M210) / 0.5, rel=1e-12
        )

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_rango_invalido(self, bad: float) -> None:
        with pytest.raises(ValueError, match="powertrain_efficiency"):
            GongEnergyModel(powertrain_efficiency=bad)


class TestVarianteDelHarness:
    def test_esta_registrada(self) -> None:
        assert "gong_coherent" in ENERGY_MODELS

    def test_devuelve_el_modelo_congelado(self) -> None:
        m = make_energy_model("gong_coherent")
        assert isinstance(m, GongEnergyModel)
        assert m.continuous_at_hover is True
        assert m.powertrain_efficiency == POWERTRAIN_EFFICIENCY_M210
        assert m.params.rotor_disc_area_m2 == pytest.approx(
            RotorParameters.dji_m210().rotor_disc_area_m2
        )

    def test_gong_sigue_siendo_la_tabla_i(self) -> None:
        """La primera corrida de Fase 2 tiene que seguir reproduciéndose."""
        m = make_energy_model("gong")
        assert m.params.rotor_disc_area_m2 == 0.214
        assert m.params.thrust_coefficient == 0.001195
        assert m.powertrain_efficiency == 1.0
