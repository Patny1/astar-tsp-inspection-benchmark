"""Tests para GongEnergyModel (Gong et al., 2023, IEEE T-AES 59(6):7409-7422).

Los valores esperados se derivan de las Ecs. (5), (8), (13), (14) y (19) con
los parámetros de la Tabla I del paper, y las aserciones cualitativas replican
los hallazgos que el paper reporta en la Sec. IV-C. Ecuaciones y secciones
numeradas igual que en el preprint arXiv:2209.04128v1; las figuras no (la
Fig. 8 publicada es la Fig. 7 del preprint).
"""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import (
    BaseEnergyModel,
    GongEnergyModel,
    RotorParameters,
)

# Peso de simulación de la Tabla I
W_TABLE_I = 20.0


@pytest.fixture
def model() -> GongEnergyModel:
    return GongEnergyModel()


class TestRotorParameters:
    def test_defaults_son_los_de_la_tabla_i(self) -> None:
        p = RotorParameters()
        assert p.profile_drag_coefficient == pytest.approx(0.011)
        assert p.air_density_kgpm3 == pytest.approx(1.168)
        assert p.rotor_solidity == pytest.approx(0.045)
        assert p.rotor_disc_area_m2 == pytest.approx(0.214)
        assert p.induced_power_correction == pytest.approx(0.11)
        assert p.thrust_coefficient == pytest.approx(0.001195)
        assert p.fuselage_area_horizontal_m2 == pytest.approx(0.009)
        assert p.fuselage_area_vertical_m2 == pytest.approx(0.377)
        assert p.n_rotors == 4

    def test_rechaza_numero_impar_de_rotores(self) -> None:
        # El paper deriva sólo para multirrotores con n par (Sec. III-A).
        with pytest.raises(ValueError, match="par"):
            RotorParameters(n_rotors=3)

    def test_rechaza_parametros_no_positivos(self) -> None:
        with pytest.raises(ValueError, match="air_density_kgpm3"):
            RotorParameters(air_density_kgpm3=0.0)


class TestPotenciaHover:
    def test_es_subclase_de_base(self) -> None:
        assert issubclass(GongEnergyModel, BaseEnergyModel)

    def test_hover_ec5(self, model: GongEnergyModel) -> None:
        # P_bl + P_in con los parámetros de la Tabla I y W = 20 N.
        assert model.power_hover_w(W_TABLE_I) == pytest.approx(204.19, abs=0.05)

    def test_hover_escala_con_w_a_la_3_medios(self, model: GongEnergyModel) -> None:
        # Ambos términos de la Ec. (5) son ∝ W^{3/2}.
        assert model.power_hover_w(2.0 * W_TABLE_I) == pytest.approx(
            model.power_hover_w(W_TABLE_I) * 2.0**1.5
        )


class TestVueloHorizontal:
    def test_continuidad_con_hover_en_v_cero(self, model: GongEnergyModel) -> None:
        assert model.power_forward_w(0.0, W_TABLE_I) == pytest.approx(
            model.power_hover_w(W_TABLE_I)
        )

    def test_curva_en_u_con_minimo_interior(self, model: GongEnergyModel) -> None:
        # Sec. IV-C / Fig. 8(a): "only slightly decreasing around 8 m/s". El
        # preprint (Fig. 7(a)) lo decía como "first decreases and then increases".
        speeds = np.linspace(0.0, 15.0, 301)
        powers = np.array([model.power_forward_w(float(v), W_TABLE_I) for v in speeds])
        v_min = float(speeds[int(np.argmin(powers))])
        assert 6.0 < v_min < 11.0

    def test_potencia_de_crucero_puede_bajar_del_hover(self, model: GongEnergyModel) -> None:
        # Hallazgo explícito del paper: la potencia inducida cae al avanzar.
        assert model.power_forward_w(8.0, W_TABLE_I) < model.power_hover_w(W_TABLE_I)

    def test_crece_fuerte_a_alta_velocidad(self, model: GongEnergyModel) -> None:
        # A partir de ~11 m/s domina la potencia parásita (∝ V³).
        assert model.power_forward_w(15.0, W_TABLE_I) > model.power_hover_w(W_TABLE_I)


class TestVueloVertical:
    def test_ascenso_siempre_por_encima_del_hover(self, model: GongEnergyModel) -> None:
        hover = model.power_hover_w(W_TABLE_I)
        for v in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
            assert model.power_vertical_w(v, W_TABLE_I) > hover

    def test_descender_no_es_gratis(self, model: GongEnergyModel) -> None:
        # Ec. (10): descender exige empuje, así que la potencia es del mismo
        # orden que el hover — muy lejos del cero que asume max(0, Δh).
        hover = model.power_hover_w(W_TABLE_I)
        for v in (0.5, 1.0, 2.0, 3.0):
            assert model.power_vertical_w(-v, W_TABLE_I) > 0.5 * hover

    def test_ascenso_cuesta_mas_que_descenso(self, model: GongEnergyModel) -> None:
        for v in (0.5, 1.0, 2.0, 3.0):
            assert model.power_vertical_w(v, W_TABLE_I) > model.power_vertical_w(-v, W_TABLE_I)

    def test_la_asimetria_crece_con_la_velocidad_vertical(self, model: GongEnergyModel) -> None:
        # "this difference will become more obvious with V increasing" (Sec. III-C).
        gaps = [
            model.power_vertical_w(v, W_TABLE_I) - model.power_vertical_w(-v, W_TABLE_I)
            for v in (0.5, 1.0, 2.0, 3.0)
        ]
        assert gaps == sorted(gaps)

    def test_la_asimetria_se_desvanece_al_bajar_la_velocidad(self, model: GongEnergyModel) -> None:
        # Corolario clave: la asimetría de la matriz no es una constante libre.
        e_up = model.power_vertical_w(0.1, W_TABLE_I) / 0.1
        e_down = model.power_vertical_w(-0.1, W_TABLE_I) / 0.1
        assert e_up / e_down == pytest.approx(1.0, abs=0.01)

    def test_velocidad_cero_devuelve_hover(self, model: GongEnergyModel) -> None:
        assert model.power_vertical_w(0.0, W_TABLE_I) == pytest.approx(
            model.power_hover_w(W_TABLE_I)
        )

    def test_rechaza_descenso_fuera_del_dominio_validado(self, model: GongEnergyModel) -> None:
        # Por encima de ~4.8 m/s la Ec. (14) cambia de signo: no extrapolar.
        with pytest.raises(ValueError, match="dominio validado"):
            model.power_vertical_w(-3.5, W_TABLE_I)

    def test_rechaza_ascenso_fuera_del_dominio_validado(self, model: GongEnergyModel) -> None:
        with pytest.raises(ValueError, match="dominio validado"):
            model.power_vertical_w(6.0, W_TABLE_I)


class TestDiscontinuidadEnHover:
    """Las Ecs. (13)/(14) no convergen al hover: cuentan la potencia inducida dos veces.

    El paper NO lo pasa por alto — su Ec. (17) define ``P_perp(0) = P_mh`` por
    casos, así que el modelo es discontinuo *por construcción*, no mal derivado.
    Lo que aporta ``continuous_at_hover=True`` es evitar la consecuencia para un
    optimizador: con la definición por tramos, un Delta-z arbitrariamente pequeño
    paga el escalón completo (63,25 W sobre 204,19 W de hover con la Tabla I, un
    31 %), lo que penaliza de forma artificial los tours con micro-ajustes de
    altitud.
    """

    def test_las_ecuaciones_publicadas_no_convergen_al_hover(self, model: GongEnergyModel) -> None:
        hover = model.power_hover_w(W_TABLE_I)
        salto = model.power_vertical_w(1e-9, W_TABLE_I) - hover
        # El escalón vale exactamente (W/2)·√(2W/(n·rho·A)) = P_in/(1+k).
        p = model.params
        esperado = (W_TABLE_I / 2.0) * np.sqrt(
            2.0 * W_TABLE_I / (p.n_rotors * p.air_density_kgpm3 * p.rotor_disc_area_m2)
        )
        assert salto == pytest.approx(esperado, rel=1e-6)
        assert salto == pytest.approx(
            model._induced_power_w(W_TABLE_I) / (1.0 + p.induced_power_correction),
            rel=1e-9,
        )

    def test_la_correccion_restaura_la_continuidad(self) -> None:
        corregido = GongEnergyModel(continuous_at_hover=True)
        hover = corregido.power_hover_w(W_TABLE_I)
        assert corregido.power_vertical_w(1e-9, W_TABLE_I) == pytest.approx(hover, rel=1e-6)
        assert corregido.power_vertical_w(-1e-9, W_TABLE_I) == pytest.approx(hover, rel=1e-6)

    def test_la_correccion_preserva_el_orden_ascenso_descenso(self) -> None:
        corregido = GongEnergyModel(continuous_at_hover=True)
        for v in (0.5, 1.0, 2.0, 3.0):
            assert corregido.power_vertical_w(v, W_TABLE_I) > corregido.power_vertical_w(
                -v, W_TABLE_I
            )

    def test_con_correccion_el_descenso_baja_del_hover(self) -> None:
        # Sin el término duplicado, descender sí resulta más barato que flotar.
        corregido = GongEnergyModel(continuous_at_hover=True)
        hover = corregido.power_hover_w(W_TABLE_I)
        assert corregido.power_vertical_w(-3.0, W_TABLE_I) < hover


class TestModeloGenerico3D:
    def test_ec19_se_reduce_a_horizontal_cuando_v_vertical_es_cero(
        self, model: GongEnergyModel
    ) -> None:
        assert model.power_total_w(5.0, 0.0, W_TABLE_I) == pytest.approx(
            model.power_forward_w(5.0, W_TABLE_I)
        )

    def test_ec19_se_reduce_a_vertical_cuando_v_horizontal_es_cero(
        self, model: GongEnergyModel
    ) -> None:
        assert model.power_total_w(0.0, 2.0, W_TABLE_I) == pytest.approx(
            model.power_vertical_w(2.0, W_TABLE_I)
        )

    def test_ascender_avanzando_cuesta_mas_que_descender_avanzando(
        self, model: GongEnergyModel
    ) -> None:
        assert model.power_total_w(5.0, 2.0, W_TABLE_I) > model.power_total_w(5.0, -2.0, W_TABLE_I)


class TestTasasVerticales:
    def test_se_derivan_del_angulo_de_ascenso_del_dron(self, model: GongEnergyModel) -> None:
        drone = DroneModel(cruise_velocity_mps=5.0, max_climb_angle_deg=30.0)
        climb, descent = model.vertical_rates_mps(drone)
        assert climb == pytest.approx(2.5)  # 5 · sin(30°)
        assert descent == pytest.approx(2.5)

    def test_se_topean_al_dominio_validado(self) -> None:
        m = GongEnergyModel()
        drone = DroneModel(
            max_velocity_mps=20.0, cruise_velocity_mps=20.0, max_climb_angle_deg=80.0
        )
        climb, descent = m.vertical_rates_mps(drone)
        assert climb == pytest.approx(5.0)
        assert descent == pytest.approx(3.0)

    def test_los_overrides_explicitos_tienen_prioridad(self) -> None:
        m = GongEnergyModel(climb_rate_mps=1.0, descent_rate_mps=0.5)
        climb, descent = m.vertical_rates_mps(DroneModel())
        assert (climb, descent) == pytest.approx((1.0, 0.5))

    def test_rechaza_tasas_no_positivas(self) -> None:
        with pytest.raises(ValueError, match="climb_rate_mps"):
            GongEnergyModel(climb_rate_mps=0.0)


class TestComputeSobreTrayectoria:
    def test_tramo_nivelado_es_potencia_horizontal_por_tiempo(self, model: GongEnergyModel) -> None:
        drone = DroneModel()
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 10.0], [30.0, 40.0, 10.0]]))
        weight = drone.total_mass_kg * GongEnergyModel.GRAVITY_MPS2
        expected = (
            model.power_forward_w(drone.cruise_velocity_mps, weight)
            * 50.0
            / drone.cruise_velocity_mps
        )
        assert model.compute(traj, drone) == pytest.approx(expected)

    def test_trayectoria_degenerada_no_consume(self, model: GongEnergyModel) -> None:
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]))
        assert model.compute(traj, DroneModel()) == 0.0

    def test_descender_no_es_gratis(self, model: GongEnergyModel) -> None:
        # Contraste directo con PhysicalEnergyModel, que usa max(0, Δh).
        drone = DroneModel()
        bajada = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 10.0]]))
        assert model.compute(bajada, drone) > 0.0

    def test_subir_cuesta_mas_que_bajar_la_misma_altura(self, model: GongEnergyModel) -> None:
        drone = DroneModel()
        subida = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 50.0]]))
        bajada = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 10.0]]))
        assert model.compute(subida, drone) > model.compute(bajada, drone)

    def test_penaliza_el_zigzag_de_altitud(self, model: GongEnergyModel) -> None:
        # Mismo desplazamiento neto, distinto vaivén vertical: el modelo lo cobra.
        drone = DroneModel()
        directo = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 30.0]]))
        zigzag = Trajectory(
            waypoints_arr=np.array(
                [[0.0, 0.0, 10.0], [0.0, 0.0, 60.0], [0.0, 0.0, 20.0], [0.0, 0.0, 30.0]]
            )
        )
        assert model.compute(zigzag, drone) > model.compute(directo, drone)

    def test_es_aditivo_por_segmentos(self, model: GongEnergyModel) -> None:
        drone = DroneModel()
        a = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 10.0], [30.0, 0.0, 20.0]]))
        b = Trajectory(waypoints_arr=np.array([[30.0, 0.0, 20.0], [30.0, 40.0, 15.0]]))
        ab = Trajectory(
            waypoints_arr=np.array([[0.0, 0.0, 10.0], [30.0, 0.0, 20.0], [30.0, 40.0, 15.0]])
        )
        assert model.compute(ab, drone) == pytest.approx(
            model.compute(a, drone) + model.compute(b, drone)
        )

    def test_energia_crece_con_la_masa(self, model: GongEnergyModel) -> None:
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 40.0]]))
        ligero = model.compute(traj, DroneModel(mass_kg=1.5))
        pesado = model.compute(traj, DroneModel(mass_kg=3.0))
        assert pesado > ligero


class TestIntegracionConLaMatrizTSP:
    """Cableado en DistanceMatrixBuilder — opt-in, sin tocar el default."""

    @staticmethod
    def _build(energy_model: GongEnergyModel | None) -> tuple[np.ndarray, np.ndarray]:
        from bioinspired.core.tsp_problem import DistanceMatrixBuilder
        from bioinspired.environment.scenarios import generate_tiny_tsp_scenario

        problem = generate_tiny_tsp_scenario(seed=0, n_targets=5)
        builder = DistanceMatrixBuilder(
            problem, voxel_size_m=10.0, verbose=False, energy_model=energy_model
        )
        dist, _d2t, _t2e, energy, _e2t, _t2ee = builder.build(compute_energy=True)  # type: ignore[misc]
        return dist, energy

    def test_el_default_deja_gratis_el_descenso(self) -> None:
        # Comportamiento heredado: E = P_hover·d/v + 10·m·g·max(0, Δz).
        # Para pares en bajada el término de altitud es 0, así que la energía
        # es exactamente proporcional a la distancia.
        dist, energy = self._build(None)
        ratio = energy / np.where(dist > 0.0, dist, np.nan)
        hover_ratio = 150.0 / 5.0  # hover_power_w / cruise_velocity_mps
        assert np.nanmin(ratio) == pytest.approx(hover_ratio, rel=1e-9)

    def test_con_gong_ningun_tramo_es_proporcional_a_la_distancia(self) -> None:
        _dist, energy = self._build(GongEnergyModel())
        assert np.isfinite(energy).all()
        assert (energy[~np.eye(5, dtype=bool)] > 0.0).all()

    def test_la_geometria_no_depende_del_modelo_energetico(self) -> None:
        dist_legacy, _ = self._build(None)
        dist_gong, _ = self._build(GongEnergyModel())
        np.testing.assert_allclose(dist_legacy, dist_gong)

    def test_la_matriz_de_gong_sigue_siendo_asimetrica(self) -> None:
        _dist, energy = self._build(GongEnergyModel())
        assert not np.allclose(energy, energy.T)

    def test_gong_cobra_el_descenso_que_el_default_regala(self) -> None:
        _d_legacy, e_legacy = self._build(None)
        _d_gong, e_gong = self._build(GongEnergyModel())
        # Pares en bajada: el default sólo cobra crucero, Gong cobra además
        # el descenso acumulado.
        bajadas = e_legacy < e_gong
        assert bajadas.any()
