"""Tests de la construcción simétrica de matriz (symmetric_pairs=True).

Propiedad central: con un A* por par no ordenado y la dirección inversa
derivada del mismo camino, la asimetría de la matriz de energía queda
EXACTAMENTE en el potencial (precio_subida - precio_bajada)·(z_j - z_i), que
telescopa a constante en todo tour con extremos fijos — el ranking de tours
depende sólo de la parte simétrica.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.core.tsp_problem import DistanceMatrixBuilder
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import GongEnergyModel
from bioinspired.environment.scenarios import generate_tiny_tsp_scenario

N = 6


def _m210() -> DroneModel:
    return DroneModel(
        mass_kg=4.8,
        max_velocity_mps=22.3,
        cruise_velocity_mps=5.0,
        battery_capacity_wh=348.0,
        hover_power_w=737.7,
    )


@pytest.fixture(scope="module")
def problema():
    return generate_tiny_tsp_scenario(seed=0, n_targets=N, drone=_m210())


@pytest.fixture(scope="module")
def matrices_sym(problema):
    builder = DistanceMatrixBuilder(
        problema,
        voxel_size_m=10.0,
        verbose=False,
        energy_model=GongEnergyModel(continuous_at_hover=True),
        symmetric_pairs=True,
    )
    return builder.build(compute_energy=True)


class TestConstruccionSimetrica:
    def test_distancias_exactamente_simetricas(self, matrices_sym) -> None:
        dist = matrices_sym[0]
        assert np.array_equal(dist, dist.T)
        assert np.isfinite(dist).all()

    def test_la_direccion_forward_coincide_con_el_modo_default(self, problema) -> None:
        """El A* i→j es el mismo en ambos modos; sólo cambia cómo se llena j→i."""
        sym = DistanceMatrixBuilder(
            problema, voxel_size_m=10.0, verbose=False, symmetric_pairs=True
        ).build()
        default = DistanceMatrixBuilder(problema, voxel_size_m=10.0, verbose=False).build()
        triu = np.triu_indices(N, k=1)
        np.testing.assert_array_equal(sym[0][triu], default[0][triu])
        np.testing.assert_array_equal(sym[1], default[1])  # depot→targets intacto
        np.testing.assert_array_equal(sym[2], default[2])  # targets→end intacto

    def test_la_asimetria_es_exactamente_el_potencial(self, problema, matrices_sym) -> None:
        energy = matrices_sym[3]
        drone = problema.drone_model
        model = GongEnergyModel(continuous_at_hover=True)
        w = drone.total_mass_kg * model.GRAVITY_MPS2
        v_up, v_dn = model.vertical_rates_mps(drone)
        precio_up = model.power_vertical_w(v_up, w) / v_up
        precio_dn = model.power_vertical_w(-v_dn, w) / v_dn
        z = problema.inspection_targets_arr[:, 2]
        potencial = (precio_up - precio_dn) * np.subtract.outer(z, z).T  # (z_j - z_i)
        off = ~np.eye(N, dtype=bool)
        np.testing.assert_allclose((energy - energy.T)[off], potencial[off], rtol=1e-9)

    def test_el_ranking_de_tours_es_invariante_bajo_el_gauge_completo(
        self, problema, matrices_sym
    ) -> None:
        """La reducción exacta ATSP→STSP para tour abierto: simetrizar el
        bloque target-target Y correr el potencial a las piernas depot/end
        (``ê2t[i] = e2t[i] - κ·z_i + κ·z_depot``, ídem end). El costo de TODO
        tour cambia en la misma constante ``κ·(z_end - z_depot)`` — la
        asimetría no puede decidir nada."""
        _d, _d2t, _t2e, energy, e2t, t2ee = matrices_sym
        drone = problema.drone_model
        model = GongEnergyModel(continuous_at_hover=True)
        w = drone.total_mass_kg * model.GRAVITY_MPS2
        v_up, v_dn = model.vertical_rates_mps(drone)
        kappa = (
            model.power_vertical_w(v_up, w) / v_up - model.power_vertical_w(-v_dn, w) / v_dn
        ) / 2.0
        z = problema.inspection_targets_arr[:, 2]
        z_depot, z_end = float(problema.start_arr[2]), float(problema.goal_arr[2])

        e_sym = 0.5 * (energy + energy.T)
        e2t_hat = e2t - kappa * z + kappa * z_depot
        t2ee_hat = t2ee + kappa * z - kappa * z_end

        rng = np.random.default_rng(3)
        diffs = []
        for _ in range(50):
            t = rng.permutation(N)
            asym = e2t[t[0]] + sum(energy[t[i], t[i + 1]] for i in range(N - 1)) + t2ee[t[-1]]
            simm = (
                e2t_hat[t[0]] + sum(e_sym[t[i], t[i + 1]] for i in range(N - 1)) + t2ee_hat[t[-1]]
            )
            diffs.append(asym - simm)
        assert np.std(diffs) == pytest.approx(0.0, abs=1e-6)
        assert diffs[0] == pytest.approx(kappa * (z_end - z_depot), rel=1e-9)

    def test_corre_la_mitad_de_astar(self, problema, monkeypatch: pytest.MonkeyPatch) -> None:
        from bioinspired.algorithms.soo.astar import AStarPlanner

        llamadas = {"n": 0}
        original = AStarPlanner._astar_search

        def contado(self, *args, **kwargs):
            llamadas["n"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(AStarPlanner, "_astar_search", contado)
        DistanceMatrixBuilder(
            problema, voxel_size_m=10.0, verbose=False, symmetric_pairs=True
        ).build()
        assert llamadas["n"] == N * (N - 1) // 2 + 2 * N

        llamadas["n"] = 0
        DistanceMatrixBuilder(problema, voxel_size_m=10.0, verbose=False).build()
        assert llamadas["n"] == N * (N - 1) + 2 * N

    def test_modo_legacy_tambien_simetrizable(self, problema) -> None:
        """Sin modelo inyectado, la energía sigue la fórmula legacy por
        dirección: la bajada neta paga 0 de término de altura."""
        m = DistanceMatrixBuilder(
            problema, voxel_size_m=10.0, verbose=False, symmetric_pairs=True
        ).build(compute_energy=True)
        dist, energy = m[0], m[3]
        z = problema.inspection_targets_arr[:, 2]
        drone = problema.drone_model
        i, j = int(np.argmin(z)), int(np.argmax(z))  # subida neta garantizada
        base = drone.hover_power_w * dist[i, j] / drone.cruise_velocity_mps
        esperado_sube = base + 10.0 * drone.total_mass_kg * 9.81 * (z[j] - z[i])
        assert energy[i, j] == pytest.approx(esperado_sube, rel=1e-9)
        assert energy[j, i] == pytest.approx(base, rel=1e-9)  # bajar es "gratis" en legacy
