"""El escenario `city`: una manzana de edificios como cajas AABB, a escala real.

Es el escenario principal de Fase 3: cajas con correlato fisico, exportables a
Gazebo, y calles con ancho garantizado
para que el A* a voxel 2,5 m pase. Lo que mas se rompe en un generador asi es
silencioso: dos edificios solapados, o un target de la fachada de A metido en
B, dejan la matriz llena de `inf` sin que nadie lo note.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from bioinspired.algorithms.soo._shared import VoxelGrid
from bioinspired.environment import scenarios
from bioinspired.environment.scenarios import (
    generate_building_tsp_scenario,
    generate_city_scenario,
    generate_city_tsp_scenario,
)
from bioinspired.metrics.path import _box_clearance


def _clearance_a_todas(p: np.ndarray, boxes: np.ndarray) -> float:
    return min(_box_clearance(p, b[:3], b[3:6]) for b in boxes)


class TestGeometria:
    def test_por_defecto_diez_edificios_y_ninguna_esfera(self) -> None:
        p = generate_city_tsp_scenario(seed=0)
        assert p.boxes_arr is not None and len(p.boxes_arr) == 10
        assert p.obstacles_arr.shape == (0, 4), "la manzana no usa esferas"

    def test_los_edificios_apoyan_en_el_suelo(self) -> None:
        p = generate_city_tsp_scenario(seed=0, n_buildings=16)
        for c in p.boxes_arr:
            assert c[2] == pytest.approx(c[5]), "cz == hz: la base esta en z=0"

    def test_tamanos_a_escala_real(self) -> None:
        p = generate_city_tsp_scenario(seed=3, n_buildings=16)
        lados_xy = 2 * p.boxes_arr[:, 3:5]
        alturas = 2 * p.boxes_arr[:, 5]
        assert np.all(lados_xy >= 10.0) and np.all(lados_xy <= 28.0)
        assert np.all(alturas >= 15.0) and np.all(alturas <= 60.0)

    def test_la_manzana_entra_en_el_dominio_con_margen(self) -> None:
        p = generate_city_tsp_scenario(seed=0, n_buildings=16)
        lo, hi = p.bounds_arr
        for c in p.boxes_arr:
            assert np.all(c[:2] - c[3:5] >= lo[:2] + 30.0)
            assert np.all(c[:2] + c[3:5] <= hi[:2] - 30.0)
            assert c[2] + c[5] <= hi[2] - 20.0, "techo demasiado cerca del techo del dominio"

    @pytest.mark.parametrize("seed", range(10))
    def test_sin_solapamiento_y_calle_minima(self, seed: int) -> None:
        """Para todo par de edificios, la separacion en x O en y es >= 15 m.

        Es lo que garantiza que el A* a 2,5 m (semidiagonal 2,17 m) tenga por
        lo menos ~6 voxels libres entre dos fachadas enfrentadas.
        """
        p = generate_city_tsp_scenario(seed=seed, n_buildings=16)
        b = p.boxes_arr
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                gap_x = abs(b[i, 0] - b[j, 0]) - (b[i, 3] + b[j, 3])
                gap_y = abs(b[i, 1] - b[j, 1]) - (b[i, 4] + b[j, 4])
                assert (
                    max(gap_x, gap_y) >= scenarios._CITY_MIN_STREET_M
                ), f"edificios {i} y {j} a {max(gap_x, gap_y):.1f} m (seed {seed})"

    def test_start_y_goal_fuera_de_la_manzana(self) -> None:
        p = generate_city_tsp_scenario(seed=0, n_buildings=16)
        assert _clearance_a_todas(p.start_arr, p.boxes_arr) > 20.0
        assert _clearance_a_todas(p.goal_arr, p.boxes_arr) > 20.0
        assert p.start_arr[2] == pytest.approx(5.0) and p.goal_arr[2] == pytest.approx(5.0)

    def test_el_subconjunto_de_edificios_anida(self) -> None:
        """Los primeros k edificios de n=16 son los k de n=k: la geometria de un
        lote no depende de cuantos lotes se pidan."""
        todos = generate_city_tsp_scenario(seed=5, n_buildings=16).boxes_arr
        pocos = generate_city_tsp_scenario(seed=5, n_buildings=6).boxes_arr
        np.testing.assert_allclose(pocos, todos[:6])

    def test_rechaza_cantidades_imposibles(self) -> None:
        with pytest.raises(ValueError, match="edificios"):
            generate_city_tsp_scenario(seed=0, n_buildings=17)
        with pytest.raises(ValueError, match="edificios"):
            generate_city_tsp_scenario(seed=0, n_buildings=0)


class TestTargets:
    @pytest.mark.parametrize("seed", range(4))
    def test_targets_a_standoff_de_una_fachada_y_libres_de_todas(self, seed: int) -> None:
        p = generate_city_tsp_scenario(seed=seed, n_targets=30)
        for t in p.inspection_targets_arr:
            clr = _clearance_a_todas(t, p.boxes_arr)
            assert clr == pytest.approx(
                scenarios._CITY_STANDOFF_M, abs=1e-6
            ), f"target {t} a {clr:.2f} m de la fachada mas cercana (seed {seed})"

    @pytest.mark.parametrize("seed", range(3))
    def test_los_targets_caen_en_voxels_libres(self, seed: int) -> None:
        """Es lo que el planificador ve, mas estricto que la caja."""
        p = generate_city_tsp_scenario(seed=seed, n_targets=30)
        grid = VoxelGrid.from_problem(p, 2.5)
        libre = grid.free_voxel_mask(p.obstacles_arr, p.boxes_arr)
        for t in p.inspection_targets_arr:
            i, j, k = grid.world_to_voxel(t)
            assert libre[i, j, k], f"target {t} cae en un voxel ocupado"

    def test_dentro_de_bounds(self) -> None:
        p = generate_city_tsp_scenario(seed=1, n_targets=30)
        lo, hi = p.bounds_arr
        assert np.all(p.inspection_targets_arr >= lo) and np.all(p.inspection_targets_arr <= hi)

    def test_el_prefijo_anida_entre_tamanos(self) -> None:
        grande = generate_city_tsp_scenario(seed=7, n_targets=30).inspection_targets_arr
        chico = generate_city_tsp_scenario(seed=7, n_targets=6).inspection_targets_arr
        np.testing.assert_allclose(chico, grande[:6])

    def test_semillas_distintas_dan_instancias_distintas(self) -> None:
        a = generate_city_tsp_scenario(seed=1, n_targets=12)
        b = generate_city_tsp_scenario(seed=2, n_targets=12)
        assert not np.allclose(a.boxes_arr, b.boxes_arr)
        assert not np.allclose(
            np.sort(a.inspection_targets_arr, axis=0), np.sort(b.inspection_targets_arr, axis=0)
        )

    def test_falla_ruidosamente_si_se_agota_el_pool(self) -> None:
        with pytest.raises(ValueError, match="densificar"):
            generate_city_tsp_scenario(seed=0, n_targets=10_000)

    def test_hay_variacion_vertical_real(self) -> None:
        z = generate_city_tsp_scenario(seed=0, n_targets=30).inspection_targets_arr[:, 2]
        assert z.max() - z.min() > 20.0


class TestVariantes:
    def test_la_variante_pp_comparte_geometria_e_id(self) -> None:
        pp = generate_city_scenario(seed=7)
        tsp = generate_city_tsp_scenario(seed=7)
        np.testing.assert_allclose(pp.boxes_arr, tsp.boxes_arr)
        assert pp.scenario_id == tsp.scenario_id == "city_seed7"
        assert pp.inspection_targets_arr is None

    def test_building_no_cambia_con_el_refactor_de_candidatos(self) -> None:
        """`_facade_candidates` se extrae de `_building_facade_targets`; los
        targets de `building` tienen que quedar bit a bit (hash capturado antes)."""
        esperado = {
            (42, 12): "251f4ce9adf9db2bd6ba4965044df9939d70872a40b2295fe66fdd9cbc9604db",
            (42, 40): "b4373e25bd7359f60fdd75c7b6250df9521c559bd2a7f6aa6c416bc96b966095",
            (0, 30): "9047d14faeed8932bfd9a0d7023c7fe6914e5a6d6ebcc97355a3283bbc07e912",
        }
        for (seed, n), h in esperado.items():
            arr = generate_building_tsp_scenario(seed=seed, n_targets=n).inspection_targets_arr
            assert hashlib.sha256(arr.tobytes()).hexdigest() == h, f"building seed={seed} n={n}"


@pytest.mark.slow
def test_todos_los_pares_alcanzables_a_voxel_2_5() -> None:
    """A N=8 la matriz simetrica sale finita y con rodeos (dist > euclidea)."""
    from bioinspired.core.tsp_problem import DistanceMatrixBuilder

    p = generate_city_tsp_scenario(seed=0, n_targets=8)
    dist, d2t, t2e = DistanceMatrixBuilder(
        p, voxel_size_m=2.5, verbose=False, symmetric_pairs=True
    ).build()
    assert np.all(np.isfinite(dist)) and np.all(np.isfinite(d2t)) and np.all(np.isfinite(t2e))
    t = p.inspection_targets_arr
    euclid = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=2)
    assert np.any(dist > euclid + 1.0), "ningun par rodea un edificio"
