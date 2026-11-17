"""El escenario de torre: geometria valida y targets alcanzables.

Lo que mas se rompe en un generador de targets es silencioso: un punto dentro de
un obstaculo hace que el A* no encuentre camino y la celda quede infactible sin
que nadie lo note hasta ver la matriz llena de `inf`.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.algorithms.soo._shared import VoxelGrid
from bioinspired.environment.scenarios import generate_tower_tsp_scenario


def _dentro_de_alguna_caja(p: np.ndarray, boxes: np.ndarray) -> bool:
    return bool(any(np.all(np.abs(p - b[:3]) <= b[3:]) for b in boxes))


class TestGeometria:
    def test_la_estructura_son_cajas_no_esferas(self) -> None:
        """Las cajas mapean a colisiones de Gazebo; las esferas no tienen correlato."""
        p = generate_tower_tsp_scenario(seed=0)
        assert p.boxes_arr is not None and len(p.boxes_arr) == 8
        assert p.obstacles_arr.shape == (0, 4), "la torre no usa esferas"

    def test_la_torre_entra_en_el_dominio(self) -> None:
        p = generate_tower_tsp_scenario(seed=0)
        lo, hi = p.bounds_arr
        for c in p.boxes_arr:
            assert np.all(c[:3] - c[3:] >= lo), "una caja se sale por abajo"
            assert np.all(c[:3] + c[3:] <= hi), "una caja se sale por arriba"


class TestTargets:
    @pytest.mark.parametrize("seed", range(4))
    def test_ningun_target_cae_dentro_de_la_estructura(self, seed: int) -> None:
        p = generate_tower_tsp_scenario(seed=seed, n_targets=21)
        for t in p.inspection_targets_arr:
            assert not _dentro_de_alguna_caja(
                t, p.boxes_arr
            ), f"target {t} dentro de la torre: el A* no va a poder alcanzarlo"

    @pytest.mark.parametrize("seed", range(3))
    def test_los_targets_caen_en_voxels_libres(self, seed: int) -> None:
        """Mas fuerte que la prueba anterior: es lo que el planificador ve."""
        p = generate_tower_tsp_scenario(seed=seed, n_targets=21)
        grid = VoxelGrid.from_problem(p, 2.5)
        libre = grid.free_voxel_mask(p.obstacles_arr, p.boxes_arr)
        for t in p.inspection_targets_arr:
            i, j, k = grid.world_to_voxel(t)
            assert libre[i, j, k], f"target {t} cae en un voxel ocupado"

    def test_el_prefijo_anida_entre_tamanos(self) -> None:
        """Igual que building/solar_farm: los N menores son prefijo del mayor.

        Permite derivar la matriz de un N chico por submatriz de la del grande.
        """
        grande = generate_tower_tsp_scenario(seed=7, n_targets=15).inspection_targets_arr
        chico = generate_tower_tsp_scenario(seed=7, n_targets=6).inspection_targets_arr
        assert np.allclose(chico, grande[:6])

    def test_el_jitter_distingue_instancias(self) -> None:
        """Sin jitter, `instance_seed` solo elegiria otro subconjunto."""
        a = generate_tower_tsp_scenario(seed=1, n_targets=12).inspection_targets_arr
        b = generate_tower_tsp_scenario(seed=2, n_targets=12).inspection_targets_arr
        assert not np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))

    def test_falla_ruidosamente_si_se_agota_el_pool(self) -> None:
        """building se agoto en silencio a N=40; acá tiene que avisar."""
        with pytest.raises(ValueError, match="densificar"):
            generate_tower_tsp_scenario(seed=0, n_targets=99)

    def test_hay_variacion_vertical_real(self) -> None:
        """Los targets cubren la altura de la torre, no un solo nivel."""
        z = generate_tower_tsp_scenario(seed=0, n_targets=21).inspection_targets_arr[:, 2]
        assert z.max() - z.min() > 60.0
