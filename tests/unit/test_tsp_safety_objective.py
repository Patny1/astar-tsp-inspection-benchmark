"""Tests del objetivo de seguridad (clearance) en la línea TSP.

Cubre la agregación por ``min``, la retrocompatibilidad del builder — quien no
pida clearance debe recibir exactamente la tupla de antes — y el requisito de
vóxel fino que hace usable la métrica como objetivo.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.core.tsp_problem import DistanceMatrixBuilder, InspectionTSPProblem
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import GongEnergyModel
from bioinspired.environment.scenarios import generate_layered_obs_tsp_scenario

VOXEL_FINO = 2.5


@pytest.fixture(scope="module")
def matrices() -> tuple:
    prob = generate_layered_obs_tsp_scenario(seed=0, drone=DroneModel.dji_m210(), n_targets=5)
    builder = DistanceMatrixBuilder(
        prob,
        voxel_size_m=VOXEL_FINO,
        verbose=False,
        energy_model=GongEnergyModel(continuous_at_hover=True),
    )
    return prob, builder


@pytest.fixture(scope="module")
def tsp(matrices: tuple) -> InspectionTSPProblem:
    prob, builder = matrices
    d, d2t, t2e, e, e2t, t2ee, c, c2t, t2ec = builder.build(
        compute_energy=True, compute_clearance=True
    )
    return InspectionTSPProblem.from_path_problem(
        prob,
        d,
        d2t,
        t2e,
        energy_matrix=e,
        depot_to_targets_energy=e2t,
        targets_to_end_energy=t2ee,
        clearance_matrix=c,
        depot_to_targets_clearance=c2t,
        targets_to_end_clearance=t2ec,
    )


class TestEscenarioLayeredObs:
    def test_tiene_obstaculos_y_dos_alturas(self) -> None:
        p = generate_layered_obs_tsp_scenario(seed=0, n_targets=12)
        assert len(p.obstacles_arr) == 5
        alturas = sorted(set(np.round(p.inspection_targets_arr[:, 2], 1)))
        assert alturas == [10.0, 35.0]

    def test_las_esferas_son_grandes_para_el_voxel(self) -> None:
        # Radio >= 8 m: por debajo de eso el vóxel de 5 m deja pasar rutas por
        # dentro del obstáculo y el clearance sale negativo.
        p = generate_layered_obs_tsp_scenario(seed=0)
        assert float(p.obstacles_arr[:, 3].min()) >= 8.0


class TestBuilderRetrocompatible:
    def test_sin_clearance_devuelve_la_tupla_de_antes(self, matrices: tuple) -> None:
        _, builder = matrices
        assert len(builder.build(compute_energy=True)) == 6
        assert len(builder.build(compute_energy=False)) == 3

    def test_el_clearance_se_agrega_al_final(self, matrices: tuple) -> None:
        _, builder = matrices
        sin = builder.build(compute_energy=True)
        con = builder.build(compute_energy=True, compute_clearance=True)
        assert len(con) == 9
        for a, b in zip(sin, con[:6], strict=True):
            assert np.allclose(a, b, equal_nan=True)


class TestTourClearance:
    def test_es_el_minimo_de_los_tramos(self, tsp: InspectionTSPProblem) -> None:
        tour = [0, 1, 2, 3, 4]
        tramos = [float(tsp.depot_to_targets_clearance[tour[0]])]  # type: ignore[index]
        tramos += [
            float(tsp.clearance_matrix[tour[i], tour[i + 1]])  # type: ignore[index]
            for i in range(len(tour) - 1)
        ]
        tramos.append(float(tsp.targets_to_end_clearance[tour[-1]]))  # type: ignore[index]
        assert tsp.tour_clearance(tour) == pytest.approx(min(tramos))

    def test_el_orden_de_visita_cambia_la_seguridad(self, tsp: InspectionTSPProblem) -> None:
        import itertools

        vals = {tsp.tour_clearance(list(t)) for t in itertools.permutations(range(5))}
        assert len(vals) > 1, "si no varía, el objetivo no discrimina"

    def test_con_voxel_fino_no_hay_clearance_negativo(self, tsp: InspectionTSPProblem) -> None:
        # Negativo = la ruta atraviesa el obstáculo. Como objetivo sería
        # perverso: el optimizador buscaría justo esas rutas.
        m = tsp.clearance_matrix
        assert m is not None
        fuera_diagonal = m[~np.eye(len(m), dtype=bool)]
        assert float(fuera_diagonal.min()) >= 0.0

    def test_moo4_niega_el_clearance_para_minimizar(self, tsp: InspectionTSPProblem) -> None:
        tour = [0, 1, 2, 3, 4]
        d, e, t, neg_c = tsp.tour_cost_moo4(tour)
        assert (d, e, t) == tsp.tour_cost_moo3(tour)
        assert neg_c == pytest.approx(-tsp.tour_clearance(tour))

    def test_falla_claro_si_no_se_construyo_la_matriz(self, matrices: tuple) -> None:
        prob, builder = matrices
        d, d2t, t2e = builder.build(compute_energy=False)
        sin_clr = InspectionTSPProblem.from_path_problem(prob, d, d2t, t2e)
        with pytest.raises(ValueError, match="clearance_matrix"):
            sin_clr.tour_clearance([0, 1, 2, 3, 4])
