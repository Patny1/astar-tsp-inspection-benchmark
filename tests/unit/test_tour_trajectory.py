"""Reconstrucción de la trayectoria completa de un tour TSP.

`DistanceMatrixBuilder` calcula los caminos A* y conserva sólo su costo
escalar: los waypoints se descartan. Sin ellos no se puede visualizar un tour
real, medir su energía punta a punta, ni volarlo en un backend físico.
"""

import numpy as np
import pytest

from bioinspired.core.trajectory import Trajectory
from bioinspired.core.tsp_problem import DistanceMatrixBuilder, build_tour_trajectory
from bioinspired.environment.scenarios import generate_building_tsp_scenario

VOXEL = 5.0


@pytest.fixture(scope="module")
def problem():
    return generate_building_tsp_scenario(seed=0, n_targets=6)


def test_devuelve_una_trayectoria_del_deposito_al_final(problem) -> None:
    tour = [0, 1, 2, 3, 4, 5]
    traj = build_tour_trajectory(problem, tour, voxel_size_m=VOXEL)
    assert isinstance(traj, Trajectory)
    np.testing.assert_allclose(traj.waypoints_arr[0], problem.start_arr)
    np.testing.assert_allclose(traj.waypoints_arr[-1], problem.goal_arr)


def test_visita_los_targets_en_el_orden_del_tour(problem) -> None:
    tour = [3, 1, 5, 0, 4, 2]
    traj = build_tour_trajectory(problem, tour, voxel_size_m=VOXEL)
    targets = problem.inspection_targets_arr
    # cada target debe aparecer como waypoint, y en el orden pedido
    posiciones = []
    for t in tour:
        d = np.linalg.norm(traj.waypoints_arr - targets[t], axis=1)
        assert d.min() < 1e-6, f"target {t} no aparece en la trayectoria"
        posiciones.append(int(d.argmin()))
    assert posiciones == sorted(posiciones), "los targets no están en el orden del tour"


def test_longitud_coincide_con_el_costo_de_la_matriz(problem) -> None:
    """La reconstrucción debe medir lo mismo que la matriz que guió la optimización.

    Si divergen, el tour que el optimizador creyó estar eligiendo no es el que
    se vuela — que es justo lo que un backend físico pondría en evidencia.
    """
    dist, d2t, t2e = DistanceMatrixBuilder(problem, voxel_size_m=VOXEL, verbose=False).build()
    tour = [2, 0, 4, 1, 3, 5]
    esperado = d2t[tour[0]] + sum(dist[tour[k], tour[k + 1]] for k in range(len(tour) - 1))
    esperado += t2e[tour[-1]]

    traj = build_tour_trajectory(problem, tour, voxel_size_m=VOXEL)
    medido = float(np.linalg.norm(np.diff(traj.waypoints_arr, axis=0), axis=1).sum())
    assert medido == pytest.approx(esperado, rel=1e-9)


def test_rechaza_un_tour_invalido(problem) -> None:
    with pytest.raises(ValueError, match="tour"):
        build_tour_trajectory(problem, [0, 1, 1, 2, 3, 4], voxel_size_m=VOXEL)
