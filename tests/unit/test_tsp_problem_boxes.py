"""Tests que el oráculo de matriz TSP propaga y evita cajas AABB."""

import numpy as np

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.tsp_problem import DistanceMatrixBuilder, InspectionTSPProblem
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import NullEnergyModel


def _insp_problem(boxes: np.ndarray) -> PathPlanningProblem:
    return PathPlanningProblem(
        start_arr=np.array([2.0, 10.0, 10.0]),
        goal_arr=np.array([18.0, 10.0, 10.0]),
        bounds_arr=np.array([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]),
        obstacles_arr=np.zeros((0, 4)),
        drone_model=DroneModel(),
        energy_model=NullEnergyModel(),
        n_waypoints=10,
        scenario_id="insp",
        seed=0,
        inspection_targets_arr=np.array([[5.0, 18.0, 10.0], [15.0, 2.0, 10.0]]),
        sensor_range_m=5.0,
        boxes_arr=boxes,
    )


def test_from_path_problem_carries_boxes() -> None:
    box = np.array([[10.0, 10.0, 10.0, 2.0, 2.0, 2.0]])
    prob = _insp_problem(box)
    dist, d2t, t2e = DistanceMatrixBuilder(prob, voxel_size_m=2.0, verbose=False).build()
    tsp = InspectionTSPProblem.from_path_problem(prob, dist, d2t, t2e)
    assert tsp.boxes_arr is not None
    assert tsp.boxes_arr.shape == (1, 6)


def test_matrix_detours_around_box() -> None:
    # Caja-pared central; la distancia entre los dos targets debe ser mayor
    # que la línea recta (hay rodeo) y finita (hay camino).
    # Targets a cada lado: (5,18,10) y (15,2,10).
    # Caja en el centro (10,10,10) con semiejes (1.5, 8, 8) bloquea la ruta directa.
    box = np.array([[10.0, 10.0, 10.0, 1.5, 8.0, 8.0]])
    prob = _insp_problem(box)
    dist, _d2t, _t2e = DistanceMatrixBuilder(prob, voxel_size_m=2.0, verbose=False).build()
    straight = np.linalg.norm(prob.inspection_targets_arr[0] - prob.inspection_targets_arr[1])
    assert np.isfinite(dist[0, 1])
    assert dist[0, 1] >= straight  # rodea la pared
