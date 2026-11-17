import numpy as np

from bioinspired.core.trajectory import Trajectory
from bioinspired.metrics.path import count_collisions, min_clearance


def _traj(points) -> Trajectory:
    return Trajectory(waypoints_arr=np.asarray(points, dtype=np.float64))


# Caja en el origen, semiejes (1,1,1) → ocupa [-1,1]^3.
BOX = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])


def test_waypoint_inside_box_counts() -> None:
    traj = _traj([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    res = count_collisions(traj, np.zeros((0, 4)), boxes_arr=BOX)
    assert res["waypoints_inside"] == 1  # solo el (0,0,0) está dentro


def test_segment_crosses_box_counts() -> None:
    traj = _traj([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])  # cruza la caja por X
    res = count_collisions(traj, np.zeros((0, 4)), boxes_arr=BOX)
    assert res["segments_crossing"] == 1
    assert res["total"] >= 1


def test_segment_misses_box() -> None:
    traj = _traj([[-5.0, 5.0, 0.0], [5.0, 5.0, 0.0]])  # pasa por y=5, lejos
    res = count_collisions(traj, np.zeros((0, 4)), boxes_arr=BOX)
    assert res["segments_crossing"] == 0


def test_clearance_outside_positive_inside_negative() -> None:
    # Dos puntos muy separados: el mínimo clearance viene del (3,0,0) vs cara x=1 → 2.0
    out = min_clearance(
        _traj([[3.0, 0.0, 0.0], [10.0, 10.0, 10.0]]), np.zeros((0, 4)), boxes_arr=BOX
    )
    assert np.isclose(out, 2.0)  # dist a la cara x=1 desde x=3
    inside = min_clearance(
        _traj([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]), np.zeros((0, 4)), boxes_arr=BOX
    )
    assert inside < 0.0  # penetración


def test_spheres_only_unchanged() -> None:
    # Regresión: sin boxes_arr, comportamiento idéntico al previo.
    sph = np.array([[0.0, 0.0, 0.0, 2.0]])
    traj = _traj([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    res = count_collisions(traj, sph)
    assert res["waypoints_inside"] == 1
