"""Test que A* rodea cajas AABB al buscar un camino."""

import numpy as np

from bioinspired.algorithms.soo.astar import AStarPlanner
from bioinspired.core.problem import PathPlanningProblem
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import NullEnergyModel
from bioinspired.metrics.path import count_collisions
from bioinspired.sim_backends.numpy_backend import NumpyBackend


def test_astar_routes_around_box() -> None:
    # Caja-pared bloqueando el camino directo este-oeste.
    # Espacio 40x40x40m, caja en el centro con semiejes (3, 15, 15):
    # cubre x=[17,23], y=[5,35], z=[5,35] — bloquea la ruta directa.
    # Con voxel_size_m=2.0 hay voxels libres en x<17 y x>23 para rodear.
    box = np.array([[20.0, 20.0, 20.0, 3.0, 15.0, 15.0]])
    problem = PathPlanningProblem(
        start_arr=np.array([2.0, 20.0, 20.0]),
        goal_arr=np.array([38.0, 20.0, 20.0]),
        bounds_arr=np.array([[0.0, 0.0, 0.0], [40.0, 40.0, 40.0]]),
        obstacles_arr=np.zeros((0, 4)),
        drone_model=DroneModel(),
        energy_model=NullEnergyModel(),
        n_waypoints=10,
        scenario_id="boxwall",
        seed=0,
        boxes_arr=box,
    )
    result = AStarPlanner(voxel_size_m=2.0).solve(problem, seed=0, backend=NumpyBackend())
    coll = count_collisions(result.trajectory, problem.obstacles_arr, problem.boxes_arr)
    assert coll["total"] == 0  # rodea la caja
