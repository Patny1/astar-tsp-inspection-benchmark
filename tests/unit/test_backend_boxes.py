import numpy as np

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import NullEnergyModel
from bioinspired.sim_backends.numpy_backend import NumpyBackend


def _problem(boxes, body_radius=0.0) -> PathPlanningProblem:
    return PathPlanningProblem(
        start_arr=np.array([-5.0, 0.0, 0.0]),
        goal_arr=np.array([5.0, 0.0, 0.0]),
        bounds_arr=np.array([[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]]),
        obstacles_arr=np.zeros((0, 4)),
        drone_model=DroneModel(body_radius_m=body_radius),
        energy_model=NullEnergyModel(),
        n_waypoints=1,
        scenario_id="t",
        seed=0,
        boxes_arr=boxes,
    )


def test_backend_counts_box_collision() -> None:
    box = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    traj = Trajectory(waypoints_arr=np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    res = NumpyBackend().evaluate(traj, _problem(box))
    assert res.collisions >= 1


def test_backend_inflates_box_by_body_radius() -> None:
    # Caja chica; trayectoria roza a 1.4m del centro en y. Sin inflado no colisiona;
    # con body_radius=0.6 el semieje y (1.0) -> 1.6 y ahora sí.
    box = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    traj = Trajectory(waypoints_arr=np.array([[-5.0, 1.4, 0.0], [5.0, 1.4, 0.0]]))
    assert NumpyBackend().evaluate(traj, _problem(box, body_radius=0.0)).collisions == 0
    assert NumpyBackend().evaluate(traj, _problem(box, body_radius=0.6)).collisions >= 1
