import numpy as np
import pytest

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import NullEnergyModel


def _base(**kw) -> dict:
    d = {
        "start_arr": np.array([0.0, 0.0, 0.0]),
        "goal_arr": np.array([10.0, 10.0, 10.0]),
        "bounds_arr": np.array([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]),
        "obstacles_arr": np.zeros((0, 4)),
        "drone_model": DroneModel(),
        "energy_model": NullEnergyModel(),
        "n_waypoints": 5,
        "scenario_id": "t",
        "seed": 0,
    }
    d.update(kw)
    return d


def test_default_boxes_empty() -> None:
    p = PathPlanningProblem(**_base())
    assert p.boxes_arr.shape == (0, 6)
    assert p.n_boxes == 0


def test_boxes_accepted_and_counted() -> None:
    boxes = np.array([[5.0, 5.0, 5.0, 1.0, 2.0, 3.0]])
    p = PathPlanningProblem(**_base(boxes_arr=boxes))
    assert p.n_boxes == 1


def test_bad_boxes_shape_raises() -> None:
    with pytest.raises(ValueError):
        PathPlanningProblem(**_base(boxes_arr=np.zeros((2, 5))))
