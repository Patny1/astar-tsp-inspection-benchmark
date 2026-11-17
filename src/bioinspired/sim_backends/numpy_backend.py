"""Backend de simulación analítico (rápido, sin física)."""

from __future__ import annotations

import numpy as np

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.trajectory import Trajectory
from bioinspired.metrics.path import count_collisions
from bioinspired.sim_backends.base import SimResult

_GOAL_TOLERANCE_M = 1e-6


class NumpyBackend:
    """Evaluador analítico de trayectorias.

    Cumple `SimulationBackend`. Velocidad típica: ~µs por evaluación.
    Apto para optimización iterativa (10⁴+ evaluaciones por corrida).
    """

    def evaluate(self, trajectory: Trajectory, problem: PathPlanningProblem) -> SimResult:
        energy = problem.energy_model.compute(trajectory, problem.drone_model)
        obstacles = problem.obstacles_arr
        if problem.drone_model.body_radius_m > 0.0 and obstacles.shape[0] > 0:
            obstacles = obstacles.copy()
            obstacles[:, 3] += problem.drone_model.body_radius_m
        boxes = problem.boxes_arr
        if problem.drone_model.body_radius_m > 0.0 and boxes.shape[0] > 0:
            boxes = boxes.copy()
            boxes[:, 3:6] += problem.drone_model.body_radius_m
        collisions = count_collisions(trajectory, obstacles, boxes)
        completion = bool(
            np.linalg.norm(trajectory.waypoints_arr[-1] - problem.goal_arr) < _GOAL_TOLERANCE_M
        )
        return SimResult(
            energy=float(energy),
            collisions=int(collisions["total"]),
            completion=completion,
            metadata={
                "backend": "numpy",
                "waypoints_inside": collisions["waypoints_inside"],
                "segments_crossing": collisions["segments_crossing"],
            },
        )
