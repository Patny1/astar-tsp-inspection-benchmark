"""Tests para BaseAlgorithm."""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np
import pytest

from bioinspired.core.algorithm import BaseAlgorithm
from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.result import OptimizationResult
from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import NullEnergyModel
from bioinspired.sim_backends.base import SimResult, SimulationBackend


class _FakeBackend:
    def evaluate(self, trajectory: Trajectory, problem: PathPlanningProblem) -> SimResult:
        return SimResult(energy=0.0, collisions=0, completion=True, metadata={})


class _FakeAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "fake"

    def solve(
        self,
        problem: PathPlanningProblem,
        seed: int,
        backend: SimulationBackend,
    ) -> OptimizationResult:
        time.sleep(0.001)
        traj = Trajectory(
            waypoints_arr=np.vstack([problem.start_arr, problem.goal_arr]),
        )
        return OptimizationResult(
            trajectory=traj,
            pareto_front=None,
            convergence_history=np.zeros((1, 1)),
            runtime_seconds=0.001,
            metadata={"seed": seed},
        )


def _make_problem() -> PathPlanningProblem:
    return PathPlanningProblem(
        start_arr=np.array([0.0, 0.0, 5.0]),
        goal_arr=np.array([10.0, 10.0, 5.0]),
        bounds_arr=np.array([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]),
        obstacles_arr=np.zeros((0, 4)),
        drone_model=DroneModel(),
        energy_model=NullEnergyModel(),
        n_waypoints=2,
        scenario_id="t",
        seed=42,
    )


class TestBaseAlgorithm:
    def test_no_se_puede_instanciar_sin_solve(self) -> None:
        with pytest.raises(TypeError):
            BaseAlgorithm()  # type: ignore[abstract]

    def test_subclase_concreta_funciona(self) -> None:
        algo = _FakeAlgorithm()
        result = algo.solve(_make_problem(), seed=42, backend=_FakeBackend())
        assert isinstance(result, OptimizationResult)
        assert result.metadata["seed"] == 42

    def test_name_es_class_var(self) -> None:
        assert _FakeAlgorithm.name == "fake"
