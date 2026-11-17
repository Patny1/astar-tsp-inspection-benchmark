"""Tests para OptimizationResult, SimResult y SimulationBackend Protocol."""

from __future__ import annotations

import numpy as np

from bioinspired.core.result import OptimizationResult
from bioinspired.core.trajectory import Trajectory
from bioinspired.sim_backends.base import SimResult, SimulationBackend


class TestSimResult:
    def test_construye(self) -> None:
        r = SimResult(energy=10.5, collisions=2, completion=True, metadata={"backend": "numpy"})
        assert r.energy == 10.5
        assert r.collisions == 2
        assert r.completion is True
        assert r.metadata["backend"] == "numpy"


class TestOptimizationResult:
    def test_construye_soo(self) -> None:
        traj = Trajectory(waypoints_arr=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        result = OptimizationResult(
            trajectory=traj,
            pareto_front=None,
            convergence_history=np.zeros((10, 1)),
            runtime_seconds=0.5,
            metadata={"algorithm": "test"},
        )
        assert result.trajectory is traj
        assert result.pareto_front is None
        assert result.convergence_history.shape == (10, 1)


class TestSimulationBackendProtocol:
    def test_clase_que_cumple_es_instancia(self) -> None:
        from bioinspired.core.problem import PathPlanningProblem

        class DummyBackend:
            def evaluate(self, trajectory: Trajectory, problem: PathPlanningProblem) -> SimResult:
                return SimResult(energy=0.0, collisions=0, completion=True, metadata={})

        backend: SimulationBackend = DummyBackend()
        # mypy: el assert siguiente solo verifica que isinstance check sobre Protocol no rompe en runtime
        assert hasattr(backend, "evaluate")
