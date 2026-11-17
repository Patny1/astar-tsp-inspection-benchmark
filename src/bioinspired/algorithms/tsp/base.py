"""Base class and result type for TSP inspection algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from bioinspired.core.tsp_problem import InspectionTSPProblem


@dataclass(frozen=True)
class TSPResult:
    """Output of a single TSP algorithm run.

    Attributes:
        best_tour:           permutation of range(n_targets), depot not included.
        best_cost:           total tour distance (depot→targets→end via A* distances).
        best_cost_moo:       (distance, energy) tuple for the best tour; None in SOO mode.
        convergence_history: best cost per iteration, shape (n_iter,).
        runtime_seconds:     wall-clock time.
        metadata:            algorithm name, seed, n_evaluations, snapshots, etc.
        pareto_tours:        list of non-dominated tours (Fase 4); None in SOO/MOO-scalar mode.
        pareto_costs:        (M, 2) array of (distance, energy) for each Pareto tour; None otherwise.
    """

    best_tour: list[int]
    best_cost: float
    convergence_history: np.ndarray
    runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
    best_cost_moo: tuple[float, float] | None = field(default=None)
    pareto_tours: list[list[int]] | None = field(default=None)
    pareto_costs: np.ndarray | None = field(default=None)


class BaseTSPAlgorithm(ABC):
    """Contract for all TSP permutation-encoding algorithms."""

    name: ClassVar[str]

    @abstractmethod
    def solve(self, problem: InspectionTSPProblem, seed: int) -> TSPResult:
        """Optimize the inspection tour.

        Args:
            problem: TSP problem with precomputed A* distance matrix.
            seed:    RNG seed for full reproducibility.

        Returns:
            TSPResult with best tour permutation and convergence history.
        """
        ...
