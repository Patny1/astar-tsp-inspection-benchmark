"""Resultado de una corrida de optimización."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bioinspired.core.trajectory import Trajectory


@dataclass(frozen=True)
class OptimizationResult:
    """Salida estándar de cualquier algoritmo (SOO o MOO).

    Attributes:
        trajectory: solución representativa (best-so-far en SOO; representante
            del frente en MOO).
        pareto_front: matriz (n_solutions, n_objectives) en MOO; None en SOO.
        convergence_history: matriz (n_iterations, n_metrics) — fitness por iter.
        runtime_seconds: tiempo wall-clock de la corrida.
        metadata: algoritmo, seed, n_evaluations, hostname, git_sha, etc.
    """

    trajectory: Trajectory
    pareto_front: np.ndarray | None
    convergence_history: np.ndarray
    runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
