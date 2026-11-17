"""Contrato del backend de simulación: SimResult + SimulationBackend Protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from bioinspired.core.problem import PathPlanningProblem
    from bioinspired.core.trajectory import Trajectory


@dataclass(frozen=True)
class SimResult:
    """Resultado de evaluar una trayectoria en un backend de simulación.

    Attributes:
        energy: consumo energético total (unidades del modelo).
        collisions: número total de colisiones detectadas.
        completion: True si la trayectoria llega al goal.
        metadata: información del backend (versión, tiempo, etc.).
    """

    energy: float
    collisions: int
    completion: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SimulationBackend(Protocol):
    """Backend que ejecuta una trayectoria y devuelve métricas reales.

    Implementaciones:
        - NumpyBackend: rápido, modelo analítico (uso típico durante optimización).
        - GazeboBackend: lento, simulación física realista (validación post-hoc).

    Ambos cumplen el mismo contrato — son intercambiables vía `backend=` en CLI.
    """

    def evaluate(self, trajectory: Trajectory, problem: PathPlanningProblem) -> SimResult:
        """Ejecuta la trayectoria y devuelve métricas observadas."""
        ...
