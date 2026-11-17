"""Clase base abstracta para todos los algoritmos de optimización."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.result import OptimizationResult
from bioinspired.sim_backends.base import SimulationBackend


class BaseAlgorithm(ABC):
    """Contrato uniforme para todos los algoritmos del workspace.

    Cumple §8.3 (Single Responsibility): el algoritmo solo optimiza. NO calcula
    métricas (eso lo hace `metrics/`), NO instancia DroneModel ni EnergyModel
    (vienen en `problem`), NO elige backend (lo recibe como parámetro).

    Attributes:
        name: identificador único del algoritmo (snake_case).
    """

    name: ClassVar[str]

    @abstractmethod
    def solve(
        self,
        problem: PathPlanningProblem,
        seed: int,
        backend: SimulationBackend,
    ) -> OptimizationResult:
        """Ejecuta el algoritmo sobre el problema dado.

        Args:
            problem: especificación inmutable del escenario.
            seed: semilla para reproducibilidad bit-a-bit.
            backend: simulador donde se evalúan las trayectorias candidatas.

        Returns:
            Resultado con la trayectoria solución, historia de convergencia y metadata.
        """
        ...
