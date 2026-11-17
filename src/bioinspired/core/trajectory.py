"""Tipo canónico de trayectoria 3D para path planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Trayectoria 3D representada como una secuencia ordenada de waypoints.

    Es el tipo CANÓNICO usado por todos los algoritmos. No hay adaptadores:
    cualquier optimizador devuelve `Trajectory`, cualquier métrica la consume.

    Attributes:
        waypoints_arr: array float64 de forma (n_points, 3) con coordenadas X/Y/Z.
        metadata: información opcional (algoritmo, seed, git_sha, timestamp).
    """

    waypoints_arr: np.ndarray
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        arr = self.waypoints_arr
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"waypoints_arr debe tener shape (N, 3), recibido {arr.shape}")
        if arr.shape[0] < 2:
            raise ValueError("Una trayectoria requiere al menos 2 puntos (inicio y meta)")
        # asegura float64 sin copia si ya lo es
        object.__setattr__(self, "waypoints_arr", np.ascontiguousarray(arr, dtype=np.float64))

    @property
    def n_points(self) -> int:
        """Número total de puntos (inicio + intermedios + meta)."""
        return int(self.waypoints_arr.shape[0])

    @property
    def start_arr(self) -> np.ndarray:
        """Punto de inicio (3,)."""
        return self.waypoints_arr[0]

    @property
    def goal_arr(self) -> np.ndarray:
        """Punto meta (3,)."""
        return self.waypoints_arr[-1]
