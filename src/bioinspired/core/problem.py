"""Definición canónica del problema de path planning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import BaseEnergyModel


@dataclass(frozen=True)
class PathPlanningProblem:
    """Especificación inmutable de un problema de path planning 3D.

    Cumple R9 (estandarización TSP-like): n_obstacles y n_waypoints son
    parámetros del escenario, fijos durante todo el experimento. Todos los
    algoritmos reciben EXACTAMENTE el mismo problema.

    Attributes:
        start_arr: posición inicial (3,).
        goal_arr: posición meta (3,).
        bounds_arr: bounding box (2, 3).
        obstacles_arr: matriz (M, 4).
        drone_model: modelo cinemático inyectado.
        energy_model: modelo energético inyectado.
        n_waypoints: número de waypoints intermedios solicitados al planner.
        scenario_id: identificador determinístico del escenario.
        seed: semilla para reproducibilidad.
        inspection_targets_arr: (K, 3) puntos de inspección que el drone debe
            observar. None para problemas de path planning genérico.
        sensor_range_m: distancia máxima a la que el sensor puede observar
            un target. Solo relevante cuando inspection_targets_arr != None.
        boxes_arr: matriz (K, 6) — cajas AABB (cx, cy, cz, semiejes hx, hy, hz).
    """

    start_arr: np.ndarray
    goal_arr: np.ndarray
    bounds_arr: np.ndarray
    obstacles_arr: np.ndarray
    drone_model: DroneModel
    energy_model: BaseEnergyModel
    n_waypoints: int
    scenario_id: str
    seed: int
    inspection_targets_arr: np.ndarray | None = field(default=None)
    sensor_range_m: float = 0.0
    boxes_arr: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))

    def __post_init__(self) -> None:
        if self.start_arr.shape != (3,):
            raise ValueError(f"start_arr shape debe ser (3,), recibido {self.start_arr.shape}")
        if self.goal_arr.shape != (3,):
            raise ValueError(f"goal_arr shape debe ser (3,), recibido {self.goal_arr.shape}")
        if self.bounds_arr.shape != (2, 3):
            raise ValueError(f"bounds_arr shape debe ser (2, 3), recibido {self.bounds_arr.shape}")
        if not np.all(self.bounds_arr[0] < self.bounds_arr[1]):
            raise ValueError("bounds_arr[0] debe ser estrictamente menor que bounds_arr[1]")
        if not np.all(
            (self.start_arr >= self.bounds_arr[0]) & (self.start_arr <= self.bounds_arr[1])
        ):
            raise ValueError("start_arr fuera de bounds_arr")
        if not np.all(
            (self.goal_arr >= self.bounds_arr[0]) & (self.goal_arr <= self.bounds_arr[1])
        ):
            raise ValueError("goal_arr fuera de bounds_arr")
        if self.obstacles_arr.ndim != 2 or self.obstacles_arr.shape[1] != 4:
            raise ValueError(
                f"obstacles_arr shape debe ser (M, 4) — (x, y, z, radius). Recibido {self.obstacles_arr.shape}"
            )
        if self.n_waypoints < 1:
            raise ValueError(f"n_waypoints debe ser >= 1, recibido {self.n_waypoints}")
        if self.inspection_targets_arr is not None:
            t = self.inspection_targets_arr
            if t.ndim != 2 or t.shape[1] != 3:
                raise ValueError(
                    f"inspection_targets_arr debe tener shape (K, 3), recibido {t.shape}"
                )
            if t.shape[0] == 0:
                raise ValueError("inspection_targets_arr no puede estar vacío")
        if self.sensor_range_m < 0:
            raise ValueError(f"sensor_range_m debe ser >= 0, recibido {self.sensor_range_m}")
        if self.inspection_targets_arr is not None and self.sensor_range_m <= 0:
            raise ValueError(
                "sensor_range_m debe ser > 0 cuando inspection_targets_arr está definido"
            )
        if self.boxes_arr.ndim != 2 or self.boxes_arr.shape[1] != 6:
            raise ValueError(
                f"boxes_arr shape debe ser (K, 6) — (cx, cy, cz, hx, hy, hz). "
                f"Recibido {self.boxes_arr.shape}"
            )

    @property
    def n_obstacles(self) -> int:
        return int(self.obstacles_arr.shape[0])

    @property
    def n_boxes(self) -> int:
        return int(self.boxes_arr.shape[0])

    @property
    def n_dim(self) -> int:
        return 3

    @property
    def is_inspection(self) -> bool:
        """True si el problema tiene targets de inspección definidos."""
        return self.inspection_targets_arr is not None

    @property
    def n_targets(self) -> int:
        """Número de targets de inspección (0 si no es problema de inspección)."""
        return (
            0 if self.inspection_targets_arr is None else int(self.inspection_targets_arr.shape[0])
        )
