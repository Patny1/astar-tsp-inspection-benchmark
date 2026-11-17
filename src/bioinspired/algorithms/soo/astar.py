"""A* path planner sobre voxel grid 3D (Hart, Nilsson, Raphael 1968).

Baseline algoritmico clasico para path planning. Discretiza el espacio de busqueda
en un voxel grid uniforme, marca como bloqueados los voxels cuyo centro cae dentro
de algun obstaculo esferico, y aplica A* con 26-conectividad y heuristica euclidiana
3D para encontrar el camino optimo entre start y goal.

Es deterministico: con el mismo problema produce siempre el mismo path,
independientemente de `seed`. Sirve como referencia de optimalidad para los
algoritmos metaheuristicos posteriores.
"""

from __future__ import annotations

import heapq
import time
from typing import ClassVar

import numpy as np

from bioinspired.algorithms.soo._shared import (
    VoxelGrid,
    evaluate_with_penalty,
    neighbors_offsets,
    resample_arclength,
)
from bioinspired.core.algorithm import BaseAlgorithm
from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.result import OptimizationResult
from bioinspired.core.trajectory import Trajectory
from bioinspired.sim_backends.base import SimulationBackend

Voxel = tuple[int, int, int]


class AStarPlanner(BaseAlgorithm):
    """A* sobre voxel grid 3D con 26-conectividad y heuristica euclidiana.

    Attributes:
        voxel_size_m: tamano del voxel en metros (default 5.0).
        weight: peso de la heuristica en f = g + weight*h (default 1.0,
            equivalente a A* estandar; valores >1 dan A* ponderado, mas rapido
            pero potencialmente sub-optimo).
    """

    name: ClassVar[str] = "astar"

    def __init__(self, voxel_size_m: float = 5.0, weight: float = 1.0) -> None:
        if voxel_size_m <= 0:
            raise ValueError(f"voxel_size_m debe ser positivo, recibido {voxel_size_m}")
        if weight <= 0:
            raise ValueError(f"weight debe ser positivo, recibido {weight}")
        self.voxel_size_m = voxel_size_m
        self.weight = weight

    def solve(
        self,
        problem: PathPlanningProblem,
        seed: int,
        backend: SimulationBackend,
    ) -> OptimizationResult:
        t0 = time.perf_counter()
        grid = VoxelGrid.from_problem(problem, self.voxel_size_m)
        free = grid.free_voxel_mask(problem.obstacles_arr, problem.boxes_arr)
        start_v = grid.world_to_voxel(problem.start_arr)
        goal_v = grid.world_to_voxel(problem.goal_arr)
        path_voxels = self._astar_search(start_v, goal_v, free)
        path_world = grid.voxels_to_world(path_voxels)

        # El path en mundo usa centros de voxel; reemplaza extremos por start/goal exactos
        if path_world.shape[0] >= 2:
            path_world[0] = problem.start_arr
            path_world[-1] = problem.goal_arr
        else:
            path_world = np.vstack([problem.start_arr, problem.goal_arr])

        waypoints_arr = resample_arclength(path_world, problem.n_waypoints + 2)
        # Asegurar exact start/goal despues del resample (np.interp puede tener jitter)
        waypoints_arr[0] = problem.start_arr
        waypoints_arr[-1] = problem.goal_arr

        traj = Trajectory(waypoints_arr=waypoints_arr)
        energy_raw, fitness = evaluate_with_penalty(traj, problem, backend)
        return OptimizationResult(
            trajectory=traj,
            pareto_front=None,
            convergence_history=np.array([[energy_raw]]),
            runtime_seconds=time.perf_counter() - t0,
            metadata={
                "algorithm": self.name,
                "seed": seed,
                "n_evaluations": 1,
                "best_energy_raw": energy_raw,
                "best_fitness": fitness,
                "best_energy": fitness,  # legacy alias = fitness
            },
        )

    # -------------------- delegates a VoxelGrid (compat con tests) --------------------

    def _grid_shape(self, problem: PathPlanningProblem) -> tuple[int, int, int]:
        """Numero de voxels por eje (delegado en `VoxelGrid.from_problem.shape`)."""
        return VoxelGrid.from_problem(problem, self.voxel_size_m).shape

    def _world_to_voxel(self, point_arr: np.ndarray, problem: PathPlanningProblem) -> Voxel:
        """Mapea un punto en mundo al indice del voxel (delegado en `VoxelGrid`)."""
        grid = VoxelGrid.from_problem(problem, self.voxel_size_m)
        return grid.world_to_voxel(point_arr)

    def _free_voxel_mask(
        self,
        problem: PathPlanningProblem,
        shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Mask booleana del grid (delegada en `VoxelGrid.free_voxel_mask`).

        El parametro ``shape`` se conserva por compatibilidad de firma con
        tests existentes; el shape real lo deriva ``VoxelGrid`` del problema.
        """
        grid = VoxelGrid.from_problem(problem, self.voxel_size_m)
        return grid.free_voxel_mask(problem.obstacles_arr, problem.boxes_arr)

    # -------------------- A* search --------------------

    def _astar_search(self, start_v: Voxel, goal_v: Voxel, free: np.ndarray) -> list[Voxel]:
        """A* clasico con heap de prioridad y heuristica euclidiana 3D.

        Si start o goal caen en voxel bloqueado, retorna fallback [start, goal]
        (linea recta) — el caller decide que hacer en ese caso degenerado.
        """
        if not free[start_v] or not free[goal_v]:
            return [start_v, goal_v]
        offsets = neighbors_offsets()
        shape = free.shape
        goal_arr = np.asarray(goal_v, dtype=np.float64)

        def h(v: Voxel) -> float:
            return float(np.linalg.norm(np.asarray(v, dtype=np.float64) - goal_arr))

        # heap de tuplas (f_score, counter, voxel) — counter rompe empates
        # determinsticamente, evitando que heapq compare tuples voxel
        counter = 0
        open_heap: list[tuple[float, int, Voxel]] = [(h(start_v), counter, start_v)]
        came_from: dict[Voxel, Voxel] = {}
        g_score: dict[Voxel, float] = {start_v: 0.0}

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == goal_v:
                # Reconstruir path desde goal hasta start
                path = [current]
                while path[-1] in came_from:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path
            cx, cy, cz = current
            for dx, dy, dz in offsets:
                nx, ny, nz = cx + int(dx), cy + int(dy), cz + int(dz)
                if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                    continue
                neighbor: Voxel = (nx, ny, nz)
                if not free[neighbor]:
                    continue
                step_cost = float(
                    np.sqrt(int(dx) * int(dx) + int(dy) * int(dy) + int(dz) * int(dz))
                )
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    counter += 1
                    f = tentative_g + self.weight * h(neighbor)
                    heapq.heappush(open_heap, (f, counter, neighbor))

        # No se encontro path: fallback linea recta start->goal
        return [start_v, goal_v]
