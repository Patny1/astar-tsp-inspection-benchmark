"""RRT* path planner (Karaman & Frazzoli 2011) — sampling-based asintóticamente óptimo.

Crece un árbol muestreando puntos en el espacio continuo `bounds_arr`,
conectándolos al vecino más cercano (limitado por `step_size_m`) y, opcionalmente,
haciendo *rewire* dentro de un radio `search_radius_m`. Devuelve `OptimizationResult`
con la trayectoria interpolada a `n_waypoints+2` puntos, igual que A*.

Es estocástico pero **bit-a-bit determinista con `np.random.default_rng(seed)`**.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np

from bioinspired.algorithms.soo._shared import (
    evaluate_with_penalty,
    resample_arclength,
    segment_free,
)
from bioinspired.core.algorithm import BaseAlgorithm
from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.result import OptimizationResult
from bioinspired.core.trajectory import Trajectory
from bioinspired.sim_backends.base import SimulationBackend


class RRTStarPlanner(BaseAlgorithm):
    """RRT* (Karaman & Frazzoli 2011) — sampling-based asintóticamente óptimo.

    Attributes:
        max_iterations: numero de iteraciones de tree-growth.
        step_size_m: paso máximo desde el nearest neighbor hacia el sample.
        goal_bias: probabilidad de samplear directamente el goal.
        search_radius_m: radio para choose-parent y rewire.
        snapshot_every: si se setea, captura un snapshot del best path-to-goal
            (o al nodo más cercano si aún no se alcanzó) cada N iteraciones
            (1-indexado). Cada snapshot es ``{iteration, best_fitness,
            best_waypoints_arr}`` y se acumula en ``metadata["snapshots"]``.
    """

    name: ClassVar[str] = "rrt_star"

    def __init__(
        self,
        max_iterations: int = 5000,
        step_size_m: float = 5.0,
        goal_bias: float = 0.1,
        search_radius_m: float = 15.0,
        snapshot_every: int | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations debe ser positivo")
        if not 0.0 <= goal_bias <= 1.0:
            raise ValueError("goal_bias debe estar en [0, 1]")
        if step_size_m <= 0 or search_radius_m <= 0:
            raise ValueError("step_size_m y search_radius_m deben ser positivos")
        if snapshot_every is not None and snapshot_every <= 0:
            raise ValueError("snapshot_every debe ser positivo o None")
        self.max_iterations = max_iterations
        self.step_size_m = step_size_m
        self.goal_bias = goal_bias
        self.search_radius_m = search_radius_m
        self.snapshot_every = snapshot_every

    def solve(
        self,
        problem: PathPlanningProblem,
        seed: int,
        backend: SimulationBackend,
    ) -> OptimizationResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        nodes_arr = problem.start_arr.reshape(1, 3).astype(np.float64).copy()
        parent_idx: list[int] = [-1]
        cost_arr: list[float] = [0.0]
        goal_idx: int | None = None
        history: list[float] = []
        snapshots: list[dict[str, Any]] = []

        for it in range(self.max_iterations):
            sample = (
                problem.goal_arr
                if rng.random() < self.goal_bias
                else rng.uniform(problem.bounds_arr[0], problem.bounds_arr[1])
            )
            dists = np.linalg.norm(nodes_arr - sample, axis=1)
            nearest = int(np.argmin(dists))
            direction = sample - nodes_arr[nearest]
            d = float(np.linalg.norm(direction))
            if d < 1e-9:
                # Aún sin avance, registramos history e (si toca) snapshot
                # para mantener cadencia 1-indexada con `it + 1`.
                history.append(cost_arr[goal_idx] if goal_idx is not None else float("inf"))
                if self.snapshot_every is not None and (it + 1) % self.snapshot_every == 0:
                    snapshots.append(
                        self._build_snapshot(
                            it + 1, nodes_arr, parent_idx, cost_arr, goal_idx, problem
                        )
                    )
                continue
            new_node = nodes_arr[nearest] + direction / d * min(self.step_size_m, d)
            if not segment_free(
                nodes_arr[nearest], new_node, problem.obstacles_arr, problem.boxes_arr
            ):
                history.append(cost_arr[goal_idx] if goal_idx is not None else float("inf"))
                if self.snapshot_every is not None and (it + 1) % self.snapshot_every == 0:
                    snapshots.append(
                        self._build_snapshot(
                            it + 1, nodes_arr, parent_idx, cost_arr, goal_idx, problem
                        )
                    )
                continue
            near_dists = np.linalg.norm(nodes_arr - new_node, axis=1)
            near_idxs = np.where(near_dists <= self.search_radius_m)[0]
            best_parent = nearest
            best_cost = cost_arr[nearest] + float(np.linalg.norm(new_node - nodes_arr[nearest]))
            for k in near_idxs:
                if not segment_free(
                    nodes_arr[k], new_node, problem.obstacles_arr, problem.boxes_arr
                ):
                    continue
                c = cost_arr[k] + float(np.linalg.norm(new_node - nodes_arr[k]))
                if c < best_cost:
                    best_parent, best_cost = int(k), c
            new_idx = len(parent_idx)
            nodes_arr = np.vstack([nodes_arr, new_node])
            parent_idx.append(best_parent)
            cost_arr.append(best_cost)
            for k in near_idxs:
                c_via_new = best_cost + float(np.linalg.norm(nodes_arr[k] - new_node))
                if c_via_new < cost_arr[k] and segment_free(
                    new_node, nodes_arr[k], problem.obstacles_arr, problem.boxes_arr
                ):
                    parent_idx[k] = new_idx
                    cost_arr[k] = c_via_new
            if (
                goal_idx is None
                and float(np.linalg.norm(new_node - problem.goal_arr)) <= self.step_size_m
                and segment_free(
                    new_node, problem.goal_arr, problem.obstacles_arr, problem.boxes_arr
                )
            ):
                nodes_arr = np.vstack([nodes_arr, problem.goal_arr])
                parent_idx.append(new_idx)
                cost_arr.append(best_cost + float(np.linalg.norm(problem.goal_arr - new_node)))
                goal_idx = len(parent_idx) - 1
            history.append(cost_arr[goal_idx] if goal_idx is not None else float("inf"))
            if self.snapshot_every is not None and (it + 1) % self.snapshot_every == 0:
                snapshots.append(
                    self._build_snapshot(it + 1, nodes_arr, parent_idx, cost_arr, goal_idx, problem)
                )

        reached_goal = goal_idx is not None
        if goal_idx is None:
            goal_idx = int(np.argmin(np.linalg.norm(nodes_arr - problem.goal_arr, axis=1)))

        path = [goal_idx]
        while parent_idx[path[-1]] != -1:
            path.append(parent_idx[path[-1]])
        path.reverse()
        path_arr = nodes_arr[path]
        # Si no se alcanzó el goal, fuerza el endpoint exacto (igual que A*).
        if not reached_goal:
            path_arr = np.vstack([path_arr, problem.goal_arr])
        traj = Trajectory(waypoints_arr=resample_arclength(path_arr, problem.n_waypoints + 2))
        energy_raw, fitness = evaluate_with_penalty(traj, problem, backend)
        return OptimizationResult(
            trajectory=traj,
            pareto_front=None,
            convergence_history=np.asarray(history, dtype=np.float64).reshape(-1, 1),
            runtime_seconds=time.perf_counter() - t0,
            metadata={
                "algorithm": self.name,
                "seed": seed,
                "n_evaluations": 1,
                "best_energy_raw": energy_raw,
                "best_fitness": fitness,
                "best_energy": fitness,  # legacy alias = fitness
                "n_nodes": int(len(parent_idx)),
                "reached_goal": bool(reached_goal),
                "snapshots": snapshots,
            },
        )

    def _build_snapshot(
        self,
        iteration: int,
        nodes_arr: np.ndarray,
        parent_idx: list[int],
        cost_arr: list[float],
        goal_idx: int | None,
        problem: PathPlanningProblem,
    ) -> dict[str, Any]:
        """Construye un snapshot del best path-to-goal vigente.

        Si `goal_idx is None` (aún no se alcanzó el goal), usa el nodo más
        cercano al goal y fuerza el endpoint exacto al goal (igual que el
        fallback final del solve).
        """
        if goal_idx is not None:
            target_idx = goal_idx
            forced_goal = False
            best_cost = float(cost_arr[target_idx])
        else:
            target_idx = int(np.argmin(np.linalg.norm(nodes_arr - problem.goal_arr, axis=1)))
            forced_goal = True
            # `cost_arr[target_idx]` es el costo hasta el nodo más cercano,
            # que es una cota inferior razonable del best-so-far (sin colisiones
            # adicionales en el segmento forzado, ya que el goal puede estar
            # bloqueado en este momento).
            best_cost = float(cost_arr[target_idx])
        path = [target_idx]
        while parent_idx[path[-1]] != -1:
            path.append(parent_idx[path[-1]])
        path.reverse()
        path_arr = nodes_arr[path]
        if forced_goal:
            path_arr = np.vstack([path_arr, problem.goal_arr])
        wp_arr = resample_arclength(path_arr, problem.n_waypoints + 2)
        return {
            "iteration": iteration,
            "best_fitness": best_cost,
            "best_waypoints_arr": wp_arr.copy(),
        }
