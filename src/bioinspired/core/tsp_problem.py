"""Inspection TSP Problem — multi-waypoint tour with real obstacle-aware distances.

Approach B1: the distance matrix d[i][j] is precomputed using A* path planning
so each entry reflects the actual obstacle-avoiding path length between two
inspection targets, not straight-line Euclidean distance.

Usage
-----
problem = generate_urban_inspection_problem(seed=42)
builder = DistanceMatrixBuilder(problem)
matrix  = builder.build(voxel_size_m=5.0)
tsp     = InspectionTSPProblem.from_path_problem(problem, matrix)

# Evaluate a tour
tour = [0, 3, 1, 5, 2, 4]   # permutation over range(n_targets)
cost = tsp.tour_cost(tour)   # sum of d[tour[i], tour[i+1]]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import GongEnergyModel


@dataclass(frozen=True)
class InspectionTSPProblem:
    """Multi-waypoint inspection tour problem with precomputed distance matrix.

    The drone starts at ``depot_arr``, visits all ``n_targets`` inspection
    points exactly once, and returns to ``depot_arr`` (closed tour) or ends
    at ``end_arr`` (open tour).

    Attributes:
        targets_arr:     (K, 3) inspection target positions.
        distance_matrix: (K, K) A*-computed path lengths between all pairs.
        energy_matrix:   (K, K) estimated energy for each segment (optional).
        depot_arr:       start position (used as tour origin).
        end_arr:         end position; if None → closed tour back to depot.
        scenario_id:     identifier string.
        drone_model:     drone model (for energy estimation).
    """

    targets_arr: np.ndarray  # (K, 3)
    distance_matrix: np.ndarray  # (K, K)  — A* path lengths
    depot_to_targets: np.ndarray  # (K,)    — A* dist from depot to each target
    targets_to_end: np.ndarray  # (K,)    — A* dist from each target to end
    depot_arr: np.ndarray  # (3,)
    end_arr: np.ndarray  # (3,)
    scenario_id: str
    drone_model: DroneModel
    obstacles_arr: np.ndarray | None = field(default=None)  # (M, 4) for visualization
    boxes_arr: np.ndarray | None = field(default=None)  # (K, 6) cajas AABB para visualización
    energy_matrix: np.ndarray | None = field(default=None)  # (K, K)  — energy per segment (J)
    clearance_matrix: np.ndarray | None = field(default=None)  # (K,K) — margen mínimo i→j
    depot_to_targets_clearance: np.ndarray | None = field(default=None)  # (K,)
    targets_to_end_clearance: np.ndarray | None = field(default=None)  # (K,)
    depot_to_targets_energy: np.ndarray | None = field(default=None)  # (K,) — energy depot→target
    targets_to_end_energy: np.ndarray | None = field(default=None)  # (K,) — energy target→end
    sensor_range_m: float = field(default=0.0)  # radio de cobertura del sensor (visualización)
    # Tiempo de vuelo por tramo (s), Fase 3 en adelante. Van al final: quien no
    # los pida no ve ningún cambio.
    time_matrix: np.ndarray | None = field(default=None)  # (K, K)
    depot_to_targets_time: np.ndarray | None = field(default=None)  # (K,)
    targets_to_end_time: np.ndarray | None = field(default=None)  # (K,)

    @property
    def n_targets(self) -> int:
        return int(self.targets_arr.shape[0])

    def battery_fraction(self, energy_j: float) -> float:
        """Fracción de la batería del aparato que consume ``energy_j`` (1,0 = toda).

        La batería no es objetivo: es la energía leída como fracción de
        ``battery_capacity_wh``, para reportarla y para el techo de factibilidad.
        """
        return float(energy_j) / (float(self.drone_model.battery_capacity_wh) * 3600.0)

    def tour_time(self, tour: list[int] | np.ndarray) -> float:
        """Tiempo de vuelo total del tour (s): depósito → targets → fin.

        Suma la matriz de tiempo (misma descomposición en piernas que la
        energía: horizontal a crucero, vertical a sus tasas). Objetivo de Fase 3
        en adelante; requiere ``DistanceMatrixBuilder.build(compute_time=True)``.

        Raises:
            ValueError: si la matriz de tiempo no fue construida.
        """
        if self.time_matrix is None:
            raise ValueError(
                "time_matrix no está construida; usar DistanceMatrixBuilder.build(compute_time=True)"
            )
        t = list(tour)
        total = float(self.depot_to_targets_time[t[0]])  # type: ignore[index]
        for i in range(len(t) - 1):
            total += float(self.time_matrix[t[i], t[i + 1]])
        total += float(self.targets_to_end_time[t[-1]])  # type: ignore[index]
        return total

    def tour_cost(self, tour: list[int] | np.ndarray) -> float:
        """Total path length of a complete tour (depot → targets → end).

        Args:
            tour: permutation of range(n_targets).

        Returns:
            Sum of A*-computed distances for all segments.
        """
        t = list(tour)
        cost = float(self.depot_to_targets[t[0]])
        for i in range(len(t) - 1):
            cost += float(self.distance_matrix[t[i], t[i + 1]])
        cost += float(self.targets_to_end[t[-1]])
        return cost

    def tour_cost_moo(self, tour: list[int] | np.ndarray) -> tuple[float, float]:
        """Return (total_distance, total_energy) for a tour.

        Energy requires ``energy_matrix`` to be set (built by
        ``DistanceMatrixBuilder`` with ``compute_energy=True``).
        Falls back to hover-power estimate when energy_matrix is None.

        Args:
            tour: permutation of range(n_targets).

        Returns:
            (distance_m, energy_J) tuple.
        """
        dist = self.tour_cost(tour)

        if self.energy_matrix is not None:
            t = list(tour)
            energy = float(self.depot_to_targets_energy[t[0]])  # type: ignore[index]
            for i in range(len(t) - 1):
                energy += float(self.energy_matrix[t[i], t[i + 1]])
            energy += float(self.targets_to_end_energy[t[-1]])  # type: ignore[index]
        else:
            # Fallback: hover-power model (energy proportional to distance)
            dm = self.drone_model
            v = float(dm.cruise_velocity_mps) if dm.cruise_velocity_mps > 0 else 1.0
            energy = float(dm.hover_power_w) * dist / v

        return dist, energy

    def tour_clearance(self, tour: list[int] | np.ndarray) -> float:
        """Margen mínimo a un obstáculo a lo largo de todo el tour, en metros.

        Tercer objetivo de la línea MOO (seguridad), a MAXIMIZAR. Se agrega con
        ``min`` y no con un promedio: lo que define el riesgo de una misión es su
        punto más expuesto, y promediar lo disimularía — un tour que roza una
        esfera en un solo tramo es peligroso aunque el resto vuele despejado.

        Requiere que la matriz se haya construido con ``compute_clearance=True``
        y **vóxel fino** (<= 2,5 m). Con 5 m algunas rutas cortan obstáculos y
        el clearance sale negativo; como objetivo eso sería perverso, porque el
        optimizador iría a buscar justo esas rutas.

        Args:
            tour: permutación de range(n_targets).

        Returns:
            Margen mínimo en metros. ``inf`` si el escenario no tiene obstáculos.

        Raises:
            ValueError: si la matriz de clearance no fue construida.
        """
        if self.clearance_matrix is None:
            raise ValueError(
                "clearance_matrix no está construida; usar "
                "DistanceMatrixBuilder.build(compute_clearance=True)"
            )
        t = list(tour)
        worst = float(self.depot_to_targets_clearance[t[0]])  # type: ignore[index]
        for i in range(len(t) - 1):
            worst = min(worst, float(self.clearance_matrix[t[i], t[i + 1]]))
        return min(worst, float(self.targets_to_end_clearance[t[-1]]))  # type: ignore[index]

    def tour_cost_moo4(self, tour: list[int] | np.ndarray) -> tuple[float, float, float, float]:
        """(distancia, energía, giro, -clearance) — los cuatro en minimización.

        El clearance se devuelve NEGADO para que los cuatro objetivos se
        minimicen por igual y el código de dominancia no tenga que llevar la
        cuenta de cuál va al revés.
        """
        dist, energy, turn = self.tour_cost_moo3(tour)
        return dist, energy, turn, -self.tour_clearance(tour)

    def tour_turning(self, tour: list[int] | np.ndarray) -> float:
        """Suma de ángulos de giro (rad) de la poligonal depot → targets → end.

        Tercer objetivo de la línea MOO. Se mide sobre la poligonal que une los
        waypoints, no sobre la ruta A*: el ordenador decide el ORDEN de visita,
        y los giros que puede optimizar son los de los targets — la forma del
        tramo entre dos targets la fija el planificador de caminos, no él.

        Motivación física: la matriz de costos asume crucero constante entre
        targets, así que no cobra el frenar y reacelerar que impone un vértice
        agudo. Este objetivo captura ese costo omitido.

        Verificado como objetivo independiente sobre 440 tours de ``layered``:
        r = -0,234 contra distancia y +0,254 contra energía, la mitad que la
        correlación entre esos dos (-0,546).

        Args:
            tour: permutación de range(n_targets).

        Returns:
            Suma de ángulos en radianes. 0 = tour perfectamente recto.
        """
        t = list(tour)
        pts = [self.depot_arr, *[self.targets_arr[i] for i in t]]
        pts.append(self.end_arr if self.end_arr is not None else self.depot_arr)
        arr = np.asarray(pts, dtype=np.float64)

        v = np.diff(arr, axis=0)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # dos waypoints coincidentes no aportan giro
        u = v / norms
        cos = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
        return float(np.sum(np.arccos(cos)))

    def tour_cost_moo3(self, tour: list[int] | np.ndarray) -> tuple[float, float, float]:
        """(distancia, energía, giro) — la variante tri-objetivo de ``tour_cost_moo``.

        Se mantiene aparte en vez de extender ``tour_cost_moo`` para no romper a
        los llamadores existentes ni invalidar los datos ya generados.
        """
        dist, energy = self.tour_cost_moo(tour)
        return dist, energy, self.tour_turning(tour)

    @classmethod
    def from_path_problem(
        cls,
        problem: PathPlanningProblem,
        distance_matrix: np.ndarray,
        depot_to_targets: np.ndarray,
        targets_to_end: np.ndarray,
        energy_matrix: np.ndarray | None = None,
        depot_to_targets_energy: np.ndarray | None = None,
        targets_to_end_energy: np.ndarray | None = None,
        clearance_matrix: np.ndarray | None = None,
        depot_to_targets_clearance: np.ndarray | None = None,
        targets_to_end_clearance: np.ndarray | None = None,
        time_matrix: np.ndarray | None = None,
        depot_to_targets_time: np.ndarray | None = None,
        targets_to_end_time: np.ndarray | None = None,
    ) -> InspectionTSPProblem:
        """Construct from an existing PathPlanningProblem with precomputed matrices.

        Args:
            problem:                   source PathPlanningProblem.
            distance_matrix:           (K, K) A* path lengths between targets.
            depot_to_targets:          (K,) A* distances from depot to each target.
            targets_to_end:            (K,) A* distances from each target to end.
            energy_matrix:             (K, K) energy (J) per segment — optional, for MOO.
            depot_to_targets_energy:   (K,) energy from depot to each target — optional.
            targets_to_end_energy:     (K,) energy from each target to end — optional.
            clearance_matrix:          (K, K) margen mínimo a obstáculos por tramo —
                                       opcional, tercer objetivo MOO (seguridad).
            depot_to_targets_clearance: (K,) ídem desde el depot.
            targets_to_end_clearance:  (K,) ídem hasta el final.
            time_matrix:               (K, K) tiempo de vuelo (s) por tramo — opcional,
                                       objetivo de Fase 3 en adelante.
            depot_to_targets_time:     (K,) ídem desde el depot.
            targets_to_end_time:       (K,) ídem hasta el final.
        """
        if problem.inspection_targets_arr is None:
            raise ValueError("problem must have inspection_targets_arr set")
        return cls(
            targets_arr=problem.inspection_targets_arr,
            distance_matrix=distance_matrix,
            depot_to_targets=depot_to_targets,
            targets_to_end=targets_to_end,
            depot_arr=problem.start_arr,
            end_arr=problem.goal_arr,
            scenario_id=problem.scenario_id,
            drone_model=problem.drone_model,
            obstacles_arr=problem.obstacles_arr,
            boxes_arr=problem.boxes_arr,
            energy_matrix=energy_matrix,
            depot_to_targets_energy=depot_to_targets_energy,
            targets_to_end_energy=targets_to_end_energy,
            clearance_matrix=clearance_matrix,
            depot_to_targets_clearance=depot_to_targets_clearance,
            targets_to_end_clearance=targets_to_end_clearance,
            sensor_range_m=problem.sensor_range_m,
            time_matrix=time_matrix,
            depot_to_targets_time=depot_to_targets_time,
            targets_to_end_time=targets_to_end_time,
        )


def build_tour_trajectory(
    problem: PathPlanningProblem,
    tour: list[int],
    voxel_size_m: float = 5.0,
) -> Trajectory:
    """Reconstruye la trayectoria completa que recorre ``tour``: depósito → targets → fin.

    ``DistanceMatrixBuilder`` calcula los caminos A* pero conserva sólo su costo
    escalar. Esta función los recalcula y los concatena, usando el mismo grid y
    el mismo planificador, de modo que la longitud de la trayectoria coincide
    con el costo que la matriz le asignó al tour.

    Es el precursor de cualquier validación física: sin los waypoints no se
    puede visualizar el tour, medir su energía punta a punta ni volarlo.

    Args:
        problem: problema con ``inspection_targets_arr`` definido.
        tour: permutación de los índices de target, en orden de visita.
        voxel_size_m: resolución del A*. Debe coincidir con la usada al construir
            la matriz, o las longitudes no van a corresponderse.

    Returns:
        ``Trajectory`` con los waypoints concatenados de todos los segmentos.

    Raises:
        ValueError: si el problema no tiene targets o ``tour`` no es una
            permutación válida de ellos.
    """
    from bioinspired.algorithms.soo._shared import VoxelGrid
    from bioinspired.algorithms.soo.astar import AStarPlanner

    targets = problem.inspection_targets_arr
    if targets is None:
        raise ValueError("el problema no tiene inspection_targets_arr")
    n = len(targets)
    if sorted(tour) != list(range(n)):
        raise ValueError(f"tour debe ser una permutación de 0..{n - 1}, recibido {tour!r}")

    planner = AStarPlanner(voxel_size_m=voxel_size_m)
    grid = VoxelGrid.from_problem(problem, voxel_size_m)
    free = grid.free_voxel_mask(problem.obstacles_arr, problem.boxes_arr)

    def _segment(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Waypoints del camino A* start→goal, con extremos exactos."""
        path = grid.voxels_to_world(
            planner._astar_search(
                grid.world_to_voxel(start.astype(np.float64)),
                grid.world_to_voxel(goal.astype(np.float64)),
                free,
            )
        )
        if path.shape[0] >= 2:
            path[0] = start.astype(np.float64)
            path[-1] = goal.astype(np.float64)
            return path
        # Sin camino por voxels: el builder cae al segmento recto, y acá también
        # para que la longitud reconstruida siga coincidiendo con la matriz.
        return np.vstack([start, goal]).astype(np.float64)

    puntos = [problem.start_arr, *(targets[i] for i in tour), problem.goal_arr]
    tramos = [_segment(puntos[k], puntos[k + 1]) for k in range(len(puntos) - 1)]
    # Concatenar sin repetir el punto de unión entre tramos consecutivos.
    waypoints = np.vstack([tramos[0], *(t[1:] for t in tramos[1:])])
    return Trajectory(
        waypoints_arr=waypoints,
        metadata={"tour": list(tour), "voxel_size_m": voxel_size_m, "n_segments": len(tramos)},
    )


class DistanceMatrixBuilder:
    """Builds the A*-based distance matrix for an InspectionTSPProblem.

    Runs A* for every ordered pair of inspection targets plus depot→targets
    and targets→end segments. Uses the same voxel grid as AStarPlanner.

    Args:
        problem: PathPlanningProblem with inspection_targets_arr set.
        voxel_size_m: voxel resolution for A* (smaller = more accurate, slower).
        verbose: print progress during matrix construction.
        energy_model: optional published power model used to fill the energy
            matrices. ``None`` (default) keeps the legacy heuristic — see
            ``build`` — so existing results stay bit-reproducible.
    """

    def __init__(
        self,
        problem: PathPlanningProblem,
        voxel_size_m: float = 5.0,
        verbose: bool = True,
        energy_model: GongEnergyModel | None = None,
        symmetric_pairs: bool = False,
    ) -> None:
        """Args (además de los documentados en la clase):

        symmetric_pairs: si True, el bloque target-target corre **un solo A***
            por par no ordenado y deriva la dirección inversa del mismo camino
            geométrico (misma longitud; ascenso y descenso intercambiados).
            Consecuencias: (a) mitad de corridas A*; (b) la matriz de
            distancias queda exactamente simétrica; (c) la asimetría de la
            matriz de energía queda EXACTAMENTE en el potencial
            ``(precio_subida - precio_bajada)·(z_j - z_i)``, que telescopa a
            una constante en todo tour con extremos fijos — el ranking de
            tours pasa a depender sólo de la parte simétrica (crucero +
            vaivén). El camino invertido es igualmente óptimo en distancia,
            así que la derivación es legítima y más consistente que correr
            A* dos veces con desempates distintos. ``False`` (default)
            conserva el comportamiento histórico corrida-por-dirección.
        """
        if problem.inspection_targets_arr is None:
            raise ValueError("problem must have inspection_targets_arr set")
        self.problem = problem
        self.voxel_size_m = voxel_size_m
        self.verbose = verbose
        self.energy_model = energy_model
        self.symmetric_pairs = symmetric_pairs

    def build(
        self,
        compute_energy: bool = False,
        compute_clearance: bool = False,
        compute_time: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Build distance (and optionally energy, clearance, time) matrices.

        Args:
            compute_time: if True, also compute flight-time matrices (s) from
                the same legs the energy uses — horizontal at cruise, climb and
                descent at their rates (``leg_time_s``). Objective from Fase 3
                on; appended AFTER clearance so nobody who does not ask for it
                sees any change. Adds no extra A* calls.
            compute_clearance: if True, also compute clearance matrices —
                the minimum distance to any obstacle along each A* path. Third
                MOO objective (safety). Like energy, adds no extra A* calls:
                it is measured on the path that was traversed anyway.

                Requires a fine voxel (<= 2.5 m). With 5 m some paths clip
                obstacles and yield NEGATIVE clearance; harmless as a post-hoc
                metric, unacceptable as an objective — the optimizer would seek
                exactly those paths.
            compute_energy: if True, also compute energy_matrix using the
                drone's hover-power model along each A* path. Needed for
                MOO phases (Fase 3 and 4). Adds no extra A* calls — energy
                is derived from the same trajectories.

        Returns:
            ``compute_energy=False`` (default, backward-compatible):
                ``(dist_matrix, depot_to_targets, targets_to_end)``
            ``compute_energy=True``:
                ``(dist_matrix, depot_to_targets, targets_to_end,
                   energy_matrix, depot_to_targets_energy, targets_to_end_energy)``
            then ``+ (clearance_matrix, depot_to_targets_clearance,
            targets_to_end_clearance)`` if ``compute_clearance``, then
            ``+ (time_matrix, depot_to_targets_time, targets_to_end_time)`` if
            ``compute_time`` — always in that order.

        All matrices return ``inf`` for infeasible A* paths.
        """
        import math

        from bioinspired.algorithms.soo._shared import VoxelGrid
        from bioinspired.algorithms.soo.astar import AStarPlanner
        from bioinspired.environment.energy_model import leg_time_s

        targets = self.problem.inspection_targets_arr
        assert targets is not None  # validado en __init__
        n = len(targets)
        depot = self.problem.start_arr
        end = self.problem.goal_arr
        dm = self.problem.drone_model
        v = float(dm.cruise_velocity_mps) if dm.cruise_velocity_mps > 0 else 1.0

        dist_matrix = np.full((n, n), np.inf)
        depot_to_targets = np.full(n, np.inf)
        targets_to_end = np.full(n, np.inf)

        if compute_energy:
            energy_matrix = np.full((n, n), np.inf)
            depot_to_targets_energy = np.full(n, np.inf)
            targets_to_end_energy = np.full(n, np.inf)

        if compute_clearance:
            clearance_matrix = np.full((n, n), -np.inf)
            depot_to_targets_clearance = np.full(n, -np.inf)
            targets_to_end_clearance = np.full(n, -np.inf)

        if compute_time:
            time_matrix = np.full((n, n), np.inf)
            depot_to_targets_time = np.full(n, np.inf)
            targets_to_end_time = np.full(n, np.inf)

        def _time_from_legs(dist: float, horizontal: float, climb: float, descent: float) -> float:
            """Tiempo (s) de un tramo en piernas; ``inf`` si el tramo es infactible.

            Con modelo publicado, las tasas verticales son las del modelo (las
            mismas por las que ``leg_energy_j`` divide). Legacy: la tasa nominal
            ``v · sin(ángulo máximo de ascenso)``, sin topes.
            """
            if not np.isfinite(dist):
                return float("inf")
            if self.energy_model is not None:
                return self.energy_model.leg_time_s(horizontal, climb, descent, dm)
            v_nom = v * math.sin(math.radians(dm.max_climb_angle_deg))
            return leg_time_s(horizontal, climb, descent, v, v_nom, v_nom)

        planner = AStarPlanner(voxel_size_m=self.voxel_size_m)
        # Grid y máscara son idénticos para todos los pares — construir una sola vez
        grid = VoxelGrid.from_problem(self.problem, self.voxel_size_m)
        free = grid.free_voxel_mask(self.problem.obstacles_arr, self.problem.boxes_arr)
        pair_runs = (n * (n - 1)) // 2 if self.symmetric_pairs else n * (n - 1)
        total = pair_runs + 2 * n
        done = 0
        t0 = time.perf_counter()

        def _run(start: np.ndarray, goal: np.ndarray) -> tuple[float, float, float, float]:
            """Run A*; return (path_length, energy_J, clearance_m, time_s).

            Returns (inf, inf, -inf, inf) on failure — el clearance se devuelve
            como -inf para que un tramo infactible nunca parezca seguro al
            agregar con ``min``.

            Two energy models, selected by ``self.energy_model``:

            ``None`` — legacy heuristic, kept for reproducibility of published
            Fase 1/2 results::

                E = P_hover · T_flight + climb_coeff · m · g · max(0, Δz)

            with ``climb_coeff = 10``. Both the coefficient and the ``max(0, ·)``
            (descending is free) are unsupported by any published model.

            ``GongEnergyModel`` — power model of Gong et al. (2023). The A* path
            is decomposed into horizontal distance, accumulated climb and
            accumulated descent, each priced by its own published equation. This
            also counts altitude churn *along* the path, which the legacy model
            misses by looking only at the net endpoint Δz.
            """
            try:
                start_v = grid.world_to_voxel(start.astype(np.float64))
                goal_v = grid.world_to_voxel(goal.astype(np.float64))
                path_voxels = planner._astar_search(start_v, goal_v, free)
                path_world = grid.voxels_to_world(path_voxels)
                if path_world.shape[0] >= 2:
                    path_world[0] = start.astype(np.float64)
                    path_world[-1] = goal.astype(np.float64)
                    diffs = np.diff(path_world, axis=0)
                    dist = float(np.sum(np.sqrt((diffs**2).sum(axis=1))))
                else:
                    path_world = np.vstack([start.astype(np.float64), goal.astype(np.float64)])
                    diffs = np.diff(path_world, axis=0)
                    dist = float(np.linalg.norm(goal - start))

                # El clearance sale de la MISMA ruta: ninguna llamada A* extra.
                clr = float("inf")
                if compute_clearance:
                    from bioinspired.core.trajectory import Trajectory
                    from bioinspired.metrics.path import min_clearance

                    clr = float(
                        min_clearance(
                            Trajectory(waypoints_arr=path_world),
                            self.problem.obstacles_arr,
                            self.problem.boxes_arr,
                        )
                    )

                # Piernas del tramo: las mismas para energía (modelo publicado)
                # y para tiempo. La rama legacy de energía NO las usa (Δz neto
                # de extremos): se deja tal cual por bit-identidad con Fase 1/2.
                delta_z_arr = diffs[:, 2]
                horizontal = float(np.linalg.norm(diffs[:, :2], axis=1).sum())
                climb = float(np.maximum(delta_z_arr, 0.0).sum())
                descent = float(np.maximum(-delta_z_arr, 0.0).sum())
                t = _time_from_legs(dist, horizontal, climb, descent)

                if self.energy_model is not None:
                    energy = self.energy_model.leg_energy_j(
                        horizontal_m=horizontal,
                        climb_m=climb,
                        descent_m=descent,
                        drone=dm,
                    )
                    return dist, energy, clr, t

                h_gain = max(0.0, float(goal[2]) - float(start[2]))
                # Cruise energy: hover power proxy (per unit distance)
                # Climbing energy: motor inefficiency during ascent makes climbing
                # approximately 10x more expensive per metre than level cruise.
                # Coefficient derived from: P_climb / v_climb >> P_cruise / v_cruise
                # for typical multirotors (P_climb ≈ 2·P_hover at v_climb ≈ 0.5 m/s).
                climb_coeff = 10.0
                energy = (
                    float(dm.hover_power_w) * dist / v
                    + climb_coeff * dm.total_mass_kg * 9.81 * h_gain
                )
                return dist, energy, clr, t
            except Exception:
                # -inf en clearance: un tramo infactible nunca debe parecer seguro
                # cuando se agrega el tour con min().
                return float("inf"), float("inf"), float("-inf"), float("inf")

        # depot → each target
        for i in range(n):
            d, e, c, t = _run(depot, targets[i])
            depot_to_targets[i] = d
            if compute_energy:
                depot_to_targets_energy[i] = e  # type: ignore[index]
            if compute_clearance:
                depot_to_targets_clearance[i] = c  # type: ignore[index]
            if compute_time:
                depot_to_targets_time[i] = t  # type: ignore[index]
            done += 1
            if self.verbose:
                self._log(done, total, t0, f"depot->T{i}")

        # each target → end
        for i in range(n):
            d, e, c, t = _run(targets[i], end)
            targets_to_end[i] = d
            if compute_energy:
                targets_to_end_energy[i] = e  # type: ignore[index]
            if compute_clearance:
                targets_to_end_clearance[i] = c  # type: ignore[index]
            if compute_time:
                targets_to_end_time[i] = t  # type: ignore[index]
            done += 1
            if self.verbose:
                self._log(done, total, t0, f"T{i}->end")

        def _run_legs(
            start: np.ndarray, goal: np.ndarray
        ) -> tuple[float, float, float, float, float]:
            """Run A*; return (dist, horizontal, climb, descent, clearance).

            El clearance es geométricamente SIMÉTRICO — la ruta i->j y la j->i
            son el mismo trazado — así que la rama simétrica lo asigna a ambas
            direcciones sin recalcular. La energía no lo es: ahí ascenso y
            descenso se intercambian.
            """
            try:
                start_v = grid.world_to_voxel(start.astype(np.float64))
                goal_v = grid.world_to_voxel(goal.astype(np.float64))
                path_voxels = planner._astar_search(start_v, goal_v, free)
                path_world = grid.voxels_to_world(path_voxels)
                if path_world.shape[0] >= 2:
                    path_world[0] = start.astype(np.float64)
                    path_world[-1] = goal.astype(np.float64)
                else:
                    path_world = np.vstack([start.astype(np.float64), goal.astype(np.float64)])
                diffs = np.diff(path_world, axis=0)
                dist = float(np.sum(np.sqrt((diffs**2).sum(axis=1))))
                delta_z_arr = diffs[:, 2]
                clr = float("inf")
                if compute_clearance:
                    from bioinspired.core.trajectory import Trajectory
                    from bioinspired.metrics.path import min_clearance

                    clr = float(
                        min_clearance(
                            Trajectory(waypoints_arr=path_world),
                            self.problem.obstacles_arr,
                            self.problem.boxes_arr,
                        )
                    )
                return (
                    dist,
                    float(np.linalg.norm(diffs[:, :2], axis=1).sum()),
                    float(np.maximum(delta_z_arr, 0.0).sum()),
                    float(np.maximum(-delta_z_arr, 0.0).sum()),
                    clr,
                )
            except Exception:
                return (
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("-inf"),
                )

        def _energy_from_legs(
            dist: float, horizontal: float, climb: float, descent: float
        ) -> float:
            """Energía de un tramo dado en piernas — mismo modelo que ``_run``."""
            if not np.isfinite(dist):
                return float("inf")
            if self.energy_model is not None:
                return self.energy_model.leg_energy_j(
                    horizontal_m=horizontal, climb_m=climb, descent_m=descent, drone=dm
                )
            # Legacy: h_gain = max(0, Δz neto) = max(0, climb - descent) por camino.
            h_gain = max(0.0, climb - descent)
            return float(dm.hover_power_w) * dist / v + 10.0 * dm.total_mass_kg * 9.81 * h_gain

        # all pairs target i → target j
        if self.symmetric_pairs:
            # Un A* por par no ordenado; la dirección inversa reusa el mismo
            # camino geométrico con ascenso y descenso intercambiados. Ver el
            # docstring de __init__ para las consecuencias (gauge/potencial).
            assert targets is not None  # validado en __init__
            for i in range(n):
                dist_matrix[i, i] = 0.0
                if compute_energy:
                    energy_matrix[i, i] = 0.0
                if compute_clearance:
                    clearance_matrix[i, i] = float("inf")  # quedarse quieto no expone
                if compute_time:
                    time_matrix[i, i] = 0.0
                for j in range(i + 1, n):
                    d, horiz, up, down, clr = _run_legs(targets[i], targets[j])
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
                    if compute_energy:
                        energy_matrix[i, j] = _energy_from_legs(d, horiz, up, down)
                        energy_matrix[j, i] = _energy_from_legs(d, horiz, down, up)
                    if compute_clearance:
                        # Simétrico: es el mismo trazado en ambos sentidos.
                        clearance_matrix[i, j] = clr
                        clearance_matrix[j, i] = clr
                    if compute_time:
                        # Simétrico si sube y baja a la misma tasa; se calcula en
                        # general por si un modelo las distingue.
                        time_matrix[i, j] = _time_from_legs(d, horiz, up, down)
                        time_matrix[j, i] = _time_from_legs(d, horiz, down, up)
                    done += 1
                    if self.verbose:
                        self._log(done, total, t0, f"T{i}<->T{j}")
        else:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        dist_matrix[i, j] = 0.0
                        if compute_energy:
                            energy_matrix[i, j] = 0.0  # type: ignore[index]
                        if compute_clearance:
                            clearance_matrix[i, j] = float("inf")  # type: ignore[index]
                        if compute_time:
                            time_matrix[i, j] = 0.0  # type: ignore[index]
                        continue
                    d, e, c, t = _run(targets[i], targets[j])
                    dist_matrix[i, j] = d
                    if compute_energy:
                        energy_matrix[i, j] = e  # type: ignore[index]
                    if compute_clearance:
                        clearance_matrix[i, j] = c  # type: ignore[index]
                    if compute_time:
                        time_matrix[i, j] = t  # type: ignore[index]
                    done += 1
                    if self.verbose:
                        self._log(done, total, t0, f"T{i}->T{j}")

        if self.verbose:
            elapsed = time.perf_counter() - t0
            print(f"\nMatrix built in {elapsed:.1f}s")

        # El clearance va SIEMPRE al final de la tupla: quien no lo pida recibe
        # exactamente la forma de antes, y los datos ya generados siguen válidos.
        base: tuple = (dist_matrix, depot_to_targets, targets_to_end)
        if compute_energy:
            base = (
                *base,
                energy_matrix,
                depot_to_targets_energy,
                targets_to_end_energy,
            )
        if compute_clearance:
            base = (
                *base,
                clearance_matrix,
                depot_to_targets_clearance,
                targets_to_end_clearance,
            )
        # El tiempo va DESPUÉS del clearance, por la misma razón: los llamadores
        # que desempaquetan 3, 6 o 9 elementos no cambian.
        if compute_time:
            base = (*base, time_matrix, depot_to_targets_time, targets_to_end_time)
        return base

    def _log(self, done: int, total: int, t0: float, label: str) -> None:
        elapsed = time.perf_counter() - t0
        eta = (elapsed / done) * (total - done) if done else 0
        print(f"  [{done:3d}/{total}]  {label}  ({elapsed:.1f}s, ETA {eta:.0f}s)")


class RRTStarMatrixBuilder:
    """Builds the RRT*-based distance matrix for an InspectionTSPProblem.

    Drop-in replacement for DistanceMatrixBuilder that uses RRTStarPlanner
    instead of A*. RRT* is stochastic and asymptotically optimal — paths
    improve with more iterations but are not guaranteed to be globally optimal.
    This enables the A* vs RRT* comparison at the matrix-generation layer.

    Each segment (i→j) gets a deterministic seed derived from ``base_seed``
    so results are fully reproducible.

    Args:
        problem: PathPlanningProblem with inspection_targets_arr set.
        max_iterations: RRT* iterations per segment (higher = better quality,
            slower). Default 3000 gives paths within ~5% of A* for typical
            inspection scenarios.
        step_size_m: RRT* step size in metres.
        search_radius_m: RRT* rewire radius in metres.
        base_seed: base seed for reproducibility; each segment gets
            ``base_seed * total_segments + segment_index``.
        verbose: print progress during matrix construction.
    """

    def __init__(
        self,
        problem: PathPlanningProblem,
        max_iterations: int = 3000,
        step_size_m: float = 5.0,
        search_radius_m: float = 15.0,
        base_seed: int = 42,
        verbose: bool = True,
    ) -> None:
        if problem.inspection_targets_arr is None:
            raise ValueError("problem must have inspection_targets_arr set")
        self.problem = problem
        self.max_iterations = max_iterations
        self.step_size_m = step_size_m
        self.search_radius_m = search_radius_m
        self.base_seed = base_seed
        self.verbose = verbose

    def build(
        self, compute_energy: bool = False
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ):
        """Build distance (and optionally energy) matrices using RRT*.

        Interface identical to DistanceMatrixBuilder.build() for drop-in use.

        Args:
            compute_energy: if True, also compute energy_matrix derived from
                the RRT* paths (same energy formula as DistanceMatrixBuilder).

        Returns:
            Same structure as DistanceMatrixBuilder.build().
        """
        from dataclasses import replace

        from bioinspired.algorithms.soo.rrt_star import RRTStarPlanner
        from bioinspired.sim_backends.numpy_backend import NumpyBackend

        targets = self.problem.inspection_targets_arr
        n = len(targets)
        depot = self.problem.start_arr
        end = self.problem.goal_arr
        dm = self.problem.drone_model
        v = float(dm.cruise_velocity_mps) if dm.cruise_velocity_mps > 0 else 1.0

        dist_matrix = np.full((n, n), np.inf)
        depot_to_targets = np.full(n, np.inf)
        targets_to_end = np.full(n, np.inf)

        if compute_energy:
            energy_matrix = np.full((n, n), np.inf)
            depot_to_targets_energy = np.full(n, np.inf)
            targets_to_end_energy = np.full(n, np.inf)

        total = n * (n - 1) + 2 * n
        done = 0
        t0 = time.perf_counter()

        def _run(start: np.ndarray, goal: np.ndarray, seg_idx: int) -> tuple[float, float]:
            """Run RRT*; return (path_length, energy_J). Returns (inf, inf) on failure."""
            planner = RRTStarPlanner(
                max_iterations=self.max_iterations,
                step_size_m=self.step_size_m,
                search_radius_m=self.search_radius_m,
            )
            sub = replace(
                self.problem,
                start_arr=start.astype(np.float64),
                goal_arr=goal.astype(np.float64),
                inspection_targets_arr=None,
                sensor_range_m=0.0,
            )
            seed = self.base_seed * total + seg_idx
            try:
                result = planner.solve(sub, seed=seed, backend=NumpyBackend())
                from bioinspired.metrics.path import length

                dist = length(result.trajectory)
                h_gain = max(0.0, float(goal[2]) - float(start[2]))
                climb_coeff = 10.0
                energy = (
                    float(dm.hover_power_w) * dist / v
                    + climb_coeff * dm.total_mass_kg * 9.81 * h_gain
                )
                return dist, energy
            except Exception:
                return float("inf"), float("inf")

        # depot → each target
        for i in range(n):
            d, e = _run(depot, targets[i], done)
            depot_to_targets[i] = d
            if compute_energy:
                depot_to_targets_energy[i] = e  # type: ignore[index]
            done += 1
            if self.verbose:
                self._log(done, total, t0, f"depot->T{i}")

        # each target → end
        for i in range(n):
            d, e = _run(targets[i], end, done)
            targets_to_end[i] = d
            if compute_energy:
                targets_to_end_energy[i] = e  # type: ignore[index]
            done += 1
            if self.verbose:
                self._log(done, total, t0, f"T{i}->end")

        # all pairs target i → target j
        for i in range(n):
            for j in range(n):
                if i == j:
                    dist_matrix[i, j] = 0.0
                    if compute_energy:
                        energy_matrix[i, j] = 0.0  # type: ignore[index]
                    continue
                d, e = _run(targets[i], targets[j], done)
                dist_matrix[i, j] = d
                if compute_energy:
                    energy_matrix[i, j] = e  # type: ignore[index]
                done += 1
                if self.verbose:
                    self._log(done, total, t0, f"T{i}->T{j}")

        if self.verbose:
            elapsed = time.perf_counter() - t0
            print(f"\nRRT* matrix built in {elapsed:.1f}s")

        if compute_energy:
            return (
                dist_matrix,
                depot_to_targets,
                targets_to_end,
                energy_matrix,
                depot_to_targets_energy,
                targets_to_end_energy,  # type: ignore[return-value]
            )
        return dist_matrix, depot_to_targets, targets_to_end

    def _log(self, done: int, total: int, t0: float, label: str) -> None:
        elapsed = time.perf_counter() - t0
        eta = (elapsed / done) * (total - done) if done else 0
        print(f"  [{done:3d}/{total}]  {label}  ({elapsed:.1f}s, ETA {eta:.0f}s)")
