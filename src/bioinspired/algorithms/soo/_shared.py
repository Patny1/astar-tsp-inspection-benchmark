"""Helpers compartidos entre los algoritmos SOO de Fase 1.

Centraliza la logica que A*, RRT*, ACO, GA, PSO y GWO duplicaban: voxel grid 3D,
re-muestreo por longitud de arco, deteccion de colision en segmento y wrappers
de evaluacion con penalizacion de colisiones. Permite un punto unico de cambio
para tunear el modelo de penalizacion o la conectividad del grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.trajectory import Trajectory
from bioinspired.sim_backends.base import SimulationBackend

COLLISION_PENALTY: float = 1000.0
N_NEIGHBORS_26: int = 26


def neighbors_offsets() -> np.ndarray:
    """26-conectividad: cubo 3x3x3 menos el centro. Shape ``(26, 3)`` int32."""
    return np.array(
        [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ],
        dtype=np.int32,
    )


@dataclass(frozen=True)
class VoxelGrid:
    """Grid 3D sobre ``bounds_arr`` con resolucion ``voxel_size_m``.

    Encapsula los helpers de A*/ACO. Construyalo via
    ``VoxelGrid.from_problem(problem, voxel_size_m)``.

    Attributes:
        voxel_size_m: tamano del voxel (lado del cubo) en metros.
        shape: numero de voxels por eje ``(nx, ny, nz)``.
        origin_arr: punto ``bounds_arr[0]`` como float64 ``(3,)``.
    """

    voxel_size_m: float
    shape: tuple[int, int, int]
    origin_arr: np.ndarray  # (3,) -- bounds_arr[0]

    @classmethod
    def from_problem(cls, problem: PathPlanningProblem, voxel_size_m: float) -> VoxelGrid:
        """Construye un grid que cubre ``problem.bounds_arr`` con voxels uniformes."""
        extent = problem.bounds_arr[1] - problem.bounds_arr[0]
        nx, ny, nz = np.ceil(extent / voxel_size_m).astype(int)
        return cls(
            voxel_size_m=voxel_size_m,
            shape=(int(nx), int(ny), int(nz)),
            origin_arr=problem.bounds_arr[0].astype(np.float64),
        )

    def world_to_voxel(self, point_arr: np.ndarray) -> tuple[int, int, int]:
        """Mapea un punto en mundo al indice ``(i, j, k)`` del voxel que lo contiene.

        Clipea al rango valido para que puntos exactamente en ``bounds[1]`` no
        caigan fuera por una unidad.
        """
        offset = (point_arr - self.origin_arr) / self.voxel_size_m
        idx = np.floor(offset).astype(int)
        idx = np.clip(
            idx,
            [0, 0, 0],
            [self.shape[0] - 1, self.shape[1] - 1, self.shape[2] - 1],
        )
        return int(idx[0]), int(idx[1]), int(idx[2])

    def voxels_to_world(self, path_voxels: list[tuple[int, int, int]]) -> np.ndarray:
        """Convierte una secuencia de voxels al centro de cada celda en mundo."""
        if not path_voxels:
            return np.empty((0, 3), dtype=np.float64)
        idx_arr = np.asarray(path_voxels, dtype=np.float64)
        return self.origin_arr + (idx_arr + 0.5) * self.voxel_size_m

    def free_voxel_mask(
        self, obstacles_arr: np.ndarray, boxes_arr: np.ndarray | None = None
    ) -> np.ndarray:
        """Mask booleana ``(nx, ny, nz)``: True donde el centro NO cae en obstaculos.

        Vectoriza por obstaculo: para cada esfera ``(cx, cy, cz, r)``, marca como
        False los voxels cuyo centro queda a distancia ``<= r`` del centro.
        Para cajas AABB ``(cx, cy, cz, hx, hy, hz)``, marca como False los voxels
        cuyo centro cae dentro de la caja.
        """
        nx, ny, nz = self.shape
        ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        centers_arr = (
            self.origin_arr
            + (np.stack([ix, iy, iz], axis=-1).astype(np.float64) + 0.5) * self.voxel_size_m
        )
        free = np.ones(self.shape, dtype=bool)
        for cx, cy, cz, r in obstacles_arr:
            d2 = ((centers_arr - np.array([cx, cy, cz])) ** 2).sum(axis=-1)
            free &= d2 > r * r
        if boxes_arr is not None and boxes_arr.shape[0] > 0:
            for cx, cy, cz, hx, hy, hz in boxes_arr:
                c = np.array([cx, cy, cz])
                h = np.array([hx, hy, hz])
                inside = np.all(np.abs(centers_arr - c) <= h, axis=-1)
                free &= ~inside
        return free


def resample_arclength(path_arr: np.ndarray, n_total: int) -> np.ndarray:
    """Re-interpola un path de M puntos a ``n_total`` puntos por longitud de arco.

    Si M < 2 (path degenerado) replica el unico punto. Si ``n_total <= M``,
    sub-muestrea uniformemente. Si ``n_total > M``, interpola linealmente entre
    los puntos del path proporcionalmente a la distancia acumulada. Si todos los
    puntos son iguales, replica el primero.
    """
    if path_arr.shape[0] < 2:
        base = path_arr[0] if path_arr.size else np.zeros(3, dtype=np.float64)
        return np.tile(base, (max(n_total, 1), 1))
    if n_total <= path_arr.shape[0]:
        idxs = np.linspace(0, path_arr.shape[0] - 1, n_total).round().astype(int)
        return path_arr[idxs]  # type: ignore[no-any-return]
    seg_lens = np.linalg.norm(np.diff(path_arr, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    if cum[-1] < 1e-12:
        return np.tile(path_arr[0], (n_total, 1))
    targets = np.linspace(0.0, cum[-1], n_total)
    return np.column_stack([np.interp(targets, cum, path_arr[:, axis]) for axis in range(3)])


def segment_free(
    a: np.ndarray,
    b: np.ndarray,
    obstacles_arr: np.ndarray,
    boxes_arr: np.ndarray | None = None,
) -> bool:
    """True si el segmento ``a->b`` no atraviesa ninguna esfera ni caja AABB.

    Calcula la distancia minima de cada centro de esfera al segmento (con
    parametro ``t`` clipeado a ``[0, 1]``) y compara contra el radio.
    Para cajas AABB usa el método de slabs importado de metrics.path.
    """
    if obstacles_arr.shape[0] > 0:
        centers = obstacles_arr[:, :3]
        radii = obstacles_arr[:, 3]
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq < 1e-18:
            if np.any(np.linalg.norm(centers - a, axis=1) < radii):
                return False
        else:
            ap = centers - a
            t = np.clip(ap @ ab / ab_len_sq, 0.0, 1.0)
            closest = a + t[:, None] * ab
            dists = np.linalg.norm(centers - closest, axis=1)
            if not bool(np.all(dists >= radii)):
                return False
    if boxes_arr is not None and boxes_arr.shape[0] > 0:
        from bioinspired.metrics.path import _segment_intersects_box

        bc = boxes_arr[:, :3]
        bh = boxes_arr[:, 3:6]
        for k in range(boxes_arr.shape[0]):
            if _segment_intersects_box(a, b, bc[k], bh[k]):
                return False
    return True


def evaluate_with_penalty(
    trajectory: Trajectory,
    problem: PathPlanningProblem,
    backend: SimulationBackend,
    penalty: float = COLLISION_PENALTY,
) -> tuple[float, float]:
    """Evalua la trayectoria; retorna ``(energy_raw, fitness_penalizado)``.

    ``fitness = energy_raw + penalty * collisions_total``. Acepta tanto
    ``sim.collisions`` como ``int`` o como ``dict`` con clave ``"total"``
    para tolerar backends futuros.
    """
    sim = backend.evaluate(trajectory, problem)
    energy_raw = float(sim.energy)
    coll_total = (
        sim.collisions.get("total", 0) if isinstance(sim.collisions, dict) else int(sim.collisions)
    )
    fitness = energy_raw + penalty * float(coll_total)
    return energy_raw, fitness


def genome_to_trajectory(genome_arr: np.ndarray, problem: PathPlanningProblem) -> Trajectory:
    """Construye ``[start, *waypoints, goal]`` desde un genome flat.

    Args:
        genome_arr: shape ``(n_waypoints * 3,)`` con coordenadas aplanadas.
        problem: aporta ``start_arr``, ``goal_arr`` y ``n_waypoints``.

    Returns:
        ``Trajectory`` con ``waypoints_arr`` shape ``(n_waypoints + 2, 3)``.
    """
    wps = genome_arr.reshape(problem.n_waypoints, 3)
    full = np.vstack(
        [
            problem.start_arr.reshape(1, 3),
            wps,
            problem.goal_arr.reshape(1, 3),
        ]
    )
    return Trajectory(waypoints_arr=full)
