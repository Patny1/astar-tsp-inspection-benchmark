"""Metricas geometricas y de colision sobre Trajectory.

Funciones puras (sin estado, sin IO, sin plot). Vectorizadas con numpy.
"""

from __future__ import annotations

import numpy as np

from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.drone_model import DroneModel


def length(trajectory: Trajectory) -> float:
    """Longitud euclidiana 3D total de la trayectoria."""
    diffs_arr = np.diff(trajectory.waypoints_arr, axis=0)
    return float(np.linalg.norm(diffs_arr, axis=1).sum())


def count_collisions(
    trajectory: Trajectory,
    obstacles_arr: np.ndarray,
    boxes_arr: np.ndarray | None = None,
) -> dict[str, int]:
    """Cuenta colisiones de la trayectoria contra esferas (M,4) y cajas AABB (K,6).

    Penaliza dos casos:
        1. waypoints_inside: un waypoint cae dentro o en el borde de cualquier obstáculo.
        2. segments_crossing: el segmento entre dos waypoints intersecta algún obstáculo
           (aunque ningun waypoint este dentro).

    Args:
        trajectory: trayectoria a evaluar.
        obstacles_arr: matriz (M, 4) - cada fila (x_centro, y_centro, z_centro, radio).
        boxes_arr: matriz (K, 6) opcional - cada fila (cx, cy, cz, hx, hy, hz).

    Returns:
        Dict con tres llaves:
            - "waypoints_inside": waypoints que caen dentro de algún obstáculo.
            - "segments_crossing": segmentos que intersectan algún obstáculo.
            - "total": suma de los dos anteriores.
    """
    waypoints_arr = trajectory.waypoints_arr
    n = waypoints_arr.shape[0]
    inside_any = np.zeros(n, dtype=bool)
    segments_crossing = 0

    if obstacles_arr.shape[0] > 0:
        centers_arr = obstacles_arr[:, :3]
        radii_arr = obstacles_arr[:, 3]
        dist_arr = np.linalg.norm(
            waypoints_arr[:, np.newaxis, :] - centers_arr[np.newaxis, :, :], axis=2
        )
        inside_any |= (dist_arr <= radii_arr[np.newaxis, :]).any(axis=1)
        for i in range(n - 1):
            p0, p1 = waypoints_arr[i], waypoints_arr[i + 1]
            for j in range(obstacles_arr.shape[0]):
                if _segment_intersects_sphere(p0, p1, centers_arr[j], radii_arr[j]):
                    segments_crossing += 1

    if boxes_arr is not None and boxes_arr.shape[0] > 0:
        bc = boxes_arr[:, :3]
        bh = boxes_arr[:, 3:6]
        inside_box = np.all(
            np.abs(waypoints_arr[:, np.newaxis, :] - bc[np.newaxis, :, :]) <= bh[np.newaxis, :, :],
            axis=2,
        )
        inside_any |= inside_box.any(axis=1)
        for i in range(n - 1):
            p0, p1 = waypoints_arr[i], waypoints_arr[i + 1]
            for k in range(boxes_arr.shape[0]):
                if _segment_intersects_box(p0, p1, bc[k], bh[k]):
                    segments_crossing += 1

    waypoints_inside = int(inside_any.sum())
    return {
        "waypoints_inside": waypoints_inside,
        "segments_crossing": int(segments_crossing),
        "total": int(waypoints_inside + segments_crossing),
    }


def _segment_intersects_sphere(
    p0: np.ndarray, p1: np.ndarray, center: np.ndarray, radius: float
) -> bool:
    """True si el segmento [p0, p1] intersecta la esfera (center, radius).

    Resuelve la ecuacion cuadratica |p0 + t*(p1-p0) - center|^2 = r^2 para t en [0,1].
    """
    d = p1 - p0
    f = p0 - center
    a = float(d @ d)
    if a < 1e-12:  # segmento degenerado (punto)
        return bool(float(f @ f) <= radius * radius)
    b = 2.0 * float(f @ d)
    c = float(f @ f) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = float(np.sqrt(disc))
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    # Hay interseccion si algun t en [0, 1]
    return bool((0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 and t2 > 1.0))


def _segment_intersects_box(
    p0: np.ndarray, p1: np.ndarray, center: np.ndarray, half: np.ndarray
) -> bool:
    """True si el segmento [p0,p1] intersecta la AABB (center±half). Método de slabs."""
    lo = center - half
    hi = center + half
    d = p1 - p0
    t_near, t_far = 0.0, 1.0
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if p0[axis] < lo[axis] or p0[axis] > hi[axis]:
                return False  # paralelo al slab y fuera
            continue
        t1 = (lo[axis] - p0[axis]) / d[axis]
        t2 = (hi[axis] - p0[axis]) / d[axis]
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far:
            return False
    return True


def _box_clearance(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> float:
    """Distancia con signo del punto a la AABB (negativa si está dentro)."""
    d = np.abs(point - center) - half
    outside = float(np.linalg.norm(np.maximum(d, 0.0)))
    inside = float(min(float(np.max(d)), 0.0))
    return float(outside + inside)


def min_clearance(
    trajectory: Trajectory,
    obstacles_arr: np.ndarray,
    boxes_arr: np.ndarray | None = None,
) -> float:
    """Mínimo clearance contra esferas (M,4) y cajas AABB (K,6). inf si no hay obstáculos.

    Clearance for a waypoint = distance to nearest obstacle surface
                               = min(dist_to_center - radius)  for all spheres.
    Clearance for a segment  = minimum over 20 sampled interior points.

    Returns the overall minimum across all waypoints and segment samples.
    A collision-free path always returns > 0. Larger values mean safer paths.
    Returns inf when there are no obstacles.

    Args:
        trajectory: the trajectory to evaluate.
        obstacles_arr: (M, 4) array — each row (cx, cy, cz, radius).
        boxes_arr: (K, 6) array optional — each row (cx, cy, cz, hx, hy, hz).

    Returns:
        Minimum clearance in metres (>0 for collision-free paths).
    """
    has_spheres = obstacles_arr.shape[0] > 0
    has_boxes = boxes_arr is not None and boxes_arr.shape[0] > 0
    if not has_spheres and not has_boxes:
        return float("inf")

    pts = trajectory.waypoints_arr  # (N, 3)

    # Build dense sample set: waypoints + segment interior samples (20 per segment)
    samples = list(pts)
    n_samples = 20
    for i in range(len(pts) - 1):
        ts = np.linspace(0.0, 1.0, n_samples + 2)[1:-1]
        for t in ts:
            samples.append(pts[i] + t * (pts[i + 1] - pts[i]))

    best = float("inf")
    if has_spheres:
        centers = obstacles_arr[:, :3]
        radii = obstacles_arr[:, 3]
        for p in samples:
            best = min(best, float((np.linalg.norm(p - centers, axis=1) - radii).min()))
    if has_boxes:
        bc = boxes_arr[:, :3]  # type: ignore[index]
        bh = boxes_arr[:, 3:6]  # type: ignore[index]
        for p in samples:
            for k in range(bc.shape[0]):
                best = min(best, _box_clearance(p, bc[k], bh[k]))
    return best


def coverage_rate(
    trajectory: Trajectory,
    targets_arr: np.ndarray,
    sensor_range_m: float,
) -> float:
    """Fraction of inspection targets observed by the trajectory.

    A target is considered observed if the drone passes within ``sensor_range_m``
    of it at any point along the path. Segment interiors are sampled (20 points
    per segment) to avoid missing targets between sparse waypoints.

    Args:
        trajectory: the planned path.
        targets_arr: (K, 3) array of inspection target positions.
        sensor_range_m: maximum sensing distance in metres.

    Returns:
        Coverage rate in [0, 1]. 1.0 means all targets observed.
    """
    if targets_arr.shape[0] == 0 or sensor_range_m <= 0:
        return 0.0

    pts = trajectory.waypoints_arr  # (N, 3)
    n_samples = 20

    # Build dense sample set: waypoints + segment interior samples
    samples = list(pts)
    for i in range(len(pts) - 1):
        ts = np.linspace(0.0, 1.0, n_samples + 2)[1:-1]
        for t in ts:
            samples.append(pts[i] + t * (pts[i + 1] - pts[i]))
    samples_arr = np.array(samples)  # (N + segs*n_samples, 3)

    covered = 0
    for target in targets_arr:
        dists = np.linalg.norm(samples_arr - target, axis=1)
        if dists.min() <= sensor_range_m:
            covered += 1

    return float(covered / targets_arr.shape[0])


def smoothness(trajectory: Trajectory) -> float:
    """Variación angular promedio entre segmentos consecutivos (radianes).

    Trajectories más suaves tienen smoothness más bajo. Si n_points <= 2 retorna 0.0.
    """
    pts = trajectory.waypoints_arr
    if pts.shape[0] < 3:
        return 0.0
    v1 = pts[1:-1] - pts[:-2]
    v2 = pts[2:] - pts[1:-1]
    cos_angles = (v1 * v2).sum(axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12
    )
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
    return float(angles.mean())


def kinematic_violations(trajectory: Trajectory, drone: DroneModel) -> dict[str, int]:
    """Cuenta violaciones de restricciones del DroneModel.

    Siempre verifica range y endurance (son independientes de enable_constraints).
    Las restricciones cinemáticas (giro, ascenso, altitud) solo se verifican
    cuando enable_constraints es True.

    Returns dict con: turn_violations, climb_violations, altitude_violations,
                      range_violations, endurance_violations, total.
    """
    pts = trajectory.waypoints_arr
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)

    # ── Range: waypoints fuera del radio máximo desde el punto de inicio ──
    range_violations = 0
    if drone.max_range_m > 0.0:
        dist_from_start = np.linalg.norm(pts - pts[0], axis=1)
        range_violations = int((dist_from_start > drone.max_range_m).sum())

    # ── Endurance: tiempo total de vuelo supera el límite ──
    endurance_violations = 0
    if drone.endurance_min > 0.0:
        total_time_min = float(seg_lengths.sum()) / (drone.cruise_velocity_mps * 60.0)
        endurance_violations = 1 if total_time_min > drone.endurance_min else 0

    if not drone.enable_constraints:
        return {
            "turn_violations": 0,
            "climb_violations": 0,
            "altitude_violations": 0,
            "range_violations": range_violations,
            "endurance_violations": endurance_violations,
            "total": range_violations + endurance_violations,
        }

    seg_len_safe = seg_lengths + 1e-12
    horizontal = np.linalg.norm(diffs[:, :2], axis=1) + 1e-12
    climb_angles_deg = np.degrees(np.arctan2(np.abs(diffs[:, 2]), horizontal))
    climb_violations = int((climb_angles_deg > drone.max_climb_angle_deg).sum())

    if diffs.shape[0] >= 2:
        v1 = diffs[:-1]
        v2 = diffs[1:]
        cos_t = (v1 * v2).sum(axis=1) / (seg_len_safe[:-1] * seg_len_safe[1:])
        turn_angles_deg = np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))
        turn_violations = int((turn_angles_deg > drone.max_turn_angle_deg).sum())
    else:
        turn_violations = 0

    altitudes = pts[:, 2]
    altitude_violations = int(
        ((altitudes < drone.min_altitude_m) | (altitudes > drone.max_altitude_m)).sum()
    )

    total = (
        climb_violations
        + turn_violations
        + altitude_violations
        + range_violations
        + endurance_violations
    )
    return {
        "turn_violations": turn_violations,
        "climb_violations": climb_violations,
        "altitude_violations": altitude_violations,
        "range_violations": range_violations,
        "endurance_violations": endurance_violations,
        "total": total,
    }
