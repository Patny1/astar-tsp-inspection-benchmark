"""Generadores determinísticos de escenarios de path planning."""

from __future__ import annotations

import numpy as np

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import BaseEnergyModel, NullEnergyModel


def generate_tiny_scenario(
    seed: int,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
) -> PathPlanningProblem:
    """Escenario pequeño 100x100x50 con 3 obstáculos esféricos.

    Cumple R V2: determinístico con `seed`, comparable entre algoritmos.
    Estilo idéntico al de la imagen de referencia (`GWO Path Planning UAV`).

    Args:
        seed: semilla para colocación de obstáculos.
        drone: opcional, default `DroneModel()`.
        energy: opcional, default `NullEnergyModel()`.
    """
    rng = np.random.default_rng(seed)
    bounds_arr = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 50.0]], dtype=np.float64)
    start_arr = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    goal_arr = np.array([95.0, 95.0, 40.0], dtype=np.float64)

    n_obstacles = 3
    centers_arr = rng.uniform(
        low=[20.0, 20.0, 5.0],
        high=[80.0, 80.0, 30.0],
        size=(n_obstacles, 3),
    )
    radii_arr = rng.uniform(low=5.0, high=12.0, size=(n_obstacles, 1))
    obstacles_arr = np.hstack([centers_arr, radii_arr]).astype(np.float64)

    return PathPlanningProblem(
        start_arr=start_arr,
        goal_arr=goal_arr,
        bounds_arr=bounds_arr,
        obstacles_arr=obstacles_arr,
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=5,
        scenario_id=f"tiny_seed{seed}",
        seed=seed,
    )


def generate_urban_scenario(
    seed: int,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
) -> PathPlanningProblem:
    """Escenario urbano 200x200x80 con 10 obstáculos esféricos (radios 8-20 m).

    Cumple R V2: determinístico con `seed`. Comparable entre algoritmos.
    Representa entornos urbanos densos con edificios distribuidos.

    Args:
        seed: semilla para colocación de obstáculos.
        drone: opcional, default `DroneModel()`.
        energy: opcional, default `NullEnergyModel()`.
    """
    rng = np.random.default_rng(seed)
    bounds_arr = np.array([[0.0, 0.0, 0.0], [200.0, 200.0, 80.0]], dtype=np.float64)
    start_arr = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    goal_arr = np.array([190.0, 190.0, 70.0], dtype=np.float64)

    n_obstacles = 10
    radius_high = 20.0
    margin = radius_high
    centers_arr = rng.uniform(
        low=[margin, margin, margin],
        high=[200.0 - margin, 200.0 - margin, 80.0 - margin],
        size=(n_obstacles, 3),
    )
    radii_arr = rng.uniform(low=8.0, high=radius_high, size=(n_obstacles, 1))
    obstacles_arr = np.hstack([centers_arr, radii_arr]).astype(np.float64)

    return PathPlanningProblem(
        start_arr=start_arr,
        goal_arr=goal_arr,
        bounds_arr=bounds_arr,
        obstacles_arr=obstacles_arr,
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=8,
        scenario_id=f"urban_seed{seed}",
        seed=seed,
    )


def generate_rural_scenario(
    seed: int,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
) -> PathPlanningProblem:
    """Escenario rural 300x300x100 con 6 obstáculos esféricos (radios 10-25 m).

    Cumple R V2: determinístico con `seed`. Comparable entre algoritmos.
    Representa entornos rurales abiertos con pocos obstáculos grandes.

    Args:
        seed: semilla para colocación de obstáculos.
        drone: opcional, default `DroneModel()`.
        energy: opcional, default `NullEnergyModel()`.
    """
    rng = np.random.default_rng(seed)
    bounds_arr = np.array([[0.0, 0.0, 0.0], [300.0, 300.0, 100.0]], dtype=np.float64)
    start_arr = np.array([15.0, 15.0, 15.0], dtype=np.float64)
    goal_arr = np.array([285.0, 285.0, 85.0], dtype=np.float64)

    n_obstacles = 6
    radius_high = 25.0
    margin = radius_high
    centers_arr = rng.uniform(
        low=[margin, margin, margin],
        high=[300.0 - margin, 300.0 - margin, 100.0 - margin],
        size=(n_obstacles, 3),
    )
    radii_arr = rng.uniform(low=10.0, high=radius_high, size=(n_obstacles, 1))
    obstacles_arr = np.hstack([centers_arr, radii_arr]).astype(np.float64)

    return PathPlanningProblem(
        start_arr=start_arr,
        goal_arr=goal_arr,
        bounds_arr=bounds_arr,
        obstacles_arr=obstacles_arr,
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=6,
        scenario_id=f"rural_seed{seed}",
        seed=seed,
    )


def generate_forest_scenario(
    seed: int,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
) -> PathPlanningProblem:
    """Escenario forestal 100x100x50 con 20 obstáculos esféricos (radios 2-5 m).

    Cumple R V2: determinístico con `seed`. Comparable entre algoritmos.
    Representa entornos boscosos densos con muchos obstáculos pequeños (árboles).

    Args:
        seed: semilla para colocación de obstáculos.
        drone: opcional, default `DroneModel()`.
        energy: opcional, default `NullEnergyModel()`.
    """
    rng = np.random.default_rng(seed)
    bounds_arr = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 50.0]], dtype=np.float64)
    start_arr = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    goal_arr = np.array([95.0, 95.0, 45.0], dtype=np.float64)

    n_obstacles = 20
    radius_high = 5.0
    margin = radius_high
    centers_arr = rng.uniform(
        low=[margin, margin, margin],
        high=[100.0 - margin, 100.0 - margin, 50.0 - margin],
        size=(n_obstacles, 3),
    )
    radii_arr = rng.uniform(low=2.0, high=radius_high, size=(n_obstacles, 1))
    obstacles_arr = np.hstack([centers_arr, radii_arr]).astype(np.float64)

    return PathPlanningProblem(
        start_arr=start_arr,
        goal_arr=goal_arr,
        bounds_arr=bounds_arr,
        obstacles_arr=obstacles_arr,
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=10,
        scenario_id=f"forest_seed{seed}",
        seed=seed,
    )


def generate_moo_scenario(
    energy: BaseEnergyModel | None = None,
) -> PathPlanningProblem:
    """Escenario diseñado para MOO con trade-off real distancia vs. energía.

    Geometría determinística (sin seed): obstáculos fijos que obligan a elegir
    entre dos corredores con diferente exposición al viento Este:

      Corredor Norte — más largo (~360 m) pero mayormente perpendicular al viento.
      Corredor Sur   — más corto (~310 m) pero más alineado con el viento (headwind).

    Drone ligero (hover=30 W, masa=0.5 kg, v=8 m/s) + viento Este 5 m/s:
      Con w_distance alto → corredor sur  (corto, más caro en energía).
      Con w_energy alto   → corredor norte (largo, más barato en energía).

    Este conflicto produce un frente Pareto genuino con la escalarización ponderada.

    Args:
        energy: modelo energético; default PhysicalEnergyModel().
    """
    from bioinspired.environment.energy_model import PhysicalEnergyModel

    bounds_arr = np.array([[0.0, 0.0, 0.0], [300.0, 200.0, 60.0]], dtype=np.float64)
    start_arr = np.array([10.0, 10.0, 20.0], dtype=np.float64)
    goal_arr = np.array([290.0, 190.0, 20.0], dtype=np.float64)

    # Obstáculos fijos diseñados para crear dos corredores distintos.
    # Bloque central — cierra la diagonal directa.
    # Corredor norte queda libre por encima de y=155.
    # Corredor sur queda libre por debajo de y=55.
    obstacles_arr = np.array(
        [
            # ── Muro central ──────────────────────────────────────────
            [150.0, 100.0, 20.0, 24.0],  # bloqueador principal
            [160.0, 75.0, 20.0, 18.0],  # extensión sur del muro
            [140.0, 125.0, 20.0, 18.0],  # extensión norte del muro
            # ── Guía corredor sur (más expuesto al viento Este) ───────
            [90.0, 45.0, 20.0, 14.0],  # empuja la ruta sur más al Este
            [210.0, 50.0, 20.0, 14.0],  # mantiene la ruta sur baja y al Este
            # ── Guía corredor norte (perpendicular al viento) ─────────
            [90.0, 155.0, 20.0, 14.0],  # empuja la ruta norte más al Norte
            [210.0, 150.0, 20.0, 14.0],  # mantiene la ruta norte alta
        ],
        dtype=np.float64,
    )

    drone = DroneModel(
        mass_kg=0.5,
        cruise_velocity_mps=8.0,
        max_velocity_mps=12.0,
        hover_power_w=30.0,
        battery_capacity_wh=20.0,
        enable_constraints=False,
        wind_speed_mps=5.0,
        wind_direction_deg=90.0,  # Este — headwind en ruta sur, neutral en ruta norte
    )

    return PathPlanningProblem(
        start_arr=start_arr,
        goal_arr=goal_arr,
        bounds_arr=bounds_arr,
        obstacles_arr=obstacles_arr,
        drone_model=drone,
        energy_model=energy if energy is not None else PhysicalEnergyModel(),
        n_waypoints=10,
        scenario_id="moo_wind_corridor",
        seed=0,
    )


def generate_tower_inspection_scenario(
    energy: BaseEnergyModel | None = None,
) -> PathPlanningProblem:
    """Inspection scenario: high-voltage transmission tower (lattice structure).

    Models a 90 m tall lattice transmission tower with 24 spherical obstacles
    arranged to approximate its real geometry:
      - 4 corner legs that taper from wide base (40x40 m) to narrow top
      - Central junction node at mid-height (blocks direct East path)
      - 4 intermediate structural levels with convergence
      - 2 cross-arm levels (lower z=52 m, upper z=70 m)
      - 3-sphere tapered mast (z=76-90 m)

    Drone flies West → East at z=30 m. The central junction (r=12 m) blocks
    the direct path, requiring every route to detour over or around the tower.

    7 inspection targets at structurally critical locations:
      1. Mast apex         (100, 100,  96) — fly above tower (+66 m)
      2. Left upper arm    ( 66, 100,  72) — wide left detour
      3. Right upper arm   (134, 100,  72) — wide right detour
      4. Left lower arm    ( 72, 100,  54) — moderate left detour
      5. Right lower arm   (128, 100,  54) — moderate right detour
      6. Front-left base   ( 83,  83,  12) — low diagonal approach
      7. Back-right base   (117, 117,  12) — low diagonal approach

    MOO trade-off (distance vs coverage):
      w_distance=1 → ~1767 m path, 29 % coverage (2/7 targets)
      w_coverage=1 → ~1801 m path, 100 % coverage (7/7 targets)  [PSO]
      Only +34 m extra distance for complete inspection coverage.

    Args:
        energy: energy model; defaults to PhysicalEnergyModel().
    """
    from bioinspired.environment.energy_model import PhysicalEnergyModel

    bounds_arr = np.array([[0.0, 0.0, 0.0], [200.0, 200.0, 120.0]], dtype=np.float64)
    start_arr = np.array([10.0, 100.0, 30.0], dtype=np.float64)
    goal_arr = np.array([190.0, 100.0, 30.0], dtype=np.float64)

    obstacles_arr = np.array(
        [
            # ── Base: 4 corner legs, wide apart (z=5) ─────────────────────
            [83.0, 83.0, 5.0, 5.5],  # front-left leg
            [83.0, 117.0, 5.0, 5.5],  # front-right leg
            [117.0, 83.0, 5.0, 5.5],  # back-left leg
            [117.0, 117.0, 5.0, 5.5],  # back-right leg
            # ── Level 1: converging (z=22) ─────────────────────────────────
            [88.0, 88.0, 22.0, 4.5],
            [88.0, 112.0, 22.0, 4.5],
            [112.0, 88.0, 22.0, 4.5],
            [112.0, 112.0, 22.0, 4.5],
            # ── Level 2: more convergence (z=40) ───────────────────────────
            [93.0, 93.0, 40.0, 3.5],
            [93.0, 107.0, 40.0, 3.5],
            [107.0, 93.0, 40.0, 3.5],
            [107.0, 107.0, 40.0, 3.5],
            # ── Central junction node — blocks direct East path (z=35) ─────
            [100.0, 100.0, 35.0, 12.0],
            # ── Level 3: nearly converged (z=58) ───────────────────────────
            [97.0, 100.0, 58.0, 3.0],
            [103.0, 100.0, 58.0, 3.0],
            [100.0, 97.0, 58.0, 3.0],
            [100.0, 103.0, 58.0, 3.0],
            # ── Lower cross-arms (z=52) ────────────────────────────────────
            [79.0, 100.0, 52.0, 3.5],  # left lower arm
            [121.0, 100.0, 52.0, 3.5],  # right lower arm
            # ── Upper cross-arms (z=70) ────────────────────────────────────
            [73.0, 100.0, 70.0, 3.0],  # left upper arm
            [127.0, 100.0, 70.0, 3.0],  # right upper arm
            # ── Tapered mast (z=76-90) ─────────────────────────────────────
            [100.0, 100.0, 76.0, 3.0],
            [100.0, 100.0, 84.0, 2.5],
            [100.0, 100.0, 90.0, 2.0],
        ],
        dtype=np.float64,
    )

    # 7 inspection targets at structurally significant points
    inspection_targets_arr = np.array(
        [
            [100.0, 100.0, 96.0],  # 1. mast apex
            [66.0, 100.0, 72.0],  # 2. left upper arm tip
            [134.0, 100.0, 72.0],  # 3. right upper arm tip
            [72.0, 100.0, 54.0],  # 4. left lower arm tip
            [128.0, 100.0, 54.0],  # 5. right lower arm tip
            [83.0, 83.0, 12.0],  # 6. front-left base leg
            [117.0, 117.0, 12.0],  # 7. back-right base leg
        ],
        dtype=np.float64,
    )

    drone = DroneModel(
        mass_kg=1.5,
        cruise_velocity_mps=5.0,
        max_velocity_mps=10.0,
        hover_power_w=150.0,
        battery_capacity_wh=100.0,
        enable_constraints=False,
        body_radius_m=0.3,
    )

    return PathPlanningProblem(
        start_arr=start_arr,
        goal_arr=goal_arr,
        bounds_arr=bounds_arr,
        obstacles_arr=obstacles_arr,
        drone_model=drone,
        energy_model=energy if energy is not None else PhysicalEnergyModel(),
        n_waypoints=15,
        scenario_id="tower_inspection",
        seed=0,
        inspection_targets_arr=inspection_targets_arr,
        sensor_range_m=12.0,
    )


def _sample_inspection_targets(
    n: int,
    bounds_arr: np.ndarray,
    obstacles_arr: np.ndarray,
    start_arr: np.ndarray,
    goal_arr: np.ndarray,
    seed: int = 99,
    clearance_factor: float = 1.5,
    min_sep: float = 8.0,
) -> np.ndarray:
    """Sample n inspection targets spread across the space, avoiding obstacles.

    Targets are placed via rejection sampling: a candidate is accepted if it
    is at least clearance_factor * r away from every obstacle surface and at
    least min_sep from every already-placed target. Falls back to best-effort
    if the budget is exhausted.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds_arr[0], bounds_arr[1]
    targets: list[np.ndarray] = []
    max_tries = 10_000

    for _ in range(max_tries):
        if len(targets) >= n:
            break
        pt = rng.uniform(lo, hi)
        # Keep away from start/goal corridors
        if np.linalg.norm(pt - start_arr) < min_sep:
            continue
        if np.linalg.norm(pt - goal_arr) < min_sep:
            continue
        # Keep away from obstacles
        ok = True
        for row in obstacles_arr:
            cx, cy, cz, r = row
            dist = np.linalg.norm(pt - np.array([cx, cy, cz]))
            if dist < r * clearance_factor + min_sep:
                ok = False
                break
        if not ok:
            continue
        # Keep away from already-placed targets
        for prev in targets:
            if np.linalg.norm(pt - prev) < min_sep:
                ok = False
                break
        if ok:
            targets.append(pt)

    # If we couldn't fill all n, add whatever we have (< n is acceptable fallback)
    return np.array(targets[:n], dtype=np.float64)


def generate_tiny_tsp_scenario(
    seed: int = 42,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
    n_targets: int = 5,
) -> PathPlanningProblem:
    """Tiny scenario (100x100x50, 3 obstacles) with TSP inspection targets.

    Deterministic with fixed seed=42 by default. Targets are placed away
    from obstacles and spread across the volume.

    Args:
        seed:      scenario seed (obstacles + target placement).
        drone:     optional DroneModel; default DroneModel().
        energy:    optional energy model; default NullEnergyModel().
        n_targets: number of inspection targets (default 5).
    """
    from dataclasses import replace

    base = generate_tiny_scenario(seed=seed, drone=drone, energy=energy)
    targets = _sample_inspection_targets(
        n_targets,
        base.bounds_arr,
        base.obstacles_arr,
        base.start_arr,
        base.goal_arr,
        seed=seed + 1000,
    )
    return replace(
        base,
        inspection_targets_arr=targets,
        sensor_range_m=8.0,
        scenario_id=f"tiny_tsp_seed{seed}",
    )


def generate_urban_tsp_scenario(
    seed: int = 42,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
    n_targets: int = 8,
) -> PathPlanningProblem:
    """Urban scenario (200x200x80, 10 obstacles) with TSP inspection targets.

    Args:
        seed:      scenario seed.
        drone:     optional DroneModel.
        energy:    optional energy model.
        n_targets: number of inspection targets (default 8).
    """
    from dataclasses import replace

    base = generate_urban_scenario(seed=seed, drone=drone, energy=energy)
    targets = _sample_inspection_targets(
        n_targets,
        base.bounds_arr,
        base.obstacles_arr,
        base.start_arr,
        base.goal_arr,
        seed=seed + 1000,
        min_sep=12.0,
    )
    return replace(
        base,
        inspection_targets_arr=targets,
        sensor_range_m=12.0,
        scenario_id=f"urban_tsp_seed{seed}",
    )


def generate_forest_tsp_scenario(
    seed: int = 42,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
    n_targets: int = 10,
) -> PathPlanningProblem:
    """Forest scenario (100x100x50, 20 obstacles) with TSP inspection targets.

    Args:
        seed:      scenario seed.
        drone:     optional DroneModel.
        energy:    optional energy model.
        n_targets: number of inspection targets (default 10).
    """
    from dataclasses import replace

    base = generate_forest_scenario(seed=seed, drone=drone, energy=energy)
    targets = _sample_inspection_targets(
        n_targets,
        base.bounds_arr,
        base.obstacles_arr,
        base.start_arr,
        base.goal_arr,
        seed=seed + 1000,
        clearance_factor=2.0,
        min_sep=6.0,
    )
    return replace(
        base,
        inspection_targets_arr=targets,
        sensor_range_m=6.0,
        scenario_id=f"forest_tsp_seed{seed}",
    )


# Esferas del escenario tri-objetivo. Radio >= 8 m a propósito: con vóxel de
# 5 m un obstáculo más chico se cuela entre centros de vóxel libres y produce
# clearance negativo — rutas que atraviesan la esfera. Como métrica post-hoc eso
# se reporta como limitación conocida (paper_v5: 7,8 % de colisiones
# residuales), pero como OBJETIVO es inaceptable: el optimizador buscaría
# justamente esas rutas para maximizar un margen que en realidad está violando.
# Medido: con vóxel 2,5 m el clearance negativo desaparece en todas las semillas.
_LAYERED_OBS_SPHERES = np.array(
    [
        [27.0, 32.0, 22.0, 10.0],
        [52.0, 68.0, 20.0, 11.0],
        [77.0, 30.0, 24.0, 9.0],
        [40.0, 15.0, 18.0, 8.0],
        [65.0, 88.0, 25.0, 9.0],
    ],
    dtype=np.float64,
)


def generate_layered_obs_tsp_scenario(
    seed: int = 42,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
    n_targets: int = 12,
) -> PathPlanningProblem:
    """Escenario TRI-OBJETIVO: altitudes alternadas + esferas (100x100x45).

    Combina los dos mecanismos que generan conflicto, que los escenarios
    existentes tenían por separado:

    - **Altitudes en damero** (z=10 / z=35, como ``layered``): el tour corto en
      XY alterna alturas y paga el vaivén, así que distancia y energía compiten.
    - **Esferas** (que ``layered`` no tiene, por ser un *clear-sky survey*): los
      tramos difieren en cuánto se acercan a un obstáculo, así que distancia y
      seguridad también compiten.

    Medido por enumeración exacta del frente 3D (N=9, vóxel 2,5, gong+m210), el
    frente pasa de 3-4 puntos sin obstáculos a **12-17 con ellos**, y los puntos
    no soportados —los que ninguna suma ponderada alcanza— de 0 a 6-7. Ese
    salto es lo que hace al escenario útil para comparar escalarización contra
    Pareto genuino.

    OJO con el vóxel: usar 2,5 m o más fino. Ver ``_LAYERED_OBS_SPHERES``.

    Args:
        seed: semilla del jitter de posiciones.
        drone: aparato; por defecto el genérico. Para MOO usar ``dji_m210``.
        energy: modelo energético; por defecto nulo. Para MOO usar ``GongEnergyModel``.
        n_targets: cuántos de los 12 puntos del grid usar.

    Returns:
        El problema de path planning con los targets de inspección.
    """
    dm = drone if drone is not None else DroneModel()
    em = energy if energy is not None else NullEnergyModel()
    rng = np.random.default_rng(seed)

    z_low, z_high = 10.0, 35.0
    col_xs = [15.0, 40.0, 65.0, 90.0]
    row_ys = [15.0, 50.0, 85.0]
    jitter = 3.0

    targets: list[np.ndarray] = []
    for row_i, ry in enumerate(row_ys):
        for col_j, cx in enumerate(col_xs):
            z = z_low if (row_i + col_j) % 2 == 0 else z_high
            dx, dy = rng.uniform(-jitter, jitter, size=2)
            targets.append(np.array([cx + dx, ry + dy, z], dtype=np.float64))

    return PathPlanningProblem(
        start_arr=np.array([5.0, 5.0, 5.0], dtype=np.float64),
        goal_arr=np.array([95.0, 95.0, 40.0], dtype=np.float64),
        bounds_arr=np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 45.0]], dtype=np.float64),
        obstacles_arr=_LAYERED_OBS_SPHERES.copy(),
        drone_model=dm,
        energy_model=em,
        n_waypoints=n_targets,
        scenario_id=f"layered_obs_tsp_seed{seed}",
        seed=seed,
        inspection_targets_arr=np.array(targets[:n_targets], dtype=np.float64),
        sensor_range_m=10.0,
    )


def generate_layered_moo_tsp_scenario(
    seed: int = 42,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
    n_targets: int = 12,
) -> PathPlanningProblem:
    """Checkerboard altitude scenario (100x100x40) designed for MOO Fase 3.

    Targets are arranged on a 4x3 grid where altitude alternates in a
    checkerboard pattern (z=10 m vs z=35 m, gap=25 m).  This ensures:

    - Adjacent targets in XY always have DIFFERENT altitudes.
    - Nearest-neighbour tour alternates altitudes -> short path, many climbs.
    - Altitude-sorted tour visits all low targets first -> long path, one climb.

    This creates a guaranteed Pareto trade-off between path length and
    climbing energy.  12 targets, no obstacle spheres (clear-sky survey).
    """
    dm = drone if drone is not None else DroneModel()
    em = energy if energy is not None else NullEnergyModel()
    rng = np.random.default_rng(seed)

    z_low, z_high = 10.0, 35.0
    bounds_arr = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 45.0]], dtype=np.float64)
    start_arr = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    goal_arr = np.array([95.0, 95.0, 40.0], dtype=np.float64)
    obstacles_arr = np.zeros((0, 4), dtype=np.float64)

    # 4 columns x 3 rows = 12 grid positions
    col_xs = [15.0, 40.0, 65.0, 90.0]
    row_ys = [15.0, 50.0, 85.0]
    jitter = 3.0

    targets: list[np.ndarray] = []
    for row_i, ry in enumerate(row_ys):
        for col_j, cx in enumerate(col_xs):
            z = z_low if (row_i + col_j) % 2 == 0 else z_high
            dx, dy = rng.uniform(-jitter, jitter, size=2)
            targets.append(np.array([cx + dx, ry + dy, z], dtype=np.float64))

    inspection_targets_arr = np.array(targets[:n_targets], dtype=np.float64)

    return PathPlanningProblem(
        start_arr=start_arr,
        goal_arr=goal_arr,
        bounds_arr=bounds_arr,
        obstacles_arr=obstacles_arr,
        drone_model=dm,
        energy_model=em,
        n_waypoints=n_targets,
        scenario_id=f"layered_moo_tsp_seed{seed}",
        seed=seed,
        inspection_targets_arr=inspection_targets_arr,
        sensor_range_m=10.0,
    )


def _facade_candidates(box: np.ndarray, standoff: float) -> np.ndarray:
    """Grilla de candidatos sobre las 4 fachadas de una caja, a `standoff` de la superficie.

    4 niveles de altura x (5 posiciones en x sobre las caras +-y, 5 en y sobre
    las caras +-x) = 80 candidatos, en un orden fijo. Extraida de
    `_building_facade_targets` sin tocar ese orden: `building` sigue bit a bit.
    """
    cx, cy, cz, hx, hy, hz = (float(v) for v in box)
    z_levels = np.linspace(cz - hz + 10.0, cz + hz - 5.0, 4)
    xs = np.linspace(cx - hx, cx + hx, 5)
    ys = np.linspace(cy - hy, cy + hy, 5)
    cands: list[list[float]] = []
    for z in z_levels:
        for x in xs:
            cands.append([float(x), cy - hy - standoff, float(z)])
            cands.append([float(x), cy + hy + standoff, float(z)])
        for y in ys:
            cands.append([cx - hx - standoff, float(y), float(z)])
            cands.append([cx + hx + standoff, float(y), float(z)])
    return np.array(cands, dtype=np.float64)


def _building_facade_targets(
    box: np.ndarray, n: int, standoff: float, rng: np.random.Generator
) -> np.ndarray:
    """Genera n targets sobre las 4 fachadas de la caja, a `standoff` de la superficie."""
    arr = _facade_candidates(box, standoff)
    if n > arr.shape[0]:
        raise ValueError(
            f"se pidieron {n} targets pero la fachada solo ofrece {arr.shape[0]} "
            "posiciones; densificá la grilla de candidatos (z_levels/xs/ys) "
            "en vez de recortar en silencio"
        )
    # permutation()[:n] y no choice(size=n): el prefijo de una permutación es
    # un subconjunto uniforme del mismo tamaño Y anida entre tamaños de
    # instancia, lo que permite derivar las matrices de los N menores por
    # submatriz de la del N máximo.
    idx: np.ndarray = rng.permutation(arr.shape[0])[:n]
    result: np.ndarray = arr[idx]
    return result


# Envolvente de una torre de celosia de 90 m, como cajas AABB (cx,cy,cz,hx,hy,hz).
#
# A voxel 2,5 m no tiene sentido modelar viga por viga: los perfiles reales miden
# 0,3-0,5 m y `free_voxel_mask` marca ocupado por CENTRO de voxel, asi que
# desaparecerian y el planificador cruzaria la estructura. La envolvente es el
# modelo correcto a esta resolucion, y ademas mapea directo a colisiones de
# Gazebo — las esferas del `tower_inspection` de Fase 0 no tienen correlato
# fisico.
_TOWER_BOXES = np.array(
    [
        [100.0, 100.0, 15.0, 14.0, 14.0, 15.0],  # cuerpo z 0-30, base ancha
        [100.0, 100.0, 45.0, 8.0, 8.0, 15.0],  # cuerpo z 30-60
        [100.0, 100.0, 68.0, 4.0, 4.0, 8.0],  # cuerpo z 60-76
        [100.0, 78.0, 52.0, 2.5, 10.0, 1.5],  # mensula inferior -Y
        [100.0, 122.0, 52.0, 2.5, 10.0, 1.5],  # mensula inferior +Y
        [100.0, 82.0, 70.0, 2.5, 8.0, 1.5],  # mensula superior -Y
        [100.0, 118.0, 70.0, 2.5, 8.0, 1.5],  # mensula superior +Y
        [100.0, 100.0, 83.0, 1.5, 1.5, 7.0],  # mastil
    ],
    dtype=np.float64,
)


def _tower_targets(n: int, standoff: float, rng: np.random.Generator) -> np.ndarray:
    """Puntos de inspeccion sobre la estructura, a `standoff` de la superficie.

    Tres familias, que es lo que un inspector real recorre: puntas de mensula
    (donde van los aisladores), caras del cuerpo a varias alturas, y el apice.

    El conflicto multiobjetivo NO viene de la energia — medido, distancia y
    energia se alinean alrededor de un eje vertical y el frente bi-objetivo
    colapsa. Viene de `clearance`: los tramos difieren mucho en cuanto se
    acercan a la estructura, y con los cuatro objetivos la fraccion no
    soportada del frente supera a la de `layered_obs`.
    """
    cands: list[list[float]] = []
    # Puntas de mensula, POR FUERA del extremo. Las mensulas inferiores van de
    # y=68 a 88 y de 112 a 132; las superiores de 74 a 90 y de 110 a 126. Poner
    # el target en el centro de ese rango lo dejaba DENTRO de la caja y el A* no
    # podia alcanzarlo — lo detecto `test_los_targets_caen_en_voxels_libres`.
    for y_punta, z in ((68.0, 52.0), (132.0, 52.0), (74.0, 70.0), (126.0, 70.0)):
        fuera = -standoff if y_punta < 100.0 else standoff
        cands.append([100.0, y_punta + fuera, z])
        cands.append([100.0, y_punta + fuera, z + 6.0])
    # Caras del cuerpo: tres alturas x cuatro caras
    # Alturas del cuerpo: la tercera baja de 68 a 62 para no rozar las mensulas
    # superiores (z 68,5-71,5). A 68 el target queda fuera de la caja pero su
    # VOXEL cae ocupado — el test de voxel es mas estricto que el de caja, y es
    # el que ve el planificador.
    for z, h in ((15.0, 14.0), (45.0, 8.0), (62.0, 4.0)):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cands.append([100.0 + dx * (h + standoff), 100.0 + dy * (h + standoff), z])
    cands.append([100.0, 100.0, 96.0])  # apice del mastil
    arr = np.array(cands, dtype=np.float64)
    # Jitter por semilla: sin esto todas las instancias comparten geometria y
    # `instance_seed` solo elegiria un subconjunto distinto.
    arr = arr + rng.uniform(-1.5, 1.5, size=arr.shape)
    if n > arr.shape[0]:
        raise ValueError(
            f"se pidieron {n} targets pero la torre ofrece {arr.shape[0]} "
            "posiciones; densificar `_tower_targets` en vez de recortar en silencio"
        )
    # permutation()[:n] y no choice(size=n): el prefijo de una permutacion anida
    # entre tamanos de instancia, asi que las matrices de los N menores salen
    # por submatriz de la del N maximo.
    idx: np.ndarray = rng.permutation(arr.shape[0])[:n]
    result: np.ndarray = arr[idx]
    return result


def generate_tower_tsp_scenario(
    seed: int = 42,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
    n_targets: int = 12,
) -> PathPlanningProblem:
    """Inspeccion de torre de transmision: celosia de 90 m como cajas AABB.

    Segundo escenario de la linea MOO, junto a `layered_obs`. Su valor es que el
    conflicto surge de un mecanismo **distinto**: en `layered_obs` son altitudes
    alternadas en damero —construidas para que el orden optimo en XY y el
    optimo en Z difieran—, y aca es la cercania a una estructura real.

    Medido con enumeracion exacta a N=8, voxel 2,5, cuatro objetivos: la torre
    deja **63%** del frente fuera del alcance de la escalarizacion contra 38% de
    `layered_obs`. En bi-objetivo, en cambio, colapsa (frente de 1,7 puntos): el
    conflicto no esta en la energia sino en la seguridad.

    Args:
        seed: semilla del jitter y de la seleccion de targets.
        drone: aparato; por defecto el generico. Para MOO usar `dji_m210`.
        energy: modelo energetico; por defecto nulo. Para MOO, `gong_coherent`.
        n_targets: cuantos de los 21 puntos disponibles usar.
    """
    rng = np.random.default_rng(seed)
    return PathPlanningProblem(
        start_arr=np.array([20.0, 20.0, 10.0], dtype=np.float64),
        goal_arr=np.array([180.0, 180.0, 10.0], dtype=np.float64),
        bounds_arr=np.array([[0.0, 0.0, 0.0], [200.0, 200.0, 120.0]], dtype=np.float64),
        obstacles_arr=np.zeros((0, 4), dtype=np.float64),
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=15,
        scenario_id=f"tower_seed{seed}",
        seed=seed,
        inspection_targets_arr=_tower_targets(n_targets, standoff=6.0, rng=rng),
        sensor_range_m=10.0,
        boxes_arr=_TOWER_BOXES.copy(),
    )


def generate_building_tsp_scenario(
    seed: int = 42,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
    n_targets: int = 12,
) -> PathPlanningProblem:
    """Escenario de inspección de fachada: un edificio (caja AABB) y targets a su alrededor.

    Args:
        seed: semilla (selección de targets de fachada).
        drone: DroneModel (default DroneModel()).
        energy: modelo energético (default NullEnergyModel()).
        n_targets: nº de puntos de inspección sobre las fachadas (default 12).
    """
    rng = np.random.default_rng(seed)
    bounds_arr = np.array([[0.0, 0.0, 0.0], [200.0, 200.0, 100.0]], dtype=np.float64)
    box = np.array([100.0, 100.0, 40.0, 20.0, 20.0, 40.0], dtype=np.float64)
    targets = _building_facade_targets(box, n_targets, standoff=8.0, rng=rng)
    return PathPlanningProblem(
        start_arr=np.array([10.0, 10.0, 5.0], dtype=np.float64),
        goal_arr=np.array([190.0, 190.0, 5.0], dtype=np.float64),
        bounds_arr=bounds_arr,
        obstacles_arr=np.zeros((0, 4), dtype=np.float64),
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=15,
        scenario_id=f"building_seed{seed}",
        seed=seed,
        inspection_targets_arr=targets,
        sensor_range_m=12.0,
        boxes_arr=box.reshape(1, 6),
    )


# ── Manzana urbana (`city`) ────────────────────────────────────────────────────
#
# Escenario principal de Fase 3: edificios como cajas AABB a
# escala real, con correlato fisico directo en Gazebo (`write_px4_world.py`).
# La geometria es el obstaculo en ESPACIO DE CONFIGURACION (fisico + radio del
# dron + margen); el exportador SDF la deflacta, el planificador no infla nada.
#
# Grilla de 4x4 lotes con paso 55 m dentro de 300x300x90. Con semiejes XY en
# [5, 14] y jitter de +-4 m, el pasillo entre vecinos nunca baja de
# 55 - 2*14 - 2*4 = 19 m (>= 15 m, ~6 voxels de 2,5 m entre fachadas
# enfrentadas); el tipico ronda 35 m. El bloque queda en ~[49, 251]^2 (unos
# 200 m de lado), con 30 m de margen al dominio. Los edificios apoyan en z=0.
#
# Presupuesto de bateria (M210, 348 Wh, gong_coherent a 122,9 J/m horizontal):
# 10 % son ~1,0 km y 25 % ~2,5 km. Un tour de N=12 sobre 200 m de manzana ronda
# 1,0-1,5 km mas lo vertical (~12-18 %); N=30, 2,2-3 km (~25-35 %). Para bajar
# N=30 del 25 %, reducir `_CITY_LOT_PITCH_M` a 50 (y el jitter a 3 para
# conservar el minimo). El sondeo de Fase 3 reporta el % real.
_CITY_BOUNDS = np.array([[0.0, 0.0, 0.0], [300.0, 300.0, 90.0]], dtype=np.float64)
_CITY_LOTS = 4
_CITY_LOT_PITCH_M = 55.0
_CITY_LOT_ORIGIN_M = 67.5  # centro del primer lote
_CITY_HALF_XY_M = (5.0, 14.0)  # huella 10-28 m
_CITY_HALF_Z_M = (7.5, 30.0)  # alturas 15-60 m
_CITY_JITTER_M = 4.0
_CITY_MIN_STREET_M = 15.0
_CITY_STANDOFF_M = 8.0  # igual que `building`, para que los frentes sean comparables
_CITY_START = (10.0, 10.0, 5.0)
_CITY_GOAL = (290.0, 290.0, 5.0)


def _city_layout(rng: np.random.Generator, n_buildings: int) -> np.ndarray:
    """(n_buildings, 6) cajas AABB de la manzana.

    Se sortean los parametros de los 16 lotes en orden fijo y DESPUES se elige
    el subconjunto con `permutation()[:n]`: asi `n_buildings` anida (prefijo)
    y la geometria de un lote no depende de cuantos lotes se pidan.
    """
    total = _CITY_LOTS**2
    if not 1 <= n_buildings <= total:
        raise ValueError(f"n_buildings={n_buildings}: la manzana tiene entre 1 y {total} edificios")
    ejes = _CITY_LOT_ORIGIN_M + _CITY_LOT_PITCH_M * np.arange(_CITY_LOTS)
    centros = np.array([(x, y) for x in ejes for y in ejes], dtype=np.float64)
    jitter = rng.uniform(-_CITY_JITTER_M, _CITY_JITTER_M, size=(total, 2))
    half_xy = rng.uniform(*_CITY_HALF_XY_M, size=(total, 2))
    half_z = rng.uniform(*_CITY_HALF_Z_M, size=total)
    idx = rng.permutation(total)[:n_buildings]
    c = centros[idx] + jitter[idx]
    # cz = hz: la base apoya en el suelo.
    return np.column_stack([c, half_z[idx], half_xy[idx], half_z[idx]])


def _city_facade_targets(
    boxes: np.ndarray,
    n: int,
    standoff: float,
    rng: np.random.Generator,
    bounds_arr: np.ndarray,
) -> np.ndarray:
    """n targets sobre las fachadas de la manzana, a `standoff` de SU edificio y
    a por lo menos `standoff` de cualquier otro.

    El filtro contra todas las cajas es lo que evita el fallo silencioso de la
    manzana: un candidato de la fachada de A metido en el pasillo de B queda
    dentro de B (o de su voxel) y el A* no lo alcanza.
    """
    from bioinspired.metrics.path import _box_clearance

    pool = np.vstack([_facade_candidates(b, standoff) for b in boxes])
    # Edificios de 15 m repiten sus 4 niveles: dedup determinista (unique ordena).
    pool = np.unique(pool, axis=0)
    lo, hi = bounds_arr[0], bounds_arr[1]
    dentro = np.all((pool >= lo + 2.5) & (pool <= hi - 2.5), axis=1)
    clr = np.array([min(_box_clearance(p, b[:3], b[3:6]) for b in boxes) for p in pool])
    pool = pool[dentro & (clr >= standoff - 1e-9)]
    if n > pool.shape[0]:
        raise ValueError(
            f"se pidieron {n} targets pero la manzana ofrece {pool.shape[0]} posiciones "
            "libres; densificar `_facade_candidates` en vez de recortar en silencio"
        )
    idx: np.ndarray = rng.permutation(pool.shape[0])[:n]
    result: np.ndarray = pool[idx]
    return result


def generate_city_tsp_scenario(
    seed: int = 42,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
    n_targets: int = 12,
    n_buildings: int = 10,
) -> PathPlanningProblem:
    """Inspeccion de fachadas en una manzana urbana: edificios como cajas AABB.

    Escenario principal de Fase 3 (MOO). Frente a `layered_obs` (esferas sin
    correlato fisico) y a `building` (una sola caja), aporta varios obstaculos
    con forma de edificio, calles de ancho garantizado y variacion de altura
    real, y se exporta a Gazebo tal cual. Ver el bloque de constantes `_CITY_*`
    para la parametrizacion y el presupuesto de bateria.

    Args:
        seed: semilla del layout (jitter, huellas, alturas) y de los targets.
        drone: aparato; por defecto el generico. Para MOO usar `dji_m210`.
        energy: modelo energetico; por defecto nulo. Para MOO, `gong_coherent`.
        n_targets: puntos de inspeccion sobre las fachadas.
        n_buildings: edificios de la manzana (1-16; los primeros k anidan).
    """
    rng = np.random.default_rng(seed)
    boxes = _city_layout(rng, n_buildings)  # antes que los targets: la PP comparte geometria
    targets = _city_facade_targets(boxes, n_targets, _CITY_STANDOFF_M, rng, _CITY_BOUNDS)
    return PathPlanningProblem(
        start_arr=np.array(_CITY_START, dtype=np.float64),
        goal_arr=np.array(_CITY_GOAL, dtype=np.float64),
        bounds_arr=_CITY_BOUNDS.copy(),
        obstacles_arr=np.zeros((0, 4), dtype=np.float64),
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=15,
        scenario_id=f"city_seed{seed}",
        seed=seed,
        inspection_targets_arr=targets,
        sensor_range_m=12.0,
        boxes_arr=boxes,
    )


def generate_city_scenario(
    seed: int,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
    n_buildings: int = 10,
) -> PathPlanningProblem:
    """Escenario PP de la manzana urbana (sin targets de inspeccion).

    Misma geometria e id que `generate_city_tsp_scenario` para la misma semilla:
    es lo que `write_px4_world.py` y Hydra invocan.
    """
    rng = np.random.default_rng(seed)
    boxes = _city_layout(rng, n_buildings)
    return PathPlanningProblem(
        start_arr=np.array(_CITY_START, dtype=np.float64),
        goal_arr=np.array(_CITY_GOAL, dtype=np.float64),
        bounds_arr=_CITY_BOUNDS.copy(),
        obstacles_arr=np.zeros((0, 4), dtype=np.float64),
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=15,
        scenario_id=f"city_seed{seed}",
        seed=seed,
        boxes_arr=boxes,
    )


def _solar_panels(
    rows: int, cols: int, n: int, standoff: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (boxes (rows*cols,6), targets (n,3)). Targets sobre n paneles elegidos."""
    panel_h = np.array(
        [8.0, 2.5, 2.5], dtype=np.float64
    )  # semiejes -> racks 16x5x5 m (resolubles a voxel=5m)
    x0, y0, x_step, y_step, cz = 20.0, 15.0, 24.0, 15.0, 2.5
    boxes: list[list[float]] = []
    centers_top: list[list[float]] = []
    for r in range(rows):
        for c in range(cols):
            cx = x0 + c * x_step
            cy = y0 + r * y_step
            boxes.append([cx, cy, cz, *panel_h.tolist()])
            centers_top.append([cx, cy, cz + float(panel_h[2]) + standoff])
    boxes_arr = np.array(boxes, dtype=np.float64)
    tops = np.array(centers_top, dtype=np.float64)
    if n > tops.shape[0]:
        raise ValueError(
            f"se pidieron {n} targets pero la granja tiene {tops.shape[0]} paneles "
            f"({rows}x{cols}); ampliá la grilla en vez de recortar en silencio"
        )
    # permutation()[:n]: ver nota en _building_facade_targets — anida entre
    # tamaños de instancia sin cambiar la distribución marginal.
    idx: np.ndarray = rng.permutation(tops.shape[0])[:n]
    selected: np.ndarray = tops[idx]
    return boxes_arr, selected


def generate_solar_farm_tsp_scenario(
    seed: int = 42,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
    n_targets: int = 15,
) -> PathPlanningProblem:
    """Escenario de inspección de granja solar: grilla de paneles (cajas bajas) + targets aéreos.

    Args:
        seed: semilla (selección de paneles a inspeccionar).
        drone: DroneModel (default DroneModel()).
        energy: modelo energético (default NullEnergyModel()).
        n_targets: nº de paneles a inspeccionar (<= 96; default 15).
    """
    rng = np.random.default_rng(seed)
    boxes_arr, targets = _solar_panels(rows=8, cols=12, n=n_targets, standoff=2.0, rng=rng)
    return PathPlanningProblem(
        start_arr=np.array([5.0, 5.0, 3.0], dtype=np.float64),
        goal_arr=np.array([315.0, 125.0, 3.0], dtype=np.float64),
        bounds_arr=np.array([[0.0, 0.0, 0.0], [320.0, 130.0, 40.0]], dtype=np.float64),
        obstacles_arr=np.zeros((0, 4), dtype=np.float64),
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=15,
        scenario_id=f"solar_farm_seed{seed}",
        seed=seed,
        inspection_targets_arr=targets,
        sensor_range_m=8.0,
        boxes_arr=boxes_arr,
    )


def generate_building_scenario(
    seed: int,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
) -> PathPlanningProblem:
    """Escenario PP de navegación en edificio (sin targets de inspección)."""
    bounds_arr = np.array([[0.0, 0.0, 0.0], [200.0, 200.0, 100.0]], dtype=np.float64)
    box = np.array([100.0, 100.0, 40.0, 20.0, 20.0, 40.0], dtype=np.float64)
    return PathPlanningProblem(
        start_arr=np.array([10.0, 10.0, 5.0], dtype=np.float64),
        goal_arr=np.array([190.0, 190.0, 5.0], dtype=np.float64),
        bounds_arr=bounds_arr,
        obstacles_arr=np.zeros((0, 4), dtype=np.float64),
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=15,
        scenario_id=f"building_seed{seed}",
        seed=seed,
        boxes_arr=box.reshape(1, 6),
    )


def generate_solar_farm_scenario(
    seed: int,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
) -> PathPlanningProblem:
    """Escenario PP de navegación en granja solar (sin targets de inspección)."""
    rng = np.random.default_rng(seed)
    boxes_arr, _ = _solar_panels(rows=8, cols=12, n=0, standoff=2.0, rng=rng)
    return PathPlanningProblem(
        start_arr=np.array([5.0, 5.0, 3.0], dtype=np.float64),
        goal_arr=np.array([315.0, 125.0, 3.0], dtype=np.float64),
        bounds_arr=np.array([[0.0, 0.0, 0.0], [320.0, 130.0, 40.0]], dtype=np.float64),
        obstacles_arr=np.zeros((0, 4), dtype=np.float64),
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=15,
        scenario_id=f"solar_farm_seed{seed}",
        seed=seed,
        boxes_arr=boxes_arr,
    )


def generate_validation_scenario(
    seed: int,
    drone: DroneModel | None = None,
    energy: BaseEnergyModel | None = None,
) -> PathPlanningProblem:
    """Escenario de validación 150x150x60 con 5 obstáculos esféricos (radios 6-14 m).

    Cumple R V2: determinístico con `seed`. Comparable entre algoritmos.
    Escenario intermedio reservado para validación cruzada de configuraciones.

    Args:
        seed: semilla para colocación de obstáculos.
        drone: opcional, default `DroneModel()`.
        energy: opcional, default `NullEnergyModel()`.
    """
    rng = np.random.default_rng(seed)
    bounds_arr = np.array([[0.0, 0.0, 0.0], [150.0, 150.0, 60.0]], dtype=np.float64)
    start_arr = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    goal_arr = np.array([140.0, 140.0, 50.0], dtype=np.float64)

    n_obstacles = 5
    radius_high = 14.0
    margin = radius_high
    centers_arr = rng.uniform(
        low=[margin, margin, margin],
        high=[150.0 - margin, 150.0 - margin, 60.0 - margin],
        size=(n_obstacles, 3),
    )
    radii_arr = rng.uniform(low=6.0, high=radius_high, size=(n_obstacles, 1))
    obstacles_arr = np.hstack([centers_arr, radii_arr]).astype(np.float64)

    return PathPlanningProblem(
        start_arr=start_arr,
        goal_arr=goal_arr,
        bounds_arr=bounds_arr,
        obstacles_arr=obstacles_arr,
        drone_model=drone if drone is not None else DroneModel(),
        energy_model=energy if energy is not None else NullEnergyModel(),
        n_waypoints=7,
        scenario_id=f"validation_seed{seed}",
        seed=seed,
    )
