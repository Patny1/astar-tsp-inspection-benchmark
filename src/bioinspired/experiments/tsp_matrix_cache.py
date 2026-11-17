"""Etapa 1: generación + cache versionado de matrices de costo (oráculo A*/RRT*)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from bioinspired.core.problem import PathPlanningProblem
from bioinspired.core.tsp_problem import DistanceMatrixBuilder, RRTStarMatrixBuilder

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

Matrices6 = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]


def targets_hash(targets_arr: np.ndarray) -> str:
    """sha256 de las posiciones de targets (detecta cache stale)."""
    return hashlib.sha256(np.ascontiguousarray(targets_arr, dtype=np.float64).tobytes()).hexdigest()


def voxel_tag(voxel_size: float) -> str:
    """Fragmento de nombre de archivo para un vóxel. Compartido por los dos cachés.

    Era ``int(voxel_size)``, y eso **colapsaba 2,5 m sobre 2,0 m**: ambos daban
    ``v2``, así que una corrida a 2,0 leía en silencio las matrices de 2,5 (o al
    revés) — el mismo mal etiquetado sin señal contra el que ya se puso un guard
    en ``build_matrix``. No era hipotético: Fase 3 corre a 2,5 m porque el
    objetivo de clearance lo exige, y 2,0 está a un flag de distancia.

    ``:g`` desambigua sin romper nada: 5,0 -> ``v5``, idéntico al nombre
    histórico, así que todos los cachés a vóxel 5 (Fases 1 y 2) siguen siendo
    válidos. Sólo cambian los de 2,5, que pasan de ``v2`` a ``v2.5`` y hay que
    reconstruir.
    """
    return f"v{voxel_size:g}"


def cache_path(
    cache_dir: Path,
    generator: str,
    scenario: str,
    n_targets: int,
    instance_seed: int,
    voxel_size: float,
) -> Path:
    """Devuelve la ruta canónica del archivo JSON de cache."""
    return cache_dir / (
        f"matrix_{generator}_{scenario}_n{n_targets}_seed{instance_seed}"
        f"_{voxel_tag(voxel_size)}.json"
    )


def load_validated(path: Path, expected_hash: str | None) -> Matrices6 | None:
    """Carga la matriz si existe, su esquema es vigente y el targets_hash coincide.

    Devuelve None si falta, está obsoleta (schema/hash) o es inconsistente
    (el caller reconstruye).
    """
    if not path.exists():
        return None
    data: dict[str, object] = json.loads(path.read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    if expected_hash is not None and data.get("targets_hash") != expected_hash:
        return None
    dist = np.array(data["distance_matrix"], dtype=np.float64)
    d2t = np.array(data["depot_to_targets"], dtype=np.float64)
    t2e = np.array(data["targets_to_end"], dtype=np.float64)
    k = dist.shape[0]
    if dist.shape != (k, k) or d2t.shape != (k,) or t2e.shape != (k,):
        return None
    energy: np.ndarray | None = (
        np.array(data["energy_matrix"], dtype=np.float64) if "energy_matrix" in data else None
    )
    e2t: np.ndarray | None = (
        np.array(data["depot_to_targets_energy"], dtype=np.float64)
        if "depot_to_targets_energy" in data
        else None
    )
    t2ee: np.ndarray | None = (
        np.array(data["targets_to_end_energy"], dtype=np.float64)
        if "targets_to_end_energy" in data
        else None
    )
    if energy is not None and energy.shape != (k, k):
        return None
    if e2t is not None and e2t.shape != (k,):
        return None
    if t2ee is not None and t2ee.shape != (k,):
        return None
    return dist, d2t, t2e, energy, e2t, t2ee


def build_or_load_matrix(
    generator: str,
    path_problem: PathPlanningProblem,
    scenario: str,
    n_targets: int,
    instance_seed: int,
    voxel_size: float,
    cache_dir: Path,
    compute_energy: bool,
) -> tuple[Matrices6, float, str]:
    """Devuelve ((matrices_6tuple), build_time_s, targets_hash). build_time_s=0.0 en cache hit."""
    if path_problem.inspection_targets_arr is None:
        raise ValueError("path_problem must have inspection_targets_arr set")
    thash = targets_hash(path_problem.inspection_targets_arr)
    path = cache_path(cache_dir, generator, scenario, n_targets, instance_seed, voxel_size)

    cached = load_validated(path, expected_hash=thash)
    if cached is not None:
        return cached, 0.0, thash

    t0 = time.perf_counter()
    if generator == "astar":
        builder: DistanceMatrixBuilder | RRTStarMatrixBuilder = DistanceMatrixBuilder(
            path_problem, voxel_size_m=voxel_size, verbose=False
        )
    elif generator == "rrtstar":
        builder = RRTStarMatrixBuilder(
            path_problem, max_iterations=3000, base_seed=instance_seed, verbose=False
        )
    else:
        raise ValueError(f"generator desconocido: {generator!r}")

    result = builder.build(compute_energy=compute_energy)
    build_time = time.perf_counter() - t0

    energy: np.ndarray | None
    e2t: np.ndarray | None
    t2ee: np.ndarray | None
    if compute_energy:
        dist, d2t, t2e, energy, e2t, t2ee = result  # type: ignore[misc]
    else:
        dist, d2t, t2e = result  # type: ignore[misc]
        energy = e2t = t2ee = None

    data: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generator": generator,
        "scenario": scenario,
        "n_targets": n_targets,
        "instance_seed": instance_seed,
        "voxel_size": voxel_size,
        "targets_hash": thash,
        "distance_matrix": dist.tolist(),
        "depot_to_targets": d2t.tolist(),
        "targets_to_end": t2e.tolist(),
    }
    if compute_energy and energy is not None and e2t is not None and t2ee is not None:
        data["energy_matrix"] = energy.tolist()
        data["depot_to_targets_energy"] = e2t.tolist()
        data["targets_to_end_energy"] = t2ee.tolist()

    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return (dist, d2t, t2e, energy, e2t, t2ee), build_time, thash


def build_matrix_job(
    generator: str,
    scenario: str,
    n_targets: int,
    instance_seed: int,
    voxel_size: float,
    cache_dir: str,
    compute_energy: bool,
) -> tuple[str, int, bool]:
    """Job picklable para prebuild paralelo. Devuelve (scenario, n_targets, ok)."""
    from bioinspired.experiments.tsp_registry import (
        SCENARIO_FACTORIES,
    )

    prob = SCENARIO_FACTORIES[scenario](instance_seed, n_targets)
    targets = prob.inspection_targets_arr
    ok = targets is not None and targets.shape[0] == n_targets
    if not ok:
        got = 0 if targets is None else int(targets.shape[0])
        logger.warning(
            "Prebuild saltado: %s n=%d seed=%d generó %d targets (geometría insuficiente).",
            scenario,
            n_targets,
            instance_seed,
            got,
        )
        return scenario, n_targets, False
    build_or_load_matrix(
        generator,
        prob,
        scenario,
        n_targets,
        instance_seed,
        voxel_size,
        Path(cache_dir),
        compute_energy,
    )
    return scenario, n_targets, True
