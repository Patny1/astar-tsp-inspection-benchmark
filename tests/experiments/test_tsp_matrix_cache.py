import json
from pathlib import Path

import numpy as np

from bioinspired.environment.scenarios import generate_tiny_tsp_scenario
from bioinspired.experiments.tsp_matrix_cache import (
    SCHEMA_VERSION,
    build_or_load_matrix,
    cache_path,
    load_validated,
    targets_hash,
    voxel_tag,
)


def test_cache_path_includes_voxel_tag(tmp_path: Path) -> None:
    p = cache_path(tmp_path, "astar", "tiny", 5, 0, 5.0)
    assert p.name == "matrix_astar_tiny_n5_seed0_v5.json"


def test_el_voxel_fraccionario_no_colapsa_sobre_el_entero(tmp_path: Path) -> None:
    """2,5 daba `v2` con int(): una corrida a 2,0 leía las matrices de 2,5.

    Fase 3 corre a 2,5 m porque el objetivo de clearance lo exige, así que la
    colisión estaba a un flag de distancia.
    """
    fino = cache_path(tmp_path, "astar", "tiny", 5, 0, 2.5)
    entero = cache_path(tmp_path, "astar", "tiny", 5, 0, 2.0)
    assert fino != entero
    assert fino.name == "matrix_astar_tiny_n5_seed0_v2.5.json"
    assert entero.name == "matrix_astar_tiny_n5_seed0_v2.json"


def test_el_voxel_5_conserva_el_nombre_historico(tmp_path: Path) -> None:
    """La condición para no invalidar los cachés de Fases 1 y 2, que son todos a 5 m."""
    assert voxel_tag(5.0) == "v5"
    assert cache_path(tmp_path, "astar", "tiny", 5, 0, 5.0).name.endswith("_v5.json")


def test_load_validated_rejects_old_schema(tmp_path: Path) -> None:
    """Un cache con schema_version distinto se rechaza (None)."""
    prob = generate_tiny_tsp_scenario(seed=0, n_targets=5)
    build_or_load_matrix("astar", prob, "tiny", 5, 0, 5.0, tmp_path, compute_energy=False)
    p = cache_path(tmp_path, "astar", "tiny", 5, 0, 5.0)
    data = json.loads(p.read_text())
    data["schema_version"] = SCHEMA_VERSION + 1
    p.write_text(json.dumps(data))
    assert load_validated(p, expected_hash=None) is None


def test_load_validated_rejects_bad_energy_shape(tmp_path: Path) -> None:
    """Un energy_matrix con shape inconsistente se rechaza (None)."""
    prob = generate_tiny_tsp_scenario(seed=0, n_targets=5)
    build_or_load_matrix("astar", prob, "tiny", 5, 0, 5.0, tmp_path, compute_energy=True)
    p = cache_path(tmp_path, "astar", "tiny", 5, 0, 5.0)
    data = json.loads(p.read_text())
    data["energy_matrix"] = [[0.0, 0.0], [0.0, 0.0]]  # (2,2) != (5,5)
    p.write_text(json.dumps(data))
    assert load_validated(p, expected_hash=None) is None


def test_build_then_load_roundtrip(tmp_path: Path) -> None:
    prob = generate_tiny_tsp_scenario(seed=0, n_targets=5)
    (dist, _d2t, _t2e, energy, _e2t, _t2ee), build_s, thash = build_or_load_matrix(
        "astar", prob, "tiny", 5, 0, 5.0, tmp_path, compute_energy=True
    )
    assert dist.shape == (5, 5)
    assert energy is not None and energy.shape == (5, 5)
    assert build_s > 0.0
    (dist2, *_), build_s2, thash2 = build_or_load_matrix(
        "astar", prob, "tiny", 5, 0, 5.0, tmp_path, compute_energy=True
    )
    assert build_s2 == 0.0
    assert np.allclose(dist, dist2)
    assert thash == thash2


def test_load_validated_detects_stale(tmp_path: Path) -> None:
    prob = generate_tiny_tsp_scenario(seed=0, n_targets=5)
    build_or_load_matrix("astar", prob, "tiny", 5, 0, 5.0, tmp_path, compute_energy=False)
    p = cache_path(tmp_path, "astar", "tiny", 5, 0, 5.0)
    wrong_hash = "deadbeef"
    out = load_validated(p, expected_hash=wrong_hash)
    assert out is None


def test_targets_hash_is_deterministic() -> None:
    a = np.array([[1.0, 2.0, 3.0]])
    assert targets_hash(a) == targets_hash(a.copy())
