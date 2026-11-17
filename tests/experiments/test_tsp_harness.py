"""Maquinaria compartida de los benchmarks TSP.

Los runners de cada fase necesitan lo mismo —catálogo del registry, caché con
modelo energético, celda paralela—; compartirlo evita que las copias diverjan.

La clave del caché es la razón dura para compartir y no copiar: si los dos
runners la calculan por separado, la corrida MOO reconstruiría matrices que el
benchmark de Fase 2 ya dejó cacheadas.
"""

from pathlib import Path

import numpy as np
import pytest

from bioinspired.experiments import tsp_harness as harness


def test_expone_la_maquinaria_compartida() -> None:
    for nombre in ("cache_path", "build_matrix", "evals_used", "run_cell", "make_energy_model"):
        assert hasattr(harness, nombre), f"falta {nombre} en el harness"


def test_legacy_conserva_el_nombre_de_cache_historico(tmp_path: Path) -> None:
    """No cambiar: hay cachés ya construidos con este nombre."""
    p = harness.cache_path(tmp_path, "astar", "building", 8, 3, 5.0, "legacy")
    assert p.name == "matrix_astar_building_n8_seed3_v5.json"


def test_el_modelo_energetico_entra_en_la_clave(tmp_path: Path) -> None:
    legacy = harness.cache_path(tmp_path, "astar", "building", 8, 3, 5.0, "legacy")
    gong = harness.cache_path(tmp_path, "astar", "building", 8, 3, 5.0, "gong")
    assert gong != legacy and "gong" in gong.name


def test_el_aparato_entra_en_la_clave(tmp_path: Path) -> None:
    """El peso cambia la matriz de energía: m210 no puede reusar la del default."""
    default = harness.cache_path(tmp_path, "astar", "tiny", 8, 3, 5.0, "gong")
    m210 = harness.cache_path(tmp_path, "astar", "tiny", 8, 3, 5.0, "gong", "m210")
    assert m210 != default and "m210" in m210.name


def test_gong_no_esta_cableado_para_rrtstar(tmp_path: Path) -> None:
    """RRTStarMatrixBuilder hardcodea la heurística vieja: sin este guard, la
    matriz quedaría cacheada bajo una clave "_gong" con energías legacy."""
    with pytest.raises(ValueError, match="astar"):
        harness.build_matrix(
            "rrtstar", None, "tiny", 5, 0, 5.0, tmp_path, True, energy_model="gong"
        )


def test_el_m210_no_esta_cableado_para_rrtstar(tmp_path: Path) -> None:
    """RRTStarMatrixBuilder evalúa con `NumpyBackend`, que infla los obstáculos por
    `body_radius_m`; con el m210 (0,45 m) la matriz saldría de una geometría
    distinta a la del escenario, que ya es C-space. Sólo A* lo respeta."""
    with pytest.raises(ValueError, match="astar"):
        harness.build_matrix(
            "rrtstar", None, "tiny", 5, 0, 5.0, tmp_path, True, energy_model="legacy", drone="m210"
        )


def test_el_prebuild_construye_con_el_aparato_pedido(tmp_path: Path) -> None:
    """Regresión: el prebuild ignoraba `drone` — chequeaba la clave sin sufijo y
    construía la matriz del dron por defecto, así que `run_cell` no encontraba
    la suya y la reconstruía entera. Todo el prebuild era trabajo tirado."""
    import importlib.util

    ruta = Path(__file__).resolve().parents[2] / "scripts" / "run_tsp_paper.py"
    spec = importlib.util.spec_from_file_location("run_tsp_paper", ruta)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod._build_matrix_job(
        "astar", "tiny", 4, 0, 10.0, str(tmp_path), True, energy_model="gong", drone="m210"
    )
    esperado = harness.cache_path(tmp_path, "astar", "tiny", 4, 0, 10.0, "gong", "m210")
    assert esperado.exists(), "el prebuild no escribió la clave del aparato pedido"

    # Y la energía es la del m210, no la del default: mismo escenario con el
    # dron por defecto tiene que dar otra matriz (pesa 3,2x menos).
    mod._build_matrix_job(
        "astar", "tiny", 4, 0, 10.0, str(tmp_path), True, energy_model="gong", drone="default"
    )
    import json

    e_m210 = np.array(json.loads(esperado.read_text())["energy_matrix"])
    ruta_default = harness.cache_path(tmp_path, "astar", "tiny", 4, 0, 10.0, "gong")
    e_default = np.array(json.loads(ruta_default.read_text())["energy_matrix"])
    off = ~np.eye(4, dtype=bool)
    assert (e_m210[off] > 2.0 * e_default[off]).all()


def test_gong_corrige_la_potencia_inducida_duplicada() -> None:
    modelo = harness.make_energy_model("gong")
    assert modelo is not None and modelo.continuous_at_hover is True
    assert harness.make_energy_model("legacy") is None
    with pytest.raises(ValueError, match="desconocido"):
        harness.make_energy_model("inexistente")


def test_evals_used_sale_del_solver_no_del_nominal() -> None:
    from bioinspired.algorithms.tsp.base import TSPResult

    r = TSPResult(
        best_tour=[0, 1],
        best_cost=1.0,
        convergence_history=np.array([1.0]),
        runtime_seconds=0.0,
        metadata={"n_evaluations": 17910},
    )
    assert harness.evals_used(r, is_deterministic=False) == 17910
    assert harness.evals_used(r, is_deterministic=True) == 0


def test_run_cell_es_top_level_y_recibe_los_pendientes() -> None:
    """joblib serializa por módulo+qualname: no puede ser anidada ni un cierre."""
    import inspect

    assert harness.run_cell.__qualname__ == "run_cell"
    assert "pendientes" in inspect.signature(harness.run_cell).parameters


def test_run_cell_mide_cpu_no_reloj_de_pared() -> None:
    """Con reloj de pared la contención infla la columna bajo paralelismo."""
    import inspect

    codigo = "\n".join(
        linea.split("#")[0] for linea in inspect.getsource(harness.run_cell).splitlines()
    )
    assert "process_time" in codigo
    assert "perf_counter" not in codigo


def test_el_script_del_paper_usa_el_harness() -> None:
    """Anti-deriva: si el script vuelve a definir lo suyo, esto falla."""
    import importlib.util

    ruta = Path(__file__).resolve().parents[2] / "scripts" / "run_tsp_paper.py"
    spec = importlib.util.spec_from_file_location("run_tsp_paper", ruta)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._cache_path is harness.cache_path
    assert mod._run_cell is harness.run_cell
    assert mod._build_matrix is harness.build_matrix
