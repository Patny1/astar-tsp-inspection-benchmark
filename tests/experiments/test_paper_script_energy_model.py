"""Selección del modelo energético en run_tsp_paper.py y su efecto sobre el caché.

El caché guarda las matrices de energía. Si la clave no distingue el modelo, una
corrida con Gong reutiliza en silencio las energías de la heurística vieja — un
error que no da ninguna señal y contamina el dataset entero.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_PAPER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_tsp_paper.py"


def _load_paper_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_tsp_paper", _PAPER_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def paper() -> ModuleType:
    return _load_paper_script()


def test_el_cli_ofrece_los_tres_modelos(paper) -> None:
    """Lo único específico del script: qué expone su interfaz.

    El comportamiento de `cache_path` y `make_energy_model` lo cubre
    `test_tsp_harness.py`, que es donde vive el código; el rotor de
    `gong_coherent`, `test_gong_coherent_rotor.py`.
    """
    assert set(paper.ENERGY_MODELS) == {"legacy", "gong", "gong_coherent"}


def test_el_script_delega_en_el_harness(paper) -> None:
    from bioinspired.experiments import tsp_harness

    assert paper._cache_path is tsp_harness.cache_path
    assert paper._build_matrix is tsp_harness.build_matrix
