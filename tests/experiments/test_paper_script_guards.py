"""Guardas de run_tsp_paper.py para re-correr sobre un dataset ya publicado.

Re-correr sin ``--voxel 5 --generators astar`` toma los defaults (10 m,
astar+rrtstar) y reconstruiría matrices de **otras** instancias sin avisar.
``--require-cache`` lo convierte en un error. Y como los tiempos sólo valen
dentro de una máquina, reanudar en otra también aborta.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from bioinspired.experiments.snapshot import machine_provenance

_PAPER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_tsp_paper.py"


@pytest.fixture(scope="module")
def paper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_tsp_paper", _PAPER_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _correr(paper: ModuleType, out: Path, cache: Path, **extra: object) -> None:
    paper.run_paper_benchmark(
        scenarios=["tiny"],
        n_targets_list=[5],
        instance_seeds=1,
        algo_seeds_per_instance=1,
        generators=["astar"],
        orderers_to_run=["nn"],
        cost_functions=["distance"],
        voxel_size=10.0,
        output_dir=out,
        cache_dir=cache,
        n_jobs=1,
        **extra,
    )


def test_require_cache_aborta_si_falta_una_matriz(paper: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--require-cache"):
        _correr(paper, tmp_path / "out", tmp_path / "cache_vacio", require_cache=True)
    assert not any((tmp_path / "cache_vacio").glob("*.json")), "no debe construir nada"


def test_la_procedencia_registra_maquina_y_n_jobs(paper: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _correr(paper, out, tmp_path / "cache")
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["n_jobs"] == 1
    assert prov["algo_seeds_per_instance"] == 1
    assert prov["maquina"]["cpu_model"] == machine_provenance()["cpu_model"]


def test_se_niega_a_reanudar_en_otra_maquina(paper: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    ajena = {**machine_provenance(), "cpu_model": "Apple M9 imaginario"}
    (out / "provenance.json").write_text(json.dumps({"maquina": ajena}), encoding="utf-8")
    with pytest.raises(SystemExit, match="otra máquina"):
        _correr(paper, out, tmp_path / "cache")


def test_las_corridas_viejas_sin_maquina_no_bloquean(paper: ModuleType, tmp_path: Path) -> None:
    """canonico no guardaba 'maquina': el guarda no aplica, para no romper reanudaciones."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "provenance.json").write_text(json.dumps({"preset": "canonico"}), encoding="utf-8")
    _correr(paper, out, tmp_path / "cache")
    assert "maquina" in json.loads((out / "provenance.json").read_text(encoding="utf-8"))
