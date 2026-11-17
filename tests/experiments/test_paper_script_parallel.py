"""Paralelización del loop de solvers en run_tsp_paper.py.

El invariante que importa: paralelo debe dar EXACTAMENTE lo mismo que serial.
Un benchmark que cambia de resultado según cuántos cores tenga la máquina no
sirve para un paper.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

_PAPER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_tsp_paper.py"

KEY = [
    "scenario",
    "n_targets",
    "instance_seed",
    "generator",
    "cost_function",
    "orderer",
    "algo_seed",
]


def _load_paper_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_tsp_paper", _PAPER_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def paper() -> ModuleType:
    return _load_paper_script()


def _corre(paper: ModuleType, out: Path, cache: Path, n_jobs: int) -> pd.DataFrame:
    paper.run_paper_benchmark(
        scenarios=["tiny"],
        n_targets_list=[5],
        instance_seeds=2,
        algo_seeds_per_instance=2,
        generators=["astar"],
        orderers_to_run=["nn", "pso"],
        cost_functions=["distance"],
        voxel_size=5.0,
        output_dir=out,
        cache_dir=cache,
        n_jobs=n_jobs,
    )
    return pd.read_csv(out / "results.csv").sort_values(KEY).reset_index(drop=True)


def test_paralelo_da_identico_a_serial(paper, tmp_path: Path) -> None:
    cache = tmp_path / "cache"  # compartido: las matrices no dependen de n_jobs
    serial = _corre(paper, tmp_path / "serial", cache, n_jobs=1)
    paralelo = _corre(paper, tmp_path / "paralelo", cache, n_jobs=2)

    assert len(serial) == len(paralelo) > 0
    for col in ["total_cost", "total_distance", "tour_order", "evals_used", "feasible"]:
        pd.testing.assert_series_equal(
            serial[col], paralelo[col], check_names=False, obj=f"columna {col}"
        )
