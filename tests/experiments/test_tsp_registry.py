"""Tests del registry TSP compartido."""

import importlib.util
from pathlib import Path
from types import ModuleType

from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm
from bioinspired.experiments.tsp_registry import (
    DETERMINISTIC_ORDERERS,
    SCENARIO_FACTORIES,
    make_orderers,
)

_PAPER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_tsp_paper.py"


def _load_paper_script() -> ModuleType:
    """Importa scripts/run_tsp_paper.py como módulo (no es un paquete instalable)."""
    spec = importlib.util.spec_from_file_location("run_tsp_paper", _PAPER_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scenarios_present() -> None:
    # Esferas (abstracto) + cajas de inspección (building, solar_farm, city).
    assert set(SCENARIO_FACTORIES) == {"tiny", "urban", "forest", "building", "solar_farm", "city"}


def test_make_orderers_returns_factories() -> None:
    orderers = make_orderers()
    assert set(orderers) == {
        "nn",
        "two_opt",
        "two_opt_random",
        "ga",
        "pso",
        "gwo",
        "aco",
        "ga_aco",
        "pso_aco",
        "ga_pso",
        "ga_2opt",
        "aco_2opt",
        "pso_2opt",
        "gwo_2opt",
    }
    inst = orderers["ga"]()
    assert isinstance(inst, BaseTSPAlgorithm)


def test_two_opt_random_esta_en_el_registry() -> None:
    """La variante estocástica de 2-opt vivía solo en el script del paper."""
    orderers = make_orderers()
    assert "two_opt_random" in orderers
    assert isinstance(orderers["two_opt_random"](), BaseTSPAlgorithm)


def test_two_opt_random_no_es_determinista() -> None:
    """Arranca de una permutación por seed, así que necesita varios algo_seeds.

    Si entrara en DETERMINISTIC_ORDERERS el runner le asignaría un solo seed y
    la variante perdería toda su varianza — justo lo que el ablation mide.
    """
    assert "two_opt_random" not in DETERMINISTIC_ORDERERS


def test_script_del_paper_no_mantiene_catalogo_propio() -> None:
    """El script debe ofrecer exactamente los ordenadores del registry.

    Tenían listas separadas que derivaron en ambos sentidos: al registry le
    faltaba two_opt_random y al script los híbridos y meméticos. Este test falla
    si alguien vuelve a hardcodear un catálogo local.
    """
    paper = _load_paper_script()
    assert set(paper.ORDERER_NAMES) == set(make_orderers())


def test_registry_includes_all_hybrids() -> None:
    o = make_orderers()
    for k in ["ga_aco", "pso_aco", "ga_pso", "ga_2opt", "aco_2opt", "pso_2opt", "gwo_2opt"]:
        assert k in o
        assert o[k]() is not None


def test_deterministic_set() -> None:
    assert frozenset({"nn", "two_opt"}) == DETERMINISTIC_ORDERERS


def test_scenario_factory_signature() -> None:
    # Llamada POSICIONAL (seed, n_targets) — debe respetar n_targets, no bindearlo a drone.
    prob = SCENARIO_FACTORIES["tiny"](0, 7)
    assert prob.inspection_targets_arr is not None
    assert prob.inspection_targets_arr.shape[0] == 7
