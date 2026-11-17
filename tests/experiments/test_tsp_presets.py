"""El preset de Fase 1 tiene que seguir describiendo lo que el registry construye.

Es el candado que separa las dos fases: el paper de Fase 1 ya está escrito, así
que sus hiperparámetros son un dato histórico. Si alguien mueve un default de
clase —o el registry— para favorecer a Fase 2, este test falla y obliga a la
decisión explícita: o se agrega un preset nuevo, o se revierte. Lo que no puede
pasar es que la reproducción de `paper_v5`/`paper_v6` cambie en silencio.
"""

from __future__ import annotations

import pytest

from bioinspired.experiments.tsp_presets import (
    PRESET_CANONICO,
    PRESET_FASE1,
    make_preset,
    parametros_efectivos,
)
from bioinspired.experiments.tsp_registry import make_orderers


@pytest.fixture(scope="module")
def catalogo() -> dict:
    return make_orderers()


def test_el_preset_cubre_todo_el_catalogo(catalogo: dict) -> None:
    assert set(PRESET_FASE1) == set(
        catalogo
    ), "el preset y el registry describen conjuntos distintos de ordenadores"


@pytest.mark.parametrize("nombre", sorted(PRESET_FASE1))
def test_los_parametros_coinciden(nombre: str, catalogo: dict) -> None:
    real = parametros_efectivos(catalogo[nombre]())
    esperado = PRESET_FASE1[nombre]
    assert real == esperado, (
        f"{nombre} ya no corre con los parámetros de Fase 1.\n"
        f"  registry: {real}\n"
        f"  preset  : {esperado}\n"
        "Si el cambio es deliberado para Fase 2+, agregar un preset nuevo en "
        "tsp_presets.py en vez de mover éste: el paper de Fase 1 ya está escrito."
    )


def test_el_presupuesto_de_los_puros_es_el_homologado() -> None:
    """18.000 evaluaciones: la condición para que la comparación signifique algo."""
    for nombre in ("ga", "pso", "gwo", "aco"):
        p = PRESET_FASE1[nombre]
        pop = p.get("pop_size") or p.get("n_particles") or p.get("n_wolves") or p["n_ants"]
        assert pop * p["max_iterations"] == 18_000, f"{nombre} no gasta el presupuesto homologado"


def test_aco_corre_con_la_poblacion_del_registry_no_con_su_default() -> None:
    """Detalle que el manuscrito debe declarar: el registry sobrescribe n_ants."""
    from bioinspired.algorithms.tsp.aco_tsp import ACOTSPSolver

    assert PRESET_FASE1["aco"]["n_ants"] == 60
    import inspect

    assert inspect.signature(ACOTSPSolver.__init__).parameters["n_ants"].default == 30


class TestPresetCanonico:
    """Los valores que las fuentes primarias recomiendan, y que llegan a todos lados."""

    def test_los_dos_valores_corregidos(self) -> None:
        assert PRESET_CANONICO["aco"]["beta"] == 5.0, "Dorigo et al. 1996, ant-cycle óptimo"
        assert PRESET_FASE1["aco"]["beta"] == 2.0, "el histórico no se toca"
        assert PRESET_CANONICO["pso"]["w"] == 0.729
        assert PRESET_CANONICO["pso"]["c1"] == PRESET_CANONICO["pso"]["c2"] == 1.49445

    def test_alcanza_a_los_memeticos(self) -> None:
        assert PRESET_CANONICO["aco_2opt"]["beta"] == 5.0
        assert PRESET_CANONICO["pso_2opt"]["w"] == 0.729

    def test_alcanza_a_las_etapas_internas_de_los_hibridos(self) -> None:
        """Era el agujero: ga_aco/pso_aco instanciaban sus componentes con defaults.

        Sin esto el preset canónico sería una mentira parcial — el ACO de adentro
        de `ga_aco` seguiría corriendo con beta=2.
        """
        cat = make_orderers(PRESET_CANONICO)
        assert cat["ga_aco"]().aco_beta == 5.0
        assert cat["pso_aco"]().aco_beta == 5.0
        assert cat["pso_aco"]().w == 0.729
        assert cat["ga_pso"]().w == 0.729

    def test_no_cambia_nada_que_no_tenga_fuente(self) -> None:
        """GA y GWO no tienen valores canónicos identificables: quedan igual."""
        for nombre in ("ga", "gwo", "nn", "two_opt", "two_opt_random"):
            assert PRESET_CANONICO[nombre] == PRESET_FASE1[nombre]

    def test_el_default_del_registry_sigue_siendo_fase1(self) -> None:
        """Nada cambia sin pedirlo explícitamente."""
        assert make_orderers()["aco"]().beta == 2.0
        assert make_orderers(make_preset("canonico"))["aco"]().beta == 5.0

    def test_nombre_desconocido(self) -> None:
        with pytest.raises(ValueError, match="preset desconocido"):
            make_preset("fase9")
