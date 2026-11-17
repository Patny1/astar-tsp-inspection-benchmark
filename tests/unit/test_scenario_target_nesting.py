"""Anidamiento de targets entre tamaños de instancia y tope explícito de pool.

Anidar habilita construir UNA matriz A* al N máximo y derivar las de todos los
N menores por submatriz — exacto, porque ``d(i,j)`` no depende de qué otros
targets existan. Además vuelve intra-sujeto la curva de escalamiento: mismo
terreno en todos los N, en vez de una geometría distinta por tamaño.
"""

import numpy as np
import pytest

from bioinspired.environment.scenarios import (
    generate_building_tsp_scenario,
    generate_city_tsp_scenario,
    generate_forest_tsp_scenario,
    generate_solar_farm_tsp_scenario,
    generate_tiny_tsp_scenario,
    generate_urban_tsp_scenario,
)

FACTORIES = {
    "tiny": generate_tiny_tsp_scenario,
    "urban": generate_urban_tsp_scenario,
    "forest": generate_forest_tsp_scenario,
    "building": generate_building_tsp_scenario,
    "solar_farm": generate_solar_farm_tsp_scenario,
    "city": generate_city_tsp_scenario,
}

# Targets disponibles como máximo en cada escenario (grillas de candidatos
# finitas en los de inspección; los de esferas muestrean por rechazo).
POOL_MAX = {"building": 80, "solar_farm": 96}


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_targets_anidan_entre_tamanios(name: str) -> None:
    """Los targets de N=k son el prefijo exacto de los de N=K, con K>k."""
    factory = FACTORIES[name]
    chico = factory(seed=0, n_targets=10).inspection_targets_arr
    grande = factory(seed=0, n_targets=30).inspection_targets_arr
    assert chico is not None and grande is not None
    np.testing.assert_allclose(chico, grande[:10])


@pytest.mark.parametrize("name", sorted(POOL_MAX))
def test_pedir_mas_targets_que_el_pool_falla(name: str) -> None:
    """Pedir más targets de los que hay debe fallar, no recortar en silencio.

    Recortar hace que el CSV diga ``n_targets=100`` sobre una instancia de 80,
    lo que corrompe cualquier análisis de escalamiento sin dar ninguna señal.
    """
    with pytest.raises(ValueError, match="targets"):
        FACTORIES[name](seed=0, n_targets=POOL_MAX[name] + 1)


@pytest.mark.parametrize("name", sorted(POOL_MAX))
def test_el_tope_del_pool_sigue_siendo_alcanzable(name: str) -> None:
    """Pedir exactamente el máximo del pool debe funcionar."""
    arr = FACTORIES[name](seed=0, n_targets=POOL_MAX[name]).inspection_targets_arr
    assert arr is not None
    assert arr.shape[0] == POOL_MAX[name]
