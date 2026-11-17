"""Calibración contra TSPLIB: instancias estándar con óptimos publicados.

Sirve para responder la objeción de que los solvers rinden mal por estar mal
implementados o mal configurados. Sobre instancias con óptimo conocido el gap
es verificable por cualquiera.
"""

from pathlib import Path

import numpy as np
import pytest

from bioinspired.experiments.tsplib import (
    KNOWN_OPTIMA,
    load_tsplib_instance,
    tsplib_to_tsp_problem,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "tsplib"

pytestmark = pytest.mark.skipif(
    not (DATA / "berlin52.tsp").is_file(),
    reason="instancias TSPLIB no descargadas (ver data/tsplib/README)",
)


def test_parsea_dimension_y_coordenadas() -> None:
    inst = load_tsplib_instance(DATA / "berlin52.tsp")
    assert inst.name == "berlin52"
    assert inst.coords_arr.shape == (52, 2)
    np.testing.assert_allclose(inst.coords_arr[0], [565.0, 575.0])
    np.testing.assert_allclose(inst.coords_arr[-1], [1740.0, 245.0])


def test_distancia_euc2d_usa_el_redondeo_de_tsplib() -> None:
    """EUC_2D redondea al entero más cercano con ``int(x + 0.5)``.

    Los óptimos publicados asumen ese redondeo; sin él los gaps no cierran.
    Ojo: no es ``round()`` de Python, que redondea al par en los empates.
    """
    inst = load_tsplib_instance(DATA / "berlin52.tsp")
    d = inst.distance_matrix
    assert d.dtype.kind in "iu" or np.allclose(d, np.rint(d))
    # ciudad 1 -> 2: dx=540, dy=390 -> sqrt=666.03... -> 666
    esperado = int(np.sqrt(540.0**2 + 390.0**2) + 0.5)
    assert int(d[0, 1]) == esperado
    assert int(d[1, 0]) == esperado  # simétrica
    assert int(d[0, 0]) == 0


def test_mapea_a_tour_cerrado_fijando_la_ciudad_0() -> None:
    """TSPLIB es un tour cerrado; el harness es depósito → targets → fin.

    Se fija la ciudad 0 como depósito Y fin, con lo que ``tour_cost`` de una
    permutación de las n-1 restantes es exactamente el largo del tour cerrado.
    """
    inst = load_tsplib_instance(DATA / "berlin52.tsp")
    p = tsplib_to_tsp_problem(inst)
    assert p.n_targets == 51  # 52 ciudades - la fijada como depósito
    np.testing.assert_allclose(p.depot_arr[:2], inst.coords_arr[0])
    np.testing.assert_allclose(p.end_arr[:2], inst.coords_arr[0])

    # el costo de un tour debe ser la suma de aristas del ciclo cerrado
    tour = list(range(51))
    manual = inst.distance_matrix[0, 1]
    manual += sum(inst.distance_matrix[i, i + 1] for i in range(1, 51))
    manual += inst.distance_matrix[51, 0]
    assert p.tour_cost(tour) == pytest.approx(float(manual))


@pytest.mark.parametrize("nombre", ["berlin52", "eil51", "st70"])
def test_ningun_solver_puede_batir_el_optimo_publicado(nombre: str) -> None:
    """Invariante que valida la función de distancia.

    Si un solver "mejora" el óptimo publicado, las distancias están mal
    calculadas (típicamente por no aplicar el redondeo de EUC_2D).
    """
    from bioinspired.algorithms.tsp.nn_tsp import NNTSPSolver
    from bioinspired.algorithms.tsp.two_opt_tsp import TwoOptTSPSolver

    inst = load_tsplib_instance(DATA / f"{nombre}.tsp")
    p = tsplib_to_tsp_problem(inst)
    opt = KNOWN_OPTIMA[nombre]
    for solver in (NNTSPSolver(), TwoOptTSPSolver()):
        costo = solver.solve(p, seed=0).best_cost
        assert costo >= opt - 1e-6, f"{solver.name} batió el óptimo de {nombre}: {costo} < {opt}"


def test_optimos_conocidos_cubren_las_instancias_descargadas() -> None:
    for f in sorted(DATA.glob("*.tsp")):
        assert f.stem in KNOWN_OPTIMA, f"falta el óptimo publicado de {f.stem}"
