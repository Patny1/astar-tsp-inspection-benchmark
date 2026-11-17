import numpy as np

from bioinspired.core.tsp_problem import DistanceMatrixBuilder
from bioinspired.environment.scenarios import generate_solar_farm_tsp_scenario
from bioinspired.metrics.path import _box_clearance


def test_solar_farm_geometry() -> None:
    p = generate_solar_farm_tsp_scenario(seed=0, n_targets=15)
    assert p.n_boxes == 96  # 8 filas x 12 columnas
    assert p.inspection_targets_arr is not None
    assert p.inspection_targets_arr.shape == (15, 3)
    # cada target en espacio libre (sobre un panel)
    for t in p.inspection_targets_arr:
        clr = min(_box_clearance(t, b[:3], b[3:6]) for b in p.boxes_arr)
        assert clr > 0.0
    lo, hi = p.bounds_arr
    assert np.all(p.inspection_targets_arr >= lo) and np.all(p.inspection_targets_arr <= hi)


def test_solar_farm_matrix_finite() -> None:
    p = generate_solar_farm_tsp_scenario(seed=0, n_targets=8)
    dist, _d2t, _t2e = DistanceMatrixBuilder(p, voxel_size_m=2.5, verbose=False).build()
    assert np.isfinite(dist).all()
