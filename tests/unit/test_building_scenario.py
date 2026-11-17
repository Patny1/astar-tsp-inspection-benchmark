import numpy as np

from bioinspired.core.tsp_problem import DistanceMatrixBuilder, InspectionTSPProblem
from bioinspired.environment.scenarios import generate_building_tsp_scenario
from bioinspired.metrics.path import _box_clearance


def test_building_geometry_and_targets() -> None:
    p = generate_building_tsp_scenario(seed=0, n_targets=12)
    assert p.n_boxes == 1
    assert p.inspection_targets_arr is not None
    assert p.inspection_targets_arr.shape == (12, 3)
    box = p.boxes_arr[0]
    # todos los targets en espacio libre (a standoff de la caja)
    for t in p.inspection_targets_arr:
        assert _box_clearance(t, box[:3], box[3:6]) > 0.0
    # dentro de bounds
    lo, hi = p.bounds_arr
    assert np.all(p.inspection_targets_arr >= lo) and np.all(p.inspection_targets_arr <= hi)


def test_building_targets_vary_by_seed() -> None:
    a = generate_building_tsp_scenario(seed=0, n_targets=12).inspection_targets_arr
    b = generate_building_tsp_scenario(seed=1, n_targets=12).inspection_targets_arr
    assert not np.array_equal(a, b)


def test_building_matrix_finite_and_detours() -> None:
    p = generate_building_tsp_scenario(seed=0, n_targets=6)
    dist, d2t, t2e = DistanceMatrixBuilder(p, voxel_size_m=2.5, verbose=False).build()
    assert np.isfinite(dist).all()  # hay camino entre todos (rodea el edificio)
    tsp = InspectionTSPProblem.from_path_problem(p, dist, d2t, t2e)
    assert tsp.n_targets == 6
