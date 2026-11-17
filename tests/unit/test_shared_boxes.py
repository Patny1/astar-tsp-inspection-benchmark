import numpy as np

from bioinspired.algorithms.soo._shared import VoxelGrid, segment_free


def test_segment_free_blocked_by_box() -> None:
    box = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    a = np.array([-5.0, 0.0, 0.0])
    b = np.array([5.0, 0.0, 0.0])
    assert segment_free(a, b, np.zeros((0, 4)), boxes_arr=box) is False


def test_segment_free_clear_with_box_aside() -> None:
    box = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    a = np.array([-5.0, 5.0, 0.0])
    b = np.array([5.0, 5.0, 0.0])
    assert segment_free(a, b, np.zeros((0, 4)), boxes_arr=box) is True


def test_free_voxel_mask_blocks_voxels_inside_box() -> None:
    """Voxels cuyo centro cae dentro de la caja deben quedar marcados como no-libres."""
    # Grid pequeño: bounds [0,10]^3 con voxel_size=2 → 5x5x5 voxels.
    # Centros de voxel en {1,3,5,7,9} x {1,3,5,7,9} x {1,3,5,7,9}.
    # Caja en (5,5,5) con half=(1.5,1.5,1.5) → cubre [3.5,6.5]^3.
    # Voxel (2,2,2) tiene centro (5,5,5) → dentro de la caja → no libre.
    import numpy as np

    from bioinspired.core.problem import PathPlanningProblem
    from bioinspired.environment.drone_model import DroneModel
    from bioinspired.environment.energy_model import NullEnergyModel

    box = np.array([[5.0, 5.0, 5.0, 1.5, 1.5, 1.5]])
    problem = PathPlanningProblem(
        start_arr=np.array([0.0, 0.0, 0.0]),
        goal_arr=np.array([9.9, 9.9, 9.9]),
        bounds_arr=np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]),
        obstacles_arr=np.zeros((0, 4)),
        drone_model=DroneModel(),
        energy_model=NullEnergyModel(),
        n_waypoints=3,
        scenario_id="t",
        seed=0,
        boxes_arr=box,
    )
    grid = VoxelGrid.from_problem(problem, voxel_size_m=2.0)
    mask = grid.free_voxel_mask(problem.obstacles_arr, problem.boxes_arr)
    # Voxel cuyo centro es (5,5,5) → índice (2,2,2) → debe ser False
    assert mask[2, 2, 2] is np.bool_(False)
    # Voxel en la esquina (0,0,0) → índice (0,0,0) → debe ser True (lejos de la caja)
    assert mask[0, 0, 0] is np.bool_(True)
