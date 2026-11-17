"""Tests para escenarios urban/rural/forest/validation."""

from __future__ import annotations

import numpy as np
import pytest

from bioinspired.environment.scenarios import (
    generate_forest_scenario,
    generate_rural_scenario,
    generate_urban_scenario,
    generate_validation_scenario,
)

SCENARIOS = {
    "urban": (
        generate_urban_scenario,
        {
            "bounds": [[0.0, 0.0, 0.0], [200.0, 200.0, 80.0]],
            "n_obstacles": 10,
            "radii_range": (8.0, 20.0),
            "n_waypoints": 8,
        },
    ),
    "rural": (
        generate_rural_scenario,
        {
            "bounds": [[0.0, 0.0, 0.0], [300.0, 300.0, 100.0]],
            "n_obstacles": 6,
            "radii_range": (10.0, 25.0),
            "n_waypoints": 6,
        },
    ),
    "forest": (
        generate_forest_scenario,
        {
            "bounds": [[0.0, 0.0, 0.0], [100.0, 100.0, 50.0]],
            "n_obstacles": 20,
            "radii_range": (2.0, 5.0),
            "n_waypoints": 10,
        },
    ),
    "validation": (
        generate_validation_scenario,
        {
            "bounds": [[0.0, 0.0, 0.0], [150.0, 150.0, 60.0]],
            "n_obstacles": 5,
            "radii_range": (6.0, 14.0),
            "n_waypoints": 7,
        },
    ),
}


@pytest.mark.parametrize("name,gen_and_spec", list(SCENARIOS.items()))
class TestAdditionalScenarios:
    def test_es_deterministico_con_mismo_seed(self, name: str, gen_and_spec: tuple) -> None:
        gen, _ = gen_and_spec
        a = gen(seed=42)
        b = gen(seed=42)
        np.testing.assert_array_equal(a.obstacles_arr, b.obstacles_arr)
        np.testing.assert_array_equal(a.start_arr, b.start_arr)
        np.testing.assert_array_equal(a.goal_arr, b.goal_arr)

    def test_seeds_distintos_dan_obstaculos_distintos(self, name: str, gen_and_spec: tuple) -> None:
        gen, _ = gen_and_spec
        a = gen(seed=1)
        b = gen(seed=2)
        assert not np.allclose(a.obstacles_arr, b.obstacles_arr)

    def test_n_obstaculos_correcto(self, name: str, gen_and_spec: tuple) -> None:
        gen, spec = gen_and_spec
        p = gen(seed=42)
        assert p.obstacles_arr.shape == (spec["n_obstacles"], 4)
        assert p.n_obstacles == spec["n_obstacles"]

    def test_radios_dentro_de_rango(self, name: str, gen_and_spec: tuple) -> None:
        gen, spec = gen_and_spec
        p = gen(seed=42)
        lo, hi = spec["radii_range"]
        radii_arr = p.obstacles_arr[:, 3]
        assert np.all(radii_arr >= lo)
        assert np.all(radii_arr <= hi)

    def test_centros_dentro_de_bounds(self, name: str, gen_and_spec: tuple) -> None:
        gen, spec = gen_and_spec
        p = gen(seed=42)
        bounds_arr = np.array(spec["bounds"], dtype=np.float64)
        centers_arr = p.obstacles_arr[:, :3]
        assert np.all(centers_arr >= bounds_arr[0])
        assert np.all(centers_arr <= bounds_arr[1])

    def test_start_y_goal_dentro_de_bounds(self, name: str, gen_and_spec: tuple) -> None:
        gen, spec = gen_and_spec
        p = gen(seed=42)
        bounds_arr = np.array(spec["bounds"], dtype=np.float64)
        for pt_arr in (p.start_arr, p.goal_arr):
            assert np.all(pt_arr >= bounds_arr[0])
            assert np.all(pt_arr <= bounds_arr[1])

    def test_n_waypoints_correcto(self, name: str, gen_and_spec: tuple) -> None:
        gen, spec = gen_and_spec
        p = gen(seed=42)
        assert p.n_waypoints == spec["n_waypoints"]

    def test_scenario_id_incluye_seed(self, name: str, gen_and_spec: tuple) -> None:
        gen, _ = gen_and_spec
        p = gen(seed=42)
        assert p.scenario_id == f"{name}_seed42"
