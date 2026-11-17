"""Tests para el generador de escenarios."""

from __future__ import annotations

import numpy as np

from bioinspired.environment.scenarios import generate_tiny_scenario


class TestGenerateTinyScenario:
    def test_es_deterministico_con_mismo_seed(self) -> None:
        p1 = generate_tiny_scenario(seed=42)
        p2 = generate_tiny_scenario(seed=42)
        np.testing.assert_array_equal(p1.start_arr, p2.start_arr)
        np.testing.assert_array_equal(p1.goal_arr, p2.goal_arr)
        np.testing.assert_array_equal(p1.obstacles_arr, p2.obstacles_arr)

    def test_seeds_distintos_dan_obstaculos_distintos(self) -> None:
        p1 = generate_tiny_scenario(seed=1)
        p2 = generate_tiny_scenario(seed=2)
        # con alta probabilidad los obstáculos son distintos
        assert not np.allclose(p1.obstacles_arr, p2.obstacles_arr)

    def test_n_obstaculos_correcto(self) -> None:
        p = generate_tiny_scenario(seed=42)
        assert p.n_obstacles == 3

    def test_start_y_goal_dentro_de_bounds(self) -> None:
        p = generate_tiny_scenario(seed=42)
        assert np.all((p.start_arr >= p.bounds_arr[0]) & (p.start_arr <= p.bounds_arr[1]))
        assert np.all((p.goal_arr >= p.bounds_arr[0]) & (p.goal_arr <= p.bounds_arr[1]))

    def test_scenario_id_incluye_seed(self) -> None:
        p = generate_tiny_scenario(seed=42)
        assert "42" in p.scenario_id
