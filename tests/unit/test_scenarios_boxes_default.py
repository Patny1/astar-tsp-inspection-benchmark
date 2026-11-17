"""Regresión: los escenarios de esferas existentes tienen cajas vacías por defecto."""

from bioinspired.environment.scenarios import (
    generate_forest_scenario,
    generate_tiny_scenario,
    generate_urban_scenario,
)


def test_existing_scenarios_have_empty_boxes() -> None:
    for gen in (generate_tiny_scenario, generate_urban_scenario, generate_forest_scenario):
        prob = gen(seed=1)
        assert prob.n_boxes == 0
        assert prob.boxes_arr.shape == (0, 6)
