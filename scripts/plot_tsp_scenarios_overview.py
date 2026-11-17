"""Overview figure of the 3 TSP scenarios (tiny, urban, forest), no tours.

Shows depot (start), end (goal) and inspection targets over the obstacle
field for each scenario — used as a paper figure.

Usage
-----
python scripts/plot_tsp_scenarios_overview.py [--output results/fase1/figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from bioinspired.environment.scenarios import (
    generate_forest_tsp_scenario,
    generate_tiny_tsp_scenario,
    generate_urban_tsp_scenario,
)

OBSTACLE_COLORS = {"tiny": "#e07b7b", "urban": "#e0a07b", "forest": "#6bbb6b"}
OBSTACLE_ALPHA = 0.4
# Estilo de los marcadores de la Fig. 1 del paper 1. Fuente unica: las demas figuras
# del paper (p. ej. scripts/plot_paper1_best_route.py) los importan de aqui. Los
# tamanos ``s`` son los de esta figura (18 x 6 in); otra figura los escala.
TARGET_STYLE = {"color": "royalblue", "marker": "o", "s": 40, "edgecolors": "black"}
START_STYLE = {"color": "limegreen", "marker": "^", "s": 160, "edgecolors": "black"}
END_STYLE = {"color": "red", "marker": "s", "s": 160, "edgecolors": "black"}
MARKER_LABELS = {"targets": "Targets", "start": "Start", "end": "End"}
SCENARIO_LABELS = {"tiny": "Tiny", "urban": "Urban", "forest": "Forest"}
SCENARIO_FACTORIES = {
    "tiny": generate_tiny_tsp_scenario,
    "urban": generate_urban_tsp_scenario,
    "forest": generate_forest_tsp_scenario,
}
N_TARGETS = {"tiny": 10, "urban": 20, "forest": 30}
SEED = 42


def _draw_scenario(ax: Axes3D, scen_name: str) -> None:
    problem = SCENARIO_FACTORIES[scen_name](seed=SEED, n_targets=N_TARGETS[scen_name])
    obs_color = OBSTACLE_COLORS[scen_name]

    for cx, cy, cz, r in problem.obstacles_arr:
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 8)
        ax.plot_surface(
            cx + r * np.outer(np.cos(u), np.sin(v)),
            cy + r * np.outer(np.sin(u), np.sin(v)),
            cz + r * np.outer(np.ones_like(u), np.cos(v)),
            color=obs_color,
            alpha=OBSTACLE_ALPHA,
            linewidth=0,
        )

    targets = problem.inspection_targets_arr
    assert targets is not None
    ax.scatter(*targets.T, **TARGET_STYLE, label=MARKER_LABELS["targets"])

    ax.scatter(*problem.start_arr, **START_STYLE, label=MARKER_LABELS["start"])
    ax.scatter(*problem.goal_arr, **END_STYLE, label=MARKER_LABELS["end"])

    ax.set_xlim(problem.bounds_arr[0, 0], problem.bounds_arr[1, 0])
    ax.set_ylim(problem.bounds_arr[0, 1], problem.bounds_arr[1, 1])
    ax.set_zlim(problem.bounds_arr[0, 2], problem.bounds_arr[1, 2])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{SCENARIO_LABELS[scen_name]} (N={len(targets)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/fase1/figures"))
    args = parser.parse_args()

    fig = plt.figure(figsize=(18, 6))
    handles_labels = None
    for i, scen_name in enumerate(["tiny", "urban", "forest"], start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        _draw_scenario(ax, scen_name)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    fig.suptitle(f"TSP inspection scenarios (seed={SEED})")
    assert handles_labels is not None
    fig.legend(*handles_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "00_scenarios_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
