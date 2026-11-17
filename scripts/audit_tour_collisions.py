"""Auditoría de colisiones residuales de las mejores rutas de un benchmark TSP de Fase 1.

Reconstruye con A* el camino 3-D de cada tour de ``instance_seed = 0`` bajo el costo
``distance`` (3 escenarios x N x 6 ordenadores = 90 rutas en Fase 1) y cuenta cuántas
cruzan un obstáculo con el segmento continuo entre dos centros de vóxel libres. Es la
cifra de §2.7, §3.1 y §4.3 del paper 1.

Límite conocido: A* sólo fija el LARGO del camino. Entre caminos igual de cortos el
desempate depende de la implementación, así que la cifra puede cambiar con el código
aunque la matriz sea idéntica.

Uso:
    uv run python scripts/audit_tour_collisions.py --dataset results/fase1/paper_v6 --voxel 5
    uv run python scripts/audit_tour_collisions.py --dataset results/fase1/paper_v10 --voxel 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_tsp_paper import _compute_tour_path

from bioinspired.algorithms.soo._shared import VoxelGrid
from bioinspired.algorithms.soo.astar import AStarPlanner
from bioinspired.core.trajectory import Trajectory
from bioinspired.experiments.tsp_harness import SCENARIO_FACTORIES
from bioinspired.metrics.path import count_collisions

ORDERERS = ["nn", "two_opt", "pso", "gwo", "aco", "ga"]
SCENARIOS = ["tiny", "urban", "forest"]


def auditar(dataset: Path, voxel_m: float, cost_function: str = "distance") -> pd.DataFrame:
    """Una fila por ruta auditada, con el total de colisiones de su camino reconstruido."""
    df = pd.read_csv(dataset / "results.csv")
    d = df[(df.cost_function == cost_function) & (df.instance_seed == 0)]
    filas = []
    for sc in SCENARIOS:
        for n in sorted(d.n_targets.unique()):
            problem = SCENARIO_FACTORIES[sc](seed=0, n_targets=int(n))
            obs = problem.obstacles_arr
            boxes = getattr(problem, "boxes_arr", None)
            grid = VoxelGrid.from_problem(problem, voxel_m)
            free = grid.free_voxel_mask(obs, boxes)
            planner = AStarPlanner(voxel_size_m=voxel_m)
            seg_cache: dict[tuple[int, int], np.ndarray] = {}
            for o in ORDERERS:
                row = d[(d.scenario == sc) & (d.n_targets == n) & (d.orderer == o)].iloc[0]
                path_arr = _compute_tour_path(
                    grid,
                    free,
                    planner,
                    problem.start_arr,
                    problem.inspection_targets_arr,
                    problem.goal_arr,
                    json.loads(row["tour_order"]),
                    seg_cache,
                )
                coll = count_collisions(Trajectory(waypoints_arr=path_arr), obs, boxes)
                filas.append(
                    {"scenario": sc, "n_targets": int(n), "orderer": o, "collisions": coll["total"]}
                )
    return pd.DataFrame(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True, help="carpeta con results.csv")
    ap.add_argument("--voxel", type=float, required=True, help="tamaño de vóxel del dataset (m)")
    ap.add_argument("--output", type=Path, help="CSV opcional con el detalle por ruta")
    args = ap.parse_args()

    res = auditar(args.dataset, args.voxel)
    malas = res[res.collisions > 0]
    print(
        f"{args.dataset.name} (vóxel {args.voxel:g} m): {len(malas)}/{len(res)} rutas con "
        f"colisiones ({len(malas) / len(res) * 100:.1f} %)"
    )
    print("por escenario:", malas.scenario.value_counts().to_dict())
    for _, r in malas.iterrows():
        print(f"  {r.scenario:6s} N={r.n_targets:2d} {r.orderer:8s} colisiones={r.collisions}")
    if args.output:
        res.to_csv(args.output, index=False)
        print(f"detalle -> {args.output}")


if __name__ == "__main__":
    main()
