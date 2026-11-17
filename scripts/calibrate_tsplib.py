"""Calibración contra TSPLIB — gap al óptimo publicado por ordenador.

Responde la objeción de que los solvers rinden mal por estar mal implementados
o mal configurados: sobre instancias con óptimo conocido el gap lo verifica
cualquiera. Es material de apéndice, no del cuerpo del paper.

Uso:
    uv run python scripts/calibrate_tsplib.py
    uv run python scripts/calibrate_tsplib.py --instances berlin52 eil51 --seeds 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from bioinspired.experiments.tsp_registry import DETERMINISTIC_ORDERERS, make_orderers
from bioinspired.experiments.tsplib import (
    KNOWN_OPTIMA,
    load_tsplib_instance,
    optimality_gap_pct,
    tsplib_to_tsp_problem,
)

DATA = Path(__file__).parent.parent / "data" / "tsplib"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", nargs="+", default=sorted(KNOWN_OPTIMA))
    ap.add_argument("--seeds", type=int, default=5, help="algo_seeds por ordenador estocástico")
    ap.add_argument("--orderers", nargs="+", default=sorted(make_orderers()))
    ap.add_argument(
        "--output", type=Path, default=Path("results/fase1/tsplib/tsplib_calibration.csv")
    )
    args = ap.parse_args()

    catalogo = make_orderers()
    filas: list[dict] = []

    for nombre in args.instances:
        path = DATA / f"{nombre}.tsp"
        if not path.is_file():
            print(f"[skip] {nombre}: falta {path}")
            continue
        inst = load_tsplib_instance(path)
        problem = tsplib_to_tsp_problem(inst)
        opt = KNOWN_OPTIMA[nombre]
        print(f"\n=== {nombre}  (n={inst.n_cities}, óptimo={opt}) ===", flush=True)

        for ord_name in args.orderers:
            seeds = [0] if ord_name in DETERMINISTIC_ORDERERS else range(args.seeds)
            costos = []
            t0 = time.perf_counter()
            for s in seeds:
                costos.append(catalogo[ord_name]().solve(problem, seed=s).best_cost)
            dt = time.perf_counter() - t0
            mejor, medio = float(np.min(costos)), float(np.mean(costos))
            filas.append(
                {
                    "instance": nombre,
                    "n_cities": inst.n_cities,
                    "optimum": opt,
                    "orderer": ord_name,
                    "best_cost": mejor,
                    "mean_cost": medio,
                    "gap_best_pct": optimality_gap_pct(mejor, nombre),
                    "gap_mean_pct": optimality_gap_pct(medio, nombre),
                    "n_seeds": len(list(seeds)),
                    "cpu_s": dt,
                }
            )
            print(
                f"  {ord_name:15s} mejor={mejor:10.1f} (gap {filas[-1]['gap_best_pct']:6.2f}%)"
                f"   medio={medio:10.1f} (gap {filas[-1]['gap_mean_pct']:6.2f}%)   {dt:5.1f}s",
                flush=True,
            )

    df = pd.DataFrame(filas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(
        f"\n\n=== GAP MEDIO AL ÓPTIMO, por ordenador (sobre {df.instance.nunique()} instancias) ==="
    )
    resumen = (
        df.groupby("orderer")[["gap_best_pct", "gap_mean_pct"]].mean().sort_values("gap_mean_pct")
    )
    print(resumen.round(2).to_string())
    print(f"\nGuardado en {args.output}")


if __name__ == "__main__":
    main()
