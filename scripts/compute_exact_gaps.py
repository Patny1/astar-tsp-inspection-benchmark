"""Gap al óptimo EXACTO sobre las instancias propias del benchmark.

TSPLIB sirve para argumentar que las implementaciones son correctas, pero es
simétrico y euclídeo, y las matrices de A* no: ``m[i,j] != m[j,i]`` en el 83-91 %
de los pares. Sobre una matriz simétrica, un delta de dos aristas de 2-opt que
ignore el sentido del segmento invertido es **algebraicamente idéntico** al
correcto: TSPLIB no puede detectar ese error, ni ningún otro propio de la
asimetría.

El óptimo exacto sí: se calcula sobre la misma matriz asimétrica que usan los
ordenadores, así que valida exactamente el camino de código que TSPLIB no toca.

Held-Karp llega a n = 16, o sea N = 5, 10 y 15 del benchmark. Más allá, los
óptimos salen de Concorde vía ``scripts/run_exact_optima.py`` y se leen con
``--optimos``: así no se resuelve nada acá y no hay cota de N.

Genera en ``<resultados>/analysis_exacto/<costo>/``:
    gaps.csv        gap por corrida
    resumen.csv     gap medio y tasa de óptimos por ordenador y N
    instancias.csv  por instancia, el mejor gap del catálogo
    gaps.png        la figura

Uso:
    uv run python scripts/compute_exact_gaps.py
    uv run python scripts/compute_exact_gaps.py --cost energy_aware --max-n 15
    # Fase 2, con los óptimos de Concorde:
    uv run python scripts/compute_exact_gaps.py --resultados results/fase2/canonico \\
        --optimos results/fase2/canonico/exact_optima.csv --cost energy_aware \\
        --energy-model gong_coherent --drone m210 --symmetric-pairs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bioinspired.algorithms.tsp.exact_tsp import ExactTSPSolver
from bioinspired.core.tsp_problem import InspectionTSPProblem
from bioinspired.environment.drone_model import DroneModel
from bioinspired.experiments.tsp_harness import add_model_args, cache_path

ORDEN = ["nn", "two_opt", "aco", "ga", "gwo", "pso"]
ETIQUETAS = {"nn": "NN", "two_opt": "2-opt", "pso": "PSO", "gwo": "GWO", "aco": "ACO", "ga": "GA"}
# Paleta validada para daltonismo (ver scripts/plot_local_optimality.py)
COLORES = {
    "nn": "#D55E00",
    "two_opt": "#999999",
    "aco": "#0072B2",
    "ga": "#009E73",
    "gwo": "#56B4E9",
    "pso": "#5D3A9B",
}
# El CSV guarda 4 decimales; por debajo de esto un gap es ruido de redondeo.
TOL_PCT = 1e-3


def _problema(cache: dict, cost_function: str) -> InspectionTSPProblem:
    """Arma el problema con la matriz que corresponde al costo optimizado.

    Se coloca la matriz elegida en ``distance_matrix`` para que ``tour_cost``
    —y con él el solver exacto— opere sobre ella sin ramificaciones.
    """
    if cost_function == "energy_aware":
        m, ini, fin = "energy_matrix", "depot_to_targets_energy", "targets_to_end_energy"
    else:
        m, ini, fin = "distance_matrix", "depot_to_targets", "targets_to_end"
    n = int(cache["n_targets"])
    return InspectionTSPProblem(
        targets_arr=np.zeros((n, 3)),
        distance_matrix=np.array(cache[m], dtype=float),
        depot_to_targets=np.array(cache[ini], dtype=float),
        targets_to_end=np.array(cache[fin], dtype=float),
        depot_arr=np.zeros(3),
        end_arr=np.zeros(3),
        scenario_id=str(cache["scenario"]),
        drone_model=DroneModel(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resultados", type=Path, default=Path("results/fase1/paper_v6"))
    ap.add_argument("--cache", type=Path, default=Path("results/fase1/cache/v6"))
    ap.add_argument("--cost", default="distance", choices=("distance", "energy_aware"))
    ap.add_argument("--max-n", type=int, default=15, help="cota de Held-Karp")
    ap.add_argument("--voxel", type=float, default=5.0, help="vóxel con que se construyó la matriz")
    ap.add_argument(
        "--optimos",
        type=Path,
        default=None,
        help="CSV de run_exact_optima.py: lee los óptimos en vez de resolverlos, sin cota de N",
    )
    add_model_args(ap, energy_default="legacy", drone_default="default", symmetric_default=False)
    args = ap.parse_args()

    df = pd.read_csv(args.resultados / "results.csv")
    df = df[df.cost_function == args.cost]
    if args.optimos is None:
        df = df[df.n_targets <= args.max_n]
    if df.empty:
        raise SystemExit(f"sin filas para cost_function={args.cost}")

    claves = ["scenario", "n_targets", "instance_seed"]
    celdas = df[claves].drop_duplicates()
    if args.optimos is not None:
        tabla = _leer_optimos(args.optimos, args.cost, celdas)
    else:
        tabla = _resolver_optimos(args, celdas)

    m = df.merge(tabla, on=claves)
    gap_crudo = (m.total_cost - m.optimo) / m.optimo * 100
    if args.optimos is not None:
        # Un ordenador no puede quedar por debajo del óptimo más allá del
        # redondeo (4 decimales del CSV + la cota de los pesos enteros). Si pasa,
        # el óptimo está mal, y el clip de abajo lo escondería.
        tol = (m.rounding_bound + 1e-4) / m.optimo * 100 + TOL_PCT
        debajo = m[gap_crudo < -tol]
        if not debajo.empty:
            raise SystemExit(
                f"{len(debajo)} corridas quedan POR DEBAJO del óptimo: el óptimo está mal.\n"
                + debajo[[*claves, "orderer", "total_cost", "optimo"]].head().to_string()
            )
    # El clip evita gaps de -2e-5 por el redondeo del CSV, que no son reales.
    m["gap_pct"] = gap_crudo.clip(lower=0.0)
    m["halla_optimo"] = m.gap_pct < TOL_PCT

    out = args.resultados / "analysis_exacto" / args.cost
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "gaps.csv", index=False)

    resumen = (
        m.groupby(["orderer", "n_targets"])
        .agg(
            gap_medio=("gap_pct", "mean"),
            gap_mediano=("gap_pct", "median"),
            gap_max=("gap_pct", "max"),
            halla_optimo_pct=("halla_optimo", lambda s: 100 * s.mean()),
            n=("gap_pct", "size"),
        )
        .reset_index()
    )
    resumen.to_csv(out / "resumen.csv", index=False)

    # Por instancia: ¿alguien del catálogo llegó al óptimo? Defiende el podio
    # mejor que el "mejor-o-empata" contra el propio catálogo.
    instancias = (
        m.groupby(claves)
        .agg(mejor_gap=("gap_pct", "min"), optimo=("optimo", "first"))
        .reset_index()
    )
    instancias["catalogo_halla_optimo"] = instancias.mejor_gap < TOL_PCT
    instancias.to_csv(out / "instancias.csv", index=False)

    print(f"\n=== GAP AL ÓPTIMO EXACTO — {args.cost} ===")
    print(
        resumen.pivot(index="orderer", columns="n_targets", values="gap_medio").round(2).to_string()
    )
    print("\n=== HALLA EL ÓPTIMO (%) ===")
    print(
        resumen.pivot(index="orderer", columns="n_targets", values="halla_optimo_pct")
        .round(1)
        .to_string()
    )
    print("\n=== INSTANCIAS DONDE EL CATÁLOGO ALCANZA EL ÓPTIMO (%) ===")
    print(
        instancias.groupby("n_targets").catalogo_halla_optimo.mean().mul(100).round(1).to_string()
    )

    _figura(resumen, args.cost, out / "gaps.png")
    print(f"\nSalida en {out}")


def _leer_optimos(ruta: Path, cost: str, celdas: pd.DataFrame) -> pd.DataFrame:
    """Óptimos persistidos por run_exact_optima.py; exige que estén todas las celdas."""
    opt = pd.read_csv(ruta)
    opt = opt[opt.cost_function == cost]
    claves = ["scenario", "n_targets", "instance_seed"]
    tabla = opt[[*claves, "opt_cost", "rounding_bound"]].rename(columns={"opt_cost": "optimo"})
    if tabla.duplicated(claves).any():
        raise SystemExit(f"{ruta}: hay instancias con más de un óptimo para {cost}")
    faltan = celdas.merge(tabla, on=claves, how="left").optimo.isna()
    if faltan.any():
        raise SystemExit(
            f"{ruta}: faltan óptimos para {int(faltan.sum())} de {len(celdas)} instancias "
            f"({cost}); terminar run_exact_optima.py primero"
        )
    print(f"Óptimos leídos de {ruta} ({len(celdas)} instancias).")
    return tabla


def _resolver_optimos(args: argparse.Namespace, celdas: pd.DataFrame) -> pd.DataFrame:
    """Held-Karp sobre las matrices cacheadas (n <= --max-n)."""
    solver = ExactTSPSolver(max_n=args.max_n)
    optimos: list[dict[str, float | int | str]] = []
    print(f"Resolviendo {len(celdas)} instancias con Held-Karp (n <= {args.max_n})...")
    for _, c in celdas.iterrows():
        # El nombre lo arma el harness. Construirlo a mano acá dejaba al script
        # ciego a los sufijos de modelo/aparato/simetría, así que no podía leer
        # los cachés de Fase 2 en adelante.
        f = cache_path(
            args.cache,
            "astar",
            str(c.scenario),
            int(c.n_targets),
            int(c.instance_seed),
            args.voxel,
            args.energy_model,
            args.drone,
            args.symmetric_pairs,
        )
        if not f.is_file():
            raise SystemExit(
                f"falta la matriz cacheada: {f}\n"
                "¿coinciden --energy-model / --drone / --symmetric-pairs / --voxel "
                "con los de la corrida que estás analizando?"
            )
        r = solver.solve(_problema(json.loads(f.read_text()), args.cost), seed=0)
        optimos.append(
            {
                "scenario": c.scenario,
                "n_targets": int(c.n_targets),
                "instance_seed": int(c.instance_seed),
                "optimo": r.best_cost,
                "rounding_bound": 0.0,
            }
        )
    return pd.DataFrame(optimos)


def _figura(resumen: pd.DataFrame, cost: str, destino: Path) -> None:
    """Dos paneles: cuán lejos del óptimo, y con qué frecuencia lo alcanzan."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    presentes = [o for o in ORDEN if o in resumen.orderer.values]

    for o in presentes:
        s = resumen[resumen.orderer == o].sort_values("n_targets")
        ax1.plot(
            s.n_targets, s.gap_medio, marker="o", ms=6, lw=1.9, color=COLORES[o], label=ETIQUETAS[o]
        )
        ax2.plot(s.n_targets, s.halla_optimo_pct, marker="o", ms=6, lw=1.9, color=COLORES[o])

    ax1.set_xlabel("Number of targets (N)", fontsize=10.5)
    ax1.set_ylabel("Mean gap to exact optimum (%)", fontsize=10.5)
    ax1.set_title("A · Distance to the optimum", fontsize=11, loc="left")
    ax1.legend(fontsize=8.5, framealpha=0.92)

    ax2.set_xlabel("Number of targets (N)", fontsize=10.5)
    ax2.set_ylabel("Runs reaching the optimum (%)", fontsize=10.5)
    ax2.set_title("B · How often it is reached", fontsize=11, loc="left")
    ax2.set_ylim(-3, 103)

    etiqueta = "distance" if cost == "distance" else "energy"
    for ax in (ax1, ax2):
        ax.set_xticks(sorted(resumen.n_targets.unique()))
        ax.grid(alpha=0.22, lw=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Optimality gap on the benchmark's own asymmetric instances — {etiqueta} cost",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(destino, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {destino.name}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
