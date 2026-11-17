"""Figura de apéndice: gap al óptimo publicado (TSPLIB) por ordenador.

El alcance depende del manuscrito, y no son intercambiables:

* **Paper 1** son los seis algoritmos sin hibridar — las dos heurísticas
  clásicas (NN, 2-opt) contra las cuatro bioinspiradas (ACO, GA, PSO, GWO).
* **Paper 2** agrega meméticos e híbridos, que son su contribución.

Mostrar los catorce en el apéndice del paper 1 adelantaría resultados que
pertenecen al segundo manuscrito, así que la figura se genera por separado.

Uso:
    uv run python scripts/plot_tsplib_calibration.py --paper 1
    uv run python scripts/plot_tsplib_calibration.py --paper 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DATA = Path("results/fase1/tsplib/tsplib_calibration.csv")
OUT = {
    1: Path("results/fase1/tsplib/tsplib_calibration_paper1.png"),
    2: Path("results/fase1/tsplib/tsplib_calibration_paper2.png"),
}

# Paper 1: clasicas contra bioinspiradas, sin hibridar.
PAPER1 = ["nn", "two_opt", "pso", "gwo", "aco", "ga"]

# Etiquetas limpias para el paper
LABELS: dict[str, str] = {
    "gwo_2opt": "GWO+2-opt",
    "pso_2opt": "PSO+2-opt",
    "two_opt": "2-opt",
    "aco_2opt": "ACO+2-opt",
    "ga_2opt": "GA+2-opt",
    "aco": "ACO",
    "two_opt_random": "2-opt-R",
    "ga_aco": "GA+ACO",
    "pso_aco": "PSO+ACO",
    "nn": "NN",
    "ga": "GA",
    "ga_pso": "GA+PSO",
    "gwo": "GWO",
    "pso": "PSO",
}

# Categorías: memético, puro, baseline
MEMETICO = {
    "gwo_2opt",
    "pso_2opt",
    "aco_2opt",
    "ga_2opt",
    "ga_aco",
    "pso_aco",
    "ga_pso",
    "two_opt_random",
}
PURO = {"aco", "ga", "gwo", "pso"}
BASELINE = {"nn", "two_opt"}

COLOR = {"memetico": "#2a6ebb", "puro": "#e07b3a", "baseline": "#888888"}


def cat_color(o: str) -> str:
    if o in MEMETICO:
        return COLOR["memetico"]
    if o in PURO:
        return COLOR["puro"]
    return COLOR["baseline"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", type=int, choices=(1, 2), default=1)
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    if args.paper == 1:
        df = df[df["orderer"].isin(PAPER1)]

    # Gap medio sobre las 6 instancias
    mean_gap = df.groupby("orderer")["gap_mean_pct"].mean().sort_values()

    # Gap por instancia (para líneas de dispersión)
    gap_by_inst = df.pivot_table(index="orderer", columns="instance", values="gap_mean_pct")
    gap_by_inst = gap_by_inst.loc[mean_gap.index]  # mismo orden

    orderers = mean_gap.index.tolist()
    x = np.arange(len(orderers))
    colors = [cat_color(o) for o in orderers]

    fig, ax = plt.subplots(figsize=(11 if args.paper == 2 else 7.2, 5))

    bars = ax.bar(x, mean_gap.values, color=colors, alpha=0.88, zorder=3, width=0.6)

    # Puntos por instancia (dispersión)
    for i, o in enumerate(orderers):
        vals = gap_by_inst.loc[o].dropna().values
        ax.scatter(
            [i] * len(vals),
            vals,
            color=cat_color(o),
            s=22,
            zorder=4,
            alpha=0.55,
            linewidths=0,
        )

    # Anotaciones de valor sobre barras que caben
    for bar, val in zip(bars, mean_gap.values, strict=False):
        if val < 15:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.3,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222",
            )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_ylim(0.5, 600)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[o] for o in orderers], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Gap to published optimum (%, log scale)", fontsize=10)
    ax.set_xlabel("Algorithm", fontsize=10)
    alcance = (
        "classical vs bioinspired" if args.paper == 1 else "including memetic and hybrid orderers"
    )
    ax.set_title(
        f"TSPLIB calibration — mean gap to optimum, {alcance} (6 instances, N = 51-150)",
        fontsize=11,
        pad=10,
    )
    ax.axhline(10, color="#aaa", lw=0.8, ls="--", zorder=2)
    ax.grid(axis="y", which="both", lw=0.4, alpha=0.4, zorder=1)
    ax.set_axisbelow(True)

    # Leyenda de categorías
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=COLOR["puro"], alpha=0.88, label="Bioinspired metaheuristics"),
        Patch(facecolor=COLOR["baseline"], alpha=0.88, label="Classical heuristics (NN, 2-opt)"),
    ]
    if args.paper == 2:
        legend_handles.insert(
            0,
            Patch(facecolor=COLOR["memetico"], alpha=0.88, label="Memetic and hybrid"),
        )
    ax.legend(handles=legend_handles, fontsize=8.5, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    destino = OUT[args.paper]
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print(f"Guardado: {destino}  ({len(orderers)} ordenadores)")


if __name__ == "__main__":
    main()
