"""Figura de la mejor ruta para el paper 1: varios algoritmos sobre una misma instancia.

Un panel por algoritmo, en vista superior (XY), de una instancia de ``paper_v6``
con objetivo de distancia. Por defecto ACO, 2-opt y PSO: primer, segundo y ultimo
puesto del paper. Con ``--algos aco,pso`` sale la version de dos paneles. Los
tramos son los caminos A* reales entre objetivos, reconstruidos como los arma
``DistanceMatrixBuilder`` (un A* por par ordenado, voxel de 5 m), no rectas.

``paper_v6`` tiene una sola corrida por algoritmo e instancia: la ruta dibujada
es la que ese algoritmo devolvio (su mejor ruta al agotar el presupuesto).

Nada se re-optimiza: el orden de visita y el costo se leen de ``results.csv``. El
script aborta si la longitud de los caminos dibujados no reproduce ese costo.

Sin ``--seed`` elige la instancia sola, con una regla declarada y no a ojo: entre
las semillas cuyas rutas dibujadas no tocan ningun obstaculo, la que tiene el cociente
PSO/ACO mas cercano a la mediana del grupo (escenario, N). Es decir, una instancia
representativa, no la mas favorable.

El estilo no se define aqui: marcadores, leyenda y color de obstaculos salen de
``plot_tsp_scenarios_overview.py`` (Fig. 1) y el color de cada algoritmo de
``analyze_tsp_paper.py`` (Fig. 2), para que las figuras del paper no diverjan.

Uso::

    uv run python scripts/plot_paper1_best_route.py                       # urban, N=30
    uv run python scripts/plot_paper1_best_route.py --scenario forest
    uv run python scripts/plot_paper1_best_route.py --scenario urban --seed 7
    uv run python scripts/plot_paper1_best_route.py --algos aco,pso        # dos paneles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).parent))
from analyze_tsp_paper import COLORS, LABELS
from plot_tsp_scenarios_overview import (
    END_STYLE,
    MARKER_LABELS,
    OBSTACLE_ALPHA,
    OBSTACLE_COLORS,
    START_STYLE,
    TARGET_STYLE,
)

from bioinspired.algorithms.soo._shared import VoxelGrid
from bioinspired.algorithms.soo.astar import AStarPlanner
from bioinspired.core.trajectory import Trajectory
from bioinspired.experiments.tsp_registry import SCENARIO_FACTORIES
from bioinspired.metrics.path import min_clearance

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "results" / "fase1" / "paper_v6" / "results.csv"
SALIDA = RAIZ / "results" / "fase1" / "figures"
VOXEL_M = 5.0
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}
ANCHO_LNCS_IN = 4.8  # ancho de texto de LNCS (12.2 cm): la figura sale a tamano real
# Los tamanos de la Fig. 1 son para 18 x 6 in; aqui se reducen en area por este factor
# (por ``k`` ademas), conservando la proporcion entre objetivos y despegue/aterrizaje.
ESCALA_MARCAS = 0.55
# Figura del paper a tamano real (--ancho): trazo fino como el de la Fig. 2, que entra
# reducida al ~36 % (lineas de ~0.5 pt, ejes de ~0.3 pt). Areas en pt^2.
RUTA_REAL_PT = 0.5
EJES_REAL_PT = 0.4
BORDE_REAL_PT = 0.35
AREA_OBJETIVO_REAL = 8.0
AREA_EXTREMO_REAL = 20.0  # despegue y aterrizaje


def estilo(base: dict[str, Any], k: float, real: bool = False) -> dict[str, Any]:
    """Estilo de marcador de la Fig. 1 con el area y el borde escalados a esta figura."""
    if real:
        area_real = AREA_OBJETIVO_REAL if base is TARGET_STYLE else AREA_EXTREMO_REAL
        return {**base, "s": area_real, "linewidths": BORDE_REAL_PT}
    area = ESCALA_MARCAS * k
    return {**base, "s": base["s"] * area, "linewidths": 1.5 * area**0.5}


class Caminos:
    """Caminos A* de una instancia, como los construye la matriz de Fase 1."""

    def __init__(self, scenario: str, n: int, seed: int) -> None:
        self.scenario = scenario
        self.problema = SCENARIO_FACTORIES[scenario](seed, n)
        self.planner = AStarPlanner(voxel_size_m=VOXEL_M)
        self.grid = VoxelGrid.from_problem(self.problema, VOXEL_M)
        self.libre = self.grid.free_voxel_mask(self.problema.obstacles_arr, self.problema.boxes_arr)

    def tramo(self, inicio_arr: np.ndarray, fin_arr: np.ndarray) -> np.ndarray:
        """Polilinea A* de ``inicio`` a ``fin`` con los extremos en su posicion exacta."""
        a = self.grid.world_to_voxel(inicio_arr.astype(np.float64))
        b = self.grid.world_to_voxel(fin_arr.astype(np.float64))
        camino_arr = self.grid.voxels_to_world(self.planner._astar_search(a, b, self.libre))
        if camino_arr.shape[0] < 2:
            return np.vstack([inicio_arr, fin_arr]).astype(np.float64)
        camino_arr[0] = inicio_arr
        camino_arr[-1] = fin_arr
        return camino_arr

    def ruta(self, orden: list[int]) -> np.ndarray:
        """Polilinea completa despegue -> objetivos en ``orden`` -> aterrizaje."""
        objetivos_arr = self.problema.inspection_targets_arr
        assert objetivos_arr is not None
        puntos = [
            self.problema.start_arr,
            *[objetivos_arr[i] for i in orden],
            self.problema.goal_arr,
        ]
        tramos = [self.tramo(puntos[k], puntos[k + 1]) for k in range(len(puntos) - 1)]
        return np.vstack([tramos[0], *[t[1:] for t in tramos[1:]]])


def longitud(camino_arr: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(camino_arr, axis=0), axis=1).sum())


def holgura(camino_arr: np.ndarray, caminos: Caminos) -> float:
    """Distancia minima de la polilinea a un obstaculo (negativa si lo atraviesa)."""
    return float(
        min_clearance(
            Trajectory(waypoints_arr=camino_arr),
            caminos.problema.obstacles_arr,
            caminos.problema.boxes_arr,
        )
    )


def filas(df: pd.DataFrame, scenario: str, n: int) -> pd.DataFrame:
    """Una fila por semilla con el costo y el orden de cada algoritmo (distancia)."""
    sub = df[(df.scenario == scenario) & (df.n_targets == n) & (df.cost_function == "distance")]
    ancho = sub.pivot(index="instance_seed", columns="orderer", values=["total_cost", "tour_order"])
    ancho.columns = [f"{b}_{'costo' if a == 'total_cost' else 'orden'}" for a, b in ancho.columns]
    ancho["cociente"] = ancho.pso_costo.astype(float) / ancho.aco_costo.astype(float)
    return ancho


def elegir_semilla(tabla: pd.DataFrame, scenario: str, n: int, algos: list[str]) -> int:
    """Semilla sin colisiones con el cociente PSO/ACO mas cercano a la mediana."""
    mediana = float(tabla.cociente.median())
    for seed in (tabla.cociente - mediana).abs().sort_values().index:
        caminos = Caminos(scenario, n, int(seed))
        libres = all(
            holgura(caminos.ruta(json.loads(tabla.loc[seed, f"{algo}_orden"])), caminos) > 0.0
            for algo in algos
        )
        if libres:
            return int(seed)
    raise SystemExit("ninguna semilla tiene todas las rutas libres de colision")


def dibujar(
    ax: Axes,
    caminos: Caminos,
    camino_arr: np.ndarray,
    algo: str,
    titulo: str,
    k: float,
    real: bool = False,
) -> None:
    """Dibuja un panel; ``k`` escala marcas y letras (1.0 = figura de dos paneles)."""
    problema = caminos.problema
    color_obstaculo = OBSTACLE_COLORS[caminos.scenario]
    for cx, cy, _cz, r in problema.obstacles_arr:
        ax.add_patch(
            Circle((cx, cy), r, facecolor=color_obstaculo, edgecolor="none", alpha=OBSTACLE_ALPHA)
        )
    trazo = RUTA_REAL_PT if real else 1.6 * k
    ax.plot(camino_arr[:, 0], camino_arr[:, 1], color=COLORS[algo], lw=trazo, zorder=3)
    objetivos_arr = problema.inspection_targets_arr
    assert objetivos_arr is not None
    ax.scatter(
        objetivos_arr[:, 0],
        objetivos_arr[:, 1],
        **estilo(TARGET_STYLE, k, real),
        zorder=4,
        label=MARKER_LABELS["targets"],
    )
    ax.scatter(
        *problema.start_arr[:2],
        **estilo(START_STYLE, k, real),
        zorder=5,
        label=MARKER_LABELS["start"],
    )
    ax.scatter(
        *problema.goal_arr[:2], **estilo(END_STYLE, k, real), zorder=5, label=MARKER_LABELS["end"]
    )
    lo, hi = problema.bounds_arr[0], problema.bounds_arr[1]
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title(titulo, fontsize=4 + 6 * k)
    ax.set_xlabel("X (m)", fontsize=4 + 6 * k)
    ax.tick_params(labelsize=4 + 4 * k)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--scenario", default="urban", choices=["tiny", "urban", "forest"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument(
        "--seed", type=int, default=None, help="semilla de instancia (por defecto, la regla)"
    )
    ap.add_argument(
        "--algos",
        default="aco,two_opt,pso",
        help="algoritmos a dibujar, separados por coma (un panel cada uno)",
    )
    ap.add_argument(
        "--ancho",
        type=float,
        default=None,
        help="ancho (pulgadas) con que la figura entra al paper: se dibuja a ese tamano real",
    )
    ap.add_argument(
        "--leyenda",
        choices=["abajo", "derecha"],
        default="abajo",
        help="con --ancho: leyenda en fila bajo los ejes o en columna a la derecha (menos alto)",
    )
    ap.add_argument("--output", type=Path, default=SALIDA)
    args = ap.parse_args()
    algos = args.algos.split(",")

    tabla = filas(pd.read_csv(RESULTADOS), args.scenario, args.n)
    seed = (
        args.seed if args.seed is not None else elegir_semilla(tabla, args.scenario, args.n, algos)
    )
    caminos = Caminos(args.scenario, args.n, seed)
    costos = tabla.loc[seed, [c for c in tabla.columns if c.endswith("_costo")]].astype(float)
    puesto = costos.rank(method="min").astype(int)
    mejor = float(costos.min())

    # Dos paneles: el formato original. Tres o mas: ancho de texto de LNCS, tamano real.
    # Con --ancho: tamano real para ese ancho, para incluirla en LaTeX sin escalar (las
    # letras quedan en el tamano en que se dibujan).
    real = args.ancho is not None
    dos = len(algos) == 2 and not real
    k = 1.0 if dos else 0.55
    if dos:
        tamano = (7.0, 3.7)
    elif real:
        tamano = (args.ancho, args.ancho * 0.64)
    else:
        tamano = (ANCHO_LNCS_IN, 2.25)
    if real:
        # Ejes, marcas y marco de la leyenda finos, en proporcion con la letra.
        plt.rcParams.update(
            {
                "axes.linewidth": EJES_REAL_PT,
                "xtick.major.width": EJES_REAL_PT,
                "ytick.major.width": EJES_REAL_PT,
                "xtick.major.size": 2.0,
                "ytick.major.size": 2.0,
                "patch.linewidth": EJES_REAL_PT,
                # Fuentes TrueType incrustadas, no Type 3: la imprenta puede rechazarlas.
                "pdf.fonttype": 42,
            }
        )
    fig, ejes = plt.subplots(1, len(algos), figsize=tamano, sharey=True)
    for ax, algo in zip(ejes, algos, strict=True):
        costo = float(costos[f"{algo}_costo"])
        camino_arr = caminos.ruta(json.loads(tabla.loc[seed, f"{algo}_orden"]))
        if abs(longitud(camino_arr) - costo) > 0.01:
            raise SystemExit(
                f"{algo}: los caminos dibujados miden {longitud(camino_arr):.2f} m y "
                f"results.csv dice {costo:.2f} m"
            )
        lugar = int(puesto[f"{algo}_costo"])
        extra = "shortest route" if lugar == 1 else f"{100 * (costo / mejor - 1):.0f}% longer"
        titulo = f"{LABELS[algo]} ({ORDINAL[lugar]} of {len(costos)})\n{costo:,.0f} m, {extra}"
        dibujar(ax, caminos, camino_arr, algo, titulo, k, real)
        if real:
            # Paneles chicos: matplotlib dejaria 3 marcas en X y 5 en Y; se fijan 5 en ambos.
            lo, hi = caminos.problema.bounds_arr[0], caminos.problema.bounds_arr[1]
            ax.set_xticks(np.linspace(lo[0], hi[0], 5))
            ax.set_yticks(np.linspace(lo[1], hi[1], 5))
        print(
            f"{args.scenario} N={args.n} semilla {seed} {algo}: {costo:.1f} m, puesto {lugar}, "
            f"holgura minima {holgura(camino_arr, caminos):.2f} m"
        )
    ejes[0].set_ylabel("Y (m)", fontsize=4 + 6 * k)
    # Misma leyenda que la Fig. 1 (Targets, Start, End, en ese orden y con marco), mas el
    # obstaculo, que en la vista superior no se reconoce sin rotulo. Los obstaculos de estos
    # escenarios son esferas: en la leyenda van como circulo (un rectangulo seria de cajas).
    marcas, etiquetas = ejes[0].get_legend_handles_labels()
    marcas.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=estilo(START_STYLE, k, real)["s"] ** 0.5,
            markerfacecolor=OBSTACLE_COLORS[args.scenario],
            markeredgecolor="none",
            alpha=OBSTACLE_ALPHA,
        )
    )
    etiquetas.append("Obstacle")
    orden = [
        etiquetas.index(e)
        for e in (
            MARKER_LABELS["targets"],
            MARKER_LABELS["start"],
            MARKER_LABELS["end"],
            "Obstacle",
        )
    ]
    leyenda = {
        "ncol": 4,
        "fontsize": 4 + 4 * k,
        "handletextpad": 0.4,
        "columnspacing": 1.2,
    }
    if real:
        # Figura del paper (--ancho): la leyenda va pegada a los ejes (debajo de las
        # etiquetas o a la derecha del ultimo panel); con paneles de aspecto igual sobra
        # alto y ``bbox_inches="tight"`` lo recorta al guardar. ``--ancho`` es el ancho
        # de los paneles: la leyenda a la derecha se suma a el.
        # Mas separacion entre paneles: en anchos chicos los titulos de dos lineas se tocan.
        fig.tight_layout(w_pad=1.8)
        a_figura = fig.transFigure.inverted()
        renderer = fig.canvas.get_renderer()
        if args.leyenda == "derecha":
            caja = ejes[-1].get_position()
            ancla = (caja.x1 + 0.03, (caja.y0 + caja.y1) / 2)
            lugar = "center left"
            leyenda["ncol"] = 1
        else:
            piso = min(ax.get_tightbbox(renderer).transformed(a_figura).y0 for ax in ejes)
            ancla = ((ejes[0].get_position().x0 + ejes[-1].get_position().x1) / 2, piso - 0.01)
            lugar = "upper center"
        fig.legend(
            [marcas[i] for i in orden],
            [etiquetas[i] for i in orden],
            loc=lugar,
            bbox_to_anchor=ancla,
            **leyenda,
        )
    else:
        # Los demas formatos quedan como estaban: leyenda al pie de la figura.
        fig.legend(
            [marcas[i] for i in orden], [etiquetas[i] for i in orden], loc="lower center", **leyenda
        )
        fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)

    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"cociente PSO/ACO {tabla.loc[seed, 'cociente']:.3f}, "
        f"mediana del grupo {tabla.cociente.median():.3f} ({len(tabla)} semillas)"
    )
    for ext in ("png", "pdf"):
        sufijo = "" if algos == ["aco", "pso"] else "_" + "_".join(algos)
        sufijo += f"_{args.ancho:g}in" if real else ""
        sufijo += "_leyenda_derecha" if real and args.leyenda == "derecha" else ""
        destino = (
            args.output / f"paper1_best_route_{args.scenario}_n{args.n}_seed{seed}{sufijo}.{ext}"
        )
        fig.savefig(destino, dpi=300, bbox_inches="tight")
        print(f"guardado: {destino}")


if __name__ == "__main__":
    main()
