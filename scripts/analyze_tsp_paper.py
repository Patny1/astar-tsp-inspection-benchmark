"""Analisis estadistico del benchmark TSP paper (TICEC 2026).

Lee results.csv y convergence.csv generados por run_tsp_paper.py y produce:
  1. Tabla medias+-std por ordenador y tamano de instancia
  2. Test de Friedman + post-hoc Holm
  3. Wilcoxon signed-rank pareado (pares clave)
  4. Critical Difference diagram
  5. Curvas costo y tiempo vs n_targets
  6. Curvas de convergencia PSO vs GWO por escenario x n_targets
  7. Visualizacion de rutas 2D por escenario

Uso:
    uv run python scripts/analyze_tsp_paper.py
    uv run python scripts/analyze_tsp_paper.py --data results/fase1/paper_v10 --cost energy_aware
    uv run python scripts/analyze_tsp_paper.py --generator astar --cost distance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ALPHA = 0.05


def estable(x: float) -> float:
    """Recorta a 10 cifras significativas, para que el CSV no dependa de la maquina.

    Los p-valores del Friedman y del post-hoc varian entre plataformas en el
    digito ~16: la misma corrida da 1.8576026028265948e-37 en una y
    ...952e-37 en otra. Numericamente da igual, pero `make_figures.py --check`
    compara el sha256 del archivo: sin el recorte, un manifiesto generado en otra
    plataforma marcaria CAMBIO para siempre.

    Diez cifras dejan seis de margen sobre el drift observado y no pierden nada:
    nadie necesita el digito 16 de un p-valor. No se usa `round()` porque estos
    valores son de orden 1e-37 a 1e-82 y redondear por decimales los aplasta a 0.
    """
    return float(f"{x:.10g}")


def salida(out_dir: Path, nombre: str, cost_fn: str) -> Path:
    """Ruta de salida, con sufijo ``_e`` cuando el costo es ``energy_aware``.

    Las dos funciones de costo producen exactamente los mismos nombres de
    archivo, asi que al copiarlas juntas a la carpeta del manuscrito se pisan.
    El sufijo las desambigua en el origen; renombrar a mano no sirve porque cada
    regeneracion del analisis vuelve a escribir el nombre sin sufijo.
    """
    if cost_fn == "energy_aware":
        ruta = Path(nombre)
        nombre = f"{ruta.stem}_e{ruta.suffix}"
    return out_dir / nombre


# Los SEIS de paper 1. Este script es el analisis de Fase 1, y su lista estaba
# clavada: corrido sobre el dataset de Fase 2 descartaba en silencio los ocho
# hibridos y memeticos — o sea, justo los algoritmos sobre los que trata ese
# paper. `--orderers` permite pedirlos; el default no cambia para no alterar la
# reproduccion de paper 1.
ORDERERS = ["nn", "two_opt", "pso", "gwo", "aco", "ga"]
LABELS = {
    "nn": "NN",
    "two_opt": "2-opt",
    "pso": "PSO",
    "gwo": "GWO",
    "aco": "ACO",
    "ga": "GA",
    "two_opt_random": "2-opt-R",
    "ga_2opt": "GA+2-opt",
    "pso_2opt": "PSO+2-opt",
    "gwo_2opt": "GWO+2-opt",
    "aco_2opt": "ACO+2-opt",
    "ga_aco": "GA+ACO",
    "pso_aco": "PSO+ACO",
    "ga_pso": "GA+PSO",
}
COLORS = {
    "nn": "#e6194b",
    "two_opt": "#f58231",
    "pso": "#3cb44b",
    "gwo": "#4363d8",
    "aco": "#911eb4",
    "ga": "#42d4f4",
    "two_opt_random": "#d35400",
    "ga_2opt": "#1abc9c",
    "pso_2opt": "#2ecc71",
    "gwo_2opt": "#3498db",
    "aco_2opt": "#9b59b6",
    "ga_aco": "#f39c12",
    "pso_aco": "#e74c3c",
    "ga_pso": "#95a5a6",
}
MARKERS = {
    "nn": "s",
    "two_opt": "D",
    "pso": "o",
    "gwo": "^",
    "aco": "P",
    "ga": "X",
    "two_opt_random": "d",
    "ga_2opt": "v",
    "pso_2opt": "<",
    "gwo_2opt": ">",
    "aco_2opt": "p",
    "ga_aco": "h",
    "pso_aco": "*",
    "ga_pso": "8",
}
SCENARIO_NAMES = {
    "tiny": "Tiny",
    "urban": "Urban",
    "forest": "Forest",
    "building": "Building",
    "solar_farm": "Solar Farm",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    }
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _load(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path)


def _filter(df: pd.DataFrame, generator: str, cost_fn: str) -> pd.DataFrame:
    sub = df[(df["generator"] == generator) & (df["cost_function"] == cost_fn)].copy()
    if sub.empty:
        raise ValueError(f"No data for generator={generator}, cost_function={cost_fn}")
    return sub


# ── 1. Tabla resumen ──────────────────────────────────────────────────────────


def table_summary(df: pd.DataFrame, out_dir: Path, cost_fn: str) -> pd.DataFrame:
    agg = (
        df.groupby(["orderer", "n_targets"])["total_cost"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(2)
        .reset_index()
    )
    agg.columns = ["orderer", "n_targets", "mean", "std", "best", "worst", "n"]
    agg["mean_std"] = agg.apply(lambda r: f"{r['mean']:.1f} +- {r['std']:.1f}", axis=1)
    pivot = agg.pivot(index="n_targets", columns="orderer", values="mean_std")
    pivot = pivot[[o for o in ORDERERS if o in pivot.columns]]
    pivot.columns = [LABELS[c] for c in pivot.columns]

    print("\n=== Summary table (mean+-std cost) ===")
    print(pivot.to_string())
    pivot.to_csv(salida(out_dir, "summary_table.csv", cost_fn))
    return agg


# ── 2. Friedman + Holm ────────────────────────────────────────────────────────


def friedman_holm(df: pd.DataFrame, out_dir: Path, cost_fn: str) -> None:
    """Friedman sobre TODOS los tamanos juntos.

    Ojo con interpretarlo solo: en N=5 la mayoria de los ordenadores encuentra
    el optimo exacto, asi que ese bloque aporta empates con el mismo peso que
    N=30 y aplana las diferencias. ``friedman_por_n`` da la vista desagregada,
    donde por ejemplo NN pasa de 5.o a 3.o al mirar N>=20.
    """
    print("\n=== Friedman test (all sizes pooled) ===")

    group_cols = ["scenario", "n_targets", "instance_seed"]
    groups: dict[str, list[float]] = {o: [] for o in ORDERERS if o in df["orderer"].unique()}

    for _, block in df.groupby(group_cols):
        block_ord = block.groupby("orderer")["total_cost"].median()
        if all(o in block_ord.index for o in groups):
            for o in groups:
                groups[o].append(float(block_ord[o]))

    if not groups or len(next(iter(groups.values()))) < 2:
        print("  Not enough blocks for Friedman.")
        return

    arrays = [np.array(v) for v in groups.values()]
    stat, p = friedmanchisquare(*arrays)
    print(f"  chi2={stat:.3f}  p={p:.4f}  {'SIGNIFICANT' if p < ALPHA else 'not significant'}")

    data_matrix = pd.DataFrame(groups)
    rankings = data_matrix.rank(axis=1).mean(axis=0).to_dict()
    print("  Average ranks:", {LABELS[o]: f"{r:.2f}" for o, r in rankings.items()})

    if p < ALPHA:
        print("\n  Post-hoc Holm (pairwise Wilcoxon, familia = 15 comparaciones):")
        long = data_matrix.melt(var_name="orderer", value_name="cost")
        # Holm, no Shaffer: corrige sobre las k(k-1)/2 = 15 comparaciones sin
        # explotar las dependencias logicas entre hipotesis, que es lo que
        # distingue al procedimiento de Shaffer. La etiqueta anterior decia
        # "Holm-Shaffer" y nombraba un metodo que no es el que se aplica.
        ph = sp.posthoc_wilcoxon(long, val_col="cost", group_col="orderer", p_adjust="holm")
        ph.columns = [LABELS.get(c, c) for c in ph.columns]
        ph.index = [LABELS.get(i, i) for i in ph.index]
        print(ph.round(4).to_string())
        ph.map(estable).to_csv(salida(out_dir, "posthoc_holm.csv", cost_fn))

    _cd_diagram(rankings, len(next(iter(groups.values()))), out_dir, cost_fn)


def _cd_diagram(rankings: dict[str, float], n_datasets: int, out_dir: Path, cost_fn: str) -> None:
    import math

    k = len(rankings)
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
    q = q_table.get(k, 2.850)
    cd = q * math.sqrt(k * (k + 1) / (6 * n_datasets))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    sorted_ranks = sorted(rankings.items(), key=lambda x: x[1])

    y_base = 0.55
    for i, (name, rank) in enumerate(sorted_ranks):
        color = COLORS.get(name, "gray")
        ax.plot(rank, y_base, "o", color=color, markersize=13, zorder=3)
        offset = 0.16 if i % 2 == 0 else -0.20
        ax.text(
            rank,
            y_base + offset,
            LABELS.get(name, name),
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=color,
        )

    best_rank = sorted_ranks[0][1]
    ax.annotate(
        "",
        xy=(best_rank + cd, 0.22),
        xytext=(best_rank, 0.22),
        arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.5},
    )
    ax.text(best_rank + cd / 2, 0.29, f"CD={cd:.2f}", ha="center", fontsize=11)

    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Average rank", fontsize=12)
    ax.set_title("Critical Difference Diagram (Nemenyi, alpha=0.05)", fontsize=13)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(salida(out_dir, "cd_diagram.png", cost_fn), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  CD={cd:.3f}  -> cd_diagram.png")


# ── 3. Wilcoxon pairwise ──────────────────────────────────────────────────────


def _instance_aligned(df: pd.DataFrame, orderers: list[str], value: str) -> pd.DataFrame:
    """Tabla instancia x orderer alineada por (scenario, n_targets, instance_seed),
    promediando algo_seeds y conservando SOLO las instancias presentes en TODOS los
    orderers pedidos (intersección). Cada fila es un bloque comparable para Wilcoxon
    pareado; evita el desalineo por orden de fila + truncado ``[:min_len]``."""
    if "feasible" in df.columns:
        df = df[df["feasible"]]
    piv = (
        df.groupby(["scenario", "n_targets", "instance_seed", "orderer"])[value]
        .mean()
        .unstack("orderer")
    )
    if any(o not in piv.columns for o in orderers):
        return piv.iloc[0:0]
    return piv[orderers].dropna()


def friedman_por_n(df: pd.DataFrame, out_dir: Path, cost_fn: str) -> None:
    """Friedman por tamano — es donde se ve que el ranking depende de N.

    Escribe ``friedman_por_n.csv`` y ``ranks_vs_n.png``.
    """
    print("\n=== Friedman test (per instance size) ===")
    filas: list[dict[str, float | int | str]] = []
    presentes = [o for o in ORDERERS if o in df["orderer"].unique()]

    for n, sub in df.groupby("n_targets"):
        piv = sub.pivot_table(
            index=["scenario", "instance_seed"], columns="orderer", values="total_cost"
        ).dropna()
        if len(piv) < 2 or not all(o in piv.columns for o in presentes):
            continue
        piv = piv[presentes]
        stat, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
        rangos = piv.rank(axis=1).mean()
        orden = "  <  ".join(LABELS[o] for o in rangos.sort_values().index)
        print(f"  N={n:<3} chi2={stat:8.2f}  p={p:.2e}   {orden}")
        for o in presentes:
            filas.append(
                {
                    "n_targets": int(n),
                    "orderer": o,
                    "rango_medio": float(rangos[o]),
                    "chi2": float(stat),
                    "p": float(p),
                    "n_bloques": len(piv),
                }
            )

    if not filas:
        print("  Not enough blocks.")
        return
    res = pd.DataFrame(filas)
    for col in ("chi2", "p"):
        if col in res.columns:
            res[col] = res[col].map(estable)
    res.to_csv(salida(out_dir, "friedman_por_n.csv", cost_fn), index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for o in presentes:
        s = res[res.orderer == o].sort_values("n_targets")
        ax.plot(
            s.n_targets,
            s.rango_medio,
            marker="o",
            ms=6,
            lw=1.9,
            color=COLORS.get(o, "gray"),
            label=LABELS[o],
        )
    ax.set_xlabel("Number of targets (N)", fontsize=10.5)
    ax.set_ylabel("Mean Friedman rank (1 = best)", fontsize=10.5)
    ax.set_title("The ranking depends on instance size", fontsize=12)
    ax.set_xticks(sorted(res.n_targets.unique()))
    ax.invert_yaxis()
    ax.grid(alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(salida(out_dir, "ranks_vs_n.png", cost_fn), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> ranks_vs_n.png")


def wilcoxon_pairwise(df: pd.DataFrame, out_dir: Path, cost_fn: str) -> None:
    pairs = [
        ("pso", "gwo"),
        ("nn", "two_opt"),
        ("two_opt", "pso"),
        ("two_opt", "aco"),
        ("two_opt", "ga"),
        ("aco", "ga"),
    ]
    all_rows = []
    for a, b in pairs:
        print(f"\n=== Wilcoxon {LABELS[a]} vs {LABELS[b]} ===")
        for n in sorted(df["n_targets"].unique()):
            sub = df[df["n_targets"] == n]
            piv = _instance_aligned(sub, [a, b], "total_cost")
            if len(piv) < 5:
                continue
            va = piv[a].to_numpy()
            vb = piv[b].to_numpy()
            if np.allclose(va, vb):
                print(f"  n={n:2d}: no difference")
                all_rows.append(
                    {
                        "pair": f"{a}_vs_{b}",
                        "n_targets": n,
                        "stat": 0,
                        "p": 1.0,
                        "significant": False,
                    }
                )
                continue
            stat, p = wilcoxon(va, vb, alternative="two-sided")
            sig = p < ALPHA
            print(f"  n={n:2d}: stat={stat:.1f}  p={p:.4f}  {'*' if sig else ''}")
            all_rows.append(
                {
                    "pair": f"{a}_vs_{b}",
                    "n_targets": n,
                    "stat": round(stat, 1),
                    "p": round(p, 4),
                    "significant": sig,
                }
            )

    pd.DataFrame(all_rows).to_csv(salida(out_dir, "wilcoxon_pairwise.csv", cost_fn), index=False)


# ── 4. Scaling curves ─────────────────────────────────────────────────────────


def plot_scaling(df: pd.DataFrame, out_dir: Path, cost_fn: str) -> None:
    agg = (
        df.groupby(["orderer", "n_targets"])
        .agg(
            cost_mean=("total_cost", "mean"),
            cost_std=("total_cost", "std"),
            time_mean=("cpu_time_s", "mean"),
            time_std=("cpu_time_s", "std"),
        )
        .reset_index()
    )

    cost_label = "Total cost (m)" if cost_fn == "distance" else "Total cost (J)"
    cost_title = "distance" if cost_fn == "distance" else "energy-aware"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ord_name in ORDERERS:
        sub = agg[agg["orderer"] == ord_name]
        if sub.empty:
            continue
        c, m = COLORS[ord_name], MARKERS[ord_name]
        axes[0].errorbar(
            sub["n_targets"],
            sub["cost_mean"],
            yerr=sub["cost_std"],
            label=LABELS[ord_name],
            color=c,
            marker=m,
            capsize=3,
            linewidth=1.5,
        )
        axes[1].errorbar(
            sub["n_targets"],
            sub["time_mean"],
            yerr=sub["time_std"],
            label=LABELS[ord_name],
            color=c,
            marker=m,
            capsize=3,
            linewidth=1.5,
        )

    axes[0].set_xlabel("Number of targets")
    axes[0].set_ylabel(cost_label)
    axes[0].set_title("Mean cost vs instance size")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks([5, 10, 15, 20, 30])

    axes[1].set_xlabel("Number of targets")
    axes[1].set_ylabel("CPU time (s)")
    axes[1].set_title("Computation time vs instance size")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks([5, 10, 15, 20, 30])

    fig.suptitle(f"Scaling — cost function: {cost_title}", fontsize=12)
    fig.tight_layout()
    fig.savefig(salida(out_dir, "scaling_curves.png", cost_fn), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> scaling_curves.png")


# ── 5. Convergence subplots: rows=scenario, cols=n_targets ───────────────────


def plot_convergence(data_dir: Path, out_dir: Path, generator: str, cost_fn: str) -> None:
    conv_path = data_dir / "convergence.csv"
    if not conv_path.exists():
        print("  [skip] convergence.csv not found")
        return

    res = pd.read_csv(data_dir / "results.csv")
    conv = pd.read_csv(conv_path)

    res_f = res[(res["generator"] == generator) & (res["cost_function"] == cost_fn)]
    merged = conv.merge(
        res_f[["run_id", "orderer", "n_targets", "scenario"]], on="run_id", how="inner"
    )

    # Los de `--orderers` que tengan historia de convergencia, no una lista fija:
    # una lista cableada deja afuera a los memeticos y los hibridos. `nn` y
    # `two_opt` son deterministas y no tienen historia, asi que quedan fuera solos.
    con_historia = [o for o in ORDERERS if o in set(merged["orderer"])]
    # Con doce series la banda de +-1 sigma tapa las medias; se dibuja solo
    # cuando son pocas.
    banda = len(con_historia) <= 4
    if merged.empty:
        print("  [skip] no convergence data for this combination")
        return

    scenarios = [
        s
        for s in ["tiny", "urban", "forest", "building", "solar_farm"]
        if s in merged["scenario"].unique()
    ]
    n_vals = sorted(merged["n_targets"].unique())
    nrows, ncols = len(scenarios), len(n_vals)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharey=False)
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    cost_label = "Best cost (m)" if cost_fn == "distance" else "Best cost (J)"

    for row, scen in enumerate(scenarios):
        for col, n in enumerate(n_vals):
            ax = axes[row][col]
            sub = merged[(merged["scenario"] == scen) & (merged["n_targets"] == n)]
            plotted = False
            for ord_name in con_historia:
                ord_data = sub[sub["orderer"] == ord_name]
                if ord_data.empty:
                    continue
                agg = ord_data.groupby("evals")["best_cost_so_far"].agg(["mean", "std"])
                ax.plot(
                    agg.index,
                    agg["mean"],
                    label=LABELS[ord_name],
                    color=COLORS[ord_name],
                    linewidth=1.5,
                )
                if banda:
                    ax.fill_between(
                        agg.index,
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        alpha=0.15,
                        color=COLORS[ord_name],
                    )
                plotted = True

            if row == 0:
                ax.set_title(f"N={n}", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{SCENARIO_NAMES[scen]}\n{cost_label}", fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel("Evaluations", fontsize=9)
            ax.grid(alpha=0.25)
            if plotted and row == 0 and col == ncols - 1:
                ax.legend(fontsize=8)

    cost_title = "distance" if cost_fn == "distance" else "energy-aware"
    fig.suptitle(
        f"Convergence of the stochastic orderers  |  {generator.upper()} generator  |  {cost_title}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fname = f"convergence_{generator}_{cost_fn}.png"
    fig.savefig(salida(out_dir, fname, cost_fn), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {fname}")


# ── 6. Route visualization 3D (todos los escenarios) ─────────────────────────

_VIEW_ANGLES: dict[str, tuple[float, float]] = {
    "building": (22, -55),  # fachada frontal y lateral
    "solar_farm": (35, -45),  # cuadrícula desde arriba-lateral
    "tiny": (30, -60),  # 100x100x50 — perspectiva diagonal
    "urban": (25, -55),  # 200x200x80 — ligeramente mas plana
    "forest": (30, -60),  # 100x100x50 — igual que tiny
}


def _draw_aabb_wireframe(
    ax: Any, boxes: np.ndarray, color: str = "#8888bb", alpha: float = 0.5
) -> None:
    """Dibuja cajas AABB como aristas 3D."""
    for cx, cy, cz, hx, hy, hz in boxes:
        xs, ys, zs = [cx - hx, cx + hx], [cy - hy, cy + hy], [cz - hz, cz + hz]
        for i in range(2):
            for j in range(2):
                ax.plot(
                    [xs[0], xs[1]], [ys[i], ys[i]], [zs[j], zs[j]], color=color, alpha=alpha, lw=0.9
                )
                ax.plot(
                    [xs[i], xs[i]], [ys[0], ys[1]], [zs[j], zs[j]], color=color, alpha=alpha, lw=0.9
                )
                ax.plot(
                    [xs[i], xs[i]], [ys[j], ys[j]], [zs[0], zs[1]], color=color, alpha=alpha, lw=0.9
                )


def _compute_tour_path(
    grid: Any,
    free: np.ndarray,
    planner: Any,
    depot: np.ndarray,
    targets: np.ndarray,
    end: np.ndarray,
    tour: list[int],
    seg_cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Reconstruye la ruta real de A* para el tour dado, reutilizando seg_cache."""
    idx_seq = [-1, *list(tour), -2]
    pt_seq = [depot] + [targets[t] for t in tour] + [end]
    segments: list[np.ndarray] = []
    for k, (fi, ti, fp, tp) in enumerate(
        zip(idx_seq[:-1], idx_seq[1:], pt_seq[:-1], pt_seq[1:], strict=False)
    ):
        key = (fi, ti)
        if key not in seg_cache:
            try:
                sv = grid.world_to_voxel(fp.astype(np.float64))
                gv = grid.world_to_voxel(tp.astype(np.float64))
                voxels = planner._astar_search(sv, gv, free)
                pw = grid.voxels_to_world(voxels)
                if pw.shape[0] >= 2:
                    pw[0] = fp.copy()
                    pw[-1] = tp.copy()
                else:
                    pw = np.vstack([fp, tp])
            except Exception:
                pw = np.vstack([fp, tp])
            seg_cache[key] = pw
        seg = seg_cache[key]
        segments.append(seg if k == 0 else seg[1:])
    return np.vstack(segments)


def _fill_panel_3d(
    ax: Any,
    targets: np.ndarray,
    depot: np.ndarray,
    end: np.ndarray,
    boxes: np.ndarray | None,
    obs: np.ndarray | None,
    tour: list[int],
    ord_name: str,
    cost: float,
    cost_fn: str,
    row: int,
    col: int,
    n: int,
    elev: float,
    azim: float,
    path_arr: np.ndarray | None = None,
) -> None:
    if boxes is not None and len(boxes) > 0:
        _draw_aabb_wireframe(ax, boxes)
    if obs is not None and len(obs) > 0:
        for cx, cy, cz, r in obs:
            u, v_m = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 8j]
            ax.plot_surface(
                cx + r * np.cos(u) * np.sin(v_m),
                cy + r * np.sin(u) * np.sin(v_m),
                cz + r * np.cos(v_m),
                color="#cccccc",
                alpha=0.2,
            )

    if path_arr is not None and len(path_arr) >= 2:
        ax.plot(
            path_arr[:, 0],
            path_arr[:, 1],
            path_arr[:, 2],
            "-",
            color=COLORS[ord_name],
            lw=1.3,
            alpha=0.85,
        )
    else:
        route_pts = np.vstack([depot, targets[tour], end])
        ax.plot(
            route_pts[:, 0],
            route_pts[:, 1],
            route_pts[:, 2],
            "-",
            color=COLORS[ord_name],
            lw=1.3,
            alpha=0.85,
        )

    for visit_idx, t in enumerate(tour):
        ax.scatter(
            targets[t, 0],
            targets[t, 1],
            targets[t, 2],
            s=45,
            color="deepskyblue",
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )
        ax.text(
            targets[t, 0],
            targets[t, 1],
            targets[t, 2] + 1.5,
            str(visit_idx + 1),
            fontsize=5.5,
            ha="center",
        )

    ax.scatter(
        *depot, s=90, marker="^", color="limegreen", edgecolors="black", linewidths=0.4, zorder=5
    )
    ax.scatter(*end, s=75, marker="s", color="red", edgecolors="black", linewidths=0.4, zorder=5)

    cost_str = f"{cost:.0f} m" if cost_fn == "distance" else f"{cost / 1000:.1f} kJ"
    ax.set_title(f"{LABELS[ord_name]}  ({cost_str})", fontsize=8, pad=2)
    if col == 0:
        ax.set_ylabel(f"N={n}", fontsize=8, labelpad=2)
    ax.set_xlabel("X", fontsize=6, labelpad=1)
    ax.set_zlabel("Z", fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=elev, azim=azim)


def plot_routes(df: pd.DataFrame, out_dir: Path, cost_fn: str, inst_seed: int = 0) -> None:
    from bioinspired.environment.scenarios import (
        generate_building_tsp_scenario,
        generate_forest_tsp_scenario,
        generate_solar_farm_tsp_scenario,
        generate_tiny_tsp_scenario,
        generate_urban_tsp_scenario,
    )

    _all_factories = {
        "tiny": generate_tiny_tsp_scenario,
        "urban": generate_urban_tsp_scenario,
        "forest": generate_forest_tsp_scenario,
        "building": generate_building_tsp_scenario,
        "solar_farm": generate_solar_farm_tsp_scenario,
    }
    factories = {s: f for s, f in _all_factories.items() if s in df["scenario"].unique()}

    n_vals = sorted(df["n_targets"].unique())
    show_orderers = ORDERERS
    cost_title = "distance" if cost_fn == "distance" else "energy-aware"

    from bioinspired.algorithms.soo._shared import VoxelGrid
    from bioinspired.algorithms.soo.astar import AStarPlanner

    for scen, factory in factories.items():
        ncols = len(show_orderers)
        nrows = len(n_vals)
        # panel más pequeño cuando hay muchas filas (paper_v10 tiene 5)
        pw = 3.8 if nrows >= 5 else 4.5
        ph = 3.6 if nrows >= 5 else 4.2
        fig = plt.figure(figsize=(pw * ncols, ph * nrows))
        axes: list[list[Any]] = [
            [
                fig.add_subplot(nrows, ncols, row * ncols + col + 1, projection="3d")
                for col in range(ncols)
            ]
            for row in range(nrows)
        ]
        elev, azim = _VIEW_ANGLES.get(scen, (28, -55))

        for row, n in enumerate(n_vals):
            problem = factory(seed=inst_seed, n_targets=n)
            targets = problem.inspection_targets_arr
            depot = problem.start_arr
            end = problem.goal_arr
            obs = problem.obstacles_arr
            boxes = getattr(problem, "boxes_arr", None)

            # Construir grid+free una sola vez por fila; cachear segmentos A*
            _grid = VoxelGrid.from_problem(problem, 10.0)
            _free = _grid.free_voxel_mask(problem.obstacles_arr, problem.boxes_arr)
            _planner = AStarPlanner(voxel_size_m=10.0)
            _seg_cache: dict[tuple[int, int], np.ndarray] = {}

            for col, ord_name in enumerate(show_orderers):
                ax = axes[row][col]
                sub = df[
                    (df["scenario"] == scen)
                    & (df["n_targets"] == n)
                    & (df["instance_seed"] == inst_seed)
                    & (df["cost_function"] == cost_fn)
                    & (df["orderer"] == ord_name)
                ]
                if sub.empty:
                    ax.set_visible(False)
                    continue

                best_row = sub.sort_values("total_cost").iloc[0]
                tour = json.loads(best_row["tour_order"])
                cost = float(best_row["total_cost"])

                path_arr = _compute_tour_path(
                    _grid, _free, _planner, depot, targets, end, tour, _seg_cache
                )
                _fill_panel_3d(
                    ax,
                    targets,
                    depot,
                    end,
                    boxes,
                    obs,
                    tour,
                    ord_name,
                    cost,
                    cost_fn,
                    row,
                    col,
                    n,
                    elev,
                    azim,
                    path_arr=path_arr,
                )

        fig.suptitle(
            f"Inspection tours  |  {SCENARIO_NAMES[scen]} scenario"
            f"  |  cost: {cost_title}  |  instance seed={inst_seed}",
            fontsize=12,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()
        fname = f"routes_{scen}_{cost_fn}.png"
        fig.savefig(salida(out_dir, fname, cost_fn), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {fname}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="TSP Paper Statistical Analysis (TICEC 2026)")
    parser.add_argument(
        "--data", default="results/fase1/paper_v6", help="Directory containing results.csv"
    )
    parser.add_argument(
        "--out", default=None, help="Output directory (default: --data/analysis/<cost>)"
    )
    parser.add_argument("--generator", default="astar", choices=["astar", "rrtstar"])
    parser.add_argument(
        "--cost", default="distance", choices=["distance", "energy_aware"], dest="cost_fn"
    )
    parser.add_argument(
        "--route-seed",
        type=int,
        default=0,
        dest="route_seed",
        help="Instance seed used for route visualization",
    )
    parser.add_argument(
        "--orderers",
        nargs="+",
        default=None,
        help="ordenadores a analizar. Sin esto usa los SEIS de paper 1, que es lo "
        "que reproduce sus figuras. Sobre el dataset de Fase 2 hay que pasar los "
        "14 o los hibridos y memeticos se descartan en silencio.",
    )
    args = parser.parse_args()

    if args.orderers:
        desconocidos = [o for o in args.orderers if o not in LABELS]
        if desconocidos:
            parser.error(f"sin etiqueta para {desconocidos}; agregarla a LABELS/COLORS/MARKERS")
        ORDERERS[:] = args.orderers

    data_dir = Path(args.data)
    out_dir = Path(args.out) if args.out else data_dir / "analysis" / args.cost_fn
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}")
    df_all = _load(data_dir)
    print(f"  {len(df_all)} runs loaded")
    print(f"  Generators : {df_all['generator'].unique().tolist()}")
    print(f"  Cost fns   : {df_all['cost_function'].unique().tolist()}")
    print(f"  Orderers   : {df_all['orderer'].unique().tolist()}")
    print(f"  n_targets  : {sorted(df_all['n_targets'].unique().tolist())}")

    df = _filter(df_all, args.generator, args.cost_fn)
    print(f"\nAnalyzing: generator={args.generator} | cost_function={args.cost_fn}")
    print(f"  {len(df)} runs in this combination")

    table_summary(df, out_dir, args.cost_fn)
    friedman_holm(df, out_dir, args.cost_fn)
    friedman_por_n(df, out_dir, args.cost_fn)
    wilcoxon_pairwise(df, out_dir, args.cost_fn)
    plot_scaling(df, out_dir, args.cost_fn)
    plot_convergence(data_dir, out_dir, args.generator, args.cost_fn)
    print("\nGenerating route visualizations...")
    plot_routes(df, out_dir, args.cost_fn, inst_seed=args.route_seed)

    print(f"\nOK Results saved to {out_dir}")


if __name__ == "__main__":
    main()
