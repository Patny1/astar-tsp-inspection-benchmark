"""TSP Inspection Paper Benchmark — arquitectura dos capas (TICEC 2026).

Implementa el diseño experimental completo:
  - Capa baja (generadores): astar, rrtstar
  - Capa alta (ordenadores): greedy, pso, gwo
  - Funciones de costo: distance, energy_aware
  - Tamaños de instancia: 5, 10, 15, 20, 30 targets
  - Escenarios: tiny, urban, forest
  - Seeds separados: instance_seed (controla posición de targets)
                     algo_seed (controla aleatoriedad del optimizador)

Produce dos archivos en --output:
  results.csv     — una fila por corrida (métricas finales)
  convergence.csv — una fila por iteración para PSO y GWO

Uso:
    uv run python scripts/run_tsp_paper.py
    uv run python scripts/run_tsp_paper.py --n-targets 5 10 15
    uv run python scripts/run_tsp_paper.py --generators astar --orderers pso gwo
    uv run python scripts/run_tsp_paper.py --instance-seeds 10 --algo-seeds-per 5
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from joblib import Parallel, delayed

from bioinspired.experiments.snapshot import machine_provenance, same_machine
from bioinspired.experiments.tsp_harness import (
    DRONES,
    ENERGY_MODELS,
    MAX_ITER,
    POP_SIZE,
    PRESETS,
    SCENARIO_FACTORIES,
)
from bioinspired.experiments.tsp_harness import build_matrix as _build_matrix
from bioinspired.experiments.tsp_harness import cache_path as _cache_path
from bioinspired.experiments.tsp_harness import make_drone as _make_drone
from bioinspired.experiments.tsp_harness import run_cell as _run_cell
from bioinspired.experiments.tsp_registry import (
    DETERMINISTIC_ORDERERS,
    make_orderers,
)

# Catálogo de ordenadores: única fuente de verdad en `tsp_registry`. No declarar
# una lista local — es lo que hizo derivar el script del registry (a este le
# faltaban los híbridos/meméticos, al registry le faltaba two_opt_random).
ORDERER_NAMES = tuple(sorted(make_orderers()))


# ── Funciones para prebuild paralelo ────────────────────────────────────────


def _build_matrix_job(
    generator: str,
    scenario: str,
    n_targets: int,
    inst_seed: int,
    voxel_size: float,
    cache_dir: str,
    compute_energy: bool,
    energy_model: str = "legacy",
    drone: str = "default",
    symmetric_pairs: bool = False,
) -> None:
    """Construye y cachea una matriz. Top-level para ser picklable por joblib.

    El aparato entra igual que en ``run_cell``: cambia la matriz de energía
    (peso → potencia) y la clave del caché. Sin él, el prebuild construía la
    matriz del dron por defecto bajo la clave sin sufijo y ``run_cell`` la
    reconstruía entera al no encontrar la suya — todo el prebuild era trabajo
    tirado.
    """
    dm = _make_drone(drone)
    factory = SCENARIO_FACTORIES[scenario]
    path_problem = (
        factory(seed=inst_seed, n_targets=n_targets)
        if dm is None
        else factory(seed=inst_seed, n_targets=n_targets, drone=dm)
    )
    _build_matrix(
        generator,
        path_problem,
        scenario,
        n_targets,
        inst_seed,
        voxel_size,
        Path(cache_dir),
        compute_energy,
        energy_model,
        drone,
        symmetric_pairs=symmetric_pairs,
    )


def _prebuild_all_matrices(
    scenarios: list[str],
    n_targets_list: list[int],
    instance_seeds: int,
    generators: list[str],
    voxel_size: float,
    cache_dir: Path,
    compute_energy: bool,
    n_jobs: int,
    energy_model: str = "legacy",
    drone: str = "default",
    symmetric_pairs: bool = False,
    require_cache: bool = False,
) -> None:
    """Pre-construye en paralelo todas las matrices que falten en caché.

    Con ``require_cache`` aborta en vez de construir: para re-correr sobre las
    instancias de un dataset ya publicado, una matriz faltante significa que
    los flags no coinciden con los de esa corrida (el caso típico: olvidar
    ``--voxel 5`` y reconstruir todo con el default de 10 m).
    """
    jobs = []
    for scenario in scenarios:
        for n_targets in n_targets_list:
            for inst_seed in range(instance_seeds):
                for generator in generators:
                    cache_file = _cache_path(
                        cache_dir,
                        generator,
                        scenario,
                        n_targets,
                        inst_seed,
                        voxel_size,
                        energy_model,
                        drone,
                        symmetric_pairs,
                    )
                    if not cache_file.exists():
                        jobs.append((generator, scenario, n_targets, inst_seed))

    if not jobs:
        print("[prebuild] Todas las matrices ya están en caché.")
        return
    if require_cache:
        gen, scen, n, seed = jobs[0]
        raise SystemExit(
            f"--require-cache: faltan {len(jobs)} matrices en {cache_dir} "
            f"(primera: {gen} {scen} N={n} seed={seed}, vóxel {voxel_size}). "
            "¿Coinciden --voxel/--generators/--energy-model/--drone/--symmetric-pairs "
            "con los de la corrida original?"
        )

    print(f"[prebuild] Construyendo {len(jobs)} matrices con n_jobs={n_jobs}...")
    t0 = time.perf_counter()
    Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
        delayed(_build_matrix_job)(
            gen,
            scen,
            n,
            seed,
            voxel_size,
            str(cache_dir),
            compute_energy,
            energy_model,
            drone,
            symmetric_pairs,
        )
        for gen, scen, n, seed in jobs
    )
    elapsed = time.perf_counter() - t0
    print(f"[prebuild] Listo en {elapsed:.1f}s\n")


def run_paper_benchmark(
    scenarios: list[str],
    n_targets_list: list[int],
    instance_seeds: int,
    algo_seeds_per_instance: int,
    generators: list[str],
    orderers_to_run: list[str],
    cost_functions: list[str],
    voxel_size: float,
    output_dir: Path,
    cache_dir: Path,
    n_jobs: int = 1,
    energy_model: str = "legacy",
    drone: str = "default",
    symmetric_pairs: bool = False,
    preset: str = "fase1",
    require_cache: bool = False,
    allow_machine_change: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Los tiempos sólo valen dentro de una máquina: no reanudar en otra. Si la
    # corrida previa no registró "maquina", no hay con qué comparar.
    maquina = machine_provenance()
    previa_path = output_dir / "provenance.json"
    if previa_path.exists():
        previa = json.loads(previa_path.read_text(encoding="utf-8"))
        if (
            "maquina" in previa
            and not same_machine(previa["maquina"], maquina)
            and not allow_machine_change
        ):
            raise SystemExit(
                f"{output_dir} se empezó en otra máquina "
                f"({previa['maquina'].get('cpu_model')}); ésta es {maquina.get('cpu_model')}. "
                "Mezclar máquinas confunde máquina con semilla y arruina los tiempos. "
                "Usar otro --output, o --allow-machine-change si se sabe lo que se hace."
            )

    need_energy = "energy_aware" in cost_functions
    _prebuild_all_matrices(
        scenarios,
        n_targets_list,
        instance_seeds,
        generators,
        voxel_size,
        cache_dir,
        need_energy,
        n_jobs,
        energy_model,
        drone,
        symmetric_pairs,
        require_cache,
    )

    # Procedencia junto a los datos. Sin esto, con que modelo/preset corrio un
    # dataset vive SOLO en el nombre de la carpeta y en el mensaje del commit —
    # y `fase2/coherent` vs `fase2/canonico` se distinguen justamente por el
    # preset, que el CSV no registra en ninguna columna.
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "energy_model": energy_model,
                "drone": drone,
                "symmetric_pairs": symmetric_pairs,
                "preset": preset,
                "voxel_size_m": voxel_size,
                "generators": generators,
                "scenarios": scenarios,
                "n_targets": n_targets_list,
                "instance_seeds": instance_seeds,
                "orderers": orderers_to_run,
                "cost_functions": cost_functions,
                "algo_seeds_per_instance": algo_seeds_per_instance,
                # n_jobs y la máquina se registran porque sin ellos los tiempos
                # absolutos no se pueden interpretar: el paralelismo los infla.
                "n_jobs": n_jobs,
                "commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip(),
                "maquina": maquina,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    results_path = output_dir / "results.csv"
    convergence_path = output_dir / "convergence.csv"

    results_fields = [
        "run_id",
        "scenario",
        "n_targets",
        "instance_seed",
        "algo_seed",
        "generator",
        "cost_function",
        "orderer",
        "total_cost",
        "total_distance",
        "total_energy",
        "cpu_time_s",
        "matrix_build_time_s",
        "evals_used",
        "feasible",
        "tour_order",
    ]
    conv_fields = ["run_id", "evals", "best_cost_so_far"]

    total_combos = (
        len(scenarios)
        * len(n_targets_list)
        * instance_seeds
        * len(generators)
        * len(cost_functions)
        * len(orderers_to_run)
        * algo_seeds_per_instance
    )

    # ── Resume support: load completed run keys from existing CSV ────────────
    run_key_cols = (
        "scenario",
        "n_targets",
        "instance_seed",
        "generator",
        "cost_function",
        "orderer",
        "algo_seed",
    )
    completed: set[tuple] = set()
    is_resume = results_path.exists() and results_path.stat().st_size > 0
    if is_resume:
        import pandas as _pd

        try:
            _existing = _pd.read_csv(results_path, usecols=list(run_key_cols))
            for _, row in _existing.iterrows():
                completed.add(
                    tuple(
                        int(row[k]) if k in ("n_targets", "instance_seed", "algo_seed") else row[k]
                        for k in run_key_cols
                    )
                )
            print(f"\n[resume] {len(completed)} corridas ya completadas — se omitirán")
        except _pd.errors.EmptyDataError:
            is_resume = False  # archivo vacío — arranque limpio

    print(
        f"\nTotal corridas estimadas: {total_combos}  (pendientes: {total_combos - len(completed)})"
    )
    print(f"  Escenarios     : {scenarios}")
    print(f"  Tamaños        : {n_targets_list}")
    print(f"  Instance seeds : {instance_seeds}")
    print(f"  Algo seeds/inst: {algo_seeds_per_instance}")
    print(f"  Generadores    : {generators}")
    print(f"  Funciones costo: {cost_functions}")
    print(f"  Ordenadores    : {orderers_to_run}")
    print(f"  PSO/GWO budget : {POP_SIZE} x {MAX_ITER} = {POP_SIZE * MAX_ITER:,} evals\n")

    rf_mode = "a" if is_resume else "w"
    cf_mode = "a" if convergence_path.exists() else "w"
    with (
        open(results_path, rf_mode, newline="") as rf,
        open(convergence_path, cf_mode, newline="") as cf,
    ):
        rw = csv.DictWriter(rf, fieldnames=results_fields)
        cw = csv.DictWriter(cf, fieldnames=conv_fields)
        if not is_resume:
            rw.writeheader()
            cw.writeheader()

        done = len(completed)
        t_global = time.perf_counter()

        # ── Armar las celdas pendientes ─────────────────────────────────────
        # Una celda = (escenario, N, instance_seed, generador). Es la unidad de
        # paralelizacion: comparte la matriz A* entre todos sus ordenadores y
        # funciones de costo, asi que se carga una sola vez por celda.
        celdas: list[tuple] = []
        for scenario in scenarios:
            for n_targets in n_targets_list:
                for inst_seed in range(instance_seeds):
                    for generator in generators:
                        pendientes = [
                            (cf, ord_n, als)
                            for cf in cost_functions
                            for ord_n in orderers_to_run
                            for als in (
                                [0]
                                if ord_n in DETERMINISTIC_ORDERERS
                                else range(algo_seeds_per_instance)
                            )
                            if (scenario, n_targets, inst_seed, generator, cf, ord_n, als)
                            not in completed
                        ]
                        if pendientes:
                            celdas.append((scenario, n_targets, inst_seed, generator, pendientes))

        n_pend = sum(len(c[4]) for c in celdas)
        print(
            f"celdas pendientes: {len(celdas)}  ({n_pend:,} corridas de {total_combos:,})"
            f"  n_jobs={n_jobs}\n",
            flush=True,
        )

        # ── Despachar ───────────────────────────────────────────────────────
        # El worker devuelve las filas; el padre las escribe. Escribir el CSV
        # desde varios procesos lo corromperia.
        salidas = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
            delayed(_run_cell)(
                scenario,
                n_targets,
                inst_seed,
                generator,
                voxel_size,
                str(cache_dir),
                energy_model,
                cost_functions,
                orderers_to_run,
                algo_seeds_per_instance,
                pendientes,
                drone,
                symmetric_pairs,
                preset,
            )
            for scenario, n_targets, inst_seed, generator, pendientes in celdas
        )

        for filas, conv in salidas:
            for fila in filas:
                rw.writerow(fila)
            for fila in conv:
                cw.writerow(fila)
            done += len(filas)
        rf.flush()
        cf.flush()

    elapsed = time.perf_counter() - t_global
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results -> {results_path}")
    print(f"Convergence -> {convergence_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TSP Paper Benchmark (TICEC 2026)")
    parser.add_argument("--scenarios", nargs="+", default=["tiny", "urban", "forest"])
    parser.add_argument(
        "--n-targets",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20, 30],
        metavar="N",
        dest="n_targets",
    )
    parser.add_argument(
        "--instance-seeds",
        type=int,
        default=30,
        dest="instance_seeds",
        help="Number of instance seeds (0..N-1)",
    )
    parser.add_argument(
        "--algo-seeds-per",
        type=int,
        default=1,
        dest="algo_seeds_per",
        help="Algo seeds per instance for PSO/GWO/ACO/GA (nn/two_opt are deterministic: always 1)",
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["astar", "rrtstar"],
        choices=["astar", "rrtstar"],
    )
    parser.add_argument(
        "--orderers",
        nargs="+",
        default=["nn", "two_opt", "pso", "gwo"],
        choices=ORDERER_NAMES,
    )
    parser.add_argument(
        "--cost-functions",
        nargs="+",
        default=["distance", "energy_aware"],
        choices=["distance", "energy_aware"],
        dest="cost_functions",
    )
    parser.add_argument("--voxel", type=float, default=10.0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        dest="n_jobs",
        help="Workers joblib para prebuild de matrices (-1 = todos los núcleos)",
    )
    parser.add_argument("--output", type=str, default="results/fase1/paper")
    parser.add_argument("--cache", type=str, default="results/fase1/cache/paper")
    parser.add_argument(
        "--energy-model",
        dest="energy_model",
        default="legacy",
        choices=ENERGY_MODELS,
        help=(
            "Modelo para la matriz de energia. 'legacy' (default) reproduce "
            "Fase 1; 'gong' usa Gong et al. (2023). El cache los separa."
        ),
    )
    parser.add_argument(
        "--drone",
        default="default",
        choices=DRONES,
        help=(
            "Aparato. 'default' es el DroneModel del repo (1,5 kg) que produjo "
            "paper_v5; 'm210' es el DJI M210 coherente con los parametros de "
            "rotor que usa GongEnergyModel. El cache los separa."
        ),
    )
    parser.add_argument(
        "--symmetric-pairs",
        dest="symmetric_pairs",
        action="store_true",
        help=(
            "Corre un solo A* por par no ordenado y deriva la direccion inversa: "
            "mitad de llamadas, matriz de distancias exactamente simetrica, y "
            "habilita el optimo exacto por Concorde. Default OFF para no cambiar "
            "la reproduccion de Fase 1; Fase 2 en adelante lo pasa explicito."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="fase1",
        help=(
            "Hiperparametros de los ordenadores. 'fase1' (default) reproduce lo "
            "que corrieron Fase 1 y la primera Fase 2; 'canonico' usa los valores "
            "que las fuentes primarias recomiendan (ACO beta=5, PSO 0.729/1.49445). "
            "El default queda en fase1 para no cambiar la reproduccion del paper 1."
        ),
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="abortar si falta alguna matriz en vez de construirla (re-corridas sobre un dataset)",
    )
    parser.add_argument(
        "--allow-machine-change",
        action="store_true",
        help="permitir reanudar una carpeta empezada en otra máquina (mezcla tiempos)",
    )
    args = parser.parse_args()

    run_paper_benchmark(
        scenarios=args.scenarios,
        n_targets_list=args.n_targets,
        instance_seeds=args.instance_seeds,
        algo_seeds_per_instance=args.algo_seeds_per,
        generators=args.generators,
        orderers_to_run=args.orderers,
        cost_functions=args.cost_functions,
        voxel_size=args.voxel,
        output_dir=Path(args.output),
        cache_dir=Path(args.cache),
        n_jobs=args.n_jobs,
        energy_model=args.energy_model,
        drone=args.drone,
        symmetric_pairs=args.symmetric_pairs,
        preset=args.preset,
        require_cache=args.require_cache,
        allow_machine_change=args.allow_machine_change,
    )


if __name__ == "__main__":
    main()
