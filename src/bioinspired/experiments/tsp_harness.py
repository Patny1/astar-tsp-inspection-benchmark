"""Maquinaria compartida de los benchmarks TSP (Fases 1, 2 y 3).

Los runners de cada fase —el benchmark y el MOO escalarizado— necesitan lo
mismo: catálogo del registry, caché con modelo energético y celda paralela.
Compartirlo evita que las copias diverjan.

La razón dura para compartir y no copiar es ``cache_path``: si cada runner
calculara la clave por su cuenta, la corrida MOO reconstruiría las matrices que
el benchmark de Fase 2 ya dejó cacheadas.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np

from bioinspired.core.tsp_problem import (
    DistanceMatrixBuilder,
    InspectionTSPProblem,
    RRTStarMatrixBuilder,
)
from bioinspired.environment.scenarios import (
    generate_building_tsp_scenario,
    generate_city_tsp_scenario,
    generate_forest_tsp_scenario,
    generate_layered_moo_tsp_scenario,
    generate_solar_farm_tsp_scenario,
    generate_tiny_tsp_scenario,
    generate_tower_tsp_scenario,
    generate_urban_tsp_scenario,
)
from bioinspired.experiments.tsp_matrix_cache import voxel_tag
from bioinspired.experiments.tsp_presets import make_preset
from bioinspired.experiments.tsp_registry import DETERMINISTIC_ORDERERS, make_orderers

SCENARIO_FACTORIES = {
    "tiny": generate_tiny_tsp_scenario,
    "urban": generate_urban_tsp_scenario,
    "forest": generate_forest_tsp_scenario,
    "building": generate_building_tsp_scenario,
    "solar_farm": generate_solar_farm_tsp_scenario,
    "layered": generate_layered_moo_tsp_scenario,
    "tower": generate_tower_tsp_scenario,
    "city": generate_city_tsp_scenario,
}

# Presupuesto de evaluaciones compartido por todos los solvers estocásticos.
POP_SIZE = 60
MAX_ITER = 300

# Aparatos disponibles. 'default' conserva el DroneModel del repo (1,5 kg) — es
# el que produjo results/fase1/paper_v5 a traves de la formula legacy, asi
# que NO se toca. 'm210' es el DJI M210 RTK V2, el aparato congelado de Fase 2
# en adelante; su rotor coherente vive en RotorParameters.dji_m210.
DRONES = ("default", "m210")


def make_drone(name: str):
    """Instancia el aparato, o None para el default de los escenarios."""
    if name == "default":
        return None
    if name == "m210":
        from bioinspired.environment.drone_model import DroneModel

        return DroneModel.dji_m210()
    raise ValueError(f"aparato desconocido: {name!r}; usar uno de {DRONES}")


ENERGY_MODELS = ("legacy", "gong", "gong_coherent")


def make_energy_model(name: str):
    """Instancia el modelo energético para la matriz, o None para la heurística vieja.

    Los tres comparten las mismas ecuaciones publicadas; se diferencian en la
    parametrización del rotor y en el nivel de potencia que devuelven:

    - ``legacy``: la heurística vieja (``climb_coeff=10`` a mano, descenso
      gratis). Reproduce Fase 1 / ``paper_v5``.
    - ``gong``: Tabla I del paper tal cual. Aplicada al m210 implica un rotor
      imposible (7.264 RPM sobre una hélice de 20,5"), con la potencia de
      perfil al doble de la inducida. Se conserva para reproducir la primera
      corrida de Fase 2.
    - ``gong_coherent``: las mismas ecuaciones con el rotor real del M210
      (hélice 1760S de 17", C_T anclado a una punta de pala de 130 m/s) y el
      factor de batería declarado. Es el modelo congelado de Fase 2 en
      adelante — ver ``RotorParameters.dji_m210`` y
      ``POWERTRAIN_EFFICIENCY_M210``.

    Los tres usan ``continuous_at_hover=True``: las Ecs. (13)/(14) publicadas
    cuentan dos veces la potencia inducida y no convergen al hover, con lo que
    descender saldría más caro que quedarse quieto dentro del rango validado.
    La corrección se declara en el paper junto con la cita.
    """
    if name == "legacy":
        return None
    if name == "gong":
        from bioinspired.environment.energy_model import GongEnergyModel

        return GongEnergyModel(continuous_at_hover=True)
    if name == "gong_coherent":
        from bioinspired.environment.energy_model import (
            POWERTRAIN_EFFICIENCY_M210,
            GongEnergyModel,
            RotorParameters,
        )

        return GongEnergyModel(
            params=RotorParameters.dji_m210(),
            continuous_at_hover=True,
            powertrain_efficiency=POWERTRAIN_EFFICIENCY_M210,
        )
    raise ValueError(f"modelo energético desconocido: {name!r}; usar uno de {ENERGY_MODELS}")


FROZEN_ENERGY_MODEL = "gong_coherent"
FROZEN_DRONE = "m210"
FROZEN_SYMMETRIC_PAIRS = True
"""Modelo y aparato congelados de Fase 2 en adelante.

`symmetric_pairs` corre **un solo A*** por par no ordenado y deriva la
dirección inversa del mismo camino: mitad de llamadas A*, matriz de distancias
exactamente simétrica, y la asimetría de energía queda reducida al potencial
que telescopa en todo tour con extremos fijos. Habilita además el óptimo exacto
por Concorde sobre la instancia reducida, que es la única forma de tener
gap-al-óptimo por encima del N<=16 de Held-Karp.

Fase 1 (`paper_v6`) queda fuera: corrió con `legacy`, el dron por defecto y sin
simetrizar. De Fase 2 en adelante —incluidas Fase 3, Fase 4 y el airframe de
Gazebo— todo comparte estos dos, o los papers no serían comparables entre sí.
"""


PRESETS = ("fase1", "canonico")
FROZEN_PRESET = "canonico"


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    energy_default: str = FROZEN_ENERGY_MODEL,
    drone_default: str = FROZEN_DRONE,
    symmetric_default: bool = FROZEN_SYMMETRIC_PAIRS,
    preset_default: str = FROZEN_PRESET,
) -> None:
    """Agrega ``--energy-model`` y ``--drone`` a un parser, con los catálogos.

    Existe para que los runners de Fase 2/3/4 no repitan la lista de opciones:
    cada copia es una oportunidad de que una fase quede en un modelo viejo sin
    que nada avise. Los defaults son los congelados.
    """
    parser.add_argument(
        "--energy-model",
        choices=ENERGY_MODELS,
        default=energy_default,
        help=(
            f"Modelo de la matriz energética. '{FROZEN_ENERGY_MODEL}' (default) es el "
            "congelado: mismas ecuaciones de Gong con el rotor real del m210. 'gong' "
            "usa la Tabla I sin corregir (reproduce corridas viejas); 'legacy', la "
            "heurística de Fase 1."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default=preset_default,
        help=(
            "Hiperparametros de los ordenadores. 'canonico' (default) usa los "
            "valores que las fuentes primarias recomiendan (ACO beta=5 de Dorigo "
            "et al. 1996, PSO 0.729/1.49445 de Eberhart & Shi 2000); 'fase1' "
            "reproduce lo que corrieron Fase 1 y la primera Fase 2."
        ),
    )
    parser.add_argument(
        "--symmetric-pairs",
        action=argparse.BooleanOptionalAction,
        default=symmetric_default,
        help=(
            "Corre un solo A* por par no ordenado y deriva la direccion inversa "
            "(mitad de llamadas, matriz de distancias exactamente simetrica). "
            "Activado por default de Fase 2 en adelante; --no-symmetric-pairs "
            "reproduce el comportamiento historico corrida-por-direccion."
        ),
    )
    parser.add_argument(
        "--drone",
        choices=DRONES,
        default=drone_default,
        help=(
            f"Aparato. '{FROZEN_DRONE}' (default) es el DJI M210 RTK V2. NO combinar "
            "'default' (1,5 kg) con una variante de Gong: ese aparato con los rotores "
            "del modelo describe algo físicamente imposible."
        ),
    )


def cache_path(
    cache_dir: Path,
    generator: str,
    scenario: str,
    n_targets: int,
    instance_seed: int,
    voxel_size: float,
    energy_model: str,
    drone: str = "default",
    symmetric_pairs: bool = False,
) -> Path:
    """Ruta del caché de matrices. Única fuente de verdad del nombre.

    El modelo energético entra en la clave porque el caché guarda las matrices
    de energía: sin él, una corrida con Gong reutilizaría las energías de la
    heurística vieja sin dar ninguna señal. ``legacy`` conserva el nombre
    histórico para no invalidar los cachés ya construidos.

    El vóxel se codifica con ``voxel_tag`` — ver ahí por qué no es ``int()``.
    """
    suffix = f"{scenario}_n{n_targets}_seed{instance_seed}_{voxel_tag(voxel_size)}"
    if energy_model != "legacy":
        suffix += f"_{energy_model}"
    if drone != "default":
        # El aparato cambia la matriz de energia: sin esto una corrida con m210
        # reutilizaria matrices calculadas para el dron por defecto.
        suffix += f"_{drone}"
    if symmetric_pairs:
        # Cambia las matrices (un A* por par en vez de dos), asi que necesita
        # clave propia. El False historico no lleva sufijo para no invalidar
        # los caches ya construidos.
        suffix += "_sym"
    return cache_dir / f"matrix_{generator}_{suffix}.json"


def evals_used(result, is_deterministic: bool) -> int:
    """Evaluaciones realmente consumidas, según reporta el solver.

    No usar POP_SIZE*MAX_ITER: es el presupuesto nominal y ningún solver cae
    exactamente ahí (ga 17.760, gwo/pso 18.060, híbridos ~17.9k). Escribir el
    nominal haría no verificable la paridad de presupuesto de la Fase 2.
    """
    if is_deterministic:
        return 0
    return int(result.metadata.get("n_evaluations", 0))


def build_matrix(
    generator: str,
    path_problem,
    scenario: str,
    n_targets: int,
    instance_seed: int,
    voxel_size: float,
    cache_dir: Path,
    compute_energy: bool,
    energy_model: str = "legacy",
    drone: str = "default",
    compute_clearance: bool = False,
    symmetric_pairs: bool = False,
    compute_time: bool = False,
) -> tuple:
    """Construye o carga las matrices. Devuelve ``(matrices, build_time_s)``.

    Con ``compute_clearance=True`` se añaden tres matrices al final de la tupla
    — el margen mínimo a obstáculos por tramo, tercer objetivo MOO. Sale de la
    misma ruta A* que ya se recorre, así que no cuesta llamadas extra. Con
    ``compute_time=True`` se añaden, DESPUÉS del clearance, las tres de tiempo
    de vuelo (Fase 3 en adelante).

    El caché es sensible a estas opciones: una matriz guardada sin clearance (o
    sin tiempo) no sirve para un pedido que lo requiera, porque devolvería
    ``None`` en silencio. Se detecta y se **reconstruye como superconjunto**:
    el archivo conserva su nombre y todas las claves que ya tenía, con los
    mismos valores (A* es determinista), más las nuevas. Fase 1/2 nunca piden
    tiempo, así que sus cachés no se tocan. Lo que se DEVUELVE depende sólo de
    lo pedido, no de lo que el archivo tenga.

    OJO con el vóxel: usar el clearance como objetivo exige <= 2,5 m. Con 5 m
    algunas rutas cortan obstáculos y el margen sale negativo.
    """
    if generator != "astar" and energy_model != "legacy":
        # RRTStarMatrixBuilder hardcodea la heurística vieja (climb_coeff=10) y
        # no acepta un modelo energético. Sin este guard, la matriz se
        # cachearía bajo una clave "_gong" conteniendo energías legacy — un
        # dataset mal etiquetado sin ninguna señal. Cablear el modelo en el
        # builder de RRT* antes de habilitar la combinación.
        raise ValueError(
            f"energy_model={energy_model!r} no está cableado para generator={generator!r}; "
            "sólo 'astar' lo soporta (RRTStarMatrixBuilder usa la fórmula legacy)"
        )
    if generator != "astar" and symmetric_pairs:
        # RRTStarMatrixBuilder no acepta la opcion; sin este guard la matriz se
        # cachearia bajo una clave "_sym" conteniendo pares direccionales.
        raise ValueError(
            f"symmetric_pairs no está cableado para generator={generator!r}; sólo 'astar'"
        )
    if generator != "astar" and drone != "default":
        # RRTStarMatrixBuilder evalúa con NumpyBackend, que infla esferas y cajas
        # por `body_radius_m`. La geometría del escenario ya es C-space (ver
        # sim_backends/sdf_world.py), así que con el m210 (0,45 m) la matriz
        # saldría de un mundo más estrecho que el que A* planifica. Sólo A*
        # (free_voxel_mask, sin inflado) respeta la convención.
        raise ValueError(
            f"drone={drone!r} no está cableado para generator={generator!r}; sólo 'astar' "
            "(RRTStarMatrixBuilder inflaría la geometría C-space por body_radius_m)"
        )
    cache_file = cache_path(
        cache_dir,
        generator,
        scenario,
        n_targets,
        instance_seed,
        voxel_size,
        energy_model,
        drone,
        symmetric_pairs,
    )

    # Qué se devuelve (lo pedido) vs qué se calcula/guarda (lo pedido más lo que
    # el archivo ya tenía): separados para que la tupla devuelta no se desplace.
    pedido_clearance, pedido_time = compute_clearance, compute_time

    def _leer(clave: str):
        return np.array(data[clave]) if clave in data else None

    def _salida(
        d: np.ndarray | None,
        d2t: np.ndarray | None,
        t2e: np.ndarray | None,
        e: np.ndarray | None,
        e2t: np.ndarray | None,
        t2ee: np.ndarray | None,
        clr: np.ndarray | None,
        c2t: np.ndarray | None,
        t2ec: np.ndarray | None,
        tm: np.ndarray | None,
        d2tt: np.ndarray | None,
        t2et: np.ndarray | None,
    ) -> tuple[np.ndarray | None, ...]:
        partes: list[np.ndarray | None] = [d, d2t, t2e, e, e2t, t2ee]
        if pedido_clearance:
            partes += [clr, c2t, t2ec]
        if pedido_time:
            partes += [tm, d2tt, t2et]
        return tuple(partes)

    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        falta = [
            nombre
            for nombre, pedido in (("clearance", pedido_clearance), ("time", pedido_time))
            if pedido and f"{nombre}_matrix" not in data
        ]
        if falta:
            print(
                f"  [cache] {cache_file.name} sin {'/'.join(falta)} — reconstruyendo; "
                "se conservan las claves existentes"
            )
            compute_energy = compute_energy or "energy_matrix" in data
            compute_clearance = compute_clearance or "clearance_matrix" in data
            compute_time = compute_time or "time_matrix" in data
        else:
            print(f"  [cache] {cache_file.name}")
            return _salida(
                np.array(data["distance_matrix"]),
                np.array(data["depot_to_targets"]),
                np.array(data["targets_to_end"]),
                _leer("energy_matrix"),
                _leer("depot_to_targets_energy"),
                _leer("targets_to_end_energy"),
                _leer("clearance_matrix"),
                _leer("depot_to_targets_clearance"),
                _leer("targets_to_end_clearance"),
                _leer("time_matrix"),
                _leer("depot_to_targets_time"),
                _leer("targets_to_end_time"),
            ), 0.0

    if (compute_clearance or compute_time) and generator != "astar":
        raise ValueError(
            "compute_clearance/compute_time sólo están cableados para generator='astar'; "
            "RRTStarMatrixBuilder no los soporta"
        )

    t0 = time.perf_counter()
    print(
        f"  [build ] {scenario} N={n_targets} seed={instance_seed} voxel={voxel_size}m ...",
        flush=True,
    )
    if generator == "astar":
        builder = DistanceMatrixBuilder(
            path_problem,
            voxel_size_m=voxel_size,
            verbose=False,
            energy_model=make_energy_model(energy_model),
            symmetric_pairs=symmetric_pairs,
        )
    else:
        builder = RRTStarMatrixBuilder(
            path_problem, max_iterations=3000, base_seed=instance_seed, verbose=True
        )

    if generator == "astar":
        result = builder.build(
            compute_energy=compute_energy,
            compute_clearance=compute_clearance,
            compute_time=compute_time,
        )
    else:
        result = builder.build(compute_energy=compute_energy)
    build_time = time.perf_counter() - t0

    # Desempaquetar con cursor: el builder devuelve, en este orden y sólo si se
    # pidieron, distancia (3), energía (3), clearance (3) y tiempo (3).
    dist, d2t, t2e = result[:3]
    cursor = 3
    energy = e2t = t2ee = clr = c2t = t2ec = tm = d2tt = t2et = None
    if compute_energy:
        energy, e2t, t2ee = result[cursor : cursor + 3]
        cursor += 3
    if compute_clearance:
        clr, c2t, t2ec = result[cursor : cursor + 3]
        cursor += 3
    if compute_time:
        tm, d2tt, t2et = result[cursor : cursor + 3]

    cache_data: dict = {
        "generator": generator,
        "scenario": scenario,
        "n_targets": n_targets,
        "instance_seed": instance_seed,
        "distance_matrix": dist.tolist(),
        "depot_to_targets": d2t.tolist(),
        "targets_to_end": t2e.tolist(),
    }
    if compute_energy and energy is not None:
        cache_data["energy_matrix"] = energy.tolist()
        cache_data["depot_to_targets_energy"] = e2t.tolist()
        cache_data["targets_to_end_energy"] = t2ee.tolist()
    if compute_clearance and clr is not None:
        cache_data["clearance_matrix"] = clr.tolist()
        cache_data["depot_to_targets_clearance"] = c2t.tolist()
        cache_data["targets_to_end_clearance"] = t2ec.tolist()
    if compute_time and tm is not None and d2tt is not None and t2et is not None:
        cache_data["time_matrix"] = tm.tolist()
        cache_data["depot_to_targets_time"] = d2tt.tolist()
        cache_data["targets_to_end_time"] = t2et.tolist()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache_data, indent=2))
    print(f"  [cache] Saved {cache_file.name}")
    return _salida(dist, d2t, t2e, energy, e2t, t2ee, clr, c2t, t2ec, tm, d2tt, t2et), build_time


def battery_capacity_j(drone: str) -> float:
    """Capacidad de batería (J) del aparato por nombre, para leer energías como fracción."""
    from bioinspired.environment.drone_model import DroneModel

    dm = make_drone(drone) or DroneModel()
    return float(dm.battery_capacity_wh) * 3600.0


def run_cell(
    scenario: str,
    n_targets: int,
    inst_seed: int,
    generator: str,
    voxel_size: float,
    cache_dir: str,
    energy_model: str,
    cost_functions: list[str],
    orderers_to_run: list[str],
    algo_seeds_per_instance: int,
    pendientes: list[tuple],
    drone: str = "default",
    symmetric_pairs: bool = False,
    preset: str = "fase1",
) -> tuple[list[dict], list[dict]]:
    """Corre todas las combinaciones pendientes de una celda (escenario, N, seed, generador).

    Unidad de paralelización. Top-level y sin cierres para que joblib la pueda
    picklear; por eso recibe el NOMBRE del ordenador y resuelve la factory
    adentro — ``make_orderers()`` devuelve lambdas, que no son picklables.

    La matriz se carga del caché (el prebuild ya la construyó), así que no hay
    que pasarla entre procesos.

    Args:
        pendientes: las tuplas ``(cost_fn, ord_name, algo_seed)`` que faltan;
            el padre ya descontó las que estaban en el CSV.

    Returns:
        ``(filas_de_resultados, filas_de_convergencia)`` para que las escriba el
        padre — escribir el CSV desde varios procesos lo corrompería.
    """
    if not pendientes:
        return [], []

    dm = make_drone(drone)
    path_problem = (
        SCENARIO_FACTORIES[scenario](seed=inst_seed, n_targets=n_targets)
        if dm is None
        else SCENARIO_FACTORIES[scenario](seed=inst_seed, n_targets=n_targets, drone=dm)
    )
    need_energy = "energy_aware" in cost_functions
    matrices, build_time = build_matrix(
        generator,
        path_problem,
        scenario,
        n_targets,
        inst_seed,
        voxel_size,
        Path(cache_dir),
        need_energy,
        energy_model,
        drone,
        symmetric_pairs=symmetric_pairs,
    )
    dist_mat, d2t, t2e, energy_mat, depot_e, end_e = matrices

    tsp_dist = InspectionTSPProblem.from_path_problem(path_problem, dist_mat, d2t, t2e)
    tsp_energy = None
    if need_energy and energy_mat is not None:
        from dataclasses import replace as dc_replace

        tsp_energy = dc_replace(
            tsp_dist,
            distance_matrix=energy_mat,
            depot_to_targets=depot_e,
            targets_to_end=end_e,
            energy_matrix=energy_mat,
            depot_to_targets_energy=depot_e,
            targets_to_end_energy=end_e,
        )

    catalogo = make_orderers(make_preset(preset))
    filas: list[dict] = []
    conv: list[dict] = []

    for cost_fn, ord_name, algo_seed in pendientes:
        tsp_problem = tsp_dist if cost_fn == "distance" else tsp_energy
        if tsp_problem is None:
            continue

        is_deterministic = ord_name in DETERMINISTIC_ORDERERS
        run_id = str(uuid.uuid4())[:8]

        # process_time y NO perf_counter: mide CPU de este proceso, así que la
        # contención no lo infla. Con reloj de pared, paralelizar arruinaría la
        # columna — es lo que le pasó a results/fase1/inspection, donde ga
        # con N=15 va de 0,7 s a 3.933 s para el mismo trabajo.
        t0 = time.process_time()
        result = catalogo[ord_name]().solve(tsp_problem, seed=algo_seed)
        cpu_time = time.process_time() - t0

        tour = result.best_tour
        total_dist = tsp_dist.tour_cost(tour)
        total_energy = (
            tour_total_energy(tour, tsp_dist, energy_mat, depot_e, end_e)
            if energy_mat is not None
            else float("nan")
        )
        total_cost = result.best_cost

        filas.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "n_targets": n_targets,
                "instance_seed": inst_seed,
                "algo_seed": algo_seed,
                "generator": generator,
                "cost_function": cost_fn,
                "orderer": ord_name,
                "total_cost": round(total_cost, 4),
                "total_distance": round(total_dist, 4),
                "total_energy": round(total_energy, 4),
                "cpu_time_s": round(cpu_time, 4),
                "matrix_build_time_s": round(build_time, 4),
                "evals_used": evals_used(result, is_deterministic),
                "feasible": not np.isinf(total_cost),
                "tour_order": json.dumps([int(x) for x in tour]),
            }
        )

        if not is_deterministic:
            for it, cost in enumerate(result.convergence_history.tolist()):
                conv.append(
                    {
                        "run_id": run_id,
                        "evals": (it + 1) * POP_SIZE,
                        "best_cost_so_far": round(cost, 4),
                    }
                )

    return filas, conv


def tour_total_energy(
    tour: list[int],
    problem: InspectionTSPProblem,
    energy_mat: np.ndarray,
    depot_e: np.ndarray,
    end_e: np.ndarray,
) -> float:
    cost = float(depot_e[tour[0]])
    for k in range(len(tour) - 1):
        cost += float(energy_mat[tour[k], tour[k + 1]])
    cost += float(end_e[tour[-1]])
    return cost
