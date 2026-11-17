"""Presets de hiperparámetros: el registro de con qué corrió cada fase.

**El problema que resuelve.** Hoy los 14 ordenadores toman sus hiperparámetros
de los defaults de clase, y Fase 1 y Fase 2 leen exactamente los mismos. El
paper de Fase 1 ya está escrito, así que esos valores son un dato histórico
fijo; Fase 2 en cambio está en análisis y podría querer moverlos —por ejemplo,
si la sintonización de `scripts/tune_tsp_orderers.py` mostrara algo que valga la
pena adoptar. Con una sola fuente compartida, cambiar un default para Fase 2
rompería en silencio la reproducción de `paper_v5`/`paper_v6`.

**Cómo se separa.** `PRESET_FASE1` es el registro explícito de lo que Fase 1
corrió, independiente de los defaults de clase. Un cambio en el código de un
solver ya no puede alterarlo sin que
`tests/experiments/test_tsp_presets.py` falle, porque ese test compara el preset
contra lo que `make_orderers()` construye de verdad.

**Cómo agregar un preset.** No editar `PRESET_FASE1` ni los defaults de clase:
agregar un diccionario nuevo (`PRESET_FASE2_TUNED`, etc.) y hacer que el runner
de esa fase lo pida. Así las dos parametrizaciones conviven, igual que conviven
`gong` y `gong_coherent` en el modelo energético.

Los valores fueron extraídos de los solvers que `make_orderers()` instancia, no
transcritos a mano.
"""

from __future__ import annotations

from typing import Any

# `memetic_elites` aparece también en las puras: lo heredan de la misma clase
# que subclasean los meméticos, y sin `local_search` es inerte. Se registra
# igual porque el preset documenta lo que se instancia, no lo que importa.
#
# Presupuesto homologado de los estocásticos: 60 x 300 = 18.000 evaluaciones.
# Los meméticos usan menos iteraciones porque su contabilidad de evaluaciones es
# distinta (ver `MEMETIC_FACTORIES`), no porque tengan menos presupuesto.
PRESET_FASE1: dict[str, dict[str, Any]] = {
    # ── Deterministas ───────────────────────────────────────────────────────
    "nn": {},
    "two_opt": {"max_passes": 200, "init": "nn"},
    "two_opt_random": {"max_passes": 200, "init": "random"},
    # ── Metaheurísticas puras ───────────────────────────────────────────────
    "ga": {
        "pop_size": 60,
        "prob_crossover": 0.9,
        "prob_mutation": 0.2,
        "tournament_k": 3,
        "n_elite": 1,
        "max_iterations": 300,
        "memetic_elites": 1,
    },
    "pso": {
        "n_particles": 60,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "v_max": 0.4,
        "max_iterations": 300,
        "memetic_elites": 1,
    },
    "gwo": {"n_wolves": 60, "max_iterations": 300, "memetic_elites": 1},
    "aco": {
        "n_ants": 60,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.5,
        "Q": 100.0,
        "max_iterations": 300,
        "memetic_elites": 1,
    },
    # ── Híbridos secuenciales (Fase 2; no aparecen en el paper de Fase 1) ───
    "ga_aco": {
        "ga_iters": 150,
        "aco_iters": 150,
        "ga_pop_size": 60,
        "n_seed_tours": 5,
        "pheromone_boost": 3.0,
        "aco_n_ants": 60,
        "aco_alpha": 1.0,
        "aco_beta": 2.0,
        "aco_rho": 0.5,
        "aco_q": 100.0,
    },
    "pso_aco": {
        "pso_iters": 150,
        "aco_iters": 150,
        "pso_n_particles": 60,
        "n_seed_tours": 5,
        "pheromone_boost": 3.0,
        "aco_n_ants": 60,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "v_max": 0.4,
        "aco_alpha": 1.0,
        "aco_beta": 2.0,
        "aco_rho": 0.5,
        "aco_q": 100.0,
    },
    "ga_pso": {
        "ga_iters": 150,
        "pso_iters": 150,
        "pop_size": 60,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "v_max": 0.4,
    },
    # ── Meméticos (Fase 2) ──────────────────────────────────────────────────
    "ga_2opt": {
        "pop_size": 60,
        "prob_crossover": 0.9,
        "prob_mutation": 0.2,
        "tournament_k": 3,
        "n_elite": 1,
        "max_iterations": 299,
        "memetic_elites": 1,
    },
    "aco_2opt": {
        "n_ants": 60,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.5,
        "Q": 100.0,
        "max_iterations": 295,
        "memetic_elites": 1,
    },
    "pso_2opt": {
        "n_particles": 60,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "v_max": 0.4,
        "max_iterations": 292,
        "memetic_elites": 1,
    },
    "gwo_2opt": {"n_wolves": 60, "max_iterations": 292, "memetic_elites": 1},
}

# ─────────────────────────────────────────────────────────────────────────────
# PRESET CANÓNICO — los valores que las fuentes primarias recomiendan
# ─────────────────────────────────────────────────────────────────────────────
#
# Motivación: el manuscrito tiene que poder decir «usamos los valores de X» y que
# sea cierto. Verificado contra los papers originales, dos valores
# de PRESET_FASE1 no son los que sus fuentes recomiendan:
#
#   ACO `beta`: el repo usa 2; Dorigo, Maniezzo & Colorni (1996) determinan
#       experimentalmente `alpha=1, beta=5, rho=0.5, Q=100` como el óptimo de
#       ant-cycle (p. 6 y Tabla I). El 2 está dentro del conjunto que probaron
#       —{0, 1, 2, 5}— pero no es su recomendación.
#   PSO `w, c1, c2`: el repo usa 0,7 y 1,5; Eberhart & Shi (2000) dan
#       `w=0.729, c1=c2=1.49445`, los equivalentes al factor de constricción.
#       El repo los tenía redondeados.
#
# Lo que NO se toca y por qué:
#
#   GA: `prob_crossover`, `prob_mutation`, `tournament_k`, `n_elite` no tienen
#       fuente canónica única — son convención. Cambiarlos por otros valores
#       convencionales no acercaría a ninguna cita.
#   ACO `Q=100`: se conserva, pero el propio paper dice que su influencia es
#       despreciable («Parameter Q is not shown because its influence was found
#       to be negligible»). No es una elección sustantiva.
#   ACO `n_ants=60`: AS ata las hormigas a las ciudades (`m = n`). Acá se
#       mantiene fijo porque lo impone la homologación del presupuesto a 18.000
#       evaluaciones; es una divergencia deliberada y hay que declararla, no
#       esconderla — a N=80 se corre con menos hormigas que ciudades.
#   PSO `v_max=0.4`: es un tope sobre la velocidad en el espacio de claves
#       aleatorias, específico de esta codificación. Eberhart & Shi hablan de
#       `Vmax = Xmax` sobre el rango dinámico de la variable, que no traduce
#       directo acá. Se declara como decisión de implementación.
#
# Consecuencia a tener presente: correr con este preset **rompe la identidad bit
# a bit con Fase 1** en los seis ordenadores compartidos. Por eso convive con
# PRESET_FASE1 en vez de reemplazarlo — uno da la cita limpia, el otro el puente
# de comparabilidad con el paper 1.
#
# El preset alcanza también las etapas internas de los híbridos: `ga_aco`,
# `pso_aco` y `ga_pso` aceptan y reenvían los parámetros de sus componentes. Con
# prefijo y sin él: los híbridos exponen los de su componente ACO como
# `aco_alpha`, `aco_beta`, … para no chocar con los suyos propios.
_ACO_CANONICO = {
    "alpha": 1.0,
    "beta": 5.0,
    "rho": 0.5,
    "Q": 100.0,
    "aco_alpha": 1.0,
    "aco_beta": 5.0,
    "aco_rho": 0.5,
    "aco_q": 100.0,
}
_PSO_CANONICO = {"w": 0.729, "c1": 1.49445, "c2": 1.49445, "v_max": 0.4}

PRESET_CANONICO: dict[str, dict[str, Any]] = {
    nombre: {
        **params,
        **({k: v for k, v in _ACO_CANONICO.items() if k in params}),
        **({k: v for k, v in _PSO_CANONICO.items() if k in params}),
    }
    for nombre, params in PRESET_FASE1.items()
}


# Atributos que los solvers guardan pero que no son hiperparámetros: no
# describen el comportamiento del algoritmo, sólo instrumentación.
NO_HIPERPARAMETROS = ("snapshot_every", "local_search")


def parametros_efectivos(solver: Any) -> dict[str, Any]:
    """Hiperparámetros con que quedó instanciado un solver.

    Se usa para contrastar un preset contra lo que el registry construye de
    verdad — que es más fuerte que mirar los defaults de clase, porque atrapa
    también los que el registry sobrescribe (`aco` corre con 60 hormigas, no
    con las 30 de su default).
    """
    return {
        k: v
        for k, v in vars(solver).items()
        if not k.startswith("_") and k not in NO_HIPERPARAMETROS and not callable(v)
    }


def make_preset(nombre: str) -> dict[str, dict[str, Any]]:
    """Resuelve un preset por nombre. Los runners lo pasan como string.

    `run_cell` viaja a workers de joblib, y un diccionario grande por celda
    seria carga inutil: se manda el nombre y se resuelve del lado del worker.
    """
    presets = {"fase1": PRESET_FASE1, "canonico": PRESET_CANONICO}
    if nombre not in presets:
        raise ValueError(f"preset desconocido: {nombre!r}; usar uno de {tuple(presets)}")
    return presets[nombre]
