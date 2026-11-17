"""Carga de instancias TSPLIB para calibrar los solvers contra óptimos publicados.

TSPLIB no es una librería: es un conjunto de instancias de referencia con
óptimos conocidos. Correr los mismos solvers sobre ellas responde la objeción
de que rinden mal por estar mal implementados o mal configurados — sobre una
instancia con óptimo publicado el gap lo verifica cualquiera.

Los archivos se descargan a ``data/tsplib/`` (ver el README de ese directorio).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bioinspired.core.tsp_problem import InspectionTSPProblem
from bioinspired.environment.drone_model import DroneModel

# Óptimos publicados, tomados del archivo `solutions` de TSPLIB (no de memoria).
KNOWN_OPTIMA: dict[str, int] = {
    "berlin52": 7542,
    "ch150": 6528,
    "eil51": 426,
    "eil76": 538,
    "kroA100": 21282,
    "st70": 675,
}


@dataclass(frozen=True)
class TSPLIBInstance:
    """Instancia TSPLIB de tipo ``EUC_2D`` ya convertida a matriz de distancias.

    Attributes:
        name: nombre de la instancia (``berlin52``, ...).
        coords_arr: ``(n, 2)`` coordenadas de las ciudades.
        distance_matrix: ``(n, n)`` distancias enteras según la convención EUC_2D.
    """

    name: str
    coords_arr: np.ndarray
    distance_matrix: np.ndarray

    @property
    def n_cities(self) -> int:
        return int(self.coords_arr.shape[0])


def euc_2d_matrix(coords_arr: np.ndarray) -> np.ndarray:
    """Matriz de distancias con la convención ``EUC_2D`` de TSPLIB.

    La distancia es ``nint(sqrt(dx² + dy²))``, donde ``nint`` redondea al entero
    más cercano como ``int(x + 0.5)``. **No** es ``round()`` de Python, que en
    los empates redondea al par: usarlo desplaza algunas aristas en 1 y los
    óptimos publicados dejan de cerrar.
    """
    diff = coords_arr[:, None, :] - coords_arr[None, :, :]
    return np.floor(np.sqrt((diff**2).sum(axis=2)) + 0.5)


def load_tsplib_instance(path: Path) -> TSPLIBInstance:
    """Lee un ``.tsp`` de tipo ``EUC_2D``.

    Args:
        path: ruta al archivo ``.tsp``.

    Returns:
        La instancia con su matriz de distancias ya calculada.

    Raises:
        ValueError: si el archivo no es ``EUC_2D`` o no declara coordenadas.
    """
    text = Path(path).read_text()

    # La clave aparece como "EDGE_WEIGHT_TYPE: X" y como "EDGE_WEIGHT_TYPE : X".
    m = re.search(r"EDGE_WEIGHT_TYPE\s*:\s*(\S+)", text)
    if m is None or m.group(1) != "EUC_2D":
        tipo = m.group(1) if m else "ausente"
        raise ValueError(f"{path.name}: sólo se soporta EUC_2D, encontrado {tipo!r}")

    if "NODE_COORD_SECTION" not in text:
        raise ValueError(f"{path.name}: no tiene NODE_COORD_SECTION")

    cuerpo = text.split("NODE_COORD_SECTION", 1)[1]
    coords: list[tuple[float, float]] = []
    for linea in cuerpo.splitlines():
        s = linea.strip()
        if not s or s in {"EOF", "DISPLAY_DATA_SECTION"}:
            if coords:
                break
            continue
        partes = s.split()
        if len(partes) < 3:
            break
        coords.append((float(partes[1]), float(partes[2])))

    nombre_m = re.search(r"NAME\s*:\s*(\S+)", text)
    nombre = nombre_m.group(1) if nombre_m else Path(path).stem
    coords_arr = np.array(coords, dtype=np.float64)
    return TSPLIBInstance(
        name=nombre,
        coords_arr=coords_arr,
        distance_matrix=euc_2d_matrix(coords_arr),
    )


def tsplib_to_tsp_problem(
    instance: TSPLIBInstance, drone_model: DroneModel | None = None
) -> InspectionTSPProblem:
    """Mapea la instancia al harness, que espera depósito → targets → fin.

    TSPLIB define un **tour cerrado** sobre las n ciudades. Fijando la ciudad 0
    como depósito *y* como fin, cualquier permutación de las n-1 restantes tiene
    ``tour_cost`` igual al largo del ciclo cerrado — fijar una ciudad no pierde
    generalidad en un tour cerrado simétrico.

    Las coordenadas se embeben en 3D con ``z = 0``: el problema es plano, así
    que ninguna métrica energética con término de altura aporta nada acá.
    """
    d = instance.distance_matrix
    coords3d = np.column_stack([instance.coords_arr, np.zeros(instance.n_cities)])
    return InspectionTSPProblem(
        targets_arr=coords3d[1:],
        distance_matrix=d[1:, 1:],
        depot_to_targets=d[0, 1:],
        targets_to_end=d[1:, 0],
        depot_arr=coords3d[0],
        end_arr=coords3d[0],
        scenario_id=f"tsplib_{instance.name}",
        drone_model=drone_model or DroneModel(),
    )


def optimality_gap_pct(cost: float, instance_name: str) -> float:
    """Gap porcentual al óptimo publicado: ``(cost - opt) / opt * 100``."""
    opt = KNOWN_OPTIMA[instance_name]
    return (cost - opt) / opt * 100.0
