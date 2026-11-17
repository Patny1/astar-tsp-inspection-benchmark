"""2-opt local search reusable (delta de sub-camino real, válido para matrices asimétricas)."""

from __future__ import annotations

from bioinspired.core.tsp_problem import InspectionTSPProblem


def two_opt_improve(
    tour: list[int], problem: InspectionTSPProblem, max_passes: int = 200
) -> tuple[list[int], int]:
    """Mejora `tour` por 2-opt hasta óptimo local. Devuelve (tour, n_evals=0).

    El costo se evalúa por aristas (sub-camino), no por llamadas a tour_cost,
    así que no consume presupuesto de tour_cost (n_evals=0). El delta incluye
    las aristas internas que se revierten (correcto para matrices asimétricas).

    Args:
        tour: permutación de range(n_targets).
        problem: instancia de InspectionTSPProblem con la matriz de distancias.
        max_passes: máximo de pasadas completas (0 = sin límite).

    Returns:
        Tupla (tour_mejorado, n_evals) donde n_evals=0 (no llama a tour_cost).
    """
    n = problem.n_targets
    dist = problem.distance_matrix
    d2t = problem.depot_to_targets
    t2e = problem.targets_to_end
    t = list(tour)

    def _edge(a: int, b: int) -> float:
        if a == -1:
            return float(d2t[b])
        if b == n:
            return float(t2e[a])
        return float(dist[a, b])

    def _path(seq: list[int]) -> float:
        return sum(_edge(seq[k], seq[k + 1]) for k in range(len(seq) - 1))

    improved = True
    passes = 0
    while improved and (max_passes == 0 or passes < max_passes):
        improved = False
        passes += 1
        for i in range(n):
            for j in range(i + 1, n):
                prev_i = t[i - 1] if i > 0 else -1
                next_j = t[j + 1] if j < n - 1 else n
                seg = t[i : j + 1]
                if _path([prev_i, *seg[::-1], next_j]) < _path([prev_i, *seg, next_j]) - 1e-10:
                    t[i : j + 1] = seg[::-1]
                    improved = True
    return t, 0
