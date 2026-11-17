"""Solver exacto para el gap-al-óptimo, por programación dinámica (Held-Karp).

La enumeración de permutaciones es ``O(n! · n)`` y muere en n = 11; Held-Karp
es ``O(2^n · n^2)`` y llega hasta n = 16 en segundos, lo que cubre tres de los
cinco tamaños del benchmark en vez de uno.

El tour es **abierto** —depósito -> objetivos -> fin— y la matriz es
**asimétrica**, porque A* explora i->j y j->i en distinto orden sobre la grilla
de vóxeles. Eso descarta los dos atajos habituales: no se puede fijar el primer
nodo ni dividir por dos el espacio de búsqueda.
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from bioinspired.algorithms.tsp.base import BaseTSPAlgorithm, TSPResult
from bioinspired.core.tsp_problem import InspectionTSPProblem


class ExactTSPSolver(BaseTSPAlgorithm):
    """Óptimo exacto por Held-Karp. Solo para análisis (n <= max_n).

    Args:
        max_n: cota de tamaño. El costo es ``2^n · n^2`` en tiempo y
            ``2^n · n`` en memoria, así que 16 son ~8 MB y unos segundos;
            subirlo a 20 son ~170 MB y varios minutos.
    """

    name: ClassVar[str] = "exact"

    def __init__(self, max_n: int = 16) -> None:
        self.max_n = max_n

    def solve(self, problem: InspectionTSPProblem, seed: int = 0) -> TSPResult:
        """Devuelve el tour de costo mínimo y su costo.

        Args:
            problem: problema TSP con matriz de distancias precomputada.
            seed:    ignorado (solver determinista); está por compatibilidad de interfaz.

        Returns:
            TSPResult con el tour óptimo y su costo.

        Raises:
            ValueError: si ``problem.n_targets > max_n``.
        """
        n = problem.n_targets
        if n > self.max_n:
            raise ValueError(f"ExactTSPSolver soporta n <= {self.max_n}; recibido n={n}")
        t0 = time.perf_counter()

        m = np.asarray(problem.distance_matrix, dtype=np.float64)
        inicio = np.asarray(problem.depot_to_targets, dtype=np.float64)
        fin = np.asarray(problem.targets_to_end, dtype=np.float64)

        n_estados = 1 << n
        # dp[mask, j] = costo minimo de cubrir mask terminando en j.
        dp = np.full((n_estados, n), np.inf, dtype=np.float64)
        padre = np.full((n_estados, n), -1, dtype=np.int16)
        for j in range(n):
            dp[1 << j, j] = inicio[j]

        for mask in range(n_estados):
            fila = dp[mask]
            if not np.isfinite(fila).any():
                continue
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                # min sobre j en mask; los j ausentes valen inf y no compiten.
                candidatos = fila + m[:, k]
                j = int(np.argmin(candidatos))
                if candidatos[j] < dp[mask | (1 << k), k]:
                    dp[mask | (1 << k), k] = candidatos[j]
                    padre[mask | (1 << k), k] = j

        completo = n_estados - 1
        totales = dp[completo] + fin
        ultimo = int(np.argmin(totales))
        mejor_costo = float(totales[ultimo])

        # Reconstruccion hacia atras desde el ultimo nodo.
        tour: list[int] = []
        mask, j = completo, ultimo
        while j >= 0:
            tour.append(j)
            j_prev = int(padre[mask, j])
            mask ^= 1 << j
            j = j_prev
        tour.reverse()

        return TSPResult(
            best_tour=tour,
            best_cost=mejor_costo,
            convergence_history=np.array([mejor_costo], dtype=np.float64),
            runtime_seconds=time.perf_counter() - t0,
            metadata={"algorithm": "exact", "n_evaluations": n_estados * n},
        )
