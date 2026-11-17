"""Compara dos ``results.csv`` corrida por corrida: cuántas coinciden bit a bit.

Sirve para comprobar una re-corrida contra el dataset publicado. Empareja las filas por
(escenario, N, semilla de instancia, semilla de algoritmo, costo, ordenador) y compara el
tour y el costo exactos. Los tiempos no se comparan: dependen del hardware.

Uso:
    uv run python scripts/verificar_reproduccion.py \
        --referencia results/fase1/paper_v6/results.csv --nueva results/fase1/rerun/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CLAVE = ["scenario", "n_targets", "instance_seed", "algo_seed", "cost_function", "orderer"]


def comparar(referencia: Path, nueva: Path) -> pd.DataFrame:
    """Una fila por corrida de ``nueva``, con si coincide o no con la referencia."""
    ref = pd.read_csv(referencia)
    new = pd.read_csv(nueva)
    m = new.merge(ref, on=CLAVE, how="left", suffixes=("", "_ref"), indicator=True)
    m["en_referencia"] = m["_merge"] == "both"
    m["identica"] = (
        m["en_referencia"]
        & (m["tour_order"] == m["tour_order_ref"])
        & (m["total_cost"] == m["total_cost_ref"])
    )
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--referencia", type=Path, required=True)
    ap.add_argument("--nueva", type=Path, required=True)
    args = ap.parse_args()

    m = comparar(args.referencia, args.nueva)
    faltan = int((~m["en_referencia"]).sum())
    print(
        f"corridas comparadas: {int(m['en_referencia'].sum())} (sin par en la referencia: {faltan})"
    )
    print(f"idénticas bit a bit: {int(m['identica'].sum())} de {int(m['en_referencia'].sum())}")
    por_ordenador = m[m["en_referencia"]].groupby("orderer")["identica"].agg(["sum", "count"])
    print(por_ordenador.rename(columns={"sum": "identicas", "count": "corridas"}).to_string())
    if not m.loc[m["en_referencia"], "identica"].all():
        print(
            "\nPSO y GWO decodifican claves aleatorias con un ordenamiento inestable: sus tours"
            "\nexactos dependen de la rutina de ordenamiento del procesador. Los demás"
            "\nordenadores deben coincidir en cualquier plataforma."
        )


if __name__ == "__main__":
    main()
