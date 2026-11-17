# Instancias TSPLIB

Instancias de referencia con óptimo publicado, usadas para **calibrar** los
solvers: sobre ellas el gap al óptimo lo puede verificar cualquiera, lo que
responde la objeción de que un solver rinde mal por estar mal implementado o
mal configurado.

Descargadas de <https://github.com/mastqe/tsplib> (espejo de la TSPLIB de
Heidelberg). `solutions.txt` es el listado de óptimos de ese mismo repo — los
valores en `KNOWN_OPTIMA` salen de ahí, no de memoria.

Todas las presentes son `EDGE_WEIGHT_TYPE: EUC_2D`, cuya distancia es
`nint(sqrt(dx²+dy²))` con `nint(x) = int(x + 0.5)`. **No** es `round()` de
Python (redondeo al par en empates): usarlo desplaza aristas en 1 y los óptimos
publicados dejan de cerrar.

Para agregar otra instancia:

```bash
curl -sSO https://raw.githubusercontent.com/mastqe/tsplib/master/<nombre>.tsp
```

y sumá su óptimo a `KNOWN_OPTIMA` en `src/bioinspired/experiments/tsplib.py`
(el test `test_optimos_conocidos_cubren_las_instancias_descargadas` falla si
falta).

Uso:

```bash
uv run python scripts/calibrate_tsplib.py
```
