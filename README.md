# A*+TSP inspection routing benchmark — code and data

Benchmark of six target-ordering algorithms (NN, 2-opt, PSO, GWO, ACO, GA) over A* cost
matrices in 3-D environments with obstacles, and the datasets behind every table of the
paper *Statistical Benchmarking of Classical and Bio-Inspired Algorithms for UAV Inspection Routing in an A*+TSP Framework*.

A* computes the pairwise cost matrix on a voxel grid; the orderers sequence the inspection
targets under two independent objectives, travelled distance and an energy-aware cost.
What the repository holds:

- 5,400 runs over 3 scenarios x 5 target counts x 30 instance seeds x 2 cost functions
- exact optima (Held-Karp) for N <= 15 and a TSPLIB calibration against published optima
- a dispersion pilot (ten algorithm seeds per instance) and single-process CPU timings
- the A* matrix cache, so a re-run recomputes no path

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -m "not slow"
```

## Reproduce the paper

All runs are deterministic given the seeds. The included `results/` hold the datasets
behind every table; the commands below regenerate or check them.

| Paper | Data | Command |
|---|---|---|
| Tables 3-4, Fig. 2 | `results/fase1/paper_v6/results.csv` | `uv run python scripts/analyze_tsp_paper.py --data results/fase1/paper_v6 --cost distance` (and `--cost energy_aware`) |
| Table 5, gap columns | exact optima (Held-Karp) over `results/fase1/cache/v6` | `uv run python scripts/compute_exact_gaps.py --cost distance` (and `--cost energy_aware`) |
| Table 5, CPU columns | `results/fase1/cronometraje_n1/results.csv` | timings of the same 5,400 runs with a single process |
| Table 6 | `results/fase1/tsplib/tsplib_calibration.csv` | `uv run python scripts/calibrate_tsplib.py --output results/fase1/tsplib/rerun.csv` |
| Fig. 1 | — | `uv run python scripts/plot_tsp_scenarios_overview.py` |
| Fig. 3 | `results/fase1/paper_v6/results.csv` | `uv run python scripts/plot_paper1_best_route.py --algos aco,pso --ancho 2.9 --leyenda derecha` |
| §2.7, §3.1, §4.3 (5 m voxels) | `results/fase1/paper_v6/analysis_colisiones/` | `uv run python scripts/audit_tour_collisions.py --dataset results/fase1/paper_v6 --voxel 5` |
| §2.7 (10 m voxels) | `results/fase1/paper_v10/` | `uv run python scripts/audit_tour_collisions.py --dataset results/fase1/paper_v10 --voxel 10`; ranking: `analyze_tsp_paper.py --data results/fase1/paper_v10` |
| §2.4, §4.2 (dispersion) | `results/fase1/piloto_v7/results.csv` | ten algorithm seeds per instance |

Re-run the full benchmark (5,400 runs; the A* matrices are cached in
`results/fase1/cache/v6`, so no A* is recomputed):

```bash
uv run python scripts/run_tsp_paper.py \
  --scenarios tiny urban forest --n-targets 5 10 15 20 30 \
  --instance-seeds 30 --algo-seeds-per 1 --generators astar \
  --orderers nn two_opt pso gwo aco ga \
  --cost-functions distance energy_aware --voxel 5 --n-jobs -1 \
  --energy-model legacy \
  --output results/fase1/rerun --cache results/fase1/cache/v6

uv run python scripts/verificar_reproduccion.py \
  --referencia results/fase1/paper_v6/results.csv --nueva results/fase1/rerun/results.csv
```

A quicker check re-runs a subset (for example `--scenarios tiny forest --n-targets 10 15
--instance-seeds 2`, 96 runs) and compares it the same way.

## What is and is not bit-identical

- **NN, 2-opt, ACO and GA** reproduce bit for bit on any platform.
- **PSO and GWO** decode random keys with NumPy's default (unstable) sort, and the order
  of tied keys depends on the CPU's sort routine. Their exact tours can therefore differ
  across platforms; the aggregate results do not change (same Friedman ordering, mean costs
  within a few percent).
- **CPU times** depend on the hardware: the CPU columns of Table 5 were measured with a
  single process on an Intel Core i7-1160G7 laptop. The ratios between methods are what
  carries over.

## License

MIT — see `LICENSE`.
