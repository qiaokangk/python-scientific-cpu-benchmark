# Python Scientific CPU Benchmark

This folder contains a cross-platform benchmark driver for common Python
scientific CPU workloads. It compares a single-thread mode with a multithread
mode that leaves numerical libraries on their default automatic thread policy.

## Run

Quick smoke test:

```bash
python3 cpu_scientific_benchmark.py --config benchmark_config_smoke.json
```

Standard run:

```bash
python3 cpu_scientific_benchmark.py --config benchmark_config.json
```

Open the UI:

```bash
python3 cpu_scientific_benchmark.py --ui
```

or:

```bash
python3 cpu_scientific_benchmark_ui.py --config benchmark_config.json
```

The UI uses PyQt5 or PyQt6 plus Matplotlib. The left panel contains the input
parameters and module-size JSON; the right panel updates timing, speedup,
metric, monitoring, and text-report views after every completed benchmark test.
`Cancel calculation` terminates the active worker subprocess and keeps any
completed partial results. `Save results` writes the current results with the
same file names and formats as the CLI:
`*_effective_config.json`, `*_results.json`, `*_report.txt`, and `*_plots.pdf`.

The driver writes:

- `*_report.txt`: plain text hardware, plan, thread validation, monitoring, and results.
- `*_results.json`: machine-readable results and effective configuration.
- `*_plots.pdf`: summary plots.
- `*_worker_*.log`: worker stdout/stderr.

## Thread control

The driver starts child processes before importing NumPy/SciPy so thread policy
is applied before BLAS/LAPACK/OpenMP libraries initialize. In `single` mode it
strictly sets:

```text
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
GOTO_NUM_THREADS
MKL_NUM_THREADS
VECLIB_MAXIMUM_THREADS
NUMEXPR_NUM_THREADS
BLIS_NUM_THREADS
NUMBA_NUM_THREADS
```

to `1`, and also applies a one-thread `threadpoolctl` limit when available.

In default `multi` mode, `benchmark.multi_thread_count` is `auto`. The worker
inherits the current process environment unchanged and does not apply a
threadpoolctl upper limit, so NumPy/SciPy/BLAS/LAPACK/OpenMP libraries use the
same automatic policy they would use in a normal script launched from that
environment. The report records the inherited thread environment,
library-reported threadpool state, and observed peak process CPU cores from
runtime monitoring.

You can still set `benchmark.multi_thread_count` to `logical`, `physical`, or
an integer if you explicitly want a fixed upper limit, but this is not the
standard default because some LAPACK workloads can become much slower when
forced to too many threads.

Numba kernels are compiled from module-level functions with `cache=True`.
The first run creates cache files under `__pycache__`; later runs with the same
Python, Numba, platform, and function signature can reuse those files. Deleting
`__pycache__` forces Numba to compile again. The report's `JIT init` column is
timed separately from the benchmark mean and includes either compilation or
cache lookup/loading.

## Configuration

Edit `benchmark_config.json`.

Important keys:

- `benchmark.repeats`: timed batches per benchmark. The standard default is `5`,
  so each case uses several moderately long timed repeats instead of one very
  long batch or many tiny fragments.
- `benchmark.warmups`: warmup calls before timing.
- `benchmark.target_case_s`: approximate timed seconds per benchmark/thread
  case. The standard default is `10.0`; with `repeats=5`, each timed repeat is
  calibrated to about 2 seconds. Each machine auto-selects call count for
  timing stability, while score comparison uses `Mean/call`.
- `benchmark.target_repeat_s`: legacy per-repeat target. Leave it at `0.0`
  when using `target_case_s`.
- `benchmark.max_memory_gb`: skip cases whose estimated working set exceeds this.
- `benchmark.thread_modes`: usually `["single", "multi"]`.
- `benchmark.multi_thread_count`: `auto`, `logical`, `physical`, or an
  integer. The standard default is `auto`: the multithread worker does not set
  or clear BLAS/LAPACK/OpenMP thread environment variables and records the
  actual threadpool and CPU usage chosen by the numerical libraries. `single`
  mode still strictly forces one numerical thread.
- `benchmark.execution_order`: `by_benchmark` runs single/multithread cases
  next to each other for each workload. This is the default because it reduces
  thermal/order bias when comparing speedups. Use `by_thread_mode` to run the
  complete single-thread suite first and the complete multithread suite second.
- `monitoring.enabled`: record runtime CPU, frequency, and memory samples.
- `monitoring.interval_s`: sampling interval for runtime monitoring.
- `modules.<name>.enabled`: enable or disable a benchmark.
- `modules.<name>.inner_loops`: optional fixed kernel calls per timed repeat.
  Omit it for the standard software-benchmark behavior, where calls are chosen
  dynamically to approach `benchmark.target_case_s`.
- `modules.<name>` size keys: matrix dimensions, FFT shape, grid sizes, etc.

The standard `benchmark_config.json` intentionally uses larger defaults for
the GEMM-heavy and dense-solve workloads, because very small single calls can
hide real multithreaded speedups behind thread scheduling overhead. The smoke
config remains small and is only for checking functionality, not for comparing
single-thread and multithread performance.

Timing plots show `Mean/call`, so shorter bars are faster and cross-machine
comparison is based on one kernel-call runtime. Bar labels show the actual timed
call count selected for that case.

## Deterministic data

Random inputs are allowed, but they are deterministic by default. Each workload
uses NumPy `Generator(PCG64)` with a seed derived from
`benchmark.random_seed` and the benchmark name. With the default seed and the
same module size parameters, the generated input data is the same across runs
and machines. Change `benchmark.random_seed` only when you intentionally want a
different deterministic data set.

## Included workloads

The benchmark covers dense matrix multiplication, Hermitian and non-Hermitian
eigensolvers, SVD, Cholesky, QR, dense linear solve, dense least-squares solve,
FFT, einsum tensor contraction, vectorized ufuncs, reductions, NumPy argsort,
Python scalar loops, Numba `njit` loops, Numba `prange` parallel loops, sparse
CSR matvec, sparse eigsh, SciPy signal FFT convolution, SciPy ndimage Gaussian
filtering, and SciPy pairwise distance computation.
