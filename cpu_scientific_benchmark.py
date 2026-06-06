#!/usr/bin/env python3
"""Cross-platform CPU benchmark for Python scientific numerical workloads.

The driver process does not import NumPy before launching worker processes.
Each worker applies thread-control policy first, then imports NumPy/SciPy so
BLAS/LAPACK/OpenMP thread settings are applied consistently. Single mode is
strictly one thread; the default multithread mode leaves library thread counts
on auto.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import io
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import threading
import textwrap
import time
import traceback
import zlib
from pathlib import Path
from typing import Any


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


BENCHMARK_ORDER = (
    "dense_matmul_float64",
    "dense_matmul_complex128",
    "dense_hermitian_eigvalsh",
    "dense_nonhermitian_eigvals",
    "dense_svd",
    "dense_cholesky",
    "dense_qr",
    "linear_solve",
    "least_squares_lstsq",
    "fft_3d_complex128",
    "einsum_tensor_contraction",
    "vectorized_elementwise",
    "reduction_norm",
    "sort_argsort",
    "python_loop",
    "numba_njit_loop",
    "numba_prange_loop",
    "sparse_matvec_csr",
    "sparse_eigsh",
    "scipy_signal_fftconvolve",
    "scipy_ndimage_gaussian_filter",
    "scipy_distance_cdist",
)


BENCHMARK_LABELS = {
    "dense_matmul_float64": "Dense matrix multiply, float64",
    "dense_matmul_complex128": "Dense matrix multiply, complex128",
    "dense_hermitian_eigvalsh": "Dense Hermitian eigenvalues",
    "dense_nonhermitian_eigvals": "Dense non-Hermitian eigenvalues",
    "dense_svd": "Dense SVD singular values",
    "dense_cholesky": "Dense Cholesky factorization",
    "dense_qr": "Dense QR factorization",
    "linear_solve": "Dense linear solve",
    "least_squares_lstsq": "Dense least-squares solve",
    "fft_3d_complex128": "3D FFT, complex128",
    "einsum_tensor_contraction": "Tensor contraction with einsum",
    "vectorized_elementwise": "Vectorized elementwise ufunc chain",
    "reduction_norm": "Vector reduction and norm",
    "sort_argsort": "NumPy argsort on float64 array",
    "python_loop": "Pure Python scalar loop",
    "numba_njit_loop": "Numba njit scalar loop",
    "numba_prange_loop": "Numba njit parallel prange loop",
    "sparse_matvec_csr": "SciPy CSR sparse matvec",
    "sparse_eigsh": "SciPy sparse eigsh",
    "scipy_signal_fftconvolve": "SciPy signal FFT convolution",
    "scipy_ndimage_gaussian_filter": "SciPy ndimage Gaussian filter",
    "scipy_distance_cdist": "SciPy pairwise Euclidean distances",
}


_numba_prange = range


def _numba_njit_loop_kernel(count: int) -> float:
    total = 0.0
    for i in range(count):
        x = (i + 1) * 1.0e-6
        total += math.sin(x) * math.cos(0.5 * x) + math.sqrt(x)
    return total


def _numba_prange_loop_kernel(count: int) -> float:
    total = 0.0
    for i in _numba_prange(count):
        x = (i + 1) * 1.0e-6
        total += math.sin(x) * math.cos(0.5 * x) + math.sqrt(x)
    return total


def validate_prange_kernel_is_scalar_only() -> None:
    """Reject nested array/BLAS-style work inside the prange benchmark kernel."""

    source = inspect.getsource(_numba_prange_loop_kernel)
    forbidden_tokens = (
        "np.",
        "numpy.",
        "scipy.",
        "linalg",
        "dot(",
        "matmul",
        "@",
        "sum(",
        "mean(",
        "norm(",
        "fft",
    )
    lowered = source.lower()
    hits = [token for token in forbidden_tokens if token in lowered]
    if hits:
        raise RuntimeError(
            "numba_prange_loop kernel must contain only scalar arithmetic and scalar math; "
            f"forbidden nested parallel/vector operations found: {', '.join(hits)}"
        )


def default_config() -> dict[str, Any]:
    return {
        "benchmark": {
            "repeats": 5,
            "warmups": 1,
            "random_seed": 20260606,
            "max_memory_gb": 4.0,
            "thread_modes": ["single", "multi"],
            "multi_thread_count": "auto",
            "execution_order": "by_benchmark",
            "enforce_threadpoolctl": True,
            "gc_between_repeats": True,
            "target_case_s": 10.0,
            "target_repeat_s": 0.0,
            "calibration_max_inner_loops": 100_000,
            "auto_thread_regression_guard": True,
            "auto_thread_regression_factor": 1.05,
        },
        "monitoring": {
            "enabled": True,
            "interval_s": 0.25,
            "per_cpu": False,
            "process_cpu": True,
            "process_memory": True,
        },
        "output": {
            "directory": "benchmark_results",
            "prefix": "cpu_python_scientific",
            "make_pdf": True,
            "make_json": True,
            "make_text": True,
        },
        "modules": {
            "dense_matmul_float64": {"enabled": True, "n": 3072},
            "dense_matmul_complex128": {"enabled": True, "n": 1536},
            "dense_hermitian_eigvalsh": {"enabled": True, "n": 1536},
            "dense_nonhermitian_eigvals": {"enabled": True, "n": 640},
            "dense_svd": {"enabled": True, "n": 1200},
            "dense_cholesky": {"enabled": True, "n": 3600},
            "dense_qr": {"enabled": True, "n": 1800},
            "linear_solve": {"enabled": True, "n": 3000, "nrhs": 32},
            "least_squares_lstsq": {"enabled": True, "m": 5000, "n": 1200, "nrhs": 8},
            "fft_3d_complex128": {
                "enabled": True,
                "shape": [256, 256, 192],
                "backend": "scipy_if_available",
                "use_scipy_workers": False,
            },
            "einsum_tensor_contraction": {
                "enabled": True,
                "a": 64,
                "b": 64,
                "c": 512,
                "d": 64,
                "e": 32,
            },
            "vectorized_elementwise": {"enabled": True, "n": 8_000_000},
            "reduction_norm": {"enabled": True, "n": 128_000_000},
            "sort_argsort": {"enabled": True, "n": 4_000_000},
            "python_loop": {"enabled": True, "iterations": 8_000_000},
            "numba_njit_loop": {"enabled": True, "n": 40_000_000},
            "numba_prange_loop": {"enabled": True, "n": 50_000_000},
            "sparse_matvec_csr": {
                "enabled": True,
                "n": 120_000,
                "nnz_per_row": 12,
                "loops": 200,
            },
            "sparse_eigsh": {
                "enabled": True,
                "n": 8000,
                "k": 6,
                "tol": 1.0e-6,
                "maxiter": 400,
            },
            "scipy_signal_fftconvolve": {
                "enabled": True,
                "shape": [3072, 3072],
                "kernel_shape": [129, 129],
            },
            "scipy_ndimage_gaussian_filter": {
                "enabled": True,
                "shape": [4096, 4096],
                "sigma": 2.0,
            },
            "scipy_distance_cdist": {
                "enabled": True,
                "m": 12000,
                "n": 8000,
                "d": 16,
            },
        },
    }


def smoke_config() -> dict[str, Any]:
    cfg = default_config()
    cfg["benchmark"].update(
        {
            "repeats": 1,
            "warmups": 0,
            "max_memory_gb": 1.5,
            "target_case_s": 0.0,
            "target_repeat_s": 0.0,
        }
    )
    cfg["output"].update(
        {
            "directory": "benchmark_results_smoke",
            "prefix": "cpu_python_scientific_smoke",
        }
    )
    modules = cfg["modules"]
    modules["dense_matmul_float64"]["n"] = 256
    modules["dense_matmul_complex128"]["n"] = 192
    modules["dense_hermitian_eigvalsh"]["n"] = 192
    modules["dense_nonhermitian_eigvals"]["n"] = 128
    modules["dense_svd"]["n"] = 160
    modules["dense_cholesky"]["n"] = 192
    modules["dense_qr"]["n"] = 192
    modules["linear_solve"].update({"n": 192, "nrhs": 4})
    modules["least_squares_lstsq"].update({"m": 360, "n": 120, "nrhs": 2})
    modules["fft_3d_complex128"]["shape"] = [48, 48, 32]
    modules["einsum_tensor_contraction"].update(
        {"a": 10, "b": 10, "c": 24, "d": 12, "e": 8}
    )
    modules["vectorized_elementwise"]["n"] = 200_000
    modules["reduction_norm"]["n"] = 300_000
    modules["sort_argsort"]["n"] = 80_000
    modules["python_loop"]["iterations"] = 200_000
    modules["numba_njit_loop"]["n"] = 500_000
    modules["numba_prange_loop"]["n"] = 1_000_000
    modules["sparse_matvec_csr"].update({"n": 8000, "nnz_per_row": 6, "loops": 3})
    modules["sparse_eigsh"].update({"n": 500, "k": 3, "maxiter": 100})
    modules["scipy_signal_fftconvolve"].update({"shape": [192, 192], "kernel_shape": [17, 17]})
    modules["scipy_ndimage_gaussian_filter"].update({"shape": [256, 256], "sigma": 1.5})
    modules["scipy_distance_cdist"].update({"m": 300, "n": 120, "d": 8})
    for module_cfg in modules.values():
        module_cfg["inner_loops"] = 1
    return cfg


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | None) -> dict[str, Any]:
    cfg = default_config()
    if path:
        config_path = Path(path)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                user_cfg = json.load(handle)
            cfg = deep_merge(cfg, user_cfg)
    return cfg


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_filename(text: str) -> str:
    safe = []
    for char in str(text):
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "item"


def human_bytes(num_bytes: float | int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "n/a"
    seconds = float(seconds)
    if seconds < 1.0e-3:
        return f"{seconds * 1.0e6:.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1.0e3:.2f} ms"
    return f"{seconds:.4f} s"


def get_cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        try:
            model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
            if model:
                return model
        except Exception:
            pass
        try:
            output = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            chip = None
            model = None
            for line in output.splitlines():
                stripped = line.strip()
                if stripped.startswith("Chip:"):
                    chip = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Processor Name:"):
                    chip = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Model Name:"):
                    model = stripped.split(":", 1)[1].strip()
            if chip and model:
                return f"{chip} ({model})"
            if chip:
                return chip
        except Exception:
            pass
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    if system == "Windows":
        name = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()
        if name:
            return name.strip()
    return platform.processor() or platform.machine() or "unknown"


def collect_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": get_cpu_model(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": None,
        "memory_total_bytes": None,
        "memory_available_bytes": None,
    }
    try:
        import psutil  # type: ignore

        info["physical_cpu_count"] = psutil.cpu_count(logical=False)
        vm = psutil.virtual_memory()
        info["memory_total_bytes"] = int(vm.total)
        info["memory_available_bytes"] = int(vm.available)
        try:
            freq = psutil.cpu_freq()
            if freq is not None:
                info["cpu_freq_mhz"] = {
                    "current": freq.current,
                    "min": freq.min,
                    "max": freq.max,
                }
        except Exception as exc:
            info["cpu_freq_error"] = str(exc)
    except Exception as exc:
        info["psutil_error"] = str(exc)
        if platform.system() == "Darwin":
            try:
                mem = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                ).strip()
                info["memory_total_bytes"] = int(mem)
            except Exception:
                pass
    return info


def resolve_thread_count(value: Any, system_info: dict[str, Any]) -> int:
    if isinstance(value, int):
        return 0 if int(value) <= 0 else int(value)
    text = str(value).strip().lower()
    logical = int(system_info.get("logical_cpu_count") or os.cpu_count() or 1)
    physical_value = system_info.get("physical_cpu_count")
    physical = int(physical_value) if physical_value else logical
    if text in {"auto", "default", "library", "unset", "none"}:
        return 0
    if text in {"single", "one", "1"}:
        return 1
    if text in {"logical", "all", "all_logical", "max"}:
        return max(1, logical)
    if text in {"physical", "all_physical"}:
        return max(1, physical)
    try:
        parsed = int(text)
        return 0 if parsed <= 0 else parsed
    except ValueError:
        return max(1, logical)


def resolve_thread_modes(config: dict[str, Any], system_info: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark_cfg = config.get("benchmark", {})
    modes = benchmark_cfg.get("thread_modes", ["single", "multi"])
    resolved: list[dict[str, Any]] = []
    for mode in modes:
        if isinstance(mode, dict):
            name = str(mode.get("name", mode.get("label", "custom")))
            threads = resolve_thread_count(mode.get("threads", 1), system_info)
        else:
            name = str(mode)
            if name.lower() == "single":
                threads = 1
            elif name.lower() == "multi":
                threads = resolve_thread_count(
                    benchmark_cfg.get("multi_thread_count", "auto"), system_info
                )
            else:
                threads = resolve_thread_count(name, system_info)
        resolved.append({"name": name, "threads": threads})
    seen: set[str] = set()
    unique = []
    for mode in resolved:
        key = str(mode["name"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(mode)
    return unique


def build_thread_env(thread_count: int, output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    configure_thread_env_dict(env, thread_count)
    env["MPLCONFIGDIR"] = str(output_dir / ".matplotlib")
    return env


def is_auto_thread_count(thread_count: Any) -> bool:
    try:
        return int(thread_count) <= 0
    except Exception:
        return str(thread_count).strip().lower() in {"auto", "default", "library", "unset", "none"}


def format_thread_count(thread_count: Any) -> str:
    if is_auto_thread_count(thread_count):
        return "auto"
    try:
        return str(max(1, int(thread_count)))
    except Exception:
        return str(thread_count)


def configure_thread_env_dict(env: dict[str, str], thread_count: int) -> None:
    if is_auto_thread_count(thread_count):
        for var in THREAD_ENV_VARS:
            env.pop(var, None)
        env.pop("OMP_DYNAMIC", None)
        env.pop("MKL_DYNAMIC", None)
        return
    for var in THREAD_ENV_VARS:
        env[var] = str(int(thread_count))
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"


def configure_process_thread_env(thread_count: int) -> None:
    configure_thread_env_dict(os.environ, thread_count)


def enabled_benchmark_names(config: dict[str, Any]) -> list[str]:
    modules = config.get("modules", {})
    names = []
    for name in BENCHMARK_ORDER:
        if bool(modules.get(name, {}).get("enabled", False)):
            names.append(name)
    return names


def benchmark_plan_lines(config: dict[str, Any]) -> list[str]:
    lines = []
    modules = config.get("modules", {})
    for name in enabled_benchmark_names(config):
        params = {k: v for k, v in modules.get(name, {}).items() if k != "enabled"}
        param_text = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"- {name}: {BENCHMARK_LABELS.get(name, name)} ({param_text})")
    return lines


class RuntimeMonitor:
    def __init__(self, config: dict[str, Any], thread_label: str, thread_count: int) -> None:
        monitor_cfg = config.get("monitoring", {})
        self.enabled = bool(monitor_cfg.get("enabled", True))
        self.interval_s = max(0.05, float(monitor_cfg.get("interval_s", 0.25)))
        self.per_cpu = bool(monitor_cfg.get("per_cpu", False))
        self.process_cpu = bool(monitor_cfg.get("process_cpu", True))
        self.process_memory = bool(monitor_cfg.get("process_memory", True))
        self.thread_label = thread_label
        self.thread_count = int(thread_count)
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._benchmark = "startup"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_perf = 0.0
        self._psutil: Any = None
        self._process: Any = None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            psutil.cpu_percent(interval=None, percpu=self.per_cpu)
            if self.process_cpu:
                self._process.cpu_percent(interval=None)
            self._started_perf = time.perf_counter()
            self._thread = threading.Thread(target=self._loop, name="benchmark-monitor", daemon=True)
            self._thread.start()
        except Exception as exc:
            self.enabled = False
            self.error = str(exc)

    def set_benchmark(self, name: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._benchmark = str(name)

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.interval_s))
        self._sample_once(final=True)

    def data(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_s": self.interval_s,
            "thread_label": self.thread_label,
            "thread_count": self.thread_count,
            "sample_count": len(self.samples),
            "error": self.error,
            "samples": self.samples,
        }

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once(final=False)

    def _sample_once(self, final: bool) -> None:
        if self._psutil is None:
            return
        try:
            with self._lock:
                benchmark = self._benchmark
            sample: dict[str, Any] = {
                "t_s": time.perf_counter() - self._started_perf,
                "wall_time": time.time(),
                "thread_label": self.thread_label,
                "thread_count": self.thread_count,
                "benchmark": benchmark,
                "final": bool(final),
            }
            cpu_percent = self._psutil.cpu_percent(interval=None, percpu=self.per_cpu)
            if self.per_cpu:
                sample["cpu_percent_per_cpu"] = list(cpu_percent)
                sample["cpu_percent"] = (
                    float(sum(cpu_percent)) / len(cpu_percent) if cpu_percent else None
                )
            else:
                sample["cpu_percent"] = float(cpu_percent)
            try:
                freq = self._psutil.cpu_freq()
                if freq is not None:
                    sample["cpu_freq_mhz"] = {
                        "current": safe_float(freq.current),
                        "min": safe_float(freq.min),
                        "max": safe_float(freq.max),
                    }
            except Exception:
                pass
            if self.process_cpu and self._process is not None:
                process_percent = safe_float(self._process.cpu_percent(interval=None))
                sample["process_cpu_percent"] = process_percent
                if process_percent is not None:
                    sample["process_cpu_cores"] = process_percent / 100.0
            if self.process_memory and self._process is not None:
                mem = self._process.memory_info()
                sample["process_rss_bytes"] = int(mem.rss)
                sample["process_vms_bytes"] = int(mem.vms)
            try:
                vm = self._psutil.virtual_memory()
                sample["memory_percent"] = safe_float(vm.percent)
                sample["memory_available_bytes"] = int(vm.available)
            except Exception:
                pass
            self.samples.append(sample)
        except Exception as exc:
            self.error = str(exc)


def summarize_monitoring(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def numeric_values(key: str) -> list[float]:
        values = []
        for sample in samples:
            value = sample.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        return values

    cpu = numeric_values("cpu_percent")
    process_cpu = numeric_values("process_cpu_percent")
    process_cores = numeric_values("process_cpu_cores")
    memory = numeric_values("memory_percent")
    rss = numeric_values("process_rss_bytes")
    freq = []
    for sample in samples:
        freq_info = sample.get("cpu_freq_mhz")
        if isinstance(freq_info, dict):
            current = freq_info.get("current")
            if isinstance(current, (int, float)) and math.isfinite(float(current)):
                freq.append(float(current))
    return {
        "samples": len(samples),
        "cpu_avg_percent": sum(cpu) / len(cpu) if cpu else None,
        "cpu_max_percent": max(cpu) if cpu else None,
        "process_cpu_avg_percent": sum(process_cpu) / len(process_cpu) if process_cpu else None,
        "process_cpu_max_percent": max(process_cpu) if process_cpu else None,
        "process_cpu_avg_cores": sum(process_cores) / len(process_cores) if process_cores else None,
        "process_cpu_max_cores": max(process_cores) if process_cores else None,
        "memory_avg_percent": sum(memory) / len(memory) if memory else None,
        "process_rss_max_bytes": max(rss) if rss else None,
        "cpu_freq_avg_mhz": sum(freq) / len(freq) if freq else None,
        "cpu_freq_min_mhz": min(freq) if freq else None,
        "cpu_freq_max_mhz": max(freq) if freq else None,
    }


def run_driver(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.output_dir:
        config["output"]["directory"] = args.output_dir
    output_cfg = config.get("output", {})
    output_dir = Path(str(output_cfg.get("directory", "benchmark_results"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "cpu_python_scientific"))
    system_info = collect_system_info()
    thread_modes = resolve_thread_modes(config, system_info)
    output_paths = cli_output_paths(output_dir, prefix)
    effective_config_path = output_paths["effective_config"]
    write_json(effective_config_path, config)

    print("System hardware")
    print(f"  CPU: {system_info.get('cpu_model')}")
    print(
        "  Cores: "
        f"logical={system_info.get('logical_cpu_count')}, "
        f"physical={system_info.get('physical_cpu_count') or 'unknown'}"
    )
    print(
        "  Memory: "
        f"total={human_bytes(system_info.get('memory_total_bytes'))}, "
        f"available={human_bytes(system_info.get('memory_available_bytes'))}"
    )
    print(f"  Platform: {system_info.get('platform')}")
    print()
    print("Benchmark flow")
    print("  1. Write effective JSON configuration.")
    execution_order = str(
        config.get("benchmark", {}).get("execution_order", "by_benchmark")
    ).strip().lower()
    if execution_order not in {"by_benchmark", "by_thread_mode", "thread_mode", "thread_modes"}:
        execution_order = "by_benchmark"
    print("  2. Launch worker subprocesses before NumPy/SciPy import.")
    print("  3. Run warmups and repeated timings for each enabled module.")
    print(f"  Execution order: {execution_order}")
    print("  4. Aggregate means, standard deviations, speedups, text report, JSON, and PDF plots.")
    print()
    print("Enabled modules")
    for line in benchmark_plan_lines(config):
        print(f"  {line}")
    print()

    names = enabled_benchmark_names(config)
    worker_runs = []
    if execution_order in {"by_thread_mode", "thread_mode", "thread_modes"}:
        for mode in thread_modes:
            label = str(mode["name"])
            threads = int(mode["threads"])
            safe_label = safe_filename(label)
            worker_input = {
                "config": config,
                "thread_label": label,
                "thread_count": threads,
            }
            worker_config_path = output_dir / f"{prefix}_worker_{safe_label}_input.json"
            worker_output_path = output_dir / f"{prefix}_worker_{safe_label}_results.json"
            worker_log_path = output_dir / f"{prefix}_worker_{safe_label}.log"
            write_json(worker_config_path, worker_input)
            env = build_thread_env(threads, output_dir)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-config",
                str(worker_config_path),
                "--worker-output",
                str(worker_output_path),
            ]
            print(f"Running thread mode '{label}' with {format_thread_count(threads)} thread(s)")
            started = time.perf_counter()
            proc = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(Path.cwd()),
            )
            elapsed = time.perf_counter() - started
            worker_log_path.write_text(
                "STDOUT\n"
                + proc.stdout
                + "\nSTDERR\n"
                + proc.stderr
                + f"\nRETURN_CODE {proc.returncode}\n",
                encoding="utf-8",
            )
            if proc.stdout:
                print(proc.stdout.rstrip())
            if proc.returncode != 0:
                print(f"Worker '{label}' failed; see {worker_log_path}")
                worker_runs.append(
                    {
                        "thread_label": label,
                        "thread_count": threads,
                        "elapsed_s": elapsed,
                        "status": "error",
                        "error": proc.stderr,
                        "results": [],
                    }
                )
                continue
            worker_data = read_json(worker_output_path)
            worker_data["elapsed_s"] = elapsed
            worker_runs.append(worker_data)
            print(f"Finished thread mode '{label}' in {format_seconds(elapsed)}")
            print()
    else:
        worker_runs = [
            {
                "status": "ok",
                "thread_label": str(mode["name"]),
                "thread_count": int(mode["threads"]),
                "elapsed_s": 0.0,
                "environment": {},
                "package_info": {},
                "threadpool_info": [],
                "monitoring": {"samples": []},
                "results": [],
            }
            for mode in thread_modes
        ]
        runs_by_label = {str(run["thread_label"]): run for run in worker_runs}
        total_cases = max(1, len(names) * len(thread_modes))
        case_index = 0
        for benchmark_index, name in enumerate(names, start=1):
            for mode in thread_modes:
                case_index += 1
                label = str(mode["name"])
                threads = int(mode["threads"])
                run = runs_by_label[label]
                offset_s = float(run.get("elapsed_s", 0.0) or 0.0)
                safe_label = safe_filename(label)
                item_tag = f"{benchmark_index:02d}_{safe_filename(name)}_{safe_label}"
                worker_input = {
                    "config": config,
                    "thread_label": label,
                    "thread_count": threads,
                    "benchmark_name": name,
                }
                worker_config_path = output_dir / f"{prefix}_worker_{item_tag}_input.json"
                worker_output_path = output_dir / f"{prefix}_worker_{item_tag}_results.json"
                worker_log_path = output_dir / f"{prefix}_worker_{item_tag}.log"
                write_json(worker_config_path, worker_input)
                env = build_thread_env(threads, output_dir)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-one",
                    "--worker-config",
                    str(worker_config_path),
                    "--worker-output",
                    str(worker_output_path),
                ]
                print(
                    f"Running {case_index}/{total_cases}: {name} "
                    f"in thread mode '{label}' with {format_thread_count(threads)} thread(s)"
                )
                started = time.perf_counter()
                proc = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=str(Path.cwd()),
                )
                elapsed = time.perf_counter() - started
                run["elapsed_s"] = offset_s + elapsed
                worker_log_path.write_text(
                    "STDOUT\n"
                    + proc.stdout
                    + "\nSTDERR\n"
                    + proc.stderr
                    + f"\nRETURN_CODE {proc.returncode}\n",
                    encoding="utf-8",
                )
                if proc.stdout:
                    print(proc.stdout.rstrip())
                if proc.returncode != 0 or not worker_output_path.exists():
                    print(f"Worker-one '{label}/{name}' failed; see {worker_log_path}")
                    run["status"] = "error"
                    run["results"].append(
                        {
                            "name": name,
                            "label": BENCHMARK_LABELS.get(name, name),
                            "status": "error",
                            "reason": f"worker-one failed, see {worker_log_path}",
                            "error": proc.stderr,
                            "sizes": copy.deepcopy(config.get("modules", {}).get(name, {})),
                        }
                    )
                    continue
                worker_data = read_json(worker_output_path)
                if not run.get("package_info"):
                    run["package_info"] = worker_data.get("package_info", {})
                if not run.get("threadpool_info"):
                    run["threadpool_info"] = worker_data.get("threadpool_info", [])
                if not run.get("environment"):
                    run["environment"] = worker_data.get("environment", {})
                for sample in worker_data.get("monitoring", {}).get("samples", []):
                    copied = copy.deepcopy(sample)
                    try:
                        copied["t_s"] = offset_s + float(copied.get("t_s", 0.0))
                    except Exception:
                        copied["t_s"] = offset_s
                    run["monitoring"]["samples"].append(copied)
                run["results"].extend(worker_data.get("results", []))
                print(f"Finished {name} / {label} in {format_seconds(elapsed)}")
                print()

    aggregate = {
        "status": "ok",
        "system": system_info,
        "thread_modes": thread_modes,
        "config": config,
        "runs": worker_runs,
    }

    output_files, report_text = write_benchmark_outputs(
        aggregate, output_dir, prefix, output_cfg
    )
    print(report_text)
    if output_files.get("pdf_error"):
        print(f"PDF plot generation skipped: {output_files['pdf_error']}")

    print("Output files")
    print(f"  Effective config: {output_files['effective_config']}")
    if output_files.get("text"):
        print(f"  Text report:      {output_files['text']}")
    if output_files.get("json"):
        print(f"  JSON results:     {output_files['json']}")
    if output_files.get("pdf"):
        print(f"  PDF plots:        {output_files['pdf']}")
    return 0


def cli_output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "effective_config": output_dir / f"{prefix}_effective_config.json",
        "json": output_dir / f"{prefix}_results.json",
        "text": output_dir / f"{prefix}_report.txt",
        "pdf": output_dir / f"{prefix}_plots.pdf",
    }


def write_benchmark_outputs(
    aggregate: dict[str, Any],
    output_dir: Path,
    prefix: str,
    output_cfg: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    apply_auto_thread_regression_guard(aggregate)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = cli_output_paths(output_dir, prefix)
    write_json(paths["effective_config"], aggregate.get("config", {}))
    if bool(output_cfg.get("make_json", True)):
        write_json(paths["json"], aggregate)

    report_text = format_plain_report(aggregate)
    if bool(output_cfg.get("make_text", True)):
        paths["text"].write_text(report_text, encoding="utf-8")

    pdf_error = None
    if bool(output_cfg.get("make_pdf", True)):
        pdf_error = make_pdf_report(aggregate, paths["pdf"])

    output_files: dict[str, Any] = {
        "effective_config": str(paths["effective_config"]),
        "json": str(paths["json"]) if bool(output_cfg.get("make_json", True)) else None,
        "text": str(paths["text"]) if bool(output_cfg.get("make_text", True)) else None,
        "pdf": str(paths["pdf"]) if bool(output_cfg.get("make_pdf", True)) and not pdf_error else None,
        "pdf_error": pdf_error,
    }
    aggregate["output_files"] = output_files
    if bool(output_cfg.get("make_json", True)):
        write_json(paths["json"], aggregate)
    return output_files, report_text


def format_plain_report(aggregate: dict[str, Any]) -> str:
    system_info = aggregate["system"]
    runs = aggregate.get("runs", [])
    config = aggregate["config"]
    lines: list[str] = []
    lines.append("Python Scientific CPU Benchmark Report")
    lines.append("=" * 40)
    lines.append("")
    lines.append("System hardware")
    lines.append(f"  Timestamp: {system_info.get('timestamp')}")
    lines.append(f"  CPU model: {system_info.get('cpu_model')}")
    lines.append(
        "  CPU count: "
        f"logical={system_info.get('logical_cpu_count')}, "
        f"physical={system_info.get('physical_cpu_count') or 'unknown'}"
    )
    lines.append(
        "  Memory: "
        f"total={human_bytes(system_info.get('memory_total_bytes'))}, "
        f"available={human_bytes(system_info.get('memory_available_bytes'))}"
    )
    lines.append(f"  Platform: {system_info.get('platform')}")
    lines.append(f"  Python: {system_info.get('python_version')}")
    lines.append("")

    first_ok_run = next((run for run in runs if run.get("status") == "ok"), None)
    if first_ok_run is not None:
        package_info = first_ok_run.get("package_info", {})
        lines.append("Python numerical packages")
        for key in ("numpy_version", "scipy_version", "numba_version"):
            lines.append(f"  {key}: {package_info.get(key, 'not available')}")
        threadpools = first_ok_run.get("threadpool_info", [])
        if threadpools:
            lines.append("  Threadpool libraries:")
            for lib in threadpools:
                api = lib.get("user_api") or lib.get("internal_api") or "unknown"
                prefix = lib.get("prefix") or lib.get("filepath") or "unknown"
                threads = lib.get("num_threads", "unknown")
                version = lib.get("version", "unknown")
                lines.append(f"    - {api}: {prefix}, threads={threads}, version={version}")
        blas_summary = package_info.get("numpy_show_config_summary")
        if blas_summary:
            lines.append("  NumPy configuration summary:")
            for item in blas_summary:
                lines.append(f"    {item}")
        lines.append("")

    lines.append("Thread validation")
    for run in runs:
        label = run.get("thread_label", "unknown")
        expected = format_thread_count(run.get("thread_count", "unknown"))
        threadpools = run.get("threadpool_info", [])
        pool_parts = []
        for lib in threadpools:
            api = lib.get("user_api") or lib.get("internal_api") or "unknown"
            prefix = lib.get("prefix") or "unknown"
            threads = lib.get("num_threads", "unknown")
            pool_parts.append(f"{api}:{prefix}={threads}")
        samples = run.get("monitoring", {}).get("samples", [])
        summary = summarize_monitoring(samples) if samples else {}
        observed_cores = summary.get("process_cpu_max_cores")
        if observed_cores is None and summary.get("process_cpu_max_percent") is not None:
            observed_cores = float(summary["process_cpu_max_percent"]) / 100.0
        pool_text = ", ".join(pool_parts) if pool_parts else "no threadpoolctl data"
        observed_text = f"{observed_cores:.2f}" if observed_cores is not None else "n/a"
        lines.append(
            f"  {label}: requested={expected}, observed_peak_process_cores~{observed_text}, "
            f"threadpools=[{pool_text}]"
        )
    lines.append("")

    lines.append("Runtime monitoring")
    for run in runs:
        monitoring = run.get("monitoring", {})
        samples = monitoring.get("samples", [])
        summary = summarize_monitoring(samples) if samples else {}
        if not monitoring:
            lines.append(f"  {run.get('thread_label', 'unknown')}: no monitoring data")
            continue
        if monitoring.get("error"):
            lines.append(f"  {run.get('thread_label', 'unknown')}: monitor error: {monitoring.get('error')}")
            continue
        lines.append(
            f"  {run.get('thread_label', 'unknown')}: "
            f"samples={summary.get('samples', 0)}, "
            f"cpu_avg={safe_optional(summary.get('cpu_avg_percent'), '.1f')}%, "
            f"cpu_max={safe_optional(summary.get('cpu_max_percent'), '.1f')}%, "
            f"process_cpu_max={safe_optional(summary.get('process_cpu_max_percent'), '.1f')}%, "
            f"observed_cores~{safe_optional((summary.get('process_cpu_max_percent') or 0) / 100.0 if summary.get('process_cpu_max_percent') is not None else None, '.2f')}, "
            f"freq_avg={safe_optional(summary.get('cpu_freq_avg_mhz'), '.0f')} MHz, "
            f"rss_max={human_bytes(summary.get('process_rss_max_bytes'))}"
        )
    lines.append("")

    benchmark_cfg = config.get("benchmark", {})
    lines.append("Benchmark controls")
    lines.append(f"  Repeats: {benchmark_cfg.get('repeats')}")
    lines.append(f"  Warmups: {benchmark_cfg.get('warmups')}")
    lines.append(f"  Random seed: {benchmark_cfg.get('random_seed')}")
    lines.append("  RNG: NumPy Generator(PCG64), with deterministic per-test derived seeds")
    target_case_s = float(benchmark_cfg.get("target_case_s", 0.0) or 0.0)
    target_repeat_s = float(benchmark_cfg.get("target_repeat_s", 0.0) or 0.0)
    if target_case_s > 0:
        repeats = max(1, int(benchmark_cfg.get("repeats", 1) or 1))
        lines.append(
            f"  Target timed case: {target_case_s} s "
            f"(auto-calibrated call count per case; about {target_case_s / repeats:.2f} s per timed repeat)"
        )
    elif target_repeat_s > 0:
        target_total = target_repeat_s * int(benchmark_cfg.get("repeats", 1) or 1)
        lines.append(
            f"  Target timed repeat: {target_repeat_s} s "
            f"(auto-calibrated, about {target_total:.2f} s total timed repeats per case)"
        )
    else:
        lines.append(
            "  Target timed case: disabled; one call per timed repeat unless inner_loops is set"
        )
    lines.append(f"  Max estimated memory per case: {benchmark_cfg.get('max_memory_gb')} GiB")
    if bool(benchmark_cfg.get("auto_thread_regression_guard", True)):
        factor = float(benchmark_cfg.get("auto_thread_regression_factor", 1.05) or 1.05)
        lines.append(
            f"  Auto thread regression guard: enabled "
            f"(skip auto multi rows slower than {factor:.2f}x single)"
        )
    lines.append(f"  Thread env vars: {', '.join(THREAD_ENV_VARS)}")
    lines.append("")

    lines.append("Enabled benchmark modules")
    lines.extend(f"  {line}" for line in benchmark_plan_lines(config))
    lines.append("")

    lines.append("Results")
    rows = result_rows(aggregate)
    if not rows:
        lines.append("  No benchmark results were produced.")
    else:
        header = (
            f"{'Benchmark':30s} {'Mode':8s} {'Thr':>4s} "
            f"{'Mean/call':>11s} {'Batch':>10s} {'Calls':>7s} "
            f"{'JIT init':>10s} {'Metric':>22s} {'Speedup':>8s}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows:
            if row["status"] != "ok":
                thread_text = format_thread_count(row.get("thread_count", "unknown"))
                lines.append(
                    f"{row['name'][:30]:30s} {row['thread_label'][:8]:8s} "
                    f"{thread_text:>4s} {'skipped/error':>11s} "
                    f"{'':>10s} {'':>7s} {'':>10s} {'':>22s} {'':>8s}"
                )
                lines.append(f"  reason: {row.get('reason') or row.get('error', '')}")
                continue
            metric = ""
            if row.get("metric_name") and row.get("metric_value") is not None:
                metric = f"{row['metric_value']:.3g} {row['metric_name']}"
            compile_text = ""
            if row.get("compile_s") is not None:
                compile_text = format_seconds(row.get("compile_s"))
            speedup = row.get("speedup")
            speedup_text = f"{speedup:.2f}x" if speedup is not None else ""
            thread_text = format_thread_count(row.get("thread_count", "unknown"))
            lines.append(
                f"{row['name'][:30]:30s} {row['thread_label'][:8]:8s} "
                f"{thread_text:>4s} {format_seconds(row.get('mean_s')):>11s} "
                f"{format_seconds(row.get('batch_mean_s', row.get('mean_s'))):>10s} "
                f"{int(row.get('total_calls', row.get('inner_loops', 1)) or 1):7d} {compile_text:>10s} "
                f"{metric[:22]:>22s} {speedup_text:>8s}"
            )
    lines.append("")
    lines.append("Notes")
    lines.append("  - Speedup is computed against the first successful single-thread result with the same benchmark name.")
    lines.append("  - Single mode strictly sets numerical thread environment variables to 1 before importing NumPy/SciPy.")
    lines.append("  - Multi mode defaults to auto: thread environment variables are left unset and libraries choose their own thread counts.")
    lines.append("  - Auto multi rows that are slower than single beyond the guard threshold are skipped from plots; measured values remain in JSON.")
    lines.append("  - Python scalar loops do not use BLAS/OpenMP threads and are included as interpreter overhead baselines.")
    lines.append("  - The Numba prange benchmark uses only scalar arithmetic inside prange; nested NumPy/SciPy/BLAS-style calls are rejected.")
    lines.append("  - Numba JIT init time is reported separately and is not included in the timed runtime mean.")
    lines.append("  - With cache=True, JIT init includes either compilation or cache lookup/loading, so it is not always zero.")
    lines.append("  - Random inputs use PCG64 and fixed derived seeds, so default runs are deterministic for the same sizes.")
    lines.append("  - Mean/call is one kernel call; Calls is the timed call count selected for that case.")
    lines.append("  - With target_case_s > 0, call counts are auto-calibrated per machine/thread mode; compare Mean/call across machines.")
    return "\n".join(lines) + "\n"


AUTO_REGRESSION_RAW_KEYS = (
    "times_s",
    "batch_times_s",
    "mean_s",
    "std_s",
    "min_s",
    "batch_mean_s",
    "batch_std_s",
    "batch_min_s",
    "timed_total_s",
    "metric_name",
    "metric_value",
    "checksum",
    "repeats",
    "warmups",
    "inner_loops",
    "total_calls",
    "target_case_s",
    "target_repeat_s",
)


def apply_auto_thread_regression_guard(aggregate: dict[str, Any]) -> None:
    """Mark auto-thread rows that regress badly versus single as skipped.

    The measured numbers remain in JSON under ``measured_auto_result`` so the
    data is auditable, but plotting/report speedup logic treats these rows as
    skipped. This avoids presenting an OpenBLAS/MKL/Accelerate auto-threading
    regression as a meaningful multithread benchmark result.
    """

    benchmark_cfg = aggregate.get("config", {}).get("benchmark", {})
    if not bool(benchmark_cfg.get("auto_thread_regression_guard", True)):
        return
    try:
        factor = float(benchmark_cfg.get("auto_thread_regression_factor", 1.05) or 1.05)
    except Exception:
        factor = 1.05
    factor = max(1.0, factor)

    single_by_name: dict[str, dict[str, Any]] = {}
    for run in aggregate.get("runs", []):
        label = str(run.get("thread_label", "")).lower()
        thread_count = run.get("thread_count", 0)
        if label != "single" and not (
            not is_auto_thread_count(thread_count) and int(thread_count or 0) == 1
        ):
            continue
        for result in run.get("results", []):
            if result.get("status") != "ok" or result.get("mean_s") is None:
                continue
            single_by_name.setdefault(str(result.get("name", "")), result)

    for run in aggregate.get("runs", []):
        label = str(run.get("thread_label", "")).lower()
        if label != "multi" or not is_auto_thread_count(run.get("thread_count", 0)):
            continue
        for result in run.get("results", []):
            if result.get("auto_thread_regression"):
                continue
            if result.get("status") != "ok" or result.get("mean_s") is None:
                continue
            baseline = single_by_name.get(str(result.get("name", "")))
            if baseline is None or baseline.get("mean_s") is None:
                continue
            try:
                mean_s = float(result["mean_s"])
                single_s = float(baseline["mean_s"])
            except Exception:
                continue
            if mean_s <= 0.0 or single_s <= 0.0:
                continue
            ratio = mean_s / single_s
            if ratio <= factor:
                continue

            result["measured_auto_result"] = {
                key: copy.deepcopy(result.get(key))
                for key in AUTO_REGRESSION_RAW_KEYS
                if key in result
            }
            result["auto_thread_regression"] = True
            result["auto_thread_regression_factor"] = ratio
            result["auto_thread_regression_single_mean_s"] = single_s
            result["status"] = "skipped"
            result["reason"] = (
                f"auto multithread was {ratio:.2f}x slower than single "
                f"({format_seconds(mean_s)} vs {format_seconds(single_s)}); "
                "excluded from timing and speedup plots. Raw measured values are in "
                "measured_auto_result."
            )


def result_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    single_by_name: dict[str, dict[str, Any]] = {}
    for run in aggregate.get("runs", []):
        label = str(run.get("thread_label", ""))
        for result in run.get("results", []):
            row = copy.deepcopy(result)
            row["thread_label"] = label
            row["thread_count"] = int(run.get("thread_count", 0) or 0)
            if result.get("status") == "ok" and (
                label.lower() == "single" or int(run.get("thread_count", 0) or 0) == 1
            ):
                single_by_name.setdefault(result.get("name", ""), row)
            rows.append(row)
    for row in rows:
        baseline = single_by_name.get(row.get("name", ""))
        if (
            baseline is not None
            and row.get("status") == "ok"
            and baseline.get("mean_s")
            and row.get("mean_s")
        ):
            row["speedup"] = float(baseline["mean_s"]) / float(row["mean_s"])
        else:
            row["speedup"] = None
    order = {name: idx for idx, name in enumerate(BENCHMARK_ORDER)}

    def mode_sort_key(row: dict[str, Any]) -> tuple[int, str]:
        label = str(row.get("thread_label", "")).lower()
        threads = int(row.get("thread_count", 0) or 0)
        if label == "single" or threads == 1:
            return (0, label)
        if label == "multi":
            return (1, label)
        return (2, label)

    rows.sort(key=lambda r: (order.get(r.get("name", ""), 999), mode_sort_key(r)))
    return rows


def pdf_label(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def compact_benchmark_label(name: str, width: int = 34) -> str:
    return pdf_label(BENCHMARK_LABELS.get(name, name), width=width)


TIMING_BAR_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")


def shade_plot_rows(ax: Any, count: int) -> None:
    for index in range(count):
        if index % 2:
            ax.axhspan(index - 0.5, index + 0.5, color="#F6F6F6", zorder=0)


def thread_label_sort_key(label: str) -> tuple[int, str]:
    lowered = str(label).lower()
    if lowered == "single":
        return (0, lowered)
    if lowered == "multi":
        return (1, lowered)
    return (2, lowered)


def thread_labels_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {str(row.get("thread_label", "")) for row in rows if row.get("thread_label")},
        key=thread_label_sort_key,
    )


def plot_timing_bars(
    ax: Any,
    rows: list[dict[str, Any]],
    names: list[str] | None = None,
    mode_labels: list[str] | None = None,
    *,
    label_width: int = 30,
) -> None:
    """Draw the shared CLI/UI mean-time timing bar chart."""
    plot_names = list(names) if names is not None else [
        name for name in BENCHMARK_ORDER if any(row.get("name") == name for row in rows)
    ]
    plot_modes = list(mode_labels) if mode_labels is not None else thread_labels_from_rows(rows)
    y_positions = list(range(len(plot_names)))
    height = 0.78 / max(1, len(plot_modes))
    all_means = [
        float(row["mean_s"])
        for row in rows
        if row.get("status") == "ok"
        and row.get("mean_s") is not None
        and float(row["mean_s"]) > 0
    ]
    x_max = max([1.0e-12] + all_means)
    shade_plot_rows(ax, len(plot_names))

    for idx, mode in enumerate(plot_modes):
        means = []
        labels = []
        skipped = []
        for name in plot_names:
            match = next(
                (
                    row
                    for row in rows
                    if row.get("name") == name and str(row.get("thread_label")) == mode
                ),
                None,
            )
            value = (
                float(match["mean_s"])
                if match
                and match.get("status") == "ok"
                and match.get("mean_s") is not None
                else math.nan
            )
            means.append(value)
            labels.append(format_seconds(value) if math.isfinite(value) else "")
            skipped.append(bool(match and match.get("auto_thread_regression")))
        offsets = [y - 0.39 + height / 2.0 + idx * height for y in y_positions]
        bars = ax.barh(
            offsets,
            means,
            height=height,
            label=mode,
            color=TIMING_BAR_COLORS[idx % len(TIMING_BAR_COLORS)],
            zorder=2,
        )
        for bar, label_text in zip(bars, labels):
            value = bar.get_width()
            if math.isfinite(value) and label_text:
                ax.text(
                    value,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"  {label_text}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#222222",
                    clip_on=False,
                    zorder=3,
                )
        for offset, is_skipped in zip(offsets, skipped):
            if is_skipped:
                ax.text(
                    x_max * 0.015,
                    offset,
                    "skipped",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#777777",
                    zorder=3,
                )

    ax.set_xlabel("Mean time per call (s, linear scale; shorter is faster)", fontsize=12)
    ax.set_title("Mean runtime by thread mode", fontsize=15)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([compact_benchmark_label(name, width=label_width) for name in plot_names], fontsize=9)
    ax.set_ylim(len(plot_names) - 0.5, -0.5)
    ax.set_xlim(0.0, x_max * 1.45)
    ax.grid(True, axis="x", alpha=0.25, zorder=1)
    ax.legend(loc="lower right", fontsize=10)


def adjust_timing_figure(fig: Any) -> None:
    fig.subplots_adjust(left=0.36, right=0.98, top=0.91, bottom=0.13)


def plot_speedup_bars(
    ax: Any,
    rows: list[dict[str, Any]],
    names: list[str] | None = None,
    *,
    label_width: int = 30,
) -> None:
    speedup_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("speedup") is not None
        and int(row.get("thread_count", 0) or 0) != 1
    ]
    plot_names = list(names) if names is not None else [
        name for name in BENCHMARK_ORDER if any(row.get("name") == name for row in speedup_rows)
    ]
    values = []
    for name in plot_names:
        match = next((row for row in speedup_rows if row.get("name") == name), None)
        values.append(float(match["speedup"]) if match and match.get("speedup") is not None else math.nan)

    shade_plot_rows(ax, len(plot_names))
    ax.barh(range(len(values)), values, color="#4C78A8", zorder=2)
    ax.axvline(1.0, color="black", linewidth=0.8, zorder=3)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels([compact_benchmark_label(name, width=label_width) for name in plot_names], fontsize=9)
    ax.set_ylim(len(plot_names) - 0.5, -0.5)
    ax.set_xlabel("Speedup vs single-thread", fontsize=12)
    ax.set_title("Multithread speedup", fontsize=15)
    ax.grid(True, axis="x", alpha=0.25, zorder=1)
    for index, value in enumerate(values):
        if math.isfinite(value):
            ax.text(value, index, f" {value:.2f}x", va="center", ha="left", fontsize=9, zorder=3)
    finite_values = [value for value in values if math.isfinite(value)]
    if finite_values:
        ax.set_xlim(0.0, max(1.05, max(finite_values) * 1.18))
    else:
        ax.set_xlim(0.0, 1.05)


def adjust_horizontal_bar_figure(fig: Any) -> None:
    fig.subplots_adjust(left=0.36, right=0.96, top=0.91, bottom=0.13)


def metric_units_by_order(rows: list[dict[str, Any]], names: list[str]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("metric_name") and row.get("metric_value") is not None:
            grouped.setdefault(str(row.get("metric_name")), []).append(row)
    def unit_order(unit: str) -> int:
        indexes = [names.index(row["name"]) for row in grouped[unit] if row.get("name") in names]
        return min(indexes) if indexes else 999

    return sorted(grouped, key=unit_order)


def plot_metric_unit_bars(
    ax: Any,
    rows: list[dict[str, Any]],
    names: list[str] | None = None,
    unit: str | None = None,
    *,
    label_width: int = 30,
    title: str | None = None,
) -> None:
    metric_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("metric_name")
        and row.get("metric_value") is not None
    ]
    if unit is None and metric_rows:
        unit = str(metric_rows[0].get("metric_name"))
    unit_rows = [
        row
        for row in metric_rows
        if unit is not None and str(row.get("metric_name")) == unit
    ]
    plot_names = list(names) if names is not None else [
        name for name in BENCHMARK_ORDER if any(row.get("name") == name for row in unit_rows)
    ]
    selected_by_name = {str(row.get("name")): row for row in unit_rows}
    values = [
        float(selected_by_name[name]["metric_value"]) if name in selected_by_name else math.nan
        for name in plot_names
    ]

    shade_plot_rows(ax, len(plot_names))
    ax.barh(range(len(values)), values, color="#54A24B", zorder=2)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels([compact_benchmark_label(name, width=label_width) for name in plot_names], fontsize=9)
    ax.set_ylim(len(plot_names) - 0.5, -0.5)
    ax.set_xlabel(unit or "Throughput metric", fontsize=12)
    ax.set_title(title or (f"Throughput metrics ({unit})" if unit else "Throughput metrics"), fontsize=15)
    ax.grid(True, axis="x", alpha=0.25, zorder=1)
    for index, value in enumerate(values):
        if math.isfinite(value):
            ax.text(value, index, f" {value:.3g}", va="center", ha="left", fontsize=9, zorder=3)
    finite_values = [value for value in values if math.isfinite(value)]
    if finite_values:
        ax.set_xlim(0.0, max(finite_values) * 1.25)
    else:
        ax.set_xlim(0.0, 1.0)


def plot_monitor_run(ax: Any, run: dict[str, Any], logical_count: int) -> None:
    samples = run.get("monitoring", {}).get("samples", [])
    times = [float(sample.get("t_s", 0.0)) for sample in samples]
    cpu = [
        float(sample["cpu_percent"]) if isinstance(sample.get("cpu_percent"), (int, float)) else math.nan
        for sample in samples
    ]
    process_cpu = [
        float(sample["process_cpu_percent"]) / max(1, logical_count)
        if isinstance(sample.get("process_cpu_percent"), (int, float))
        else math.nan
        for sample in samples
    ]
    freq = []
    for sample in samples:
        freq_info = sample.get("cpu_freq_mhz")
        current = freq_info.get("current") if isinstance(freq_info, dict) else None
        freq.append(float(current) if isinstance(current, (int, float)) else math.nan)

    ax.plot(times, cpu, label="system CPU %", color="#4C78A8", linewidth=1.8)
    if any(math.isfinite(value) for value in process_cpu):
        ax.plot(times, process_cpu, label="process CPU % / logical cores", color="#F58518", linewidth=1.6)
    ax.set_xlabel("Time in worker (s)", fontsize=12)
    ax.set_ylabel("CPU utilization (%)", fontsize=12)
    ax.set_ylim(bottom=0)
    thread_text = format_thread_count(run.get("thread_count"))
    ax.set_title(
        f"Runtime monitoring: {run.get('thread_label')} ({thread_text} threads)",
        fontsize=15,
    )
    ax.grid(True, alpha=0.25)
    if any(math.isfinite(value) for value in freq):
        ax2 = ax.twinx()
        ax2.plot(times, freq, label="CPU frequency MHz", color="#54A24B", alpha=0.8)
        ax2.set_ylabel("CPU frequency (MHz)", fontsize=12)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
    else:
        ax.legend(loc="upper right", fontsize=10)

    previous = None
    for sample in samples:
        benchmark = sample.get("benchmark")
        if benchmark != previous:
            ax.axvline(float(sample.get("t_s", 0.0)), color="black", linewidth=0.6, alpha=0.18)
            previous = benchmark


def adjust_monitor_figure(fig: Any) -> None:
    fig.subplots_adjust(left=0.12, right=0.88, top=0.91, bottom=0.14)


def make_pdf_report(aggregate: dict[str, Any], pdf_path: Path) -> str | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(pdf_path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:
        return str(exc)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
        }
    )

    rows = result_rows(aggregate)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return "no successful benchmark rows"
    names = [name for name in BENCHMARK_ORDER if any(row["name"] == name for row in rows)]

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        report = format_plain_report(aggregate)
        short_report = "\n".join(report.splitlines()[:38])
        ax.text(
            0.02,
            0.98,
            short_report,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        plot_timing_bars(ax, rows, names=names, label_width=30)
        adjust_timing_figure(fig)
        pdf.savefig(fig)
        plt.close(fig)

        speedup_rows = [
            row
            for row in ok_rows
            if row.get("speedup") is not None and int(row.get("thread_count", 0)) != 1
        ]
        if speedup_rows:
            fig, ax = plt.subplots(figsize=(11, 7))
            plot_speedup_bars(ax, ok_rows, names=names, label_width=30)
            adjust_horizontal_bar_figure(fig)
            pdf.savefig(fig)
            plt.close(fig)

        metric_rows = [
            row
            for row in ok_rows
            if row.get("metric_name")
            and row.get("metric_value") is not None
            and str(row.get("thread_label", "")).lower() != "single"
        ]
        if metric_rows:
            metric_units = metric_units_by_order(metric_rows, names)
            for start in range(0, len(metric_units), 3):
                units = metric_units[start : start + 3]
                fig, axes = plt.subplots(len(units), 1, figsize=(11, 8.5), squeeze=False)
                fig.suptitle("Throughput metrics grouped by unit", fontsize=15)
                for ax, unit in zip([item for row_axes in axes for item in row_axes], units):
                    unit_names = [
                        name
                        for name in names
                        if any(row.get("name") == name and str(row.get("metric_name")) == unit for row in metric_rows)
                    ]
                    plot_metric_unit_bars(
                        ax,
                        metric_rows,
                        names=unit_names,
                        unit=unit,
                        label_width=30,
                        title=f"Throughput metrics ({unit})",
                    )
                fig.subplots_adjust(left=0.36, right=0.95, top=0.90, bottom=0.08, hspace=0.62)
                pdf.savefig(fig)
                plt.close(fig)

        logical_count = int(aggregate.get("system", {}).get("logical_cpu_count") or os.cpu_count() or 1)
        for run in aggregate.get("runs", []):
            monitoring = run.get("monitoring", {})
            samples = monitoring.get("samples", [])
            if not samples:
                continue
            fig, ax = plt.subplots(figsize=(11, 7.5))
            plot_monitor_run(ax, run, logical_count)
            adjust_monitor_figure(fig)
            pdf.savefig(fig)
            plt.close(fig)
    return None


def run_worker(args: argparse.Namespace) -> int:
    worker_input = read_json(Path(args.worker_config))
    config = worker_input["config"]
    thread_label = str(worker_input["thread_label"])
    thread_count = int(worker_input["thread_count"])
    configure_process_thread_env(thread_count)

    import numpy as np

    scipy_module = None
    scipy_sparse = None
    scipy_sparse_linalg = None
    scipy_fft = None
    scipy_special = None
    scipy_signal = None
    scipy_ndimage = None
    scipy_spatial_distance = None
    try:
        import scipy as scipy_module  # type: ignore
        import scipy.fft as scipy_fft  # type: ignore
        import scipy.ndimage as scipy_ndimage  # type: ignore
        import scipy.signal as scipy_signal  # type: ignore
        import scipy.sparse as scipy_sparse  # type: ignore
        import scipy.sparse.linalg as scipy_sparse_linalg  # type: ignore
        import scipy.spatial.distance as scipy_spatial_distance  # type: ignore
        import scipy.special as scipy_special  # type: ignore
    except Exception:
        scipy_module = None

    numba_module = None
    try:
        import numba as numba_module  # type: ignore

        if not is_auto_thread_count(thread_count):
            try:
                numba_module.set_num_threads(thread_count)
            except Exception:
                pass
        globals()["_numba_prange"] = numba_module.prange
    except Exception:
        numba_module = None

    threadpool_limits_cm = contextlib.nullcontext()
    threadpool_info_func = None
    if bool(config.get("benchmark", {}).get("enforce_threadpoolctl", True)):
        try:
            from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore

            threadpool_info_func = threadpool_info
            if not is_auto_thread_count(thread_count):
                threadpool_limits_cm = threadpool_limits(limits=thread_count)
        except Exception:
            threadpool_limits_cm = contextlib.nullcontext()
    else:
        try:
            from threadpoolctl import threadpool_info  # type: ignore

            threadpool_info_func = threadpool_info
        except Exception:
            threadpool_info_func = None

    with threadpool_limits_cm:
        package_info = collect_package_info(np, scipy_module, numba_module)
        results = []
        started = time.perf_counter()
        names = enabled_benchmark_names(config)
        monitor = RuntimeMonitor(config, thread_label, thread_count)
        monitor.start()
        try:
            for index, name in enumerate(names, start=1):
                monitor.set_benchmark(name)
                print(f"[{thread_label}] {index}/{len(names)} {name}")
                result = run_one_benchmark(
                    name,
                    config,
                    np,
                    scipy_module,
                    scipy_sparse,
                    scipy_sparse_linalg,
                    scipy_fft,
                    scipy_special,
                    scipy_signal,
                    scipy_ndimage,
                    scipy_spatial_distance,
                    numba_module,
                    thread_count,
                )
                results.append(result)
                if result.get("status") == "ok":
                    compile_text = ""
                    if result.get("compile_s") is not None:
                        compile_text = f" jit_init={format_seconds(result.get('compile_s'))}"
                    print(
                        f"[{thread_label}]   mean={format_seconds(result.get('mean_s'))} "
                        f"timed_total={format_seconds(result.get('timed_total_s'))} "
                        f"calls={result.get('total_calls', result.get('inner_loops', 1))} "
                        f"metric={result.get('metric_value')} {result.get('metric_name') or ''}"
                        f"{compile_text}"
                    )
                else:
                    print(f"[{thread_label}]   {result.get('status')}: {result.get('reason')}")
        finally:
            monitor.set_benchmark("finished")
            monitor.stop()
        elapsed = time.perf_counter() - started
        try:
            _ = np.ones((8, 8)) @ np.ones((8, 8))
            threadpool_info = threadpool_info_func() if threadpool_info_func else []
        except Exception:
            threadpool_info = []
        monitoring = monitor.data()

    worker_output = {
        "status": "ok",
        "thread_label": thread_label,
        "thread_count": thread_count,
        "elapsed_s": elapsed,
        "environment": {var: os.environ.get(var) for var in THREAD_ENV_VARS},
        "package_info": package_info,
        "threadpool_info": threadpool_info,
        "monitoring": monitoring,
        "results": results,
    }
    write_json(Path(args.worker_output), worker_output)
    return 0


def run_worker_one(args: argparse.Namespace) -> int:
    worker_input = read_json(Path(args.worker_config))
    config = worker_input["config"]
    thread_label = str(worker_input["thread_label"])
    thread_count = int(worker_input["thread_count"])
    benchmark_name = str(worker_input["benchmark_name"])
    configure_process_thread_env(thread_count)

    import numpy as np

    scipy_module = None
    scipy_sparse = None
    scipy_sparse_linalg = None
    scipy_fft = None
    scipy_special = None
    scipy_signal = None
    scipy_ndimage = None
    scipy_spatial_distance = None
    try:
        import scipy as scipy_module  # type: ignore
        import scipy.fft as scipy_fft  # type: ignore
        import scipy.ndimage as scipy_ndimage  # type: ignore
        import scipy.signal as scipy_signal  # type: ignore
        import scipy.sparse as scipy_sparse  # type: ignore
        import scipy.sparse.linalg as scipy_sparse_linalg  # type: ignore
        import scipy.spatial.distance as scipy_spatial_distance  # type: ignore
        import scipy.special as scipy_special  # type: ignore
    except Exception:
        scipy_module = None

    numba_module = None
    try:
        import numba as numba_module  # type: ignore

        if not is_auto_thread_count(thread_count):
            try:
                numba_module.set_num_threads(thread_count)
            except Exception:
                pass
        globals()["_numba_prange"] = numba_module.prange
    except Exception:
        numba_module = None

    threadpool_limits_cm = contextlib.nullcontext()
    threadpool_info_func = None
    if bool(config.get("benchmark", {}).get("enforce_threadpoolctl", True)):
        try:
            from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore

            threadpool_info_func = threadpool_info
            if not is_auto_thread_count(thread_count):
                threadpool_limits_cm = threadpool_limits(limits=thread_count)
        except Exception:
            threadpool_limits_cm = contextlib.nullcontext()
    else:
        try:
            from threadpoolctl import threadpool_info  # type: ignore

            threadpool_info_func = threadpool_info
        except Exception:
            threadpool_info_func = None

    with threadpool_limits_cm:
        package_info = collect_package_info(np, scipy_module, numba_module)
        started = time.perf_counter()
        monitor = RuntimeMonitor(config, thread_label, thread_count)
        monitor.set_benchmark(benchmark_name)
        monitor.start()
        try:
            print(f"[{thread_label}] {benchmark_name}")
            result = run_one_benchmark(
                benchmark_name,
                config,
                np,
                scipy_module,
                scipy_sparse,
                scipy_sparse_linalg,
                scipy_fft,
                scipy_special,
                scipy_signal,
                scipy_ndimage,
                scipy_spatial_distance,
                numba_module,
                thread_count,
            )
            if result.get("status") == "ok":
                compile_text = ""
                if result.get("compile_s") is not None:
                    compile_text = f" jit_init={format_seconds(result.get('compile_s'))}"
                print(
                    f"[{thread_label}]   mean={format_seconds(result.get('mean_s'))} "
                    f"timed_total={format_seconds(result.get('timed_total_s'))} "
                    f"calls={result.get('total_calls', result.get('inner_loops', 1))} "
                    f"metric={result.get('metric_value')} {result.get('metric_name') or ''}"
                    f"{compile_text}"
                )
            else:
                print(f"[{thread_label}]   {result.get('status')}: {result.get('reason')}")
        finally:
            monitor.set_benchmark("finished")
            monitor.stop()
        elapsed = time.perf_counter() - started
        try:
            _ = np.ones((8, 8)) @ np.ones((8, 8))
            threadpool_info = threadpool_info_func() if threadpool_info_func else []
        except Exception:
            threadpool_info = []
        monitoring = monitor.data()

    worker_output = {
        "status": "ok",
        "thread_label": thread_label,
        "thread_count": thread_count,
        "benchmark_name": benchmark_name,
        "elapsed_s": elapsed,
        "environment": {var: os.environ.get(var) for var in THREAD_ENV_VARS},
        "package_info": package_info,
        "threadpool_info": threadpool_info,
        "monitoring": monitoring,
        "results": [result],
    }
    write_json(Path(args.worker_output), worker_output)
    return 0


def collect_package_info(np: Any, scipy_module: Any, numba_module: Any) -> dict[str, Any]:
    info = {
        "numpy_version": getattr(np, "__version__", "unknown"),
        "scipy_version": getattr(scipy_module, "__version__", None)
        if scipy_module is not None
        else None,
        "numba_version": getattr(numba_module, "__version__", None)
        if numba_module is not None
        else None,
        "numpy_show_config_summary": [],
    }
    if numba_module is not None:
        try:
            info["numba_num_threads"] = int(numba_module.get_num_threads())
            info["numba_threading_layer"] = str(numba_module.threading_layer())
        except Exception:
            pass
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            np.show_config()
        lines = [line.rstrip() for line in buffer.getvalue().splitlines() if line.strip()]
        info["numpy_show_config_summary"] = lines[:16]
    except Exception as exc:
        info["numpy_show_config_error"] = str(exc)
    return info


def run_one_benchmark(
    name: str,
    config: dict[str, Any],
    np: Any,
    scipy_module: Any,
    scipy_sparse: Any,
    scipy_sparse_linalg: Any,
    scipy_fft: Any,
    scipy_special: Any,
    scipy_signal: Any,
    scipy_ndimage: Any,
    scipy_spatial_distance: Any,
    numba_module: Any,
    thread_count: int,
) -> dict[str, Any]:
    module_cfg = config.get("modules", {}).get(name, {})
    global_cfg = config.get("benchmark", {})
    try:
        if name == "dense_matmul_float64":
            return bench_dense_matmul_float64(name, module_cfg, global_cfg, np)
        if name == "dense_matmul_complex128":
            return bench_dense_matmul_complex128(name, module_cfg, global_cfg, np)
        if name == "dense_hermitian_eigvalsh":
            return bench_dense_hermitian_eigvalsh(name, module_cfg, global_cfg, np)
        if name == "dense_nonhermitian_eigvals":
            return bench_dense_nonhermitian_eigvals(name, module_cfg, global_cfg, np)
        if name == "dense_svd":
            return bench_dense_svd(name, module_cfg, global_cfg, np)
        if name == "dense_cholesky":
            return bench_dense_cholesky(name, module_cfg, global_cfg, np)
        if name == "dense_qr":
            return bench_dense_qr(name, module_cfg, global_cfg, np)
        if name == "linear_solve":
            return bench_linear_solve(name, module_cfg, global_cfg, np)
        if name == "least_squares_lstsq":
            return bench_least_squares_lstsq(name, module_cfg, global_cfg, np)
        if name == "fft_3d_complex128":
            return bench_fft_3d_complex128(
                name, module_cfg, global_cfg, np, scipy_fft, thread_count
            )
        if name == "einsum_tensor_contraction":
            return bench_einsum_tensor_contraction(name, module_cfg, global_cfg, np)
        if name == "vectorized_elementwise":
            return bench_vectorized_elementwise(name, module_cfg, global_cfg, np)
        if name == "reduction_norm":
            return bench_reduction_norm(name, module_cfg, global_cfg, np)
        if name == "sort_argsort":
            return bench_sort_argsort(name, module_cfg, global_cfg, np)
        if name == "python_loop":
            return bench_python_loop(name, module_cfg, global_cfg)
        if name == "numba_njit_loop":
            return bench_numba_njit_loop(name, module_cfg, global_cfg, numba_module)
        if name == "numba_prange_loop":
            return bench_numba_prange_loop(
                name, module_cfg, global_cfg, numba_module, thread_count
            )
        if name == "sparse_matvec_csr":
            return bench_sparse_matvec_csr(name, module_cfg, global_cfg, np, scipy_sparse)
        if name == "sparse_eigsh":
            return bench_sparse_eigsh(
                name, module_cfg, global_cfg, np, scipy_sparse, scipy_sparse_linalg
            )
        if name == "scipy_signal_fftconvolve":
            return bench_scipy_signal_fftconvolve(name, module_cfg, global_cfg, np, scipy_signal)
        if name == "scipy_ndimage_gaussian_filter":
            return bench_scipy_ndimage_gaussian_filter(name, module_cfg, global_cfg, np, scipy_ndimage)
        if name == "scipy_distance_cdist":
            return bench_scipy_distance_cdist(name, module_cfg, global_cfg, np, scipy_spatial_distance)
        return skipped_result(name, "unknown benchmark module", module_cfg)
    except MemoryError:
        return skipped_result(name, "MemoryError during allocation or execution", module_cfg)
    except Exception as exc:
        return {
            "name": name,
            "label": BENCHMARK_LABELS.get(name, name),
            "status": "error",
            "reason": str(exc),
            "traceback": traceback.format_exc(),
            "sizes": dict(module_cfg),
        }


def skipped_result(name: str, reason: str, module_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "label": BENCHMARK_LABELS.get(name, name),
        "status": "skipped",
        "reason": reason,
        "sizes": dict(module_cfg),
    }


def data_seed_for(global_cfg: dict[str, Any], name: str, salt: str = "") -> int:
    base_seed = int(global_cfg.get("random_seed", 0))
    token = f"{name}:{salt}".encode("utf-8")
    offset = zlib.crc32(token) % 1_000_000_000
    return (base_seed + offset) & ((1 << 63) - 1)


def rng_for(np: Any, global_cfg: dict[str, Any], name: str, salt: str = "") -> Any:
    seed = data_seed_for(global_cfg, name, salt=salt)
    return np.random.Generator(np.random.PCG64(seed))


def repeats_for(module_cfg: dict[str, Any], global_cfg: dict[str, Any]) -> tuple[int, int]:
    repeats = int(module_cfg.get("repeats", global_cfg.get("repeats", 3)))
    warmups = int(module_cfg.get("warmups", global_cfg.get("warmups", 1)))
    return max(1, repeats), max(0, warmups)


def memory_limit_bytes(global_cfg: dict[str, Any]) -> float | None:
    max_gb = global_cfg.get("max_memory_gb", None)
    if max_gb is None:
        return None
    max_gb = float(max_gb)
    if max_gb <= 0:
        return None
    return max_gb * (1024.0**3)


def check_memory_or_skip(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], estimate: float) -> dict[str, Any] | None:
    limit = memory_limit_bytes(global_cfg)
    if limit is not None and estimate > limit:
        result = skipped_result(
            name,
            f"estimated working set {human_bytes(estimate)} exceeds max_memory_gb={global_cfg.get('max_memory_gb')}",
            module_cfg,
        )
        result["memory_estimate_bytes"] = int(estimate)
        return result
    return None


def timed_result(
    name: str,
    module_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    sizes: dict[str, Any],
    memory_estimate: float,
    func: Any,
    metric_name: str | None = None,
    metric_work: float | None = None,
) -> dict[str, Any]:
    repeats, warmups = repeats_for(module_cfg, global_cfg)
    size_info = dict(sizes)
    size_info.setdefault("data_seed", data_seed_for(global_cfg, name))
    size_info.setdefault("rng_algorithm", "numpy.random.PCG64")
    for _ in range(warmups):
        func()
    target_case_s_value = module_cfg.get("target_case_s", global_cfg.get("target_case_s", None))
    target_case_s = (
        float(target_case_s_value)
        if target_case_s_value is not None and float(target_case_s_value) > 0
        else 0.0
    )
    target_repeat_s = float(module_cfg.get("target_repeat_s", global_cfg.get("target_repeat_s", 0.0)) or 0.0)
    max_inner_loops = int(
        module_cfg.get(
            "calibration_max_inner_loops",
            global_cfg.get("calibration_max_inner_loops", 100_000),
        )
    )
    explicit_inner_loops = module_cfg.get("inner_loops")
    if explicit_inner_loops is not None:
        inner_loops = max(1, int(explicit_inner_loops))
    elif target_case_s > 0 or target_repeat_s > 0:
        if bool(global_cfg.get("gc_between_repeats", True)):
            gc.collect()
        calibration_started = time.perf_counter()
        func()
        calibration_elapsed = time.perf_counter() - calibration_started
        target_batch_s = target_case_s / repeats if target_case_s > 0 else target_repeat_s
        if calibration_elapsed > 0:
            inner_loops = max(1, int(math.ceil(target_batch_s / calibration_elapsed)))
        else:
            inner_loops = 1
        inner_loops = min(inner_loops, max(1, max_inner_loops))
    else:
        inner_loops = 1
    batch_times = []
    checksum = None
    for _ in range(repeats):
        if bool(global_cfg.get("gc_between_repeats", True)):
            gc.collect()
        started = time.perf_counter()
        for _inner in range(inner_loops):
            checksum = func()
        elapsed = time.perf_counter() - started
        batch_times.append(float(elapsed))
    per_call_times = [value / inner_loops for value in batch_times]
    mean = sum(per_call_times) / len(per_call_times)
    batch_mean = sum(batch_times) / len(batch_times)
    if len(per_call_times) > 1:
        variance = sum((value - mean) ** 2 for value in per_call_times) / (len(per_call_times) - 1)
        std = math.sqrt(variance)
        batch_variance = sum((value - batch_mean) ** 2 for value in batch_times) / (len(batch_times) - 1)
        batch_std = math.sqrt(batch_variance)
    else:
        std = 0.0
        batch_std = 0.0
    metric_value = None
    if metric_name and metric_work is not None and mean > 0:
        metric_value = float(metric_work) / mean
    return {
        "name": name,
        "label": BENCHMARK_LABELS.get(name, name),
        "status": "ok",
        "sizes": size_info,
        "memory_estimate_bytes": int(memory_estimate),
        "repeats": repeats,
        "warmups": warmups,
        "inner_loops": inner_loops,
        "total_calls": inner_loops * repeats,
        "target_case_s": target_case_s,
        "target_repeat_s": target_repeat_s,
        "times_s": per_call_times,
        "batch_times_s": batch_times,
        "mean_s": mean,
        "std_s": std,
        "min_s": min(per_call_times),
        "batch_mean_s": batch_mean,
        "batch_std_s": batch_std,
        "batch_min_s": min(batch_times),
        "timed_total_s": sum(batch_times),
        "checksum": safe_float(checksum),
        "metric_name": metric_name,
        "metric_value": metric_value,
    }


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if math.isfinite(result):
        return result
    return None


def safe_optional(value: Any, fmt: str = ".3g") -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    return format(number, fmt)


def random_float64(np: Any, rng: Any, shape: tuple[int, ...]) -> Any:
    return rng.standard_normal(shape).astype(np.float64, copy=False)


def random_complex128(np: Any, rng: Any, shape: tuple[int, ...]) -> Any:
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag).astype(np.complex128, copy=False)


def bench_dense_matmul_float64(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 1536))
    estimate = 3.0 * n * n * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    a = random_float64(np, rng, (n, n))
    b = random_float64(np, rng, (n, n))

    def func() -> float:
        c = a @ b
        return float(c[0, 0])

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "dtype": "float64"},
        estimate,
        func,
        "GFLOP/s",
        2.0 * n**3 / 1.0e9,
    )


def bench_dense_matmul_complex128(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 768))
    estimate = 3.0 * n * n * 16
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    a = random_complex128(np, rng, (n, n))
    b = random_complex128(np, rng, (n, n))

    def func() -> float:
        c = a @ b
        return float(c[0, 0].real)

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "dtype": "complex128"},
        estimate,
        func,
        "approx GFLOP/s",
        8.0 * n**3 / 1.0e9,
    )


def bench_dense_hermitian_eigvalsh(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 768))
    estimate = 2.5 * n * n * 16
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    x = random_complex128(np, rng, (n, n)) / math.sqrt(max(1, n))
    h = 0.5 * (x + x.conj().T)
    h[np.diag_indices(n)] += np.linspace(0.0, 1.0, n)

    def func() -> float:
        values = np.linalg.eigvalsh(h)
        return float(values[0])

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "dtype": "complex128"},
        estimate,
        func,
        "N^3/s",
        float(n**3),
    )


def bench_dense_nonhermitian_eigvals(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 448))
    estimate = 3.0 * n * n * 16
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    a = random_complex128(np, rng, (n, n)) / math.sqrt(max(1, n))

    def func() -> float:
        values = np.linalg.eigvals(a)
        return float(values[0].real)

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "dtype": "complex128"},
        estimate,
        func,
        "N^3/s",
        float(n**3),
    )


def bench_dense_svd(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 640))
    estimate = 2.5 * n * n * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    a = random_float64(np, rng, (n, n)) / math.sqrt(max(1, n))

    def func() -> float:
        values = np.linalg.svd(a, compute_uv=False, full_matrices=False)
        return float(values[0])

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "dtype": "float64", "compute_uv": False},
        estimate,
        func,
        "N^3/s",
        float(n**3),
    )


def bench_dense_cholesky(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 1400))
    estimate = 3.5 * n * n * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    x = random_float64(np, rng, (n, n)) / math.sqrt(max(1, n))
    a = x @ x.T
    a[np.diag_indices(n)] += 0.1 + n * 1.0e-3

    def func() -> float:
        factor = np.linalg.cholesky(a)
        return float(factor[0, 0])

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "dtype": "float64"},
        estimate,
        func,
        "GFLOP/s",
        (1.0 / 3.0) * n**3 / 1.0e9,
    )


def bench_dense_qr(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 1200))
    estimate = 3.0 * n * n * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    a = random_float64(np, rng, (n, n)) / math.sqrt(max(1, n))

    def func() -> float:
        q, r = np.linalg.qr(a, mode="reduced")
        return float(r[0, 0] + q[0, 0])

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "dtype": "float64", "mode": "reduced"},
        estimate,
        func,
        "GFLOP/s",
        (4.0 / 3.0) * n**3 / 1.0e9,
    )


def bench_linear_solve(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 1024))
    nrhs = int(module_cfg.get("nrhs", 16))
    estimate = (2.5 * n * n + 2.0 * n * nrhs) * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    x = random_float64(np, rng, (n, n)) / math.sqrt(max(1, n))
    a = x @ x.T
    a[np.diag_indices(n)] += 0.1 + n * 1.0e-3
    b = random_float64(np, rng, (n, nrhs))

    def func() -> float:
        y = np.linalg.solve(a, b)
        return float(y[0, 0])

    flops = (2.0 / 3.0) * n**3 + 2.0 * n * n * nrhs
    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "nrhs": nrhs, "dtype": "float64"},
        estimate,
        func,
        "GFLOP/s",
        flops / 1.0e9,
    )


def bench_least_squares_lstsq(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    m = int(module_cfg.get("m", 2400))
    n = int(module_cfg.get("n", 700))
    nrhs = int(module_cfg.get("nrhs", 8))
    if m < n:
        raise ValueError("least_squares_lstsq requires m >= n")
    estimate = (2.5 * m * n + 2.0 * n * nrhs + m * nrhs) * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    a = random_float64(np, rng, (m, n)) / math.sqrt(max(1, n))
    b = random_float64(np, rng, (m, nrhs))

    def func() -> float:
        x, residuals, rank, singular = np.linalg.lstsq(a, b, rcond=None)
        return float(x[0, 0] + rank + singular[0])

    flops = 2.0 * m * n * n + 2.0 * m * n * nrhs
    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"m": m, "n": n, "nrhs": nrhs, "dtype": "float64"},
        estimate,
        func,
        "GFLOP/s",
        flops / 1.0e9,
    )


def bench_fft_3d_complex128(
    name: str,
    module_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    np: Any,
    scipy_fft: Any,
    thread_count: int,
) -> dict[str, Any]:
    shape = tuple(int(v) for v in module_cfg.get("shape", [160, 160, 96]))
    if len(shape) != 3:
        raise ValueError("fft_3d_complex128 shape must contain exactly 3 integers")
    size = int(math.prod(shape))
    estimate = 2.5 * size * 16
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    x = random_complex128(np, rng, shape)
    backend = str(module_cfg.get("backend", "scipy_if_available"))
    use_scipy = scipy_fft is not None and backend in {"scipy", "scipy_if_available"}
    workers = (
        thread_count
        if bool(module_cfg.get("use_scipy_workers", False)) and not is_auto_thread_count(thread_count)
        else None
    )

    def func() -> float:
        if use_scipy:
            if workers is None:
                y = scipy_fft.fftn(x)
            else:
                y = scipy_fft.fftn(x, workers=workers)
        else:
            y = np.fft.fftn(x)
        return float(y.flat[0].real)

    work = 5.0 * size * math.log2(max(2, size))
    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {
            "shape": list(shape),
            "dtype": "complex128",
            "backend": "scipy.fft" if use_scipy else "numpy.fft",
            "workers": workers if use_scipy else None,
        },
        estimate,
        func,
        "approx GFLOP/s",
        work / 1.0e9,
    )


def bench_einsum_tensor_contraction(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    a_dim = int(module_cfg.get("a", 28))
    b_dim = int(module_cfg.get("b", 28))
    c_dim = int(module_cfg.get("c", 96))
    d_dim = int(module_cfg.get("d", 32))
    e_dim = int(module_cfg.get("e", 20))
    a_size = a_dim * b_dim * c_dim
    b_size = c_dim * d_dim * e_dim
    out_size = a_dim * b_dim * d_dim * e_dim
    estimate = (a_size + b_size + out_size) * 8 * 1.5
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    left = random_float64(np, rng, (a_dim, b_dim, c_dim))
    right = random_float64(np, rng, (c_dim, d_dim, e_dim))
    expression = "abc,cde->abde"

    def func() -> float:
        out = np.einsum(expression, left, right, optimize=True)
        return float(out.flat[0])

    flops = 2.0 * a_dim * b_dim * c_dim * d_dim * e_dim
    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {
            "expression": expression,
            "a": a_dim,
            "b": b_dim,
            "c": c_dim,
            "d": d_dim,
            "e": e_dim,
            "dtype": "float64",
        },
        estimate,
        func,
        "GFLOP/s",
        flops / 1.0e9,
    )


def bench_vectorized_elementwise(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 8_000_000))
    estimate = 5.0 * n * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    a = random_float64(np, rng, (n,))
    b = random_float64(np, rng, (n,))
    c = random_float64(np, rng, (n,))

    def func() -> float:
        y = np.sin(a) * np.exp(-0.1 * b) + np.sqrt(np.abs(c) + 1.0e-12)
        return float(y[0])

    bytes_moved = 4.0 * n * 8
    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "dtype": "float64"},
        estimate,
        func,
        "GB/s",
        bytes_moved / 1.0e9,
    )


def bench_reduction_norm(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 16_000_000))
    estimate = 2.5 * n * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    x = random_float64(np, rng, (n,))

    def func() -> float:
        return float(np.sum(x) + np.linalg.norm(x))

    bytes_moved = 3.0 * n * 8
    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "dtype": "float64"},
        estimate,
        func,
        "GB/s",
        bytes_moved / 1.0e9,
    )


def bench_sort_argsort(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any) -> dict[str, Any]:
    n = int(module_cfg.get("n", 4_000_000))
    estimate = (n * 8 + n * 8) * 2.0
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    x = random_float64(np, rng, (n,))

    def func() -> float:
        order = np.argsort(x.copy(), kind="quicksort")
        return float(order[0])

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "dtype": "float64", "kind": "quicksort"},
        estimate,
        func,
        "Mitem/s",
        n / 1.0e6,
    )


def bench_python_loop(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any]) -> dict[str, Any]:
    iterations = int(module_cfg.get("iterations", 8_000_000))
    estimate = 1024.0
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped

    def func() -> float:
        total = 0.0
        for i in range(iterations):
            total += (i % 1009) * 1.0000001 - (i % 97) * 0.5
        return total

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"iterations": iterations},
        estimate,
        func,
        "Miter/s",
        iterations / 1.0e6,
    )


def bench_numba_njit_loop(
    name: str,
    module_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    numba_module: Any,
) -> dict[str, Any]:
    if numba_module is None:
        return skipped_result(name, "Numba is not available", module_cfg)
    n = int(module_cfg.get("n", 20_000_000))
    estimate = 1024.0
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    kernel = numba_module.njit(fastmath=True, cache=True)(_numba_njit_loop_kernel)

    compile_started = time.perf_counter()
    kernel(16)
    compile_s = time.perf_counter() - compile_started

    def func() -> float:
        return float(kernel(n))

    result = timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "mode": "njit", "parallel": False, "cache": True},
        estimate,
        func,
        "Miter/s",
        n / 1.0e6,
    )
    result["compile_s"] = compile_s
    return result


def bench_numba_prange_loop(
    name: str,
    module_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    numba_module: Any,
    thread_count: int,
) -> dict[str, Any]:
    if numba_module is None:
        return skipped_result(name, "Numba is not available", module_cfg)
    n = int(module_cfg.get("n", 50_000_000))
    estimate = 1024.0
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    validate_prange_kernel_is_scalar_only()
    if not is_auto_thread_count(thread_count):
        try:
            numba_module.set_num_threads(int(thread_count))
        except Exception:
            pass
    globals()["_numba_prange"] = numba_module.prange
    kernel = numba_module.njit(parallel=True, fastmath=True, cache=True)(
        _numba_prange_loop_kernel
    )

    compile_started = time.perf_counter()
    kernel(16)
    compile_s = time.perf_counter() - compile_started
    actual_threads = None
    threading_layer = None
    try:
        actual_threads = int(numba_module.get_num_threads())
    except Exception:
        pass
    try:
        threading_layer = str(numba_module.threading_layer())
    except Exception:
        pass

    def func() -> float:
        return float(kernel(n))

    result = timed_result(
        name,
        module_cfg,
        global_cfg,
        {
            "n": n,
            "mode": "njit_prange",
            "parallel": True,
            "cache": True,
            "kernel_body": "scalar math only",
            "requested_threads": format_thread_count(thread_count),
            "actual_numba_threads": actual_threads,
            "threading_layer": threading_layer,
        },
        estimate,
        func,
        "Miter/s",
        n / 1.0e6,
    )
    result["compile_s"] = compile_s
    return result


def bench_sparse_matvec_csr(name: str, module_cfg: dict[str, Any], global_cfg: dict[str, Any], np: Any, scipy_sparse: Any) -> dict[str, Any]:
    if scipy_sparse is None:
        return skipped_result(name, "SciPy sparse is not available", module_cfg)
    n = int(module_cfg.get("n", 120_000))
    nnz_per_row = int(module_cfg.get("nnz_per_row", 12))
    loops = int(module_cfg.get("loops", 10))
    nnz = n * nnz_per_row
    estimate = nnz * (8 + 4 + 4) + (n + 1) * 4 + 3.0 * n * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    rows = np.repeat(np.arange(n, dtype=np.int32), nnz_per_row)
    cols = rng.integers(0, n, size=nnz, dtype=np.int32)
    data = rng.standard_normal(nnz).astype(np.float64, copy=False)
    matrix = scipy_sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    matrix.sum_duplicates()
    x = random_float64(np, rng, (n,))

    def func() -> float:
        y = x
        for _ in range(loops):
            y = matrix @ y
        return float(y[0])

    actual_nnz = int(matrix.nnz)
    flops = 2.0 * actual_nnz * loops
    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "nnz": actual_nnz, "nnz_per_row": nnz_per_row, "loops": loops},
        estimate,
        func,
        "GFLOP/s",
        flops / 1.0e9,
    )


def bench_sparse_eigsh(
    name: str,
    module_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    np: Any,
    scipy_sparse: Any,
    scipy_sparse_linalg: Any,
) -> dict[str, Any]:
    if scipy_sparse is None or scipy_sparse_linalg is None:
        return skipped_result(name, "SciPy sparse.linalg is not available", module_cfg)
    n = int(module_cfg.get("n", 2500))
    k = int(module_cfg.get("k", 6))
    maxiter = int(module_cfg.get("maxiter", 400))
    tol = float(module_cfg.get("tol", 1.0e-6))
    estimate = 12.0 * n * 8 + 10.0 * n * k * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    main = 2.0 + 0.01 * rng.standard_normal(n)
    off = -1.0 + 0.01 * rng.standard_normal(n - 1)
    matrix = scipy_sparse.diags([off, main, off], offsets=[-1, 0, 1], format="csr")

    def func() -> float:
        values = scipy_sparse_linalg.eigsh(
            matrix, k=k, which="SA", return_eigenvectors=False, tol=tol, maxiter=maxiter
        )
        return float(values[0])

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"n": n, "k": k, "tol": tol, "maxiter": maxiter},
        estimate,
        func,
        "eigs/s",
        float(k),
    )


def bench_scipy_signal_fftconvolve(
    name: str,
    module_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    np: Any,
    scipy_signal: Any,
) -> dict[str, Any]:
    if scipy_signal is None:
        return skipped_result(name, "SciPy signal is not available", module_cfg)
    shape = tuple(int(v) for v in module_cfg.get("shape", [1536, 1536]))
    kernel_shape = tuple(int(v) for v in module_cfg.get("kernel_shape", [65, 65]))
    if len(shape) != 2 or len(kernel_shape) != 2:
        raise ValueError("scipy_signal_fftconvolve shape and kernel_shape must contain 2 integers")
    size = int(math.prod(shape))
    kernel_size = int(math.prod(kernel_shape))
    estimate = (2.5 * size + kernel_size) * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    image = random_float64(np, rng, shape)
    kernel = random_float64(np, rng, kernel_shape)

    def func() -> float:
        y = scipy_signal.fftconvolve(image, kernel, mode="same")
        return float(y[0, 0])

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"shape": list(shape), "kernel_shape": list(kernel_shape), "dtype": "float64"},
        estimate,
        func,
        "Mpixel/s",
        size / 1.0e6,
    )


def bench_scipy_ndimage_gaussian_filter(
    name: str,
    module_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    np: Any,
    scipy_ndimage: Any,
) -> dict[str, Any]:
    if scipy_ndimage is None:
        return skipped_result(name, "SciPy ndimage is not available", module_cfg)
    shape = tuple(int(v) for v in module_cfg.get("shape", [2048, 2048]))
    if len(shape) != 2:
        raise ValueError("scipy_ndimage_gaussian_filter shape must contain 2 integers")
    sigma = float(module_cfg.get("sigma", 2.0))
    size = int(math.prod(shape))
    estimate = 3.0 * size * 8
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    image = random_float64(np, rng, shape)

    def func() -> float:
        y = scipy_ndimage.gaussian_filter(image, sigma=sigma, mode="reflect")
        return float(y[0, 0])

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"shape": list(shape), "sigma": sigma, "dtype": "float64"},
        estimate,
        func,
        "Mpixel/s",
        size / 1.0e6,
    )


def bench_scipy_distance_cdist(
    name: str,
    module_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    np: Any,
    scipy_spatial_distance: Any,
) -> dict[str, Any]:
    if scipy_spatial_distance is None:
        return skipped_result(name, "SciPy spatial.distance is not available", module_cfg)
    m = int(module_cfg.get("m", 3000))
    n = int(module_cfg.get("n", 1000))
    d = int(module_cfg.get("d", 16))
    estimate = (m * d + n * d + m * n) * 8 * 1.5
    skipped = check_memory_or_skip(name, module_cfg, global_cfg, estimate)
    if skipped:
        return skipped
    rng = rng_for(np, global_cfg, name)
    a = random_float64(np, rng, (m, d))
    b = random_float64(np, rng, (n, d))

    def func() -> float:
        distances = scipy_spatial_distance.cdist(a, b, metric="euclidean")
        return float(distances[0, 0])

    return timed_result(
        name,
        module_cfg,
        global_cfg,
        {"m": m, "n": n, "d": d, "metric": "euclidean", "dtype": "float64"},
        estimate,
        func,
        "Mdist/s",
        (m * n) / 1.0e6,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Python scientific CPU workloads in single-thread and multithread modes."
    )
    parser.add_argument(
        "--config",
        default="benchmark_config.json",
        help="JSON configuration file. Missing keys use built-in defaults.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output.directory from the config.",
    )
    parser.add_argument(
        "--write-default-config",
        default=None,
        help="Write the standard default configuration to this path and exit.",
    )
    parser.add_argument(
        "--write-smoke-config",
        default=None,
        help="Write a tiny fast test configuration to this path and exit.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ui", action="store_true", help="Open the PyQt benchmark UI.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.write_default_config:
        write_json(Path(args.write_default_config), default_config())
        print(f"Wrote default config to {args.write_default_config}")
        return 0
    if args.write_smoke_config:
        write_json(Path(args.write_smoke_config), smoke_config())
        print(f"Wrote smoke config to {args.write_smoke_config}")
        return 0
    if args.worker:
        if not args.worker_config or not args.worker_output:
            print("--worker requires --worker-config and --worker-output", file=sys.stderr)
            return 2
        return run_worker(args)
    if args.worker_one:
        if not args.worker_config or not args.worker_output:
            print("--worker-one requires --worker-config and --worker-output", file=sys.stderr)
            return 2
        return run_worker_one(args)
    if args.ui:
        from cpu_scientific_benchmark_ui import main as ui_main

        return ui_main([])
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
