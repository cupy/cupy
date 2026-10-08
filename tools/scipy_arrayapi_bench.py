#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Callable


# ----------------------------------------------------------------------
# IMPORTANT:
#
# SCIPY_ARRAY_API must be set BEFORE importing scipy.
# Therefore this file deliberately does NOT import scipy at module level.
# ----------------------------------------------------------------------


DEFAULT_SIZE = 10_000_000


# ======================================================================
# Benchmark cases
# ======================================================================

@dataclass
class Case:
    name: str
    module: str
    function: str
    dtype: str = "float32"
    kind: str = "elementwise"
    description: str = ""


CASES = [
    # ------------------------------------------------------------------
    # scipy.special
    # ------------------------------------------------------------------

    Case(
        name="special.i0e",
        module="scipy.special",
        function="i0e",
        kind="special",
        description="Exponentially scaled modified Bessel I0",
    ),

    Case(
        name="special.gammaln",
        module="scipy.special",
        function="gammaln",
        kind="special",
        description="Log gamma",
    ),

    Case(
        name="special.gammaincc",
        module="scipy.special",
        function="gammaincc",
        kind="special",
        description="Regularized upper incomplete gamma",
    ),

    Case(
        name="special.expit",
        module="scipy.special",
        function="expit",
        kind="special",
        description="Logistic sigmoid",
    ),

    Case(
        name="special.entr",
        module="scipy.special",
        function="entr",
        kind="special",
        description="Elementwise entropy",
    ),

    Case(
        name="special.softmax",
        module="scipy.special",
        function="softmax",
        kind="special",
        description="Softmax",
    ),

    # ------------------------------------------------------------------
    # scipy.ndimage
    # ------------------------------------------------------------------

    Case(
        name="ndimage.gaussian_filter",
        module="scipy.ndimage",
        function="gaussian_filter",
        kind="ndimage",
        description="Gaussian filter",
    ),

    Case(
        name="ndimage.uniform_filter",
        module="scipy.ndimage",
        function="uniform_filter",
        kind="ndimage",
        description="Uniform filter",
    ),

    Case(
        name="ndimage.median_filter",
        module="scipy.ndimage",
        function="median_filter",
        kind="ndimage",
        description="Median filter",
    ),

    Case(
        name="ndimage.zoom",
        module="scipy.ndimage",
        function="zoom",
        kind="ndimage",
        description="Zoom/interpolation",
    ),

    # ------------------------------------------------------------------
    # scipy.signal
    # ------------------------------------------------------------------

    Case(
        name="signal.windows.hann",
        module="scipy.signal.windows",
        function="hann",
        kind="signal",
        description="Hann window",
    ),

    Case(
        name="signal.windows.hamming",
        module="scipy.signal.windows",
        function="hamming",
        kind="signal",
        description="Hamming window",
    ),

    # ------------------------------------------------------------------
    # scipy.fft
    # ------------------------------------------------------------------

    Case(
        name="fft.fft",
        module="scipy.fft",
        function="fft",
        kind="fft",
        description="1-D FFT",
    ),

    Case(
        name="fft.rfft",
        module="scipy.fft",
        function="rfft",
        kind="fft",
        description="1-D real FFT",
    ),
]


# ======================================================================
# Backend abstraction
# ======================================================================

class Backend:
    """
    Backend adapter.

    The benchmark itself only knows:
        - xp
        - synchronize()
        - to_cpu()
        - device_info()

    This means numpy-ascend does not need to imitate CuPy's API.
    """

    def __init__(self, name: str, module_name: str | None = None):
        self.name = name
        self.module_name = module_name
        self.xp = None
        self.module = None

    def initialize(self):
        if self.name == "cpu":
            import numpy as np

            self.xp = np
            return

        if self.name == "ascend":
            if not self.module_name:
                raise RuntimeError(
                    "Ascend backend requires --module, e.g. "
                    "--module cupy"  # may also rename pkg to numpy_ascend
                )

            self.module = importlib.import_module(self.module_name)

            # We assume numpy-ascend exposes the NumPy-like API.
            self.xp = self.module

            return

        raise ValueError(f"Unknown backend: {self.name}")

    def synchronize(self):
        """
        Synchronize device work.

        For CPU this is a no-op.

        numpy-ascend may expose one of:
            xpu.runtime.deviceSynchronize()
            device.synchronize()
            cuda.runtime.deviceSynchronize()

        Add your real Ascend synchronization entry here.
        """

        if self.name == "cpu":
            return

        # numpy-ascend explicit API
        if hasattr(self.module, "synchronize"):
            self.module.synchronize()
            return

        # xpu synchronize()
        device = getattr(self.module, "xpu", None)
        if device is not None:
            rt = getattr(device, "runtime", None)
            if rt is not None:
                sync = getattr(rt, "deviceSynchronize", None)
                if sync:
                    sync()
                    return

        # cuda synchronize()
        device = getattr(self.module, "cuda", None)
        if device is not None:
            rt = getattr(device, "runtime", None)
            if rt is not None:
                sync = getattr(rt, "deviceSynchronize", None)
                if sync:
                    sync()
                    return

        # IMPORTANT:
        # If numpy-ascend operations are asynchronous, you MUST implement
        # synchronization here. Otherwise the benchmark is invalid.
        raise RuntimeError(
            "Cannot synchronize Ascend backend. "
            "Please implement Backend.synchronize() for numpy-ascend."
        )

    def to_cpu(self, x):
        """
        Convert result to NumPy.

        This function is NEVER called during timed region.
        """

        if self.name == "cpu":
            return x

        # Common numpy-ascend convention
        if hasattr(x, "get"):
            return x.get()

        # Common conversion API
        if hasattr(self.module, "asnumpy"):
            return self.module.asnumpy(x)

        # Array API backend may expose __array__
        try:
            import numpy as np

            return np.asarray(x)
        except Exception:
            pass

        raise RuntimeError(
            "Cannot convert Ascend result to NumPy. "
            "Please implement Backend.to_cpu()."
        )

    def device_info(self):
        if self.name == "cpu":
            return {
                "backend": "numpy",
                "device": "CPU",
                "device_name": platform.processor(),
            }

        info = {
            "backend": self.module_name,
            "device": "Ascend",
        }

        # Optional user-defined information
        if hasattr(self.module, "device_info"):
            try:
                info["device_name"] = self.module.device_info()
            except Exception:
                pass

        return info


# ======================================================================
# Input generation
# ======================================================================

def create_input(backend: Backend, size: int, dtype: str):
    xp = backend.xp

    # Keep values numerically stable for special functions.
    if dtype == "float32":
        x = xp.linspace(
            -8.0,
            8.0,
            size,
            dtype=xp.float32,
        )
    elif dtype == "float64":
        x = xp.linspace(
            -8.0,
            8.0,
            size,
            dtype=xp.float64,
        )
    else:
        raise ValueError(f"Unsupported benchmark dtype: {dtype}")

    return x


# ======================================================================
# Function arguments
# ======================================================================

def make_call(case: Case, x):
    """
    Return:
        callable

    We deliberately construct arguments here rather than trying to
    introspect arbitrary SciPy signatures.
    """

    if case.name == "special.i0e":
        return lambda fn: fn(x)

    if case.name == "special.gammaln":
        # gammaln requires positive x
        xx = x + 9.0
        return lambda fn: fn(xx)

    if case.name == "special.gammaincc":
        # gammaincc(a, x)
        a = x * 0.0 + 2.5
        xx = (x + 8.0) * 2.0
        return lambda fn: fn(a, xx)

    if case.name == "special.expit":
        return lambda fn: fn(x)

    if case.name == "special.entr":
        # entr requires positive input for finite result
        xx = (x + 8.1) / 16.1
        return lambda fn: fn(xx)

    if case.name == "special.softmax":
        # 10M 1-D softmax is valid but intentionally expensive.
        return lambda fn: fn(x)

    if case.name == "ndimage.gaussian_filter":
        return lambda fn: fn(x, sigma=2.0)

    if case.name == "ndimage.uniform_filter":
        return lambda fn: fn(x, size=31)

    if case.name == "ndimage.median_filter":
        return lambda fn: fn(x, size=7)

    if case.name == "ndimage.zoom":
        # Avoid doubling 10M -> 20M output.
        return lambda fn: fn(x, zoom=0.5)

    if case.name == "signal.windows.hann":
        # Windows create their own array.
        return lambda fn: fn(x.size)

    if case.name == "signal.windows.hamming":
        return lambda fn: fn(x.size)

    if case.name == "fft.fft":
        return lambda fn: fn(x)

    if case.name == "fft.rfft":
        return lambda fn: fn(x)

    raise KeyError(case.name)


# ======================================================================
# Timing
# ======================================================================

def benchmark_one(
    backend: Backend,
    case: Case,
    x,
    warmup: int,
    repeat: int,
):
    module = importlib.import_module(case.module)
    fn = getattr(module, case.function)

    call = make_call(case, x)

    # --------------------------------------------------------------
    # Warmup
    # --------------------------------------------------------------

    for _ in range(warmup):
        y = call(fn)

    backend.synchronize()

    # --------------------------------------------------------------
    # Timed iterations
    #
    # Synchronize BEFORE and AFTER each iteration.
    #
    # This is critical for asynchronous NPU execution.
    # --------------------------------------------------------------

    times = []

    for _ in range(repeat):
        backend.synchronize()

        t0 = time.perf_counter()

        y = call(fn)

        backend.synchronize()

        t1 = time.perf_counter()

        times.append(t1 - t0)

    # --------------------------------------------------------------
    # Validation summary
    #
    # Conversion is deliberately outside timing.
    # --------------------------------------------------------------

    try:
        y_cpu = backend.to_cpu(y)

        import numpy as np

        flat = np.asarray(y_cpu).reshape(-1)

        # Don't hash/transfer the whole 10M result.
        # Take deterministic samples.
        if flat.size:
            indices = np.linspace(
                0,
                flat.size - 1,
                min(1024, flat.size),
                dtype=np.int64,
            )

            sample = flat[indices]

            checksum = float(
                np.sum(
                    np.nan_to_num(
                        sample.astype(np.float64),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                )
            )

            sample_abs_max = float(
                np.max(
                    np.abs(
                        np.nan_to_num(
                            sample.astype(np.float64),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                    )
                )
            )
        else:
            checksum = 0.0
            sample_abs_max = 0.0

        output_shape = list(np.asarray(y_cpu).shape)
        output_dtype = str(np.asarray(y_cpu).dtype)

    except Exception as e:
        checksum = None
        sample_abs_max = None
        output_shape = None
        output_dtype = None

    return {
        "status": "ok",
        "times_sec": times,
        "min_sec": min(times),
        "median_sec": statistics.median(times),
        "mean_sec": statistics.mean(times),
        "p95_sec": percentile(times, 95),
        "output_shape": output_shape,
        "output_dtype": output_dtype,
        "sample_checksum": checksum,
        "sample_abs_max": sample_abs_max,
    }


def percentile(values, p):
    values = sorted(values)

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] * (c - k) + values[c] * (k - f)


# ======================================================================
# Run
# ======================================================================

def run_benchmark(args):
    # --------------------------------------------------------------
    # Set Array API mode BEFORE scipy import.
    # --------------------------------------------------------------

    if args.array_api:
        os.environ["SCIPY_ARRAY_API"] = "1"

    backend = Backend(
        name=args.backend,
        module_name=args.module,
    )

    backend.initialize()

    print()
    print("=" * 80)
    print("SciPy Array API Benchmark")
    print("=" * 80)
    print(f"Backend       : {args.backend}")
    print(f"Array module  : {args.module}")
    print(f"Array size    : {args.size:,}")
    print(f"dtype         : {args.dtype}")
    print(f"warmup        : {args.warmup}")
    print(f"repeat        : {args.repeat}")
    print(f"Array API     : {args.array_api}")
    print()

    print("Device:")
    print(json.dumps(
        backend.device_info(),
        indent=2,
        ensure_ascii=False,
    ))
    print()

    # Import SciPy only now.
    import scipy

    print(f"SciPy version : {scipy.__version__}")
    print()

    x = create_input(
        backend,
        args.size,
        args.dtype,
    )

    results = []

    for case in CASES:
        print(
            f"[{case.kind:10s}] "
            f"{case.name:30s}",
            end=" ",
            flush=True,
        )

        start = time.perf_counter()

        try:
            result = benchmark_one(
                backend,
                case,
                x,
                warmup=args.warmup,
                repeat=args.repeat,
            )

            result.update({
                "name": case.name,
                "module": case.module,
                "function": case.function,
                "kind": case.kind,
                "dtype": case.dtype,
                "description": case.description,
            })

            print(
                f"{result['median_sec'] * 1000:10.3f} ms"
            )

        except Exception as exc:
            result = {
                "name": case.name,
                "module": case.module,
                "function": case.function,
                "kind": case.kind,
                "dtype": case.dtype,
                "description": case.description,
                "status": "unsupported",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

            if args.verbose:
                traceback.print_exc()

            print(
                f"UNSUPPORTED: "
                f"{type(exc).__name__}: {exc}"
            )

        result["wall_setup_sec"] = (
            time.perf_counter() - start
        )

        results.append(result)

    output = {
        "schema_version": 1,
        "timestamp": time.time(),

        "benchmark": {
            "size": args.size,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "array_api": args.array_api,
        },

        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "scipy_version": scipy.__version__,
            "backend": args.backend,
            "array_module": args.module,
            "device": backend.device_info(),
        },

        "results": results,
    }

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Results written to: {args.output}")

    return output


# ======================================================================
# Merge
# ======================================================================

def merge_results(ascend_file, cpu_file, output_file):
    with open(ascend_file, encoding="utf-8") as f:
        ascend = json.load(f)

    with open(cpu_file, encoding="utf-8") as f:
        cpu = json.load(f)

    a_map = {
        r["name"]: r
        for r in ascend["results"]
    }

    c_map = {
        r["name"]: r
        for r in cpu["results"]
    }

    all_names = sorted(
        set(a_map) | set(c_map)
    )

    merged = []

    for name in all_names:
        a = a_map.get(name)
        c = c_map.get(name)

        row = {
            "name": name,
            "kind": (
                a or c
            ).get("kind"),

            "ascend_status": (
                a or {}
            ).get("status"),

            "cpu_status": (
                c or {}
            ).get("status"),
        }

        if (
            a
            and c
            and a.get("status") == "ok"
            and c.get("status") == "ok"
        ):
            ascend_time = a["median_sec"]
            cpu_time = c["median_sec"]

            row.update({
                "cpu_median_sec": cpu_time,
                "ascend_median_sec": ascend_time,

                "speedup": (
                    cpu_time / ascend_time
                    if ascend_time > 0
                    else None
                ),

                "cpu_min_sec": c["min_sec"],
                "ascend_min_sec": a["min_sec"],

                "cpu_p95_sec": c["p95_sec"],
                "ascend_p95_sec": a["p95_sec"],

                "output_shape_cpu": c["output_shape"],
                "output_shape_ascend": a["output_shape"],

                "output_dtype_cpu": c["output_dtype"],
                "output_dtype_ascend": a["output_dtype"],

                "sample_checksum_cpu": c[
                    "sample_checksum"
                ],
                "sample_checksum_ascend": a[
                    "sample_checksum"
                ],

                "sample_checksum_relative_error":
                    relative_error(
                        c["sample_checksum"],
                        a["sample_checksum"],
                    ),
            })

        merged.append(row)

    report = {
        "schema_version": 1,

        "benchmark": {
            "size": ascend["benchmark"]["size"],
            "dtype": ascend["benchmark"]["dtype"],
            "warmup": ascend["benchmark"]["warmup"],
            "repeat": ascend["benchmark"]["repeat"],
            "array_api": True,
        },

        "ascend_environment": ascend["environment"],
        "cpu_environment": cpu["environment"],

        "results": merged,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print_report(report)

    print()
    print(f"Final report written to: {output_file}")

    return report


def relative_error(a, b):
    if a is None or b is None:
        return None

    denom = max(abs(a), abs(b), 1e-30)

    return abs(a - b) / denom


# ======================================================================
# Report
# ======================================================================

def print_report(report):
    print()
    print("=" * 100)
    print("SciPy Array API CPU vs Ascend")
    print("=" * 100)

    print(
        f"{'Case':35s} "
        f"{'CPU(ms)':>12s} "
        f"{'Ascend(ms)':>12s} "
        f"{'Speedup':>10s} "
        f"{'Status':>12s}"
    )

    print("-" * 100)

    for r in report["results"]:

        if (
            r.get("cpu_status") == "ok"
            and r.get("ascend_status") == "ok"
        ):
            cpu_ms = r["cpu_median_sec"] * 1000
            ascend_ms = r["ascend_median_sec"] * 1000
            speedup = r["speedup"]

            print(
                f"{r['name']:35s} "
                f"{cpu_ms:12.3f} "
                f"{ascend_ms:12.3f} "
                f"{speedup:10.2f}x "
                f"{'OK':>12s}"
            )

        else:
            print(
                f"{r['name']:35s} "
                f"{'-':>12s} "
                f"{'-':>12s} "
                f"{'-':>10s} "
                f"{'UNSUPPORTED':>12s}"
            )


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SciPy Array API backends "
            "using CPU and Ascend."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------------
    # run
    # --------------------------------------------------------------

    run = sub.add_parser(
        "run",
        help="Run one backend",
    )

    run.add_argument(
        "--backend",
        choices=["cpu", "ascend"],
        required=True,
    )

    run.add_argument(
        "--module",
        default="numpy_ascend",
        help="Array backend Python module",
    )

    run.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Number of elements",
    )

    run.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
    )

    run.add_argument(
        "--warmup",
        type=int,
        default=2,
    )

    run.add_argument(
        "--repeat",
        type=int,
        default=10,
    )

    run.add_argument(
        "--output",
        required=True,
    )

    run.add_argument(
        "--array-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable SciPy Array API mode. "
            "Must be enabled for this benchmark."
        ),
    )

    run.add_argument(
        "--verbose",
        action="store_true",
    )

    # --------------------------------------------------------------
    # merge
    # --------------------------------------------------------------

    merge = sub.add_parser(
        "merge",
        help="Merge CPU and Ascend benchmark JSON files",
    )

    merge.add_argument(
        "--ascend",
        required=True,
    )

    merge.add_argument(
        "--cpu",
        required=True,
    )

    merge.add_argument(
        "--output",
        required=True,
    )

    # --------------------------------------------------------------
    # report
    # --------------------------------------------------------------

    report = sub.add_parser(
        "report",
        help="Print an existing report",
    )

    report.add_argument(
        "file",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_benchmark(args)

    elif args.command == "merge":
        merge_results(
            args.ascend,
            args.cpu,
            args.output,
        )

    elif args.command == "report":
        with open(
            args.file,
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        print_report(data)


if __name__ == "__main__":
    main()