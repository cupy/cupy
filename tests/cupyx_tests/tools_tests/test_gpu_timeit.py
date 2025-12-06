from __future__ import annotations

import cupyx.tools._gpu_timeit as magic_module
from IPython.terminal.interactiveshell import TerminalInteractiveShell
import types
import sys

# ---- Fake cupy and profiler modules ----
fake_cupy = types.ModuleType("cupy")
fake_cupy.arange = lambda n: list(range(n))  # simple CPU fallback

fake_profiler = types.ModuleType("cupyx.profiler")


class DummyBenchmarkResult:
    def __str__(self):
        return "times: 1\nmean: 0.001"  # mimic real benchmark output


def dummy_benchmark(func):
    func()
    return DummyBenchmarkResult()


fake_profiler.benchmark = dummy_benchmark

sys.modules["cupy"] = fake_cupy
sys.modules["cupyx"] = types.ModuleType("cupyx")
sys.modules["cupyx.profiler"] = fake_profiler

# ---- Now import your magic implementation ----


def test_gpu_timeit_magic():
    shell = TerminalInteractiveShell.instance()

    # Load ONLY the magic (do NOT load CuPy extension)
    magic_module.load_ipython_extension(shell)

    output = shell.run_cell_magic(
        "gpu_timeit", "", "import cupy as cp; cp.arange(10)")

    text = str(output)
    assert "times" in text or "mean" in text
