from __future__ import annotations

from cupyx.profiler import benchmark
import textwrap


def _get_ipython_magics():
    """Import IPython lazily to avoid dependency at module import time."""
    from IPython.core.magic import Magics, magics_class, cell_magic, line_magic

    @magics_class
    class GPUTimeitMagics(Magics):

        @line_magic
        def gpu_timeit(self, line):
            code = line.strip()

            def run():
                return eval(code, self.shell.user_ns)

            result = benchmark(run)
            print(result)

        @cell_magic
        def gpu_timeit_cell(self, line, cell):
            cell = textwrap.dedent(cell)

            def run():
                exec(cell, self.shell.user_ns)

            result = benchmark(run)
            print(result)

    # ensure IPython registers this as %gpu_timeit for both line & cell magics
    GPUTimeitMagics.gpu_timeit_cell.__name__ = "gpu_timeit"
    return GPUTimeitMagics


def load_ipython_extension(ipython):
    """Registers %gpu_timeit magic when loaded inside IPython."""
    GPUTimeitMagics = _get_ipython_magics()
    ipython.register_magics(GPUTimeitMagics)
