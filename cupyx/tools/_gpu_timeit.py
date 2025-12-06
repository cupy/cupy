from __future__ import annotations

from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
from cupyx.profiler import benchmark
import textwrap


@magics_class
class GPUTimeitMagics(Magics):
    """A GPU-aware timing magic using cupyx.profiler.benchmark."""

    @line_magic
    def gpu_timeit(self, line):
        """
        Line magic: %gpu_timeit expression
        """
        code = line.strip()

        def run():
            return eval(code, self.shell.user_ns)

        result = benchmark(run)
        print(result)

    @cell_magic
    def gpu_timeit_cell(self, line, cell):
        """
        Cell magic: %%gpu_timeit
        <code>
        """
        cell = textwrap.dedent(cell)

        def run():
            exec(cell, self.shell.user_ns)

        result = benchmark(run)
        print(result)


# Make the cell magic accessible as %%gpu_timeit
GPUTimeitMagics.gpu_timeit_cell.__name__ = "gpu_timeit"


def load_ipython_extension(ipython):
    """Registers %gpu_timeit magic in IPython."""
    ipython.register_magics(GPUTimeitMagics)
