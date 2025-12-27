from __future__ import annotations

import textwrap
from cupyx.profiler import benchmark


def _get_ipython_magics():
    """Lazy import of IPython to avoid hard dependency."""
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

    GPUTimeitMagics.gpu_timeit_cell.__name__ = "gpu_timeit"
    return GPUTimeitMagics


def try_register_ipython_magic():
    """Register magic only if running inside IPython."""
    try:
        from IPython import get_ipython
    except Exception:
        return

    ip = get_ipython()
    if ip is None:
        return

    GPUTimeitMagics = _get_ipython_magics()
    ip.register_magics(GPUTimeitMagics)
