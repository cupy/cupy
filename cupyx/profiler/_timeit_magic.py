"""IPython magic for GPU-aware timing using cupyx.profiler.benchmark.

To use this extension in IPython/Jupyter:
    %load_ext cupyx.profiler

Then you can use:
    %gpu_timeit my_array.sum()

Or as a cell magic:
    %%gpu_timeit
    x = cp.random.random((1000, 1000))
    x @ x.T
"""
from __future__ import annotations

import textwrap

try:
    from IPython.core.magic import Magics, magics_class, line_cell_magic
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False


if _IPYTHON_AVAILABLE:
    @magics_class
    class GPUTimeitMagics(Magics):
        """GPU-aware timing magics using cupyx.profiler.benchmark."""

        @line_cell_magic
        def gpu_timeit(self, line, cell=None):
            """Time code with GPU synchronization.

            Works as both line and cell magic.

            Line magic usage:
                %gpu_timeit expression

            Cell magic usage:
                %%gpu_timeit
                <code>

            Examples:
                %gpu_timeit cp.random.random((1000, 1000)).sum()

                %%gpu_timeit
                x = cp.random.random((1000, 1000))
                result = x @ x.T
            """
            from cupyx.profiler import benchmark

            if cell is None:
                # Line magic
                code = line.strip()

                def run():
                    return eval(code, self.shell.user_ns, self.shell.user_ns)
            else:
                # Cell magic
                code = textwrap.dedent(cell)

                def run():
                    exec(code, self.shell.user_ns, self.shell.user_ns)

            result = benchmark(run, n_warmup=1)
            print(result)
            return None


def load_ipython_extension(ipython):
    """Load the %gpu_timeit magic.

    This function is called when the extension is loaded via:
        %load_ext cupyx.profiler
    """
    if not _IPYTHON_AVAILABLE:
        raise ImportError(
            'IPython is not available. '
            'Please install IPython to use %gpu_timeit magic.'
        )
    ipython.register_magics(GPUTimeitMagics)


def unload_ipython_extension(ipython):
    """Unload the %gpu_timeit magic (optional)."""
    pass
