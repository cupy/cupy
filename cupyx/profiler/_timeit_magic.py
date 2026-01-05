from __future__ import annotations

import functools
import inspect
import textwrap

from IPython.core.magic import (
    Magics, magics_class, line_cell_magic, needs_local_scope
)
from IPython.core.magic_arguments import (
    argument, magic_arguments, parse_argstring
)

from cupyx.profiler import benchmark


@functools.cache
def _get_benchmark_defaults():
    """Extract default parameter values from benchmark function."""
    sig = inspect.signature(benchmark)
    defaults = {}
    for param_name, param in sig.parameters.items():
        if param.default != inspect.Parameter.empty:
            defaults[param_name] = param.default
    return defaults


@magics_class
class GPUTimeitMagics(Magics):
    """GPU-aware timing magics using cupyx.profiler.benchmark."""

    @line_cell_magic
    @needs_local_scope
    @magic_arguments()
    @argument(
        '-n', '--n-repeat', type=int,
        default=_get_benchmark_defaults()['n_repeat'],
        help=f'Number of repeats '
             f'(default: {_get_benchmark_defaults()["n_repeat"]})'
    )
    @argument(
        '-w', '--n-warmup', type=int,
        default=_get_benchmark_defaults()['n_warmup'],
        help=f'Number of warmup runs '
             f'(default: {_get_benchmark_defaults()["n_warmup"]})'
    )
    @argument(
        '--max-duration', type=float,
        default=_get_benchmark_defaults()['max_duration'],
        help='Maximum duration in seconds (default: '
             f'{_get_benchmark_defaults()["max_duration"]})'
    )
    @argument(
        'code', nargs='*',
        help='Code to benchmark'
    )
    def gpu_timeit(self, line, cell=None, local_ns=None):
        """Time code with GPU synchronization.

        Works as both line and cell magic.

        Line magic usage:
            %gpu_timeit [-n N_REPEAT] [-w N_WARMUP] [--max-duration SEC] \
expression

        Cell magic usage:
            %%gpu_timeit [-n N_REPEAT] [-w N_WARMUP] [--max-duration SEC]
            <code>

        Examples:
            %gpu_timeit cp.random.random((1000, 1000)).sum()
            %gpu_timeit -n 100 --max-duration 5 cp.arange(1024**3).sum()

            %%gpu_timeit -n 100
            x = cp.random.random((1000, 1000))
            result = x @ x.T
        """
        args = parse_argstring(self.gpu_timeit, line)

        if cell is None:
            # Line magic
            code = ' '.join(args.code).strip()

            def run():
                return eval(code, self.shell.user_ns, local_ns)
        else:
            # Cell magic
            code = textwrap.dedent(cell)

            def run():
                exec(code, self.shell.user_ns, local_ns)

        result = benchmark(
            run,
            n_repeat=args.n_repeat,
            n_warmup=args.n_warmup,
            max_duration=args.max_duration
        )
        print(result)
