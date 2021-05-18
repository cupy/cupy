from cupyx.jit._interface import rawkernel  # NOQA

from cupyx.jit._interface import threadIdx  # NOQA
from cupyx.jit._interface import blockDim  # NOQA
from cupyx.jit._interface import blockIdx  # NOQA
from cupyx.jit._interface import gridDim  # NOQA

from cupyx.jit._builtin_funcs import syncthreads  # NOQA
from cupyx.jit._builtin_funcs import shared_memory  # NOQA
from cupyx.jit._builtin_funcs import atomic_add  # NOQA


import inspect as _inspect

_getsource_func = _inspect.getsource  # NOQA
