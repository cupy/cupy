import cupy
from cupy import util


cdef class SimpleKernel:

    """User-defined custom kernel.

    This class can be used to define a custom kernel.
    """

    cdef:
        readonly str code
        readonly str name
        readonly tuple options

    def __init__(self, code, name='kernel', options=()):
        self.code = code
        self.name = name
        self.options = options

    def __call__(self, grid, block, args, **kwargs):
        kern = _get_simple_kernel(self.code, self.name, self.options)
        kern(grid, block, args, **kwargs)


@cupy.util.memoize(for_each_device=True)
def _get_simple_kernel(code, name, options=()):
    module = compile_with_cache(code, options)
    return module.get_function(name)
