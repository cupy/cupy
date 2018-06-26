import cupy
from cupy import util


cdef class RawKernel:

    """User-defined custom kernel.

    This class can be used to define a custom kernel.
    """

    def __init__(self, code, name='kernel', options=()):
        self.code = code
        self.name = name
        self.options = options

    def __call__(self, grid, block, args, **kwargs):
        kern = _get_raw_kernel(self.code, self.name, self.options)
        kern(grid, block, args, **kwargs)


@cupy.util.memoize(for_each_device=True)
def _get_raw_kernel(code, name, options=()):
    module = cupy.core.core.compile_with_cache(
        code, options, prepend_cupy_headers=False)
    return module.get_function(name)
