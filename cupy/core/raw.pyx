import cupy
from cupy import util


cdef class RawKernel:

    """User-defined custom kernel.

    This class can be used to define a custom kernel using raw CUDA source.

    The kernel is compiled at an invocation of the :meth:`~RawKernel.__call__`
    method, which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Alternatively, this class can also compile an existing codebase (ex:
    reading a .cu source file) by the :meth:`~RawKernel.compile` method, which
    returns a cp.cuda.function.Module instance that contains the kernels in the
    source.

    Args:
        code (str): CUDA source code.
        name (str): Name of the kernel function.
        options (str): Compile options passed to NVRTC. For details, see
            https://docs.nvidia.com/cuda/nvrtc/index.html#group__options.

    """

    def __init__(self, code, name, options=()):
        self.code = code
        self.name = name
        self.options = options

    def __call__(self, grid, block, args, **kwargs):
        """__call__(self, grid, block, args, *, shared_mem=0)

        Compiles and invokes the kernel.

        The compilation runs only if the kernel is not cached.

        Args:
            grid (tuple): Size of grid in blocks.
            block (tuple): Dimensions of each thread block.
            args (tuple): Arguments of the kernel.
            shared_mem (int): Dynamic shared-memory size per thread block in
                bytes.

        """
        kern = _get_raw_kernel(self.code, self.name, self.options)
        kern(grid, block, args, **kwargs)

    def compile(self, code=None, options=()):
        """compile(self, code=None, options=())

        Compiles the kernels in code, and returns a Module instance. If the
        arguments are not given, the corresponding attributes of the RawKernel
        instance that were given during initialization are used.

        The compilation runs only if the kernels are not cached.
        """
        if code is None:
            code = self.code

        if options == ():
            options = self.options

        module = cupy.core.core.compile_with_cache(
            code, options, prepend_cupy_headers=False)
        return module


@cupy.util.memoize(for_each_device=True)
def _get_raw_kernel(code, name, options=()):
    module = cupy.core.core.compile_with_cache(
        code, options, prepend_cupy_headers=False)
    return module.get_function(name)
