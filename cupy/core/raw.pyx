import cupy
from cupy import util
from cupy.cuda cimport driver

import six


cdef class RawKernel:

    """User-defined custom kernel.

    This class can be used to define a custom kernel using raw CUDA source.

    The kernel is compiled at an invocation of the :meth:`~RawKernel.__call__`
    method, which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        code (str): CUDA source code.
        name (str): Name of the kernel function.
        options (str): Compile options passed to NVRTC. For details, see
            https://docs.nvidia.com/cuda/nvrtc/index.html#group__options.

    """

    def __init__(self, code, name, options=()):
        if isinstance(code, six.binary_type):
            code = code.decode('UTF-8')
        if isinstance(name, six.binary_type):
            name = name.decode('UTF-8')
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

    @property
    def attributes(self):
        """Returns an object containing runtime kernel attributes.

        Returns:
            attributes (FuncAttributes): A python class containing the
                kernel's attributes. For example, ``attributes.numRegs``
                corresponds to the number of registers used by the kernel.
        """
        kern = _get_raw_kernel(self.code, self.name, self.options)
        return _get_func_attributes(kern.ptr)


@cupy.util.memoize(for_each_device=True)
def _get_raw_kernel(code, name, options=()):
    module = cupy.core.core.compile_with_cache(
        code, options, prepend_cupy_headers=False)
    return module.get_function(name)


@cupy.util.memoize(for_each_device=True)
def _get_func_attributes(func):
    cdef:
        int sharedSizeBytes, constSizeBytes, localSizeBytes
        int maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion
        int cacheModeCA, maxDynamicSharedSizeBytes, preferredShmemCarveout

    sharedSizeBytes = driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func)
    constSizeBytes = driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func)
    localSizeBytes = driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func)
    maxThreadsPerBlock = driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func)
    numRegs = driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_NUM_REGS, func)
    ptxVersion = driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_PTX_VERSION, func)
    binaryVersion = driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_BINARY_VERSION, func)
    cacheModeCA = driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, func)
    maxDynamicSharedSizeBytes = driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, func)
    preferredShmemCarveout = driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, func)

    return dict(sharedSizeBytes=sharedSizeBytes,
                constSizeBytes=constSizeBytes,
                localSizeBytes=localSizeBytes,
                maxThreadsPerBlock=maxThreadsPerBlock,
                numRegs=numRegs,
                ptxVersion=ptxVersion,
                binaryVersion=binaryVersion,
                cacheModeCA=cacheModeCA,
                maxDynamicSharedSizeBytes=maxDynamicSharedSizeBytes,
                preferredShmemCarveout=preferredShmemCarveout)
