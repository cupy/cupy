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
        self.kernel(grid, block, args, **kwargs)

    @property
    def kernel(self):
        return _get_raw_kernel(self.code, self.name, self.options)

    @property
    def attributes(self):
        """Returns a dictionary containing runtime kernel attributes. This is
        a read-only property; to overwrite the attributes, use

        .. code-block:: python

            kernel = RawKernel(...)  # arguments omitted
            kernel.max_dynamic_shared_size_bytes = ...
            kernel.preferred_shared_memory_carveout = ...

        Note that the two attributes shown in the above example are the only
        two currently settable in CUDA.

        Any attribute not existing in the present CUDA toolkit version will
        have the value -1.

        Returns:
            dict: A dictionary containing the kernel's attributes.
        """
        cdef dict attrs = {}
        cdef list keys = ['max_threads_per_block', 'shared_size_bytes',
                          'const_size_bytes', 'local_size_bytes',
                          'num_regs', 'ptx_version', 'binary_version',
                          'cache_mode_ca', 'max_dynamic_shared_size_bytes',
                          'preferred_shared_memory_carveout']
        for attr in keys:
            attrs[attr] = getattr(self, attr)
        return attrs

    @property
    def max_threads_per_block(self):
        """The maximum number of threads per block that can successfully
        launch the function on the device.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def shared_size_bytes(self):
        """The size in bytes of the statically-allocated shared memory
        used by the function. This is separate from any dynamically-allocated
        shared memory, which must be specified when the function is called.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def const_size_bytes(self):
        """The size in bytes of constant memory used by the function."""
        attr = driver.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def local_size_bytes(self):
        """The size in bytes of local memory used by the function."""
        attr = driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def num_regs(self):
        """The number of registers used by the function."""
        attr = driver.CU_FUNC_ATTRIBUTE_NUM_REGS
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def ptx_version(self):
        """The PTX virtual architecture version that was used during
        compilation, in the format: 10*major + minor.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_PTX_VERSION
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def binary_version(self):
        """The binary architecture version that was used during compilation,
        in the format: 10*major + minor.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_BINARY_VERSION
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def cache_mode_ca(self):
        """Indicates whether option "-Xptxas --dlcm=ca" was set during
        compilation.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def max_dynamic_shared_size_bytes(self):
        """The maximum dynamically-allocated shared memory size in bytes that
        can be used by the function. Can be set.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @max_dynamic_shared_size_bytes.setter
    def max_dynamic_shared_size_bytes(self, bytes):
        attr = driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        driver.funcSetAttribute(self.kernel.ptr, attr, bytes)

    @property
    def preferred_shared_memory_carveout(self):
        """On devices that have a unified L1 cache and shared memory,
        indicates the fraction to be used for shared memory as a
        `percentage` of the total. If the fraction does not exactly equal a
        supported shared memory capacity, then the next larger supported
        capacity is used. Can be set.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @preferred_shared_memory_carveout.setter
    def preferred_shared_memory_carveout(self, fraction):
        attr = driver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        driver.funcSetAttribute(self.kernel.ptr, attr, fraction)


@cupy.util.memoize(for_each_device=True)
def _get_raw_kernel(code, name, options=()):
    module = cupy.core.core.compile_with_cache(code, options,
                                               prepend_cupy_headers=False)
    return module.get_function(name)
