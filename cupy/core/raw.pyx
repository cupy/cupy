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
    def attributes(self):
        """Returns an object containing runtime kernel attributes.

        Returns:
            attributes (FunctionAttributes): A python class containing the
                kernel's attributes. For example, ``attributes.num_regs``
                corresponds to the number of registers used by the kernel.
        """
        return FunctionAttributes(self.kernel)

    @property
    def kernel(self):
        return _get_raw_kernel(self.code, self.name, self.options)


@cupy.util.memoize(for_each_device=True)
def _get_raw_kernel(code, name, options=()):
    module = cupy.core.core.compile_with_cache(
        code, options, prepend_cupy_headers=False)
    return module.get_function(name)


class FunctionAttributes:

    class Read:
        def __init__(self, func_attribute):
            self.func_attribute = func_attribute

        def __get__(self, instance, owner):
            return driver.funcGetAttribute(self.func_attribute, instance.kern.ptr)

    class ReadWrite(Read):
        def __set__(self, instance, value):
            driver.funcSetAttribute(instance.kern.ptr, self.func_attribute, value)

    shared_size_bytes = Read(driver.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    const_size_bytes = Read(driver.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)
    local_size_bytes = Read(driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
    max_threads_per_block = Read(driver.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    num_regs = Read(driver.CU_FUNC_ATTRIBUTE_NUM_REGS)
    ptx_version = Read(driver.CU_FUNC_ATTRIBUTE_PTX_VERSION)
    binary_version = Read(driver.CU_FUNC_ATTRIBUTE_BINARY_VERSION)
    cache_mode_ca = Read(driver.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA)
    max_dynamic_shared_size_bytes = ReadWrite(driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
    preferred_shared_memory_carveout = ReadWrite(driver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT)
    
    def __init__(self, kern):
        self.kern = kern
