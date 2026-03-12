# distutils: language = c++

import numpy
import re
import warnings

from libc.stdint cimport intptr_t
from libc.stdint cimport uintmax_t
from libc.stdlib cimport free
from libcpp cimport vector

from cupy._core cimport _carray
from cupy._core cimport _scalar
from cupy._core.core cimport _ndarray_base
from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime
from cupy.cuda cimport stream as stream_module
from cupy.cuda.memory cimport MemoryPointer
from cupy.cuda.texture cimport TextureObject, SurfaceObject
from cupy.cuda import device


# C++ demangling using __cu_demangle from NVIDIA's libcufilt
# See: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#library-availability
cdef extern from "nv_decode.h" nogil:
    char* __cu_demangle(const char* mangled_name, char* output_buffer)


cdef str demangle_cxx_name(str mangled):
    """Demangle a C++ mangled name using __cu_demangle from libcufilt.

    Args:
        mangled: The mangled C++ name.

    Returns:
        The demangled name, or the original name if demangling fails.
    """
    cdef bytes mangled_bytes = mangled.encode('utf-8')
    cdef const char* mangled_ptr = mangled_bytes
    cdef char* demangled_ptr = NULL
    cdef str result

    with nogil:
        demangled_ptr = __cu_demangle(mangled_ptr, NULL)

    if demangled_ptr != NULL:
        try:
            result = demangled_ptr.decode('utf-8')
        finally:
            free(demangled_ptr)
        return result
    else:
        # Demangling failed, return original
        return mangled


cdef str normalize_name(str name):
    """Normalize a C++ function name for comparison.

    Removes spaces and standardizes formatting to make comparison easier.
    """
    # Remove all whitespace
    name = re.sub(r'\s+', '', name)
    return name


cdef str match_name_expression(str name_expr, list mangled_names):
    """Match a name expression to a list of mangled names.

    Args:
        name_expr: User-provided name expression (e.g., "kernel<float>")
        mangled_names: List of (mangled_name, demangled_name) tuples

    Returns:
        The matching mangled name, or None if no match found.
    """
    # Try exact match first (for already mangled names)
    for mangled, _ in mangled_names:
        if name_expr == mangled:
            return mangled

    # Normalize the user's name expression
    normalized_expr = normalize_name(name_expr)

    # Try matching against demangled names
    for mangled, demangled in mangled_names:
        if demangled:
            normalized_demangled = normalize_name(demangled)
            if normalized_expr == normalized_demangled:
                return mangled

    return None


cdef class CPointer:
    def __init__(self, p=0):
        self.ptr = p


cdef class CUIntMax(CPointer):
    cdef:
        uintmax_t val

    def __init__(self, uintmax_t v):
        self.val = v
        self.ptr = <intptr_t><void*>(&self.val)


cdef class CIntptr(CPointer):
    cdef:
        intptr_t val

    def __init__(self, intptr_t v):
        self.val = v
        self.ptr = <intptr_t><void*>(&self.val)


cdef class CNumpyArray(CPointer):
    cdef:
        object val

    def __init__(self, v):
        self.val = v
        self.ptr = <intptr_t><void*><size_t>v.__array_interface__['data'][0]


cdef inline CPointer _pointer(x):
    if x is None:
        return CIntptr(0)
    if isinstance(x, _ndarray_base):
        return (<_ndarray_base>x).get_pointer()
    if isinstance(x, _carray.Indexer):
        return (<_carray.Indexer>x).get_pointer()
    if isinstance(x, MemoryPointer):
        return CIntptr(x.ptr)
    if isinstance(x, CPointer):
        return x
    if isinstance(x, (TextureObject, SurfaceObject)):
        return CUIntMax(x.ptr)
    if isinstance(x, numpy.ndarray):
        # All numpy.ndarray work with CNumpyArray to pass a kernel argument by
        # value. Here we allow only arrays of size one so that users do not
        # mistakenly send numpy.ndarrays instead of cupy.ndarrays to kernels.
        # This may happen if they forget to convert numpy arrays to cupy arrays
        # prior to kernel call and would pass silently without this check.
        if (x.size == 1):
            return CNumpyArray(x)
        else:
            msg = ('You are trying to pass a numpy.ndarray of shape {} as a '
                   'kernel parameter. Only numpy.ndarrays of size one can be '
                   'passed by value. If you meant to pass a pointer to __glob'
                   'al__ memory, you need to pass a cupy.ndarray instead.')
            raise TypeError(msg.format(x.shape))

    # This should be a scalar understood by NumPy (and us).
    return _scalar.CScalar(x)


cdef inline size_t _get_stream(stream) except *:
    if stream is None:
        return stream_module.get_current_stream_ptr()
    else:
        return stream.ptr


cdef _launch(intptr_t func, Py_ssize_t grid0, int grid1, int grid2,
             Py_ssize_t block0, int block1, int block2,
             args, Py_ssize_t shared_mem, size_t stream,
             bint enable_cooperative_groups=False):
    cdef list pargs = []
    cdef vector.vector[void*] kargs
    cdef CPointer cp
    kargs.reserve(len(args))
    for a in args:
        cp = _pointer(a)
        pargs.append(cp)  # keep the CPointer objects alive
        kargs.push_back(<void*>(cp.ptr))

    runtime._ensure_context()

    cdef int dev_id
    cdef int num_sm
    cdef int max_grid_size
    if enable_cooperative_groups:
        dev_id = device.get_device_id()
        num_sm = device._get_attributes(dev_id)['MultiProcessorCount']
        max_grid_size = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            func, block0 * block1 * block2, shared_mem) * num_sm
        if grid0 * grid1 * grid2 > max_grid_size:
            if grid1 == grid2 == 1:
                warnings.warn('The grid size will be reduced from {} to {}, '
                              'as the specified grid size exceeds the limit.'.
                              format(grid0, max_grid_size))
                grid0 = max_grid_size
            else:
                raise ValueError('The specified grid size ({} * {} * {}) '
                                 'exceeds the limit ({}).'.
                                 format(grid0, grid1, grid2, max_grid_size))
        driver.launchCooperativeKernel(
            func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
            <int>shared_mem, stream, <intptr_t>kargs.data())
    else:
        driver.launchKernel(
            func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
            <int>shared_mem, stream, <intptr_t>kargs.data(), <intptr_t>0)


cdef class Function:

    """CUDA kernel function."""

    def __init__(self, Module module, str funcname):
        self.module = module  # to keep module loaded
        self.ptr = driver.moduleGetFunction(module.ptr, funcname)

    def __call__(self, tuple grid, tuple block, args, size_t shared_mem=0,
                 stream=None, enable_cooperative_groups=False):
        grid = (grid + (1, 1))[:3]
        block = (block + (1, 1))[:3]
        s = _get_stream(stream)
        _launch(
            self.ptr,
            max(1, grid[0]), max(1, grid[1]), max(1, grid[2]),
            max(1, block[0]), max(1, block[1]), max(1, block[2]),
            args, shared_mem, s, enable_cooperative_groups)

    cpdef linear_launch(self, size_t size, args, size_t shared_mem=0,
                        size_t block_max_size=128, stream=None,
                        bint enable_cooperative_groups=False):
        # TODO(beam2d): Tune it
        cdef size_t gridx = min(
            0x7fffffffUL, (size + block_max_size - 1) // block_max_size)
        cdef size_t blockx = min(block_max_size, size)
        IF CUPY_HIP_VERSION > 0:
            # The HSA AQL dispatch packet stores the grid size as total
            # work-items (gridDimX * blockDimX) in a uint32_t field, so the
            # product must not overflow.
            if blockx > 0:
                gridx = min(gridx, 0xffffffffUL // blockx)
        s = _get_stream(stream)
        _launch(
            self.ptr,
            gridx, 1, 1, blockx, 1, 1,
            args,
            shared_mem, s, enable_cooperative_groups)


cdef class Module:

    """CUDA kernel module."""

    def __init__(self):
        self.ptr = 0
        self.mapping = None

    def __dealloc__(self):
        if self.ptr:
            driver.moduleUnload(self.ptr)
            self.ptr = 0

    cpdef load_file(self, filename):
        if isinstance(filename, bytes):
            filename = filename.decode()
        runtime._ensure_context()
        self.ptr = driver.moduleLoad(filename)

    cpdef load(self, bytes cubin):
        runtime._ensure_context()
        self.ptr = driver.moduleLoadData(cubin)

    cpdef get_global_var(self, name):
        if isinstance(name, bytes):
            name = name.decode()
        return driver.moduleGetGlobal(self.ptr, name)

    cpdef get_function(self, name):
        if isinstance(name, bytes):
            name = name.decode()
        return Function(self, name)

    cpdef _set_mapping(self, dict mapping):
        self.mapping = mapping

    cpdef _enumerate_and_build_mapping(self, tuple name_expressions):
        """Enumerate functions and build mapping (CUDA 11.6+).

        This method enumerates all functions in the loaded CUBIN and builds
        a mapping from user-provided name expressions to mangled names.

        Args:
            name_expressions: Tuple of name expressions to match.

        .. note::
            This function requires CUDA 11.6 or later. On HIP/ROCm, this
            raises NotImplementedError if function enumeration is not supported.
        """
        if self.mapping is not None:
            return  # Already built

        IF CUPY_HIP_VERSION > 0:
            # HIP/ROCm does not support function enumeration yet
            raise NotImplementedError(
                'Function enumeration is not supported on HIP/ROCm')

        try:
            # Enumerate all functions in the module
            function_handles = driver.moduleEnumerateFunctions(self.ptr)

            # Get mangled names and demangle them
            mangled_names = []
            for func_handle in function_handles:
                mangled = driver.funcGetName(func_handle)
                demangled = demangle_cxx_name(mangled)
                mangled_names.append((mangled, demangled))

            # Build mapping from user expressions to mangled names
            self.mapping = {}
            for name_expr in name_expressions:
                matched = match_name_expression(name_expr, mangled_names)
                if matched:
                    self.mapping[name_expr] = matched
        except Exception:
            # On error (e.g., CUDA < 11.6), mapping stays None
            # Caller should handle this by recompiling
            pass


cdef class LinkState:

    """CUDA link state."""

    def __init__(self):
        runtime._ensure_context()
        self.ptr = driver.linkCreate()

    def __dealloc__(self):
        if self.ptr:
            driver.linkDestroy(self.ptr)
            self.ptr = 0

    cpdef add_ptr_data(self, bytes data, unicode name):
        driver.linkAddData(self.ptr, driver.CU_JIT_INPUT_PTX, data, name)

    cpdef add_ptr_file(self, unicode path):
        driver.linkAddFile(self.ptr, driver.CU_JIT_INPUT_LIBRARY, path)

    cpdef bytes complete(self):
        cubin = driver.linkComplete(self.ptr)
        return cubin
