# distutils: language = c++

import re
import numpy
import warnings

from libc.stdint cimport intptr_t
from libc.stdint cimport uintmax_t
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


IF CUPY_CUDA_VERSION > 0:
    cdef extern from "../../cupy_backends/cupy_backend.h" nogil:
        pass

    # Platform-abstracted C++ demangling via NVIDIA's libcufilt.
    # Linux: __cu_demangle is linked directly from libcufilt.a.
    # Windows: cupy_cufilt.dll (a /MT trampoline) is loaded at runtime
    #          to avoid the /MT vs /MD CRT mismatch with cufilt.lib.
    cdef extern from *:
        r"""
        #ifdef _WIN32
        #include <windows.h>
        #include <cstddef>
        #include <cstring>
        #include <mutex>

        typedef char* (*cupy_demangle_fn)(
            const char*, char*, size_t*, int*);
        typedef void (*cupy_free_fn)(char*);

        static cupy_demangle_fn _cupy_demangle = NULL;
        static cupy_free_fn _cupy_dfree = NULL;
        static std::once_flag _cupy_cufilt_flag;

        static void _cupy_cufilt_do_init(void) {
            HMODULE hSelf, hDll;
            wchar_t path[MAX_PATH];
            wchar_t *sep;
            DWORD len;

            /* Locate cupy_cufilt.dll next to this extension module. */
            if (!GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    (LPCWSTR)&_cupy_cufilt_do_init, &hSelf))
                return;
            len = GetModuleFileNameW(hSelf, path, MAX_PATH);
            if (len == 0 || len >= MAX_PATH)
                return;
            sep = wcsrchr(path, L'\\');
            if (sep) *(sep + 1) = L'\0';
            wcscat(path, L"cupy_cufilt.dll");

            hDll = LoadLibraryW(path);
            if (!hDll) return;
            _cupy_demangle = (cupy_demangle_fn)GetProcAddress(
                hDll, "cupy_cu_demangle");
            _cupy_dfree = (cupy_free_fn)GetProcAddress(
                hDll, "cupy_free");
            if (!_cupy_demangle || !_cupy_dfree) {
                _cupy_demangle = NULL;
                _cupy_dfree = NULL;
            }
        }

        static char* cupy_demangle(
                const char* id, char* buf,
                size_t* len, int* status) {
            std::call_once(_cupy_cufilt_flag, _cupy_cufilt_do_init);
            if (!_cupy_demangle) {
                if (status) *status = -1;
                return NULL;
            }
            return _cupy_demangle(id, buf, len, status);
        }

        static void cupy_demangle_free(char* ptr) {
            if (_cupy_dfree) _cupy_dfree(ptr);
        }
        #else
        #include "nv_decode.h"
        #include <cstdlib>

        static char* cupy_demangle(
                const char* id, char* buf,
                size_t* len, int* status) {
            return __cu_demangle(id, buf, len, status);
        }

        static void cupy_demangle_free(char* ptr) {
            free(ptr);
        }
        #endif
        """
        char* cupy_demangle(const char* id, char* output_buffer,
                            size_t* length, int* status) nogil
        void cupy_demangle_free(char* ptr) nogil


cdef str demangle_cxx_name(
        const char* mangled_cstr, str mangled_str):
    """Demangle a C++ mangled name using libcufilt.

    Args:
        mangled_cstr: C string of the mangled name.
        mangled_str: Same name as a Python string.

    Returns:
        The demangled name, or *mangled_str* if
        demangling fails.
    """
    IF CUPY_CUDA_VERSION > 0:
        cdef char* result_ptr = NULL
        cdef int status = 0
        cdef str result

        with nogil:
            result_ptr = cupy_demangle(
                mangled_cstr, NULL, NULL, &status)

        if status == 0 and result_ptr != NULL:
            try:
                result = result_ptr.decode('utf-8')
                return result
            finally:
                cupy_demangle_free(result_ptr)
        else:
            return mangled_str
    ELSE:
        return mangled_str


# Matches CuPy's bundled Thrust internal namespace on the complex type,
# e.g. "thrust::THRUST_300102_SM_890_NS::complex" or "thrust::complex".
cdef object _thrust_complex_re = re.compile(r'thrust::(?:\w+::)?complex\b')


cdef inline str normalize_name_expr(str demangled):
    """Normalize a demangled full signature to a short name expression.

    Converts e.g.
      ``"void square<thrust::NS::complex<double> >(const T1 *, T1 *, int)"``
    to
      ``"square<complex<double>>"``.

    Steps:
      1. Strip the ``void`` return-type prefix (CUDA kernels always return
         void; non-template kernels are demangled without one).
      2. Strip the parameter list ``(...)`` that follows the function name
         (and any template arguments).
      3. Replace ``thrust::*::complex`` with ``complex`` (CuPy's bundled
         Thrust namespace).
      4. Collapse whitespace before ``>`` so ``<double >`` becomes
         ``<double>``.

    Args:
        demangled: Full demangled signature from ``__cu_demangle``.

    Returns:
        Normalised short name expression.
    """
    cdef str s = demangled

    if s.startswith('void '):
        s = s[5:]

    # Find the opening '(' of the parameter list at angle-bracket depth 0.
    cdef int depth = 0
    cdef Py_ssize_t idx
    cdef Py_ssize_t slen = len(s)
    for idx in range(slen):
        if s[idx] == '<':
            depth += 1
        elif s[idx] == '>':
            depth -= 1
        elif s[idx] == '(' and depth == 0:
            s = s[:idx]
            break

    s = _thrust_complex_re.sub('complex', s)
    s = s.replace(' >', '>')

    return s


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
        """Enumerate functions and build mapping (CUDA driver 12.4+).

        This method enumerates all functions in the loaded CUBIN and builds
        a mapping from user-provided name expressions to mangled names.

        Args:
            name_expressions: Tuple of name expressions to match.

        .. note::
            This function requires CUDA driver 12.4 or later.
            On HIP/ROCm, this raises NotImplementedError.
        """
        IF CUPY_CUDA_VERSION > 0:
            if self.mapping is not None:
                return  # Already built

            cdef vector.vector[driver.Function] function_handles
            cdef Py_ssize_t i, num_functions
            cdef const char* mangled_cstr

            cdef dict mapping = {}
            # Reverse lookup: normalized user name → original user name.
            # This lets us match demangled symbols back to the user's
            # original name expressions (which may contain thrust:: etc.).
            cdef dict norm_to_user = {
                normalize_name_expr(ne): ne for ne in name_expressions}

            try:
                num_functions = driver.moduleGetFunctionCount(self.ptr)
                driver.moduleEnumerateFunctions(
                    self.ptr, num_functions, function_handles)

                for i in range(function_handles.size()):
                    mangled_cstr = driver.funcGetName(
                        <intptr_t>(function_handles[i]))
                    mangled_str = mangled_cstr.decode('utf-8')
                    demangled = demangle_cxx_name(mangled_cstr, mangled_str)
                    normalized = normalize_name_expr(demangled)
                    original = norm_to_user.get(normalized)
                    if original is not None:
                        mapping[original] = mangled_str
            except Exception:
                pass
            else:
                if len(mapping) == len(name_expressions):
                    self._set_mapping(mapping)
        ELSE:
            raise NotImplementedError(
                'Function enumeration is not supported on HIP/ROCm')


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
