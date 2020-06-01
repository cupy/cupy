# distutils: language = c++

import numpy

cimport cpython  # NOQA
from libc.stdint cimport int8_t
from libc.stdint cimport int16_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdint cimport intptr_t
from libc.stdint cimport uintmax_t
from libcpp cimport vector

from cupy.core cimport _carray
from cupy.core cimport core
from cupy.cuda cimport driver
from cupy.cuda cimport runtime
from cupy.cuda cimport stream as stream_module
from cupy.cuda.texture cimport TextureObject, SurfaceObject


cdef class CPointer:
    def __init__(self, p=0):
        self.ptr = <void*>p


cdef class CInt8(CPointer):
    cdef:
        int8_t val

    def __init__(self, int8_t v):
        self.val = v
        self.ptr = <void*>&self.val


cdef class CInt16(CPointer):
    cdef:
        int16_t val

    def __init__(self, int16_t v):
        self.val = v
        self.ptr = <void*>&self.val


cdef class CInt32(CPointer):
    cdef:
        int32_t val

    def __init__(self, int32_t v):
        self.val = v
        self.ptr = <void*>&self.val


cdef class CInt64(CPointer):
    cdef:
        int64_t val

    def __init__(self, int64_t v):
        self.val = v
        self.ptr = <void*>&self.val


cdef class CInt128(CPointer):
    cdef:
        double complex val

    def __init__(self, double complex v):
        self.val = v
        self.ptr = <void*>&self.val


cdef class CUIntMax(CPointer):
    cdef:
        uintmax_t val

    def __init__(self, uintmax_t v):
        self.val = v
        self.ptr = <void*>&self.val


cdef set _pointer_numpy_types = {numpy.dtype(i).type
                                 for i in '?bhilqBHILQefdFD'}


cdef inline CPointer _pointer(x):
    cdef Py_ssize_t itemsize
    if x is None:
        return CPointer()
    if isinstance(x, core.ndarray):
        return (<core.ndarray>x).get_pointer()
    if isinstance(x, _carray.Indexer):
        return (<_carray.Indexer>x).get_pointer()

    if isinstance(x, CPointer):
        return x

    if isinstance(x, TextureObject):
        return CUIntMax(x.ptr)

    if isinstance(x, SurfaceObject):
        return CUIntMax(x.ptr)

    if type(x) not in _pointer_numpy_types:
        if isinstance(x, int):
            x = numpy.int64(x)
        elif isinstance(x, float):
            x = numpy.float64(x)
        elif isinstance(x, bool):
            x = numpy.bool_(x)
        elif isinstance(x, complex):
            x = numpy.complex128(x)
        else:
            raise TypeError('Unsupported type %s' % type(x))

    itemsize = x.itemsize
    if itemsize == 1:
        return CInt8(x.view(numpy.int8))
    if itemsize == 2:
        return CInt16(x.view(numpy.int16))
    if itemsize == 4:
        return CInt32(x.view(numpy.int32))
    if itemsize == 8:
        return CInt64(x.view(numpy.int64))
    if itemsize == 16:
        return CInt128(x.view(numpy.complex128))
    raise TypeError('Unsupported type %s. (size=%d)', type(x), itemsize)


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
        pargs.append(cp)
        kargs.push_back(cp.ptr)

    runtime._ensure_context()

    if enable_cooperative_groups:
        driver.launchCooperativeKernel(
            func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
            <int>shared_mem, stream, <intptr_t>&(kargs[0]))
    else:
        driver.launchKernel(
            func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
            <int>shared_mem, stream, <intptr_t>&(kargs[0]), <intptr_t>0)


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
                        size_t block_max_size=128, stream=None):
        # TODO(beam2d): Tune it
        cdef size_t gridx = min(
            0x7fffffffUL, (size + block_max_size - 1) // block_max_size)
        cdef size_t blockx = min(block_max_size, size)
        s = _get_stream(stream)
        _launch(self.ptr,
                gridx, 1, 1, blockx, 1, 1, args, shared_mem, s)


cdef class Module:

    """CUDA kernel module."""

    def __init__(self):
        self.ptr = 0

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

    cpdef get_texref(self, name):
        if isinstance(name, bytes):
            name = name.decode()
        return driver.moduleGetTexRef(self.ptr, name)


cdef class LinkState:

    """CUDA link state."""

    def __init__(self):
        runtime._ensure_context()
        self.ptr = driver.linkCreate()

    def __dealloc__(self):
        if self.ptr:
            driver.linkDestroy(self.ptr)
            self.ptr = 0

    cpdef add_ptr_data(self, unicode data, unicode name):
        cdef bytes data_byte = data.encode()
        driver.linkAddData(self.ptr, driver.CU_JIT_INPUT_PTX, data_byte, name)

    cpdef add_ptr_file(self, unicode path):
        driver.linkAddFile(self.ptr, driver.CU_JIT_INPUT_LIBRARY, path)

    cpdef bytes complete(self):
        cubin = driver.linkComplete(self.ptr)
        return cubin
