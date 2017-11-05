# distutils: language = c++

import numpy
import six

cimport cpython  # NOQA
from libcpp cimport vector


from cupy.cuda cimport driver
from cupy.cuda cimport runtime
from cupy.core cimport _scalar
from cupy.core cimport core
from cupy.cuda cimport stream as stream_module


cdef inline core.CPointer _pointer(x):
    if isinstance(x, core.CPointer):
        return x
    if isinstance(x, core.ndarray):
        return (<core.ndarray>x).get_pointer()
    return _scalar.convert_scalar(x, True)


cdef inline size_t _get_stream(stream) except *:
    if stream is None:
        return stream_module.get_current_stream_ptr()
    else:
        return stream.ptr


cdef _launch(size_t func, Py_ssize_t grid0, int grid1, int grid2,
             Py_ssize_t block0, int block1, int block2,
             args, Py_ssize_t shared_mem, size_t stream):
    cdef list pargs = []
    cdef vector.vector[void*] kargs
    cdef core.CPointer cp
    kargs.reserve(len(args))
    for a in args:
        if a is None:
            kargs.push_back(<void*>0)
        cp = _pointer(a)
        pargs.append(cp)
        kargs.push_back(cp.ptr)

    runtime._ensure_context()
    driver.launchKernel(
        func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
        <int>shared_mem, stream, <size_t>&(kargs[0]), <size_t>0)


cdef class Function:

    """CUDA kernel function."""

    def __init__(self, Module module, str funcname):
        self.module = module  # to keep module loaded
        self.ptr = driver.moduleGetFunction(module.ptr, funcname)

    def __call__(self, tuple grid, tuple block, args, size_t shared_mem=0,
                 stream=None):
        grid = (grid + (1, 1))[:3]
        block = (block + (1, 1))[:3]
        s = _get_stream(stream)
        _launch(
            self.ptr,
            max(1, grid[0]), max(1, grid[1]), max(1, grid[2]),
            max(1, block[0]), max(1, block[1]), max(1, block[2]),
            args, shared_mem, s)

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

    cpdef load_file(self, str filename):
        runtime._ensure_context()
        self.ptr = driver.moduleLoad(filename)

    cpdef load(self, bytes cubin):
        runtime._ensure_context()
        self.ptr = driver.moduleLoadData(cubin)

    cpdef get_global_var(self, str name):
        return driver.moduleGetGlobal(self.ptr, name)

    cpdef get_function(self, str name):
        return Function(self, name)


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

    cpdef bytes complete(self):
        cubin = driver.linkComplete(self.ptr)
        return cubin
