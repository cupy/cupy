cimport cpython
import numpy
cimport numpy
import six

from cupy.cuda import driver

cdef:
    class CPointer:
        def __init__(self, p=0):
            self.ptr = <void*>p

        @property
        def intp(self):
            return <numpy.npy_intp>self.ptr

    class CInt8(CPointer):
        cdef numpy.npy_int8 val

        def __init__(self, numpy.npy_int8 v):
            self.val = v
            self.ptr = <void*>&self.val

    class CInt16(CPointer):
        cdef numpy.npy_int16 val

        def __init__(self, numpy.npy_int16 v):
            self.val = v
            self.ptr = <void*>&self.val

    class CInt32(CPointer):
        cdef numpy.npy_int32 val

        def __init__(self, numpy.npy_int32 v):
            self.val = v
            self.ptr = <void*>&self.val

    class CInt64(CPointer):
        cdef numpy.npy_int64 val

        def __init__(self, numpy.npy_int64 v):
            self.val = v
            self.ptr = <void*>&self.val


cdef CPointer _pointer(x):
    if x is None:
        return CPointer()
    if isinstance(x, six.integer_types):
        x = numpy.int64(x)
    elif isinstance(x, float):
        x = numpy.float64(x)
    elif isinstance(x, bool):
        x = numpy.bool_(x)

    if isinstance(x, (numpy.number, numpy.bool_)):
        if x.itemsize == 1:
            return CInt8(x.view(numpy.int8))
        elif x.itemsize == 2:
            return CInt16(x.view(numpy.int16))
        elif x.itemsize == 4:
            return CInt32(x.view(numpy.int32))
        elif x.itemsize == 8:
            return CInt64(x.view(numpy.int64))
    elif hasattr(x, 'ctypes'):  # TODO(okuta): fix name
        return x.ctypes

    raise TypeError('Unsupported type %s' % type(x))

cdef size_t _get_stream(strm):
    if strm is None or strm.ptr is None:
        return 0
    else:
        return strm.ptr

cdef _launch(size_t func, int grid0, int grid1, int grid2,
             int block0, int block1, int block2,
             args, int shared_mem, size_t stream):
    pargs = [_pointer(a) for a in args]

    cdef int i, n = len(args)
    cdef void** kargs = <void**>0
    cdef CPointer arg

    try:
        kargs = <void**>cpython.PyMem_Malloc(sizeof(void*) * n)
        for i in range(n):
            arg = pargs[i]
            kargs[i] = arg.ptr
        driver.launchKernel(
            func, grid0, grid1, grid2, block0, block1, block2,
            shared_mem, stream, <size_t>kargs, <size_t>0)
    finally:
        cpython.PyMem_Free(kargs)

class Function(object):

    """CUDA kernel function."""

    def __init__(self, module, funcname):
        self.module = module  # to keep module loaded
        self.ptr = driver.moduleGetFunction(module.ptr, funcname)

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        grid = (grid + (1, 1))[:3]
        block = (block + (1, 1))[:3]
        s = _get_stream(stream)
        _launch(
            self.ptr,
            grid[0], grid[1], grid[2], block[0], block[1], block[2],
            args, shared_mem, s)

    def linear_launch(self, size_t size, args, size_t shared_mem=0,
                        size_t block_max_size=128, stream=None):
        # TODO(beam2d): Tune it
        gridx = size // block_max_size + 1
        if gridx > 65536:
            gridx = 65536
        if size > block_max_size:
            size = block_max_size
        s = _get_stream(stream)
        _launch(self.ptr,
                gridx, 1, 1, size, 1, 1, args, shared_mem, s)



class Module(object):

    """CUDA kernel module."""

    def __init__(self):
        self.ptr = None

    def __del__(self):
        if self.ptr:
            driver.moduleUnload(self.ptr)
            self.ptr = None

    def load_file(self, filename):
        self.ptr = driver.moduleLoad(filename)

    def load(self, cubin):
        self.ptr = driver.moduleLoadData(cubin)

    def get_global_var(self, name):
        return driver.moduleGetGlobal(self.ptr, name)

    def get_function(self, name):
        return Function(self, name)
