import ctypes

import numpy

from cupy.cuda import driver

_native = {
    int: ctypes.c_int,
    float: ctypes.c_float,
    bool: ctypes.c_bool,
    type(None): lambda x: ctypes.c_void_p(),
    numpy.bool_: ctypes.c_bool,
    numpy.int8: ctypes.c_int8,
    numpy.uint8: ctypes.c_uint8,
    numpy.int16: ctypes.c_int16,
    numpy.uint16: ctypes.c_uint16,
    numpy.int32: ctypes.c_int32,
    numpy.uint32: ctypes.c_uint32,
    numpy.int64: ctypes.c_int64,
    numpy.uint64: ctypes.c_uint64,
    numpy.float16: lambda x: numpy.ctypeslib.as_ctypes(x.view(numpy.uint16)),
    numpy.float32: ctypes.c_float,
    numpy.float64: ctypes.c_double,
}
_ptrarray_types = [ctypes.c_void_p * l for l in range(32)]


def _get_ctypes(x):
    return x.ctypes


class Function(object):

    """CUDA kernel function."""

    def __init__(self, module, funcname):
        self.module = module  # to keep module loaded
        self.ptr = driver.moduleGetFunction(module.ptr, funcname)

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        a_src = [_native.get(type(x), _get_ctypes)(x) for x in args]
        a = _ptrarray_types[len(a_src)](
            *[ctypes.addressof(x) for x in a_src])

        driver.launchKernel(self.ptr, grid[0], grid[1], grid[2],
                            block[0], block[1], block[2], shared_mem,
                            stream and stream.ptr, a, None)

    def linear_launch(self, size, args, shared_mem=0, block_max_size=128,
                      stream=None):
        # TODO(beam2d): Tune it
        gridx = size // block_max_size + 1
        if gridx > 65536:
            gridx = 65536
        if size > block_max_size:
            size = block_max_size
        self((gridx, 1, 1), (size, 1, 1), args, shared_mem, stream)


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
