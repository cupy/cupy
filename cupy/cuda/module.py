import ctypes

import numpy

from cupy.cuda import driver
from cupy.cuda import stream

_native_ctypes = {
    int: ctypes.c_int,
    float: ctypes.c_float,
    bool: ctypes.c_bool,
    type(None): lambda x: ctypes.c_void_p()
}


def _pointer(x):
    conv = _native_ctypes.get(type(x), None)
    if conv is not None:
        x = conv(x)
    elif isinstance(x, numpy.bool_):
        x = ctypes.c_bool(x)
    elif isinstance(x, numpy.number):
        if isinstance(x, numpy.float16):
            x = numpy.ctypeslib.as_ctypes(x.view(numpy.uint16))
        else:
            x = numpy.ctypeslib.as_ctypes(x)
    else:
        try:
            x = x.ctypes
        except AttributeError:
            pass
    return ctypes.pointer(x)


def _get_stream(strm):
    if strm is None:
        return stream.Stream(null=True)
    else:
        return strm


class Function(object):

    """CUDA kernel function."""

    def __init__(self, module, funcname):
        self.module = module  # to keep module loaded
        self.ptr = driver.moduleGetFunction(module.ptr, funcname)

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        grid = (grid + (1, 1))[:3]
        block = (block + (1, 1))[:3]

        a = (ctypes.c_void_p * len(args))()
        for i, arg in enumerate(args):
            a[i] = ctypes.cast(_pointer(arg), ctypes.c_void_p)
        arg_ptr = ctypes.cast(a, ctypes.POINTER(ctypes.c_void_p))

        stream = _get_stream(stream)
        driver.launchKernel(self.ptr, grid[0], grid[1], grid[2],
                            block[0], block[1], block[2], shared_mem,
                            stream.ptr, arg_ptr, ctypes.c_void_p())

    def linear_launch(self, size, args, shared_mem=0, block_max_size=128,
                      stream=None):
        # TODO(beam2d): Tune it
        gridx = min(65536, size // block_max_size + 1)
        blockx = min(size, block_max_size)
        self((gridx,), (blockx,), args, shared_mem, stream)


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
