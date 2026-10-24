import pickle

import cupy

from cupy.backends.backend.api cimport driver
from cupy.backends.backend.api cimport runtime
from cupy.xpu.function cimport Function, Module

cdef class RawKernel:
    """User-defined custom kernel.
    """
    def __cinit__(self):
        # this is only for pickling: if any change is made such that the old
        # pickles cannot be reused, we bump this version number
        self.raw_ver = 2

    def __init__(self, str code, str name, tuple options=(),
                 str backend='nvrtc', *, bint translate_cucomplex=False,
                 bint enable_cooperative_groups=False, bint jitify=False):
        pass

    def __call__(self, grid, block, args, **kwargs):
        pass


cdef class RawModule:
    def __init__(self, *, str code=None, str path=None, tuple options=(),
                 str backend='nvrtc', bint translate_cucomplex=False,
                 bint enable_cooperative_groups=False,
                 name_expressions=None, bint jitify=False):
        pass