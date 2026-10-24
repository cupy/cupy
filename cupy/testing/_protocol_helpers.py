from __future__ import annotations

import cupy
from cupy import cuda


max_cuda_array_interface_version = 3


class DummyObjectWithCudaArrayInterface:
    # Private helper used for testing cuda array interface support.
    def __init__(
        self, a, ver=3, include_strides=False, mask=None, stream=None
    ):
        assert ver in tuple(range(max_cuda_array_interface_version+1))
        self.a = None
        if isinstance(a, cupy.ndarray):
            self.a = a
        else:
            self.shape, self.strides, self.typestr, self.descr, self.data = a
        self.ver = ver
        self.include_strides = include_strides
        self.mask = mask
        self.stream = stream

    @property
    def __cuda_array_interface__(self):
        if self.a is not None:
            desc = {
                'shape': self.a.shape,
                'typestr': self.a.dtype.str,
                'descr': self.a.dtype.descr,
                'data': (self.a.data.ptr, False),
                'version': self.ver,
            }
            if self.a.flags.c_contiguous:
                if self.include_strides is True:
                    desc['strides'] = self.a.strides
                elif self.include_strides is None:
                    desc['strides'] = None
                else:  # self.include_strides is False
                    pass
            else:  # F contiguous or neither
                desc['strides'] = self.a.strides
        else:
            desc = {
                'shape': self.shape,
                'typestr': self.typestr,
                'descr': self.descr,
                'data': (self.data, False),
                'version': self.ver,
            }
            if self.include_strides is True:
                desc['strides'] = self.strides
            elif self.include_strides is None:
                desc['strides'] = None
            else:  # self.include_strides is False
                pass
        if self.mask is not None:
            desc['mask'] = self.mask
        # The stream field is kept here for compliance. However, since the
        # synchronization is done via calling a cpdef function, which cannot
        # be mock-tested.
        if self.stream is not None:
            if self.stream is cuda.Stream.null:
                desc['stream'] = cuda.runtime.streamLegacy
            elif (not cuda.runtime.is_hip) and self.stream is cuda.Stream.ptds:
                desc['stream'] = cuda.runtime.streamPerThread
            else:
                desc['stream'] = self.stream.ptr
        return desc


class DummyObjectWithCuPyGetNDArray:
    # Private helper used for testing `__cupy_get_ndarray__` support.
    def __init__(self, a):
        self.a = a

    def __cupy_get_ndarray__(self):
        return self.a
