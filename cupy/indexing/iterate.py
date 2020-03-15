import numpy

import cupy
from cupy import core
from cupy.core import internal


class flatiter():

    def __init__(self, a):
        self._base = a

    def __setitem__(self, ind, value):
        if ind is Ellipsis:
            self[:] = value
            return

        if isinstance(ind, tuple):
            raise IndexError('unsupported iterator index')

        if isinstance(ind, bool):
            raise IndexError('unsupported iterator index')

        if numpy.isscalar(ind):
            ind = int(ind)
            base = self._base
            size = base.size
            indices = []
            for s in base.shape:
                size = size // s
                indices.append(ind // size)
                ind %= size
            base[tuple(indices)] = value
            return

        if isinstance(ind, slice):
            base = self._base
            s = internal.complete_slice(ind, base.size)
            s_start = s.start
            s_stop = s.stop
            s_step = s.step
            if s_step > 0:
                size = (s_stop - s_start - 1) // s_step + 1
            else:
                size = (s_stop - s_start + 1) // s_step + 1
            value = cupy.asarray(value, dtype=base.dtype)
            _flatiter_setitem_slice(value, s_start, s_step, base, size=size)
            return

        raise IndexError('unsupported iterator index')

    def __getitem__(self, ind):
        if ind is Ellipsis:
            return self[:]

        if isinstance(ind, tuple):
            raise IndexError('unsupported iterator index')

        if isinstance(ind, bool):
            raise IndexError('unsupported iterator index')

        if numpy.isscalar(ind):
            ind = int(ind)
            base = self._base
            size = base.size
            indices = []
            for s in base.shape:
                size = size // s
                indices.append(ind // size)
                ind %= size
            return base[tuple(indices)].copy()

        if isinstance(ind, slice):
            base = self._base
            s = internal.complete_slice(ind, base.size)
            s_start = s.start
            s_stop = s.stop
            s_step = s.step
            if s_step > 0:
                size = (s_stop - s_start - 1) // s_step + 1
            else:
                size = (s_stop - s_start + 1) // s_step + 1
            return _flatiter_getitem_slice(base, s_start, s_step, size=size)

        raise IndexError('unsupported iterator index')

    # TODO(Takagi): Implement __iter__ just raising NotImplementedError

    # TODO(Takagi): Implement __next__ just raising NotImplementedError

    # TODO(Takagi): Implement copy

    # TODO(Takagi): Implement base

    # TODO(Takagi): Implement coords

    # TODO(Takagi): Implement index

    # TODO(Takagi): Implement __lt__

    # TODO(Takagi): Implement __le__

    # TODO(Takagi): Implement __eq__

    # TODO(Takagi): Implement __ne__

    # TODO(Takagi): Implement __ge__

    # TODO(Takagi): Implement __gt__

    # TODO(Takagi): Implement __len__


_flatiter_setitem_slice = core.ElementwiseKernel(
    'raw T val, int64 start, int64 step', 'raw T a',
    'a[start + i * step] = val[i % val.size()]',
    'cupy_flatiter_setitem_slice')


_flatiter_getitem_slice = core.ElementwiseKernel(
    'raw T a, int64 start, int64 step', 'T o',
    'o = a[start + i * step]',
    'cupy_flatiter_getitem_slice')
