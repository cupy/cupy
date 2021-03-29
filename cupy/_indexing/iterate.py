import numpy

import cupy
from cupy import _core
from cupy._core import internal


class flatiter:
    """Flat iterator object to iterate over arrays.

    A flatiter iterator is returned by ``x.flat`` for any array ``x``. It
    allows iterating over the array as if it were a 1-D array, either in a
    for-loop or by calling its ``next`` method.

    Iteration is done in row-major, C-style order (the last index varying the
    fastest).

    Attributes:
        base (cupy.ndarray): A reference to the array that is iterated over.

    .. note::
       Restricted support of basic slicing is currently supplied. Advanced
       indexing is not supported yet.

    .. seealso:: :func:`numpy.flatiter`

    """

    def __init__(self, a):
        self._base = a
        self._index = 0

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
            s_step = s.step
            size = s.stop - s.start
            if s_step > 0:
                size = (size - 1) // s_step + 1
            else:
                size = (size + 1) // s_step + 1
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
            s_step = s.step
            size = s.stop - s.start
            if s_step > 0:
                size = (size - 1) // s_step + 1
            else:
                size = (size + 1) // s_step + 1
            return _flatiter_getitem_slice(base, s_start, s_step, size=size)

        raise IndexError('unsupported iterator index')

    def __iter__(self):
        return self

    def __next__(self):
        index = self._index
        if index == len(self):
            raise StopIteration()
        self._index += 1
        return self[index]

    def copy(self):
        """Get a copy of the iterator as a 1-D array."""
        return self.base.flatten()

    @property
    def base(self):
        """A reference to the array that is iterated over."""
        return self._base

    # TODO(Takagi): Implement coords

    # TODO(Takagi): Implement index

    # TODO(Takagi): Implement __lt__

    # TODO(Takagi): Implement __le__

    # TODO(Takagi): Implement __eq__

    # TODO(Takagi): Implement __ne__

    # TODO(Takagi): Implement __ge__

    # TODO(Takagi): Implement __gt__

    def __len__(self):
        return self.base.size


_flatiter_setitem_slice = _core.ElementwiseKernel(
    'raw T val, int64 start, int64 step', 'raw T a',
    'a[start + i * step] = val[i % val.size()]',
    'cupy_flatiter_setitem_slice')


_flatiter_getitem_slice = _core.ElementwiseKernel(
    'raw T a, int64 start, int64 step', 'T o',
    'o = a[start + i * step]',
    'cupy_flatiter_getitem_slice')
