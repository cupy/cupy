import ctypes

import six

from cupy import carray
from cupy import internal


def _make_cindexer(n):
    class CIndexer(ctypes.Structure):
        _fields_ = [('size', ctypes.c_int),
                    ('shape', ctypes.c_int * n),
                    ('index', ctypes.c_int * n)]
    return CIndexer


_cindexers = [_make_cindexer(i) for i in six.moves.range(carray.MAX_NDIM)]


def to_cindexer(size, shape):
    return _cindexers[len(shape)](size, shape, (0,) * len(shape))


class Indexer(object):
    def __init__(self, shape):
        self.shape = shape
        self.size = internal.prod(shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def ctypes(self):
        return to_cindexer(self.size, self.shape)
