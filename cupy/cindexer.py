import ctypes

import numpy
import six

from cupy import carray


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
        self.size = numpy.prod(shape, dtype=int)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def ctypes(self):
        return to_cindexer(self.size, self.shape)
