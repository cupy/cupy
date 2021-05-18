import unittest

import pytest
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

from cupy import testing
from cupyx.scipy import sparse


if scipy_available:
    class DummySparseCPU(scipy.sparse.spmatrix):

        def __init__(self, maxprint=50, shape=None, nnz=0):
            super(DummySparseCPU, self).__init__(maxprint)
            self._shape = shape
            self._nnz = nnz

        def getnnz(self):
            return self._nnz


class DummySparseGPU(sparse.spmatrix):

    def __init__(self, maxprint=50, shape=None, nnz=0):
        super(DummySparseGPU, self).__init__(maxprint)
        self._shape = shape
        self._nnz = nnz

    def get_shape(self):
        return self._shape

    def getnnz(self):
        return self._nnz


@testing.with_requires('scipy')
class TestSpmatrix(unittest.TestCase):

    def dummy_class(self, sp):
        if sp is sparse:
            return DummySparseGPU
        else:
            return DummySparseCPU

    def test_instantiation(self):
        for sp in (scipy.sparse, sparse):
            with pytest.raises(ValueError):
                sp.spmatrix()

    def test_len(self):
        for sp in (scipy.sparse, sparse):
            s = self.dummy_class(sp)()
            with pytest.raises(TypeError):
                len(s)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_bool_true(self, xp, sp):
        s = self.dummy_class(sp)(shape=(1, 1), nnz=1)
        return bool(s)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_bool_false(self, xp, sp):
        s = self.dummy_class(sp)(shape=(1, 1), nnz=0)
        return bool(s)

    def test_bool_invalid(self):
        for sp in (scipy.sparse, sparse):
            s = self.dummy_class(sp)(shape=(2, 1))
            with pytest.raises(ValueError):
                bool(s)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_asformat_none(self, xp, sp):
        s = self.dummy_class(sp)()
        assert s.asformat(None) is s

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_maxprint(self, xp, sp):
        s = self.dummy_class(sp)(maxprint=30)
        return s.getmaxprint()
