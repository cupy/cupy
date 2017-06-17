import unittest

import scipy.sparse

import cupy.sparse
from cupy import testing


class DummySparseCPU(scipy.sparse.spmatrix):

    def __init__(self, maxprint=50, shape=None, nnz=0):
        super(DummySparseCPU, self).__init__(maxprint)
        self._shape = shape
        self._nnz = nnz

    def getnnz(self):
        return self._nnz


class DummySparseGPU(cupy.sparse.spmatrix):

    def __init__(self, maxprint=50, shape=None, nnz=0):
        super(DummySparseGPU, self).__init__(maxprint)
        self._shape = shape
        self._nnz = nnz

    def get_shape(self):
        return self._shape

    def getnnz(self):
        return self._nnz


dummies = {
    scipy.sparse: DummySparseCPU,
    cupy.sparse: DummySparseGPU,
}


class TestSpmatrix(unittest.TestCase):

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_instantiation(self, xp, sp):
        sp.spmatrix()

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=TypeError)
    def test_len(self, xp, sp):
        s = dummies[sp]()
        len(s)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_bool_true(self, xp, sp):
        s = dummies[sp](shape=(1, 1), nnz=1)
        return bool(s)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_bool_false(self, xp, sp):
        s = dummies[sp](shape=(1, 1), nnz=0)
        return bool(s)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_bool_invalid(self, xp, sp):
        s = dummies[sp](shape=(2, 1))
        bool(s)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_asformat_none(self, xp, sp):
        s = dummies[sp]()
        self.assertIs(s.asformat(None), s)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_maxprint(self, xp, sp):
        s = dummies[sp](maxprint=30)
        return s.getmaxprint()
