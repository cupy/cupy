import unittest


import numpy
try:
    import scipy.sparse  # NOQA
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
import cupy.sparse
from cupy import testing


def _make(xp, sp, dtype):
    data = xp.array([[0, 1, 2], [3, 4, 5]], dtype)
    offsets = xp.array([0, -1], 'i')
    # 0, 0, 0, 0
    # 3, 1, 0, 0
    # 0, 4, 2, 0
    return sp.dia_matrix((data, offsets), shape=(3, 4))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
class TestDiaMatrix(unittest.TestCase):

    def setUp(self):
        self.m = _make(cupy, cupy.sparse, self.dtype)

    def test_dtype(self):
        self.assertEqual(self.m.dtype, self.dtype)

    def test_data(self):
        self.assertEqual(self.m.data.dtype, self.dtype)
        testing.assert_array_equal(
            self.m.data, cupy.array([[0, 1, 2], [3, 4, 5]], self.dtype))

    def test_offsets(self):
        self.assertEqual(self.m.offsets.dtype, numpy.int32)
        testing.assert_array_equal(
            self.m.offsets, cupy.array([0, -1], self.dtype))

    def test_shape(self):
        self.assertEqual(self.m.shape, (3, 4))

    def test_ndim(self):
        self.assertEqual(self.m.ndim, 2)

    def test_nnz(self):
        self.assertEqual(self.m.nnz, 5)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestDiaMatrixInit(unittest.TestCase):

    def setUp(self):
        self.shape = (3, 4)

    def data(self, xp):
        return xp.array([[1, 2, 3], [4, 5, 6]], self.dtype)

    def offsets(self, xp):
        return xp.array([0, -1], 'i')

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_shape_none(self, xp, sp):
        sp.dia_matrix(
            (self.data(xp), self.offsets(xp)), shape=None)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_large_rank_offset(self, xp, sp):
        sp.dia_matrix(
            (self.data(xp), self.offsets(xp)[None]))

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_large_rank_data(self, xp, sp):
        sp.dia_matrix(
            (self.data(xp)[None], self.offsets(xp)))

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_data_offsets_different_size(self, xp, sp):
        offsets = xp.array([0, -1, 1], 'i')
        sp.dia_matrix(
            (self.data(xp), offsets))

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_duplicated_offsets(self, xp, sp):
        offsets = xp.array([1, 1], 'i')
        sp.dia_matrix(
            (self.data(xp), offsets))
