import unittest


import numpy
try:
    import scipy.sparse  # NOQA
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import testing
from cupyx.scipy import sparse


def _make(xp, sp, dtype):
    data = xp.array([[0, 1, 2], [3, 4, 5]], dtype)
    offsets = xp.array([0, -1], 'i')
    # 0, 0, 0, 0
    # 3, 1, 0, 0
    # 0, 4, 2, 0
    return sp.dia_matrix((data, offsets), shape=(3, 4))


def _make_empty(xp, sp, dtype):
    data = xp.array([[]], 'f')
    offsets = xp.array([0], 'i')
    return sp.dia_matrix((data, offsets), shape=(3, 4))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
class TestDiaMatrix(unittest.TestCase):

    def setUp(self):
        self.m = _make(cupy, sparse, self.dtype)

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

    @unittest.skipUnless(scipy_available, 'requires scipy')
    def test_str(self):
        if numpy.dtype(self.dtype).kind == 'f':
            expect = '''  (1, 1)\t1.0
  (2, 2)\t2.0
  (1, 0)\t3.0
  (2, 1)\t4.0'''
        else:
            expect = '''  (1, 1)\t(1+0j)
  (2, 2)\t(2+0j)
  (1, 0)\t(3+0j)
  (2, 1)\t(4+0j)'''
        self.assertEqual(str(self.m), expect)

    def test_toarray(self):
        m = self.m.toarray()
        expect = [
            [0, 0, 0, 0],
            [3, 1, 0, 0],
            [0, 4, 2, 0]
        ]
        self.assertTrue(m.flags.c_contiguous)
        cupy.testing.assert_allclose(m, expect)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
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
            (self.data(xp), self.offsets(xp)[None]), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_large_rank_data(self, xp, sp):
        sp.dia_matrix(
            (self.data(xp)[None], self.offsets(xp)), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_data_offsets_different_size(self, xp, sp):
        offsets = xp.array([0, -1, 1], 'i')
        sp.dia_matrix(
            (self.data(xp), offsets), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_duplicated_offsets(self, xp, sp):
        offsets = xp.array([1, 1], 'i')
        sp.dia_matrix(
            (self.data(xp), offsets), shape=self.shape)


@testing.parameterize(*testing.product({
    'make_method': ['_make', '_make_empty'],
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestDiaMatrixScipyComparison(unittest.TestCase):

    @property
    def make(self):
        return globals()[self.make_method]

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_nnz_axis(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.nnz

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=NotImplementedError)
    def test_nnz_axis_not_none(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.getnnz(axis=0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_A(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.A

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_sum_tuple_axis(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        m.sum(axis=(0, 1))

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_sum_float_axis(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        m.sum(axis=0.0)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_sum_too_large_axis(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        m.sum(axis=3)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocoo(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.tocoo()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocoo_copy(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = m.tocoo(copy=True)
        self.assertIsNot(m.data, n.data)
        return n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.tocsc()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc_copy(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = m.tocsc(copy=True)
        self.assertIsNot(m.data, n.data)
        return n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.tocsr()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsr_copy(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = m.tocsr(copy=True)
        self.assertIsNot(m.data, n.data)
        return n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_transpose(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.transpose()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'ret_dtype': [None, numpy.float32, numpy.float64],
    'axis': [None, 0, 1, -1, -2],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestDiaMatrixSum(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.sum(axis=self.axis, dtype=self.ret_dtype)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_with_out(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        if self.axis is None:
            shape = ()
        else:
            shape = list(m.shape)
            shape[self.axis] = 1
            shape = tuple(shape)
        out = xp.empty(shape, dtype=self.ret_dtype)
        if xp is numpy:
            # TODO(unno): numpy.matrix is used for scipy.sparse though
            # cupy.ndarray is used for cupyx.scipy.sparse.
            out = xp.asmatrix(out)
        return m.sum(axis=self.axis, dtype=self.ret_dtype, out=out)


class TestIsspmatrixDia(unittest.TestCase):

    def test_dia(self):
        x = sparse.dia_matrix(
            (cupy.array([], 'f'),
             cupy.array([0], 'i')),
            shape=(0, 0), dtype='f')
        self.assertTrue(sparse.isspmatrix_dia(x))

    def test_csr(self):
        x = sparse.csr_matrix(
            (cupy.array([], 'f'),
             cupy.array([], 'i'),
             cupy.array([0], 'i')),
            shape=(0, 0), dtype='f')
        self.assertFalse(sparse.isspmatrix_dia(x))
