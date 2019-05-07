import unittest

import numpy
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import testing
from cupyx.scipy import sparse


def _make(xp, sp, dtype):
    data = xp.array([0, 1, 2, 3], dtype)
    row = xp.array([0, 0, 1, 2], 'i')
    col = xp.array([0, 1, 3, 2], 'i')
    # 0, 1, 0, 0
    # 0, 0, 0, 2
    # 0, 0, 3, 0
    return sp.coo_matrix((data, (row, col)), shape=(3, 4))


def _make_complex(xp, sp, dtype):
    data = xp.array([0, 1, 2, 3], dtype)
    if dtype in [numpy.complex64, numpy.complex128]:
        data = data - 1j
    row = xp.array([0, 0, 1, 2], 'i')
    col = xp.array([0, 1, 3, 2], 'i')
    # 0, 1 - 1j, 0, 0
    # 0, 0, 0, 2 - 1j
    # 0, 0, 3 - 1j, 0
    return sp.coo_matrix((data, (row, col)), shape=(3, 4))


def _make2(xp, sp, dtype):
    data = xp.array([1, 2, 3, 4], dtype)
    row = xp.array([0, 1, 1, 2], 'i')
    col = xp.array([2, 1, 2, 2], 'i')
    # 0, 0, 1, 0
    # 0, 2, 3, 0
    # 0, 0, 4, 0
    return sp.coo_matrix((data, (row, col)), shape=(3, 4))


def _make3(xp, sp, dtype):
    data = xp.array([1, 2, 3, 4, 5], dtype)
    row = xp.array([0, 1, 1, 3, 3], 'i')
    col = xp.array([0, 2, 1, 0, 2], 'i')
    # 1, 0, 0
    # 0, 3, 2
    # 0, 0, 0
    # 4, 0, 5
    return sp.coo_matrix((data, (row, col)), shape=(4, 3))


def _make_unordered(xp, sp, dtype):
    data = xp.array([1, 4, 3, 2], dtype)
    row = xp.array([0, 2, 1, 0], 'i')
    col = xp.array([0, 2, 3, 1], 'i')
    # 1, 2, 0, 0
    # 0, 0, 0, 3
    # 0, 0, 4, 0
    return sp.coo_matrix((data, (row, col)), shape=(3, 4))


def _make_duplicate(xp, sp, dtype):
    data = xp.array([0, 1, 2, 3, 4, 5], dtype)
    row = xp.array([1, 1, 1, 1, 0, 1], 'i')
    col = xp.array([0, 0, 2, 0, 0, 2], 'i')
    # 4, 0, 0, 0
    # 4, 0, 7, 0
    # 0, 0, 0, 0
    return sp.coo_matrix((data, (row, col)), shape=(3, 4))


def _make_empty(xp, sp, dtype):
    data = xp.array([], dtype)
    row = xp.array([], 'i')
    col = xp.array([], 'i')
    return sp.coo_matrix((data, (row, col)), shape=(3, 4))


def _make_square(xp, sp, dtype):
    data = xp.array([0, 1, 2, 3], dtype)
    row = xp.array([0, 0, 1, 2], 'i')
    col = xp.array([0, 2, 0, 2], 'i')
    # 0, 1, 0
    # 2, 0, 0
    # 0, 0, 3
    return sp.coo_matrix((data, (row, col)), shape=(3, 3))


def _make_shape(xp, sp, dtype):
    return sp.coo_matrix((3, 4))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
class TestCooMatrix(unittest.TestCase):

    def setUp(self):
        self.m = _make(cupy, sparse, self.dtype)

    def test_dtype(self):
        self.assertEqual(self.m.dtype, self.dtype)

    def test_data(self):
        self.assertEqual(self.m.data.dtype, self.dtype)
        testing.assert_array_equal(
            self.m.data, cupy.array([0, 1, 2, 3], self.dtype))

    def test_row(self):
        self.assertEqual(self.m.row.dtype, numpy.int32)
        testing.assert_array_equal(
            self.m.row, cupy.array([0, 0, 1, 2], self.dtype))

    def test_col(self):
        self.assertEqual(self.m.col.dtype, numpy.int32)
        testing.assert_array_equal(
            self.m.col, cupy.array([0, 1, 3, 2], self.dtype))

    def test_init_copy(self):
        n = sparse.coo_matrix(self.m)
        self.assertIsNot(n, self.m)
        cupy.testing.assert_array_equal(n.toarray(), self.m.toarray())

    def test_init_copy_other_sparse(self):
        n = sparse.coo_matrix(self.m.tocsr())
        cupy.testing.assert_array_equal(n.toarray(), self.m.toarray())

    @unittest.skipUnless(scipy_available, 'requires scipy')
    def test_init_copy_scipy_sparse(self):
        m = _make(numpy, scipy.sparse, self.dtype)
        n = sparse.coo_matrix(m)
        self.assertIsInstance(n.data, cupy.ndarray)
        self.assertIsInstance(n.row, cupy.ndarray)
        self.assertIsInstance(n.col, cupy.ndarray)
        cupy.testing.assert_array_equal(n.data, m.data)
        cupy.testing.assert_array_equal(n.row, m.row)
        cupy.testing.assert_array_equal(n.col, m.col)
        self.assertEqual(n.shape, m.shape)

    @unittest.skipUnless(scipy_available, 'requires scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_init_copy_other_scipy_sparse(self, xp, sp):
        m = _make(numpy, scipy.sparse, self.dtype)
        n = sp.coo_matrix(m.tocsc())
        assert len(n.data) == len(m.data)
        assert len(n.row) == len(m.row)
        assert len(n.col) == len(m.col)
        assert n.shape == m.shape
        return n

    def test_shape(self):
        self.assertEqual(self.m.shape, (3, 4))

    def test_ndim(self):
        self.assertEqual(self.m.ndim, 2)

    def test_nnz(self):
        self.assertEqual(self.m.nnz, 4)

    def test_conj(self):
        n = _make_complex(cupy, sparse, self.dtype)
        cupy.testing.assert_array_equal(n.conj().data, n.data.conj())

    def test_has_canonical_format(self):
        self.assertFalse(self.m.has_canonical_format)

    @unittest.skipUnless(scipy_available, 'requires scipy')
    def test_get(self):
        m = self.m.get()
        self.assertIsInstance(m, scipy.sparse.coo_matrix)
        expect = [
            [0, 1, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 3, 0]
        ]
        numpy.testing.assert_allclose(m.toarray(), expect)

    @unittest.skipUnless(scipy_available, 'requires scipy')
    def test_str(self):
        if numpy.dtype(self.dtype).kind == 'f':
            expect = '''  (0, 0)\t0.0
  (0, 1)\t1.0
  (1, 3)\t2.0
  (2, 2)\t3.0'''
        elif numpy.dtype(self.dtype).kind == 'c':
            expect = '''  (0, 0)\t0j
  (0, 1)\t(1+0j)
  (1, 3)\t(2+0j)
  (2, 2)\t(3+0j)'''
        self.assertEqual(str(self.m), expect)

    def test_toarray(self):
        m = self.m.toarray()
        expect = [
            [0, 1, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 3, 0]
        ]
        cupy.testing.assert_allclose(m, expect)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCooMatrixInit(unittest.TestCase):

    def setUp(self):
        self.shape = (3, 4)

    def data(self, xp):
        return xp.array([0, 1, 2, 3], self.dtype)

    def row(self, xp):
        return xp.array([0, 0, 1, 2], 'i')

    def col(self, xp):
        return xp.array([0, 1, 3, 2], 'i')

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_shape_none(self, xp, sp):
        x = sp.coo_matrix(
            (self.data(xp), (self.row(xp), self.col(xp))), shape=None)
        self.assertEqual(x.shape, (3, 4))

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_dtype(self, xp, sp):
        data = self.data(xp).real.astype('i')
        x = sp.coo_matrix(
            (data, (self.row(xp), self.col(xp))), dtype=self.dtype)
        self.assertEqual(x.dtype, self.dtype)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_copy_true(self, xp, sp):
        data = self.data(xp)
        row = self.row(xp)
        col = self.col(xp)
        x = sp.coo_matrix((data, (row, col)), copy=True)

        self.assertIsNot(data, x.data)
        self.assertIsNot(row, x.row)
        self.assertIsNot(col, x.col)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_invalid_format(self, xp, sp):
        sp.coo_matrix(
            (self.data(xp), self.row(xp)), shape=self.shape)

    @testing.numpy_cupy_allclose(sp_name='sp', atol=1e-5)
    def test_intlike_shape(self, xp, sp):
        s = sp.coo_matrix((self.data(xp), (self.row(xp), self.col(xp))),
                          shape=(xp.array(self.shape[0]),
                                 xp.int32(self.shape[1])))
        assert isinstance(s.shape[0], int)
        assert isinstance(s.shape[1], int)
        return s

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_shape_invalid(self, xp, sp):
        sp.coo_matrix(
            (self.data(xp), (self.row(xp), self.col(xp))), shape=(2,))

    def test_data_invalid(self):
        with self.assertRaises(ValueError):
            sparse.coo_matrix(
                ('invalid', (self.row(cupy), self.col(cupy))),
                shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_data_invalid_ndim(self, xp, sp):
        sp.coo_matrix(
            (self.data(xp)[None], (self.row(xp), self.col(xp))),
            shape=self.shape)

    def test_row_invalid(self):
        with self.assertRaises(ValueError):
            sparse.coo_matrix(
                (self.data(cupy), ('invalid', self.col(cupy))),
                shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_row_invalid_ndim(self, xp, sp):
        sp.coo_matrix(
            (self.data(xp), (self.row(xp)[None], self.col(xp))),
            shape=self.shape)

    def test_col_invalid(self):
        with self.assertRaises(ValueError):
            sparse.coo_matrix(
                (self.data(cupy), (self.row(cupy), 'invalid')),
                shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_col_invalid_ndim(self, xp, sp):
        sp.coo_matrix(
            (self.data(xp), (self.row(xp), self.col(xp)[None])),
            shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_data_different_length(self, xp, sp):
        data = xp.arange(5, dtype=self.dtype)
        sp.coo_matrix(
            (data(xp), (self.row(xp), self.col(xp))), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_row_different_length(self, xp, sp):
        row = xp.arange(5, dtype=self.dtype)
        sp.coo_matrix(
            (self.data(xp), (row(xp), self.col(xp))), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_col_different_length(self, xp, sp):
        col = xp.arange(5, dtype=self.dtype)
        sp.coo_matrix(
            (self.data(xp), (self.row(xp), col(xp))), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_fail_to_infer_shape(self, xp, sp):
        data = xp.array([], dtype=self.dtype)
        row = xp.array([], dtype='i')
        col = xp.array([], dtype='i')
        sp.coo_matrix((data, (row, col)), shape=None)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_row_too_large(self, xp, sp):
        row = xp.array([0, 0, 1, 3], 'i')
        sp.coo_matrix(
            (self.data(xp), (row, self.col(xp))), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_row_too_small(self, xp, sp):
        row = xp.array([0, -1, 1, 2], 'i')
        sp.coo_matrix(
            (self.data(xp), (row, self.col(xp))), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_col_too_large(self, xp, sp):
        col = xp.array([0, 1, 4, 2], 'i')
        sp.coo_matrix(
            (self.data(xp), (self.row(xp), col)), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_col_too_small(self, xp, sp):
        col = xp.array([0, -1, 3, 2], 'i')
        sp.coo_matrix(
            (self.data(xp), (self.row(xp), col)), shape=self.shape)

    def test_unsupported_dtype(self):
        with self.assertRaises(ValueError):
            sparse.coo_matrix(
                (self.data(cupy), (self.row(cupy), self.col(cupy))),
                shape=self.shape, dtype='i')

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_conj(self, xp, sp):
        n = _make_complex(xp, sp, self.dtype)
        cupy.testing.assert_array_equal(n.conj().data, n.data.conj())


@testing.parameterize(*testing.product({
    'make_method': [
        '_make', '_make_unordered', '_make_empty', '_make_duplicate',
        '_make_shape'],
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCooMatrixScipyComparison(unittest.TestCase):

    @property
    def make(self):
        return globals()[self.make_method]

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_dtype(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.dtype

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_nnz(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.getnnz()

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_asfptype(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.asfptype()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_A(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.A

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocoo(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.tocoo()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocoo_copy(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = m.tocoo(copy=True)
        self.assertIsNot(m.data, n.data)
        self.assertIsNot(m.row, n.row)
        self.assertIsNot(m.col, n.col)
        return n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.tocsc()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc_copy(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = m.tocsc(copy=True)
        self.assertIsNot(m.data, n.data)
        return n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.tocsr()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsr_copy(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = m.tocsr(copy=True)
        self.assertIsNot(m.data, n.data)
        return n

    # dot
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.dot(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_numpy_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.dot(numpy.dtype(self.dtype).type(2.0))

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_csr(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return m.dot(x)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_dot_csr_invalid_shape(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = sp.csr_matrix((5, 3), dtype=self.dtype)
        m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_csc(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocsc()
        return m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_sparse(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocoo()
        return m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_zero_dim(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.array(2, dtype=self.dtype)
        return m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_dense_vector(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(4).astype(self.dtype)
        return m.dot(x)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_dot_dense_vector_invalid_shape(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(5).astype(self.dtype)
        m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_dense_matrix(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(8).reshape(4, 2).astype(self.dtype)
        return m.dot(x)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_dot_dense_matrix_invalid_shape(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(10).reshape(5, 2).astype(self.dtype)
        m.dot(x)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_dot_dense_ndim3(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
        m.dot(x)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_dot_unsupported(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        m.dot(None)

    # __add__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add_zero(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m + 0

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_add_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        m + 1

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add_csr(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = _make2(xp, sp, self.dtype)
        return m + n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add_coo(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = _make2(xp, sp, self.dtype).tocoo()
        return m + n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add_dense(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return m + n

    # __radd__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_radd_zero(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return 0 + m

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_radd_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        1 + m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_radd_dense(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return n + m

    # __sub__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_zero(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m - 0

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_sub_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        m - 1

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_csr(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = _make2(xp, sp, self.dtype)
        return m - n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_coo(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = _make2(xp, sp, self.dtype).tocoo()
        return m - n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_dense(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return m - n

    # __rsub__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rsub_zero(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return 0 - m

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_rsub_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        1 - m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rsub_dense(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return n - m

    # __mul__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m * 2.0

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_numpy_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m * numpy.dtype(self.dtype).type(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_csr(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return m * x

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_mul_csr_invalid_shape(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = sp.csr_matrix((5, 3), dtype=self.dtype)
        m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_csc(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocsc()
        return m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_sparse(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocoo()
        return m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_zero_dim(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.array(2, dtype=self.dtype)
        return m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_dense_vector(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(4).astype(self.dtype)
        return m * x

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_mul_dense_vector_invalid_shape(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(5).astype(self.dtype)
        m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_dense_matrix(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(8).reshape(4, 2).astype(self.dtype)
        return m * x

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_mul_dense_matrix_invalid_shape(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(10).reshape(5, 2).astype(self.dtype)
        m * x

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_mul_dense_ndim3(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
        m * x

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_mul_unsupported(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        m * None

    # __rmul__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return 2.0 * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_numpy_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return numpy.dtype(self.dtype).type(2.0) * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_csr(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_csc(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocsc()
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_sparse(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocoo()
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_zero_dim(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.array(2, dtype=self.dtype)
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_dense_matrix(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(12).reshape(4, 3).astype(self.dtype)
        return x * m

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_rmul_dense_ndim3(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
        x * m

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_rmul_unsupported(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        None * m

    # __pow__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_pow_0(self, xp, sp):
        m = _make_square(xp, sp, self.dtype)
        return m ** 0

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_pow_1(self, xp, sp):
        m = _make_square(xp, sp, self.dtype)
        return m ** 1

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_pow_2(self, xp, sp):
        m = _make_square(xp, sp, self.dtype)
        return m ** 2

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_pow_3(self, xp, sp):
        m = _make_square(xp, sp, self.dtype)
        return m ** 3

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_pow_neg(self, xp, sp):
        m = _make_square(xp, sp, self.dtype)
        m ** -1

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
    def test_transpose(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.transpose()

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_transpose_axes_int(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        m.transpose(axes=0)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_eliminate_zeros(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.eliminate_zeros()
        return m.nnz


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'ret_dtype': [None, numpy.float32, numpy.float64],
    'axis': [None, 0, 1, -1, -2],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCooMatrixSum(unittest.TestCase):

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


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCooMatrixSumDuplicates(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_duplicates(self, xp, sp):
        m = _make_duplicate(xp, sp, self.dtype)
        self.assertFalse(m.has_canonical_format)
        m.sum_duplicates()
        self.assertTrue(m.has_canonical_format)
        self.assertEqual(m.nnz, 3)

        m.sum_duplicates()
        self.assertTrue(m.has_canonical_format)
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_duplicates_canonical(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        self.assertFalse(m.has_canonical_format)
        m.sum_duplicates()
        self.assertTrue(m.has_canonical_format)
        self.assertEqual(m.nnz, 4)
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_duplicates_empty(self, xp, sp):
        m = _make_empty(xp, sp, self.dtype)
        self.assertFalse(m.has_canonical_format)
        m.sum_duplicates()
        self.assertTrue(m.has_canonical_format)
        self.assertEqual(m.nnz, 0)
        return m


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'ufunc': [
        'arcsin', 'arcsinh', 'arctan', 'arctanh', 'ceil', 'deg2rad', 'expm1',
        'floor', 'log1p', 'rad2deg', 'rint', 'sign', 'sin', 'sinh', 'sqrt',
        'tan', 'tanh', 'trunc',
    ],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestUfunc(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp', atol=1e-5)
    def test_ufun(self, xp, sp):
        x = _make(xp, sp, self.dtype)
        x.data *= 0.1
        func = getattr(x, self.ufunc)
        complex_unsupported = {'ceil', 'deg2rad', 'floor', 'rad2deg', 'trunc'}
        if (numpy.dtype(self.dtype).kind == 'c' and
                self.ufunc in complex_unsupported):
            with self.assertRaises(TypeError):
                func()
            return numpy.array(0)
        else:
            return func()


class TestIsspmatrixCoo(unittest.TestCase):

    def test_coo(self):
        x = sparse.coo_matrix(
            (cupy.array([0], 'f'),
             (cupy.array([0], 'i'), cupy.array([0], 'i'))),
            shape=(1, 1), dtype='f')
        self.assertTrue(sparse.isspmatrix_coo(x))

    def test_csr(self):
        x = sparse.csr_matrix(
            (cupy.array([], 'f'),
             cupy.array([], 'i'),
             cupy.array([0], 'i')),
            shape=(0, 0), dtype='f')
        self.assertFalse(sparse.isspmatrix_coo(x))
