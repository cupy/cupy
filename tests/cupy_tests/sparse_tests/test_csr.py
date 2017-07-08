import unittest

import numpy
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
import cupy.sparse
from cupy import testing


def _make(xp, sp, dtype):
    data = xp.array([0, 1, 2, 3], dtype)
    indices = xp.array([0, 1, 3, 2], 'i')
    indptr = xp.array([0, 2, 3, 4], 'i')
    # 0, 1, 0, 0
    # 0, 0, 0, 2
    # 0, 0, 3, 0
    return sp.csr_matrix((data, indices, indptr), shape=(3, 4))


def _make_unordered(xp, sp, dtype):
    data = xp.array([1, 2, 3, 4], dtype)
    indices = xp.array([1, 0, 1, 2], 'i')
    indptr = xp.array([0, 2, 3, 4], 'i')
    # 2, 1, 0, 0
    # 0, 3, 0, 0
    # 0, 0, 4, 0
    return sp.csr_matrix((data, indices, indptr), shape=(3, 4))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
class TestCsrMatrix(unittest.TestCase):

    def setUp(self):
        self.m = _make(cupy, cupy.sparse, self.dtype)

    def test_dtype(self):
        self.assertEqual(self.m.dtype, self.dtype)

    def test_data(self):
        self.assertEqual(self.m.data.dtype, self.dtype)
        testing.assert_array_equal(
            self.m.data, cupy.array([0, 1, 2, 3], self.dtype))

    def test_indices(self):
        self.assertEqual(self.m.indices.dtype, numpy.int32)
        testing.assert_array_equal(
            self.m.indices, cupy.array([0, 1, 3, 2], self.dtype))

    def test_indptr(self):
        self.assertEqual(self.m.indptr.dtype, numpy.int32)
        testing.assert_array_equal(
            self.m.indptr, cupy.array([0, 2, 3, 4], self.dtype))

    def test_shape(self):
        self.assertEqual(self.m.shape, (3, 4))

    def test_ndim(self):
        self.assertEqual(self.m.ndim, 2)

    def test_nnz(self):
        self.assertEqual(self.m.nnz, 4)

    @unittest.skipUnless(scipy_available, 'requires scipy')
    def test_get(self):
        m = self.m.get()
        self.assertIsInstance(m, scipy.sparse.csr_matrix)
        expect = [
            [0, 1, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 3, 0]
        ]
        numpy.testing.assert_allclose(m.toarray(), expect)

    @unittest.skipUnless(scipy_available, 'requires scipy')
    def test_str(self):
        self.assertEqual(str(self.m), '''  (0, 0)\t0.0
  (0, 1)\t1.0
  (1, 3)\t2.0
  (2, 2)\t3.0''')

    def test_toarray(self):
        m = self.m.toarray()
        expect = [
            [0, 1, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 3, 0]
        ]
        cupy.testing.assert_allclose(m, expect)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCsrMatrixInvalidInit(unittest.TestCase):

    def setUp(self):
        self.shape = (3, 4)

    def data(self, xp):
        return xp.array([1, 2, 3, 4], self.dtype)

    def indices(self, xp):
        return xp.array([0, 1, 3, 2], 'i')

    def indptr(self, xp):
        return xp.array([0, 2, 3, 4], 'i')

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_shape_invalid(self, xp, sp):
        sp.csr_matrix(
            (self.data(xp), self.indices(xp), self.indptr(xp)), shape=(2,))

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_data_invalid(self, xp, sp):
        sp.csr_matrix(
            ('invalid', self.indices(xp), self.indptr(xp)), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_data_invalid_ndim(self, xp, sp):
        sp.csr_matrix(
            (self.data(xp)[None], self.indices(xp), self.indptr(xp)),
            shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_indices_invalid(self, xp, sp):
        sp.csr_matrix(
            (self.data(xp), 'invalid', self.indptr(xp)), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_indices_invalid_ndim(self, xp, sp):
        sp.csr_matrix(
            (self.data(xp), self.indices(xp)[None], self.indptr(xp)),
            shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_indptr_invalid(self, xp, sp):
        sp.csr_matrix(
            (self.data(xp), self.indices(xp), 'invalid'), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_indptr_invalid_ndim(self, xp, sp):
        sp.csr_matrix(
            (self.data(xp), self.indices(xp), self.indptr(xp)[None]),
            shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_data_indices_different_length(self, xp, sp):
        data = xp.arange(5, dtype=self.dtype)
        sp.csr_matrix(
            (data, self.indices(xp), self.indptr(xp)), shape=self.shape)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_indptr_invalid_length(self, xp, sp):
        indptr = xp.array([0, 1], 'i')
        sp.csr_matrix(
            (self.data(xp), self.indices(xp), indptr), shape=self.shape)

    def test_unsupported_dtype(self):
        with self.assertRaises(ValueError):
            cupy.sparse.csr_matrix(
                (self.data(cupy), self.indices(cupy), self.indptr(cupy)),
                shape=self.shape, dtype='i')


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCsrMatrixScipyComparison(unittest.TestCase):

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=TypeError)
    def test_len(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        len(m)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.tocsc().toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsr(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.tocsr().toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sort_indices(self, xp, sp):
        m = _make_unordered(xp, sp, self.dtype)
        m.sort_indices()
        return m.toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_transpose(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.transpose().toarray()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCsrMatrixScipyCompressed(unittest.TestCase):

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_get_shape(self, xp, sp):
        return _make(xp, sp, self.dtype).get_shape()

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_getnnz(self, xp, sp):
        return _make(xp, sp, self.dtype).getnnz()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCsrMatrixData(unittest.TestCase):

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_dtype(self, xp, sp):
        return _make(xp, sp, self.dtype).dtype

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_abs(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return abs(m).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_neg(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return (-m).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_astype(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.astype('d').toarray()

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_count_nonzero(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.count_nonzero()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.power(2).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_with_dtype(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.power(2, 'd').toarray()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
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
        return getattr(x, self.ufunc)().toarray()
