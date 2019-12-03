import unittest

import numpy
import pytest
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
    indices = xp.array([0, 1, 3, 2], 'i')
    indptr = xp.array([0, 2, 3, 4], 'i')
    # 0, 1, 0, 0
    # 0, 0, 0, 2
    # 0, 0, 3, 0
    return sp.csr_matrix((data, indices, indptr), shape=(3, 4))


def _make_complex(xp, sp, dtype):
    data = xp.array([0, 1, 2, 3], dtype)
    if dtype in [numpy.complex64, numpy.complex128]:
        data = data - 1j
    indices = xp.array([0, 1, 3, 2], 'i')
    indptr = xp.array([0, 2, 3, 4], 'i')
    # 0, 1 - 1j, 0, 0
    # 0, 0, 0, 2 - 1j
    # 0, 0, 3 - 1j, 0
    return sp.csr_matrix((data, indices, indptr), shape=(3, 4))


def _make2(xp, sp, dtype):
    data = xp.array([1, 2, 3, 4], dtype)
    indices = xp.array([2, 1, 2, 2], 'i')
    indptr = xp.array([0, 1, 3, 4], 'i')
    # 0, 0, 1, 0
    # 0, 2, 3, 0
    # 0, 0, 4, 0
    return sp.csr_matrix((data, indices, indptr), shape=(3, 4))


def _make3(xp, sp, dtype):
    data = xp.array([1, 2, 3, 4, 5], dtype)
    indices = xp.array([0, 2, 1, 0, 2], 'i')
    indptr = xp.array([0, 1, 3, 3, 5], 'i')
    # 1, 0, 0
    # 0, 3, 2
    # 0, 0, 0
    # 4, 0, 5
    return sp.csr_matrix((data, indices, indptr), shape=(4, 3))


def _make_unordered(xp, sp, dtype):
    data = xp.array([1, 2, 3, 4], dtype)
    indices = xp.array([1, 0, 1, 2], 'i')
    indptr = xp.array([0, 2, 3, 4], 'i')
    # 2, 1, 0, 0
    # 0, 3, 0, 0
    # 0, 0, 4, 0
    return sp.csr_matrix((data, indices, indptr), shape=(3, 4))


def _make_duplicate(xp, sp, dtype):
    data = xp.array([0, 1, 3, 2, 4, 5], dtype)
    indices = xp.array([0, 0, 0, 2, 0, 2], 'i')
    indptr = xp.array([0, 3, 6, 6], 'i')
    # 4, 0, 0, 0
    # 4, 0, 7, 0
    # 0, 0, 0, 0
    return sp.csr_matrix((data, indices, indptr), shape=(3, 4))


def _make_empty(xp, sp, dtype):
    data = xp.array([], dtype)
    indices = xp.array([], 'i')
    indptr = xp.array([0, 0, 0, 0], 'i')
    return sp.csr_matrix((data, indices, indptr), shape=(3, 4))


def _make_square(xp, sp, dtype):
    data = xp.array([0, 1, 2, 3], dtype)
    indices = xp.array([0, 1, 0, 2], 'i')
    indptr = xp.array([0, 2, 3, 4], 'i')
    # 0, 1, 0
    # 2, 0, 0
    # 0, 0, 3
    return sp.csr_matrix((data, indices, indptr), shape=(3, 3))


def _make_shape(xp, sp, dtype):
    return sp.csr_matrix((3, 4))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
class TestCsrMatrix(unittest.TestCase):

    def setUp(self):
        self.m = _make(cupy, sparse, self.dtype)

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

    def test_init_copy(self):
        n = sparse.csr_matrix(self.m)
        self.assertIsNot(n, self.m)
        cupy.testing.assert_array_equal(n.data, self.m.data)
        cupy.testing.assert_array_equal(n.indices, self.m.indices)
        cupy.testing.assert_array_equal(n.indptr, self.m.indptr)
        self.assertEqual(n.shape, self.m.shape)

    def test_init_copy_other_sparse(self):
        n = sparse.csr_matrix(self.m.tocsc())
        cupy.testing.assert_array_equal(n.data, self.m.data)
        cupy.testing.assert_array_equal(n.indices, self.m.indices)
        cupy.testing.assert_array_equal(n.indptr, self.m.indptr)
        self.assertEqual(n.shape, self.m.shape)

    @unittest.skipUnless(scipy_available, 'requires scipy')
    def test_init_copy_scipy_sparse(self):
        m = _make(numpy, scipy.sparse, self.dtype)
        n = sparse.csr_matrix(m)
        self.assertIsInstance(n.data, cupy.ndarray)
        self.assertIsInstance(n.indices, cupy.ndarray)
        self.assertIsInstance(n.indptr, cupy.ndarray)
        cupy.testing.assert_array_equal(n.data, m.data)
        cupy.testing.assert_array_equal(n.indices, m.indices)
        cupy.testing.assert_array_equal(n.indptr, m.indptr)
        self.assertEqual(n.shape, m.shape)

    @unittest.skipUnless(scipy_available, 'requires scipy')
    def test_init_copy_other_scipy_sparse(self):
        m = _make(numpy, scipy.sparse, self.dtype)
        n = sparse.csr_matrix(m.tocsc())
        self.assertIsInstance(n.data, cupy.ndarray)
        self.assertIsInstance(n.indices, cupy.ndarray)
        self.assertIsInstance(n.indptr, cupy.ndarray)
        cupy.testing.assert_array_equal(n.data, m.data)
        cupy.testing.assert_array_equal(n.indices, m.indices)
        cupy.testing.assert_array_equal(n.indptr, m.indptr)
        self.assertEqual(n.shape, m.shape)

    def test_init_dense(self):
        m = cupy.array([[0, 1, 0, 2],
                        [0, 0, 0, 0],
                        [0, 0, 3, 0]], dtype=self.dtype)
        n = sparse.csr_matrix(m)
        self.assertEqual(n.nnz, 3)
        self.assertEqual(n.shape, (3, 4))
        cupy.testing.assert_array_equal(n.data, [1, 2, 3])
        cupy.testing.assert_array_equal(n.indices, [1, 3, 2])
        cupy.testing.assert_array_equal(n.indptr, [0, 2, 2, 3])

    def test_init_dense_empty(self):
        m = cupy.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]], dtype=self.dtype)
        n = sparse.csr_matrix(m)
        self.assertEqual(n.nnz, 0)
        self.assertEqual(n.shape, (3, 4))
        cupy.testing.assert_array_equal(n.data, [])
        cupy.testing.assert_array_equal(n.indices, [])
        cupy.testing.assert_array_equal(n.indptr, [0, 0, 0, 0])

    def test_init_dense_one_dim(self):
        m = cupy.array([0, 1, 0, 2], dtype=self.dtype)
        n = sparse.csr_matrix(m)
        self.assertEqual(n.nnz, 2)
        self.assertEqual(n.shape, (1, 4))
        cupy.testing.assert_array_equal(n.data, [1, 2])
        cupy.testing.assert_array_equal(n.indices, [1, 3])
        cupy.testing.assert_array_equal(n.indptr, [0, 2])

    def test_init_dense_zero_dim(self):
        m = cupy.array(1, dtype=self.dtype)
        n = sparse.csr_matrix(m)
        self.assertEqual(n.nnz, 1)
        self.assertEqual(n.shape, (1, 1))
        cupy.testing.assert_array_equal(n.data, [1])
        cupy.testing.assert_array_equal(n.indices, [0])
        cupy.testing.assert_array_equal(n.indptr, [0, 1])

    @unittest.skipUnless(scipy_available, 'requires scipy')
    @testing.numpy_cupy_raises(sp_name='sp', accept_error=TypeError)
    def test_init_dense_invalid_ndim(self, xp, sp):
        m = xp.zeros((1, 1, 1), dtype=self.dtype)
        sp.csr_matrix(m)

    def test_copy(self):
        n = self.m.copy()
        self.assertIsInstance(n, sparse.csr_matrix)
        self.assertIsNot(n, self.m)
        self.assertIsNot(n.data, self.m.data)
        self.assertIsNot(n.indices, self.m.indices)
        self.assertIsNot(n.indptr, self.m.indptr)
        cupy.testing.assert_array_equal(n.data, self.m.data)
        cupy.testing.assert_array_equal(n.indices, self.m.indices)
        cupy.testing.assert_array_equal(n.indptr, self.m.indptr)
        self.assertEqual(n.shape, self.m.shape)

    def test_shape(self):
        self.assertEqual(self.m.shape, (3, 4))

    def test_ndim(self):
        self.assertEqual(self.m.ndim, 2)

    def test_nnz(self):
        self.assertEqual(self.m.nnz, 4)

    def test_conj(self):
        n = _make_complex(cupy, sparse, self.dtype)
        cupy.testing.assert_array_equal(n.conj().data, n.data.conj())

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
        self.assertTrue(m.flags.c_contiguous)
        cupy.testing.assert_allclose(m, expect)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCsrMatrixInit(unittest.TestCase):

    def setUp(self):
        self.shape = (3, 4)

    def data(self, xp):
        return xp.array([1, 2, 3, 4], self.dtype)

    def indices(self, xp):
        return xp.array([0, 1, 3, 2], 'i')

    def indptr(self, xp):
        return xp.array([0, 2, 3, 4], 'i')

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_shape_none(self, xp, sp):
        x = sp.csr_matrix(
            (self.data(xp), self.indices(xp), self.indptr(xp)), shape=None)
        self.assertEqual(x.shape, (3, 4))

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_dtype(self, xp, sp):
        data = self.data(xp).real.astype('i')
        x = sp.csr_matrix(
            (data, self.indices(xp), self.indptr(xp)), dtype=self.dtype)
        self.assertEqual(x.dtype, self.dtype)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_copy_true(self, xp, sp):
        data = self.data(xp)
        indices = self.indices(xp)
        indptr = self.indptr(xp)
        x = sp.csr_matrix((data, indices, indptr), copy=True)

        self.assertIsNot(data, x.data)
        self.assertIsNot(indices, x.indices)
        self.assertIsNot(indptr, x.indptr)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_init_with_shape(self, xp, sp):
        s = sp.csr_matrix(self.shape)
        self.assertEqual(s.shape, self.shape)
        self.assertEqual(s.dtype, 'd')
        self.assertEqual(s.size, 0)
        return s

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_init_with_shape_and_dtype(self, xp, sp):
        s = sp.csr_matrix(self.shape, dtype=self.dtype)
        self.assertEqual(s.shape, self.shape)
        self.assertEqual(s.dtype, self.dtype)
        self.assertEqual(s.size, 0)
        return s

    @testing.numpy_cupy_allclose(sp_name='sp', atol=1e-5)
    def test_intlike_shape(self, xp, sp):
        s = sp.csr_matrix((self.data(xp), self.indices(xp), self.indptr(xp)),
                          shape=(xp.array(self.shape[0]),
                                 xp.int32(self.shape[1])))
        assert isinstance(s.shape[0], int)
        assert isinstance(s.shape[1], int)
        return s

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
            sparse.csr_matrix(
                (self.data(cupy), self.indices(cupy), self.indptr(cupy)),
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
class TestCsrMatrixScipyComparison(unittest.TestCase):

    @property
    def make(self):
        return globals()[self.make_method]

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=TypeError)
    def test_len(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        len(m)

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_iter(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        rows = []
        for r in m:
            rows.append(r)
            self.assertIsInstance(r, sp.spmatrix)
        self.assertEqual(len(rows), 3)
        return xp.concatenate([r.toarray() for r in rows])

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_asfptype(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.asfptype().toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        a = m.toarray()
        self.assertTrue(a.flags.c_contiguous)
        return a

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray_c_order(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        a = m.toarray(order='C')
        self.assertTrue(a.flags.c_contiguous)
        return a

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray_f_order(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        a = m.toarray(order='F')
        self.assertTrue(a.flags.f_contiguous)
        return a

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=TypeError)
    def test_toarray_unknown_order(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.toarray(order='#')

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_A(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.A

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
        m = self.make(xp, sp, self.dtype)
        return m.tocsc()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc_copy(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = m.tocsc(copy=True)
        self.assertIsNot(m.data, n.data)
        self.assertIsNot(m.indices, n.indices)
        self.assertIsNot(m.indptr, n.indptr)
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
        self.assertIsNot(m.indices, n.indices)
        self.assertIsNot(m.indptr, n.indptr)
        return n

    # dot
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.dot(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_numpy_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.dot(numpy.dtype(self.dtype).type(2.0)).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_csr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return m.dot(x)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_dot_csr_invalid_shape(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = sp.csr_matrix((5, 3), dtype=self.dtype)
        m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_csc(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocsc()
        return m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_sparse(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocoo()
        return m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_zero_dim(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.array(2, dtype=self.dtype)
        return m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_dense_vector(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(4).astype(self.dtype)
        return m.dot(x)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_dot_dense_vector_invalid_shape(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(5).astype(self.dtype)
        m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_dot_dense_matrix(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(8).reshape(4, 2).astype(self.dtype)
        return m.dot(x)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_dot_dense_matrix_invalid_shape(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(10).reshape(5, 2).astype(self.dtype)
        m.dot(x)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_dot_dense_ndim3(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
        m.dot(x)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_dot_unsupported(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.dot(None)

    # __add__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add_zero(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m + 0

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_add_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m + 1

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add_csr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = _make2(xp, sp, self.dtype)
        return m + n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add_coo(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = _make2(xp, sp, self.dtype).tocoo()
        return m + n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add_dense(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return m + n

    # __radd__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_radd_zero(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return (0 + m).toarray()

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_radd_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        1 + m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_radd_dense(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return n + m

    # __sub__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_zero(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return (m - 0).toarray()

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_sub_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m - 1

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_csr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = _make2(xp, sp, self.dtype)
        return (m - n).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_coo(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = _make2(xp, sp, self.dtype).tocoo()
        return m - n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_dense(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return m - n

    # __rsub__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rsub_zero(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return 0 - m

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_rsub_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        1 - m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rsub_dense(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return n - m

    # __mul__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m * 2.0

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_numpy_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m * numpy.dtype(self.dtype).type(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_csr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return m * x

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_mul_csr_invalid_shape(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = sp.csr_matrix((5, 3), dtype=self.dtype)
        m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_csc(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocsc()
        return m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_sparse(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocoo()
        return m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_zero_dim(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.array(2, dtype=self.dtype)
        return m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_dense_vector(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(4).astype(self.dtype)
        return m * x

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_mul_dense_vector_invalid_shape(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(5).astype(self.dtype)
        m * x

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_mul_dense_matrix(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(8).reshape(4, 2).astype(self.dtype)
        return m * x

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_mul_dense_matrix_invalid_shape(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(10).reshape(5, 2).astype(self.dtype)
        m * x

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_mul_dense_ndim3(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
        m * x

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_mul_unsupported(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m * None

    # __rmul__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return 2.0 * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_numpy_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return numpy.dtype(self.dtype).type(2.0) * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_csr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_csc(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocsc()
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_sparse(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocoo()
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_zero_dim(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.array(2, dtype=self.dtype)
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_rmul_dense_matrix(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(12).reshape(4, 3).astype(self.dtype)
        return x * m

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_rmul_dense_ndim3(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
        x * m

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_rmul_unsupported(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        if m.nnz == 0:
            # When there is no element, a SciPy's sparse matrix does not raise
            # an error when it is multiplied with None.
            pytest.skip()
        None * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sort_indices(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.sort_indices()
        return m

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_sum_tuple_axis(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.sum(axis=(0, 1))

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_sum_str_axis(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.sum(axis='test')

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_sum_too_large_axis(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.sum(axis=3)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_duplicates(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.sum_duplicates()
        self.assertTrue(m.has_canonical_format)
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_transpose(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.transpose()

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_transpose_axes_int(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.transpose(axes=0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_eliminate_zeros(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.eliminate_zeros()
        return m

    @testing.numpy_cupy_equal(sp_name='sp')
    @unittest.skipIf(
        cupy.cuda.runtime.runtimeGetVersion() < 8000,
        'CUDA <8 cannot keep number of non-zero entries ')
    def test_eliminate_zeros_nnz(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.eliminate_zeros()
        return m.nnz


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCsrMatrixPowScipyComparison(unittest.TestCase):

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
        return m ** 3

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_pow_neg(self, xp, sp):
        m = _make_square(xp, sp, self.dtype)
        m ** -1

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=TypeError)
    def test_pow_not_square(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        m ** 2

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=ValueError)
    def test_pow_float(self, xp, sp):
        m = _make_square(xp, sp, self.dtype)
        m ** 1.5

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=TypeError)
    def test_pow_list(self, xp, sp):
        m = _make_square(xp, sp, self.dtype)
        m ** []


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'ret_dtype': [None, numpy.float32, numpy.float64],
    'axis': [None, 0, 1, -1, -2],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCsrMatrixSum(unittest.TestCase):

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
class TestCsrMatrixScipyCompressed(unittest.TestCase):

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_get_shape(self, xp, sp):
        return _make(xp, sp, self.dtype).get_shape()

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_getnnz(self, xp, sp):
        return _make(xp, sp, self.dtype).getnnz()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCsrMatrixData(unittest.TestCase):

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_dtype(self, xp, sp):
        return _make(xp, sp, self.dtype).dtype

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_abs(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return abs(m)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_neg(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return -m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_astype(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        if numpy.dtype(self.dtype).kind == 'c':
            t = 'D'
        else:
            t = 'd'
        return m.astype(t)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_count_nonzero(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.count_nonzero()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.power(2)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_with_dtype(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        if numpy.dtype(self.dtype).kind == 'c':
            t = 'D'
        else:
            t = 'd'
        return m.power(2, t)


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


class TestIsspmatrixCsr(unittest.TestCase):

    def test_csr(self):
        x = sparse.csr_matrix(
            (cupy.array([], 'f'),
             cupy.array([], 'i'),
             cupy.array([0], 'i')),
            shape=(0, 0), dtype='f')
        self.assertTrue(sparse.isspmatrix_csr(x))

    def test_csc(self):
        x = sparse.csr_matrix(
            (cupy.array([], 'f'),
             cupy.array([], 'i'),
             cupy.array([0], 'i')),
            shape=(0, 0), dtype='f')
        self.assertFalse(sparse.isspmatrix_csc(x))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, cupy.complex64, cupy.complex128],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestCsrMatrixGetitem(unittest.TestCase):

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_getitem_int_int(self, xp, sp):
        self.assertEqual(_make(xp, sp, self.dtype)[0, 1], 1)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_getitem_int_int_not_found(self, xp, sp):
        self.assertEqual(_make(xp, sp, self.dtype)[1, 1], 0)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_getitem_int_int_negative(self, xp, sp):
        self.assertEqual(_make(xp, sp, self.dtype)[-1, -2], 3)

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=IndexError)
    def test_getitem_int_int_too_small_row(self, xp, sp):
        _make(xp, sp, self.dtype)[-4, 0]

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=IndexError)
    def test_getitem_int_int_too_large_row(self, xp, sp):
        _make(xp, sp, self.dtype)[3, 0]

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=IndexError)
    def test_getitem_int_int_too_small_col(self, xp, sp):
        _make(xp, sp, self.dtype)[0, -5]

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=IndexError)
    def test_getitem_int_int_too_large_col(self, xp, sp):
        _make(xp, sp, self.dtype)[0, 4]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_int(self, xp, sp):
        return _make(xp, sp, self.dtype)[1]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_int_negative(self, xp, sp):
        return _make(xp, sp, self.dtype)[-1]

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=IndexError)
    def test_getitem_int_to_small(self, xp, sp):
        _make(xp, sp, self.dtype)[-4]

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=IndexError)
    def test_getitem_int_to_large(self, xp, sp):
        _make(xp, sp, self.dtype)[3]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_int_none_slice(self, xp, sp):
        return _make(xp, sp, self.dtype)[1, :]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_negative_int_none_slice(self, xp, sp):
        return _make(xp, sp, self.dtype)[-1, :]

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=IndexError)
    def test_getitem_int_too_small_none_slice(self, xp, sp):
        _make(xp, sp, self.dtype)[-4, :]

    @testing.numpy_cupy_raises(sp_name='sp', accept_error=IndexError)
    def test_getitem_int_too_large_none_slice(self, xp, sp):
        _make(xp, sp, self.dtype)[3, :]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice(self, xp, sp):
        return _make(xp, sp, self.dtype)[1:3]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_negative(self, xp, sp):
        return _make(xp, sp, self.dtype)[-2:-1]

    # avoid segfault: https://github.com/scipy/scipy/issues/11125
    @testing.with_requires('scipy!=1.3.*')
    @testing.numpy_cupy_raises(sp_name='sp', accept_error=IndexError)
    def test_getitem_slice_start_larger_than_stop(self, xp, sp):
        _make(xp, sp, self.dtype)[3:2]

    def test_getitem_slice_step_2(self):
        with self.assertRaises(ValueError):
            _make(cupy, sparse, self.dtype)[0::2]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_ellipsis(self, xp, sp):
        return _make(xp, sp, self.dtype)[...]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_int_ellipsis(self, xp, sp):
        return _make(xp, sp, self.dtype)[1, ...]


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, cupy.complex64, cupy.complex128],
}))
@testing.with_requires('scipy>=1.0')
class TestCsrMatrixGetitem2(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_start_too_small(self, xp, sp):
        return _make(xp, sp, self.dtype)[-4:None]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_start_too_large(self, xp, sp):
        return _make(xp, sp, self.dtype)[4:None]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_stop_too_small(self, xp, sp):
        return _make(xp, sp, self.dtype)[None:-4]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_stop_too_large(self, xp, sp):
        return _make(xp, sp, self.dtype)[None:4]
