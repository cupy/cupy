import pickle
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


def _make_sum_dup(xp, sp, dtype):
    # 1 0 0
    # 1 1 0
    # 1 1 1
    data = xp.array([1, 1, 1, 1, 1, 1], dtype)
    row = xp.array([0, 1, 1, 2, 2, 2], 'i')
    col = xp.array([0, 0, 1, 0, 1, 2], 'i')
    return sp.coo_matrix((data, (row, col)), shape=(3, 3))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
class TestCooMatrix(unittest.TestCase):

    def setUp(self):
        self.m = _make(cupy, sparse, self.dtype)

    def test_dtype(self):
        assert self.m.dtype == self.dtype

    def test_data(self):
        assert self.m.data.dtype == self.dtype
        testing.assert_array_equal(
            self.m.data, cupy.array([0, 1, 2, 3], self.dtype))

    def test_row(self):
        assert self.m.row.dtype == numpy.int32
        testing.assert_array_equal(
            self.m.row, cupy.array([0, 0, 1, 2], self.dtype))

    def test_col(self):
        assert self.m.col.dtype == numpy.int32
        testing.assert_array_equal(
            self.m.col, cupy.array([0, 1, 3, 2], self.dtype))

    def test_init_copy(self):
        n = sparse.coo_matrix(self.m)
        assert n is not self.m
        cupy.testing.assert_array_equal(n.toarray(), self.m.toarray())

    def test_init_copy_other_sparse(self):
        n = sparse.coo_matrix(self.m.tocsr())
        cupy.testing.assert_array_equal(n.toarray(), self.m.toarray())

    @unittest.skipUnless(scipy_available, 'requires scipy')
    def test_init_copy_scipy_sparse(self):
        m = _make(numpy, scipy.sparse, self.dtype)
        n = sparse.coo_matrix(m)
        assert isinstance(n.data, cupy.ndarray)
        assert isinstance(n.row, cupy.ndarray)
        assert isinstance(n.col, cupy.ndarray)
        cupy.testing.assert_array_equal(n.data, m.data)
        cupy.testing.assert_array_equal(n.row, m.row)
        cupy.testing.assert_array_equal(n.col, m.col)
        assert n.shape == m.shape

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

    def test_pickle_roundtrip(self):
        s = _make(cupy, sparse, self.dtype)
        s2 = pickle.loads(pickle.dumps(s))
        assert s.shape == s2.shape
        assert s.dtype == s2.dtype
        if scipy_available:
            assert (s.get() != s2.get()).count_nonzero() == 0

    def test_shape(self):
        assert self.m.shape == (3, 4)

    def test_ndim(self):
        assert self.m.ndim == 2

    def test_nnz(self):
        assert self.m.nnz == 4

    def test_conj(self):
        n = _make_complex(cupy, sparse, self.dtype)
        cupy.testing.assert_array_equal(n.conj().data, n.data.conj())

    def test_has_canonical_format(self):
        assert self.m.has_canonical_format is False

    @unittest.skipUnless(scipy_available, 'requires scipy')
    def test_get(self):
        m = self.m.get()
        assert isinstance(m, scipy.sparse.coo_matrix)
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
        assert str(self.m) == expect

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
        assert x.shape == (3, 4)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_dtype(self, xp, sp):
        data = self.data(xp).real.astype('i')
        x = sp.coo_matrix(
            (data, (self.row(xp), self.col(xp))), dtype=self.dtype)
        assert x.dtype == self.dtype

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_copy_true(self, xp, sp):
        data = self.data(xp)
        row = self.row(xp)
        col = self.col(xp)
        x = sp.coo_matrix((data, (row, col)), copy=True)

        assert data is not x.data
        assert row is not x.row
        assert col is not x.col

    def test_init_dense(self):
        m = cupy.array([[0, 1, 0, 2],
                        [0, 0, 0, 0],
                        [0, 0, 3, 0]], dtype=self.dtype)
        n = sparse.coo_matrix(m)
        assert n.nnz == 3
        assert n.shape == (3, 4)
        cupy.testing.assert_array_equal(n.data, [1, 2, 3])
        cupy.testing.assert_array_equal(n.row, [0, 0, 2])
        cupy.testing.assert_array_equal(n.col, [1, 3, 2])

    def test_init_dense_allzero(self):
        m = cupy.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]], dtype=self.dtype)
        n = sparse.coo_matrix(m)
        assert n.nnz == 0
        assert n.shape == (3, 4)
        cupy.testing.assert_array_equal(n.data, [])
        cupy.testing.assert_array_equal(n.row, [])
        cupy.testing.assert_array_equal(n.col, [])

    def test_init_dense_check_if_row_major(self):
        rows, cols = 10, 9
        for order in ('C', 'F'):
            d = testing.shaped_random((rows, cols), dtype=self.dtype,
                                      order=order)
            mask = testing.shaped_random((rows, cols), scale=1.0)
            d[mask > 0.5] = 0
            s = sparse.coo_matrix(d)
            for i in range(s.nnz):
                assert 0 <= s.row[i] < rows
                assert 0 <= s.col[i] < cols
                assert s.data[i] == d[s.row[i], s.col[i]]
                if i == 0:
                    continue
                assert ((s.row[i-1] < s.row[i]) or
                        (s.row[i-1] == s.row[i] and s.col[i-1] < s.col[i]))
            assert s.has_canonical_format

    def test_invalid_format(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(TypeError):
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

    def test_shape_invalid(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.coo_matrix(
                    (self.data(xp), (self.row(xp), self.col(xp))),
                    shape=(2,))

    def test_data_invalid(self):
        with self.assertRaises(ValueError):
            sparse.coo_matrix(
                ('invalid', (self.row(cupy), self.col(cupy))),
                shape=self.shape)

    def test_data_invalid_ndim(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.coo_matrix(
                    (self.data(xp)[None], (self.row(xp), self.col(xp))),
                    shape=self.shape)

    def test_row_invalid(self):
        with self.assertRaises(ValueError):
            sparse.coo_matrix(
                (self.data(cupy), ('invalid', self.col(cupy))),
                shape=self.shape)

    def test_row_invalid_ndim(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.coo_matrix(
                    (self.data(xp), (self.row(xp)[None], self.col(xp))),
                    shape=self.shape)

    def test_col_invalid(self):
        with self.assertRaises(ValueError):
            sparse.coo_matrix(
                (self.data(cupy), (self.row(cupy), 'invalid')),
                shape=self.shape)

    def test_col_invalid_ndim(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.coo_matrix(
                    (self.data(xp), (self.row(xp), self.col(xp)[None])),
                    shape=self.shape)

    def test_data_different_length(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            data = xp.arange(5, dtype=self.dtype)
            with pytest.raises(TypeError):
                sp.coo_matrix(
                    (data(xp), (self.row(xp), self.col(xp))),
                    shape=self.shape)

    def test_row_different_length(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            row = xp.arange(5, dtype=self.dtype)
            with pytest.raises(TypeError):
                sp.coo_matrix(
                    (self.data(xp), (row(xp), self.col(xp))),
                    shape=self.shape)

    def test_col_different_length(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            col = xp.arange(5, dtype=self.dtype)
            with pytest.raises(TypeError):
                sp.coo_matrix(
                    (self.data(xp), (self.row(xp), col(xp))),
                    shape=self.shape)

    def test_fail_to_infer_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            data = xp.array([], dtype=self.dtype)
            row = xp.array([], dtype='i')
            col = xp.array([], dtype='i')
            with pytest.raises(ValueError):
                sp.coo_matrix((data, (row, col)), shape=None)

    def test_row_too_large(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            row = xp.array([0, 0, 1, 3], 'i')
            with pytest.raises(ValueError):
                sp.coo_matrix(
                    (self.data(xp), (row, self.col(xp))),
                    shape=self.shape)

    def test_row_too_small(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            row = xp.array([0, -1, 1, 2], 'i')
            with pytest.raises(ValueError):
                sp.coo_matrix(
                    (self.data(xp), (row, self.col(xp))),
                    shape=self.shape)

    def test_col_too_large(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            col = xp.array([0, 1, 4, 2], 'i')
            with pytest.raises(ValueError):
                sp.coo_matrix(
                    (self.data(xp), (self.row(xp), col)),
                    shape=self.shape)

    def test_col_too_small(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            col = xp.array([0, -1, 3, 2], 'i')
            with pytest.raises(ValueError):
                sp.coo_matrix(
                    (self.data(xp), (self.row(xp), col)),
                    shape=self.shape)

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
        assert m.data is not n.data
        assert m.row is not n.row
        assert m.col is not n.col
        return n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        out = m.tocsc()
        assert out.has_canonical_format
        return out

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc_copy(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = m.tocsc(copy=True)
        assert m.data is not n.data
        assert n.has_canonical_format
        return n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        out = m.tocsr()
        assert out.has_canonical_format
        return out

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsr_copy(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = m.tocsr(copy=True)
        assert m.data is not n.data
        assert n.has_canonical_format
        return n

    # dot
    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_dot_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.dot(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_dot_numpy_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.dot(numpy.dtype(self.dtype).type(2.0))

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_csr(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return m.dot(x)

    def test_dot_csr_invalid_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            x = sp.csr_matrix((5, 3), dtype=self.dtype)
            with pytest.raises(ValueError):
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

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_dot_zero_dim(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.array(2, dtype=self.dtype)
        return m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_dense_vector(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(4).astype(self.dtype)
        return m.dot(x)

    def test_dot_dense_vector_invalid_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            x = xp.arange(5).astype(self.dtype)
            with pytest.raises(ValueError):
                m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_dense_matrix(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(8).reshape(4, 2).astype(self.dtype)
        return m.dot(x)

    def test_dot_dense_matrix_invalid_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            x = xp.arange(10).reshape(5, 2).astype(self.dtype)
            with pytest.raises(ValueError):
                m.dot(x)

    def test_dot_dense_ndim3(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
            with pytest.raises(ValueError):
                m.dot(x)

    def test_dot_unsupported(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(TypeError):
                m.dot(None)

    # __add__
    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_add_zero(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m + 0

    def test_add_scalar(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(NotImplementedError):
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
    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_radd_zero(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return 0 + m

    def test_radd_scalar(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(NotImplementedError):
                1 + m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_radd_dense(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return n + m

    # __sub__
    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_sub_zero(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m - 0

    def test_sub_scalar(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(NotImplementedError):
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
    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_rsub_zero(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return 0 - m

    def test_rsub_scalar(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(NotImplementedError):
                1 - m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rsub_dense(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return n - m

    # __mul__
    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_mul_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m * 2.0

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_mul_numpy_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m * numpy.dtype(self.dtype).type(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_csr(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return m * x

    def test_mul_csr_invalid_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            x = sp.csr_matrix((5, 3), dtype=self.dtype)
            with pytest.raises(ValueError):
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

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_mul_zero_dim(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.array(2, dtype=self.dtype)
        return m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_dense_vector(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(4).astype(self.dtype)
        return m * x

    def test_mul_dense_vector_invalid_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            x = xp.arange(5).astype(self.dtype)
            with pytest.raises(ValueError):
                m * x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_dense_matrix(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(8).reshape(4, 2).astype(self.dtype)
        return m * x

    def test_mul_dense_matrix_invalid_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            x = xp.arange(10).reshape(5, 2).astype(self.dtype)
            with pytest.raises(ValueError):
                m * x

    def test_mul_dense_ndim3(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
            with pytest.raises(ValueError):
                m * x

    def test_mul_unsupported(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(TypeError):
                m * None

    # __rmul__
    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_rmul_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return 2.0 * m

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_rmul_numpy_scalar(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return numpy.dtype(self.dtype).type(2.0) * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_csr(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_rmul_csc(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocsc()
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_sparse(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocoo()
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_rmul_zero_dim(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.array(2, dtype=self.dtype)
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rmul_dense_matrix(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        x = xp.arange(12).reshape(4, 3).astype(self.dtype)
        return x * m

    def test_rmul_dense_ndim3(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
            with pytest.raises(ValueError):
                x * m

    def test_rmul_unsupported(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(TypeError):
                None * m

    # Note: '@' operator is almost equivalent to '*' operator. Only test the
    # cases where '@' raises an exception and '*' does not.
    def test_matmul_scalar(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            x = 2.0
            with pytest.raises(ValueError):
                m @ x
            with pytest.raises(ValueError):
                x @ m

    def test_matmul_numpy_scalar(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            x = numpy.dtype(self.dtype).type(2.0)
            with pytest.raises(ValueError):
                m @ x
            with pytest.raises(ValueError):
                x @ m

    def test_matmul_scalar_like_array(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            x = xp.array(2.0, self.dtype)
            with pytest.raises(ValueError):
                m @ x
            with pytest.raises(ValueError):
                x @ m

    # __pow__
    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
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

    def test_pow_neg(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make_square(xp, sp, self.dtype)
            with pytest.raises(ValueError):
                m ** -1

    def test_sum_tuple_axis(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(TypeError):
                m.sum(axis=(0, 1))

    def test_sum_float_axis(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(TypeError):
                m.sum(axis=0.0)

    def test_sum_too_large_axis(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(ValueError):
                m.sum(axis=3)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_transpose(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.transpose()

    def test_transpose_axes_int(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            with pytest.raises(ValueError):
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
        assert not m.has_canonical_format
        m.sum_duplicates()
        assert m.has_canonical_format
        assert m.nnz == 3

        m.sum_duplicates()
        assert m.has_canonical_format
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_duplicates_canonical(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        assert not m.has_canonical_format
        m.sum_duplicates()
        assert m.has_canonical_format
        assert m.nnz == 4
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_duplicates_empty(self, xp, sp):
        m = _make_empty(xp, sp, self.dtype)
        assert not m.has_canonical_format
        m.sum_duplicates()
        assert m.has_canonical_format
        assert m.nnz == 0
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_duplicates_incompatibility(self, xp, sp):
        # See #3620 and #3624. CuPy's and SciPy's COO indices could mismatch
        # due to the order of lexsort, but the matrix is correct.
        m = _make_sum_dup(xp, sp, self.dtype)
        if xp is cupy:
            sorted_first = m.row.copy()
        else:
            sorted_first = m.col.copy()
        assert not m.has_canonical_format
        m.sum_duplicates()
        assert m.has_canonical_format
        # Here we ensure this sorting order is not altered by future PRs...
        sorted_first.sort()
        if xp is cupy:
            assert (m.row == sorted_first).all()
        else:
            assert (m.col == sorted_first).all()
        assert m.has_canonical_format
        # ...and now we make sure the dense matrix is the same
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
            return xp.array(0)
        else:
            return func()


class TestIsspmatrixCoo(unittest.TestCase):

    def test_coo(self):
        x = sparse.coo_matrix(
            (cupy.array([0], 'f'),
             (cupy.array([0], 'i'), cupy.array([0], 'i'))),
            shape=(1, 1), dtype='f')
        assert sparse.isspmatrix_coo(x) is True

    def test_csr(self):
        x = sparse.csr_matrix(
            (cupy.array([], 'f'),
             cupy.array([], 'i'),
             cupy.array([0], 'i')),
            shape=(0, 0), dtype='f')
        assert sparse.isspmatrix_coo(x) is False


@testing.parameterize(*testing.product({
    'shape': [(8, 5), (5, 5), (5, 8)],
}))
@testing.with_requires('scipy>=1.5.0')
@testing.gpu
class TestCooMatrixDiagonal(unittest.TestCase):
    density = 0.5

    def _make_matrix(self, dtype):
        a = testing.shaped_random(self.shape, numpy, dtype=dtype)
        mask = testing.shaped_random(self.shape, numpy, dtype='f', scale=1.0)
        a[mask > self.density] = 0
        scipy_a = scipy.sparse.coo_matrix(a)
        cupyx_a = sparse.coo_matrix(cupy.array(a))
        return scipy_a, cupyx_a

    @testing.for_dtypes('fdFD')
    def test_diagonal(self, dtype):
        scipy_a, cupyx_a = self._make_matrix(dtype)
        m, n = self.shape
        for k in range(-m, n+1):
            scipy_diag = scipy_a.diagonal(k=k)
            cupyx_diag = cupyx_a.diagonal(k=k)
            testing.assert_allclose(scipy_diag, cupyx_diag)

    def _test_setdiag(self, scipy_a, cupyx_a, x, k):
        scipy_a = scipy_a.copy()
        cupyx_a = cupyx_a.copy()
        scipy_a.setdiag(x, k=k)
        cupyx_a.setdiag(cupy.array(x), k=k)
        testing.assert_allclose(scipy_a.data, cupyx_a.data)
        testing.assert_array_equal(scipy_a.row, cupyx_a.row)
        testing.assert_array_equal(scipy_a.col, cupyx_a.col)

    @testing.for_dtypes('fdFD')
    def test_setdiag(self, dtype):
        scipy_a, cupyx_a = self._make_matrix(dtype)
        m, n = self.shape
        for k in range(-m+1, n):
            m_st, n_st = max(0, -k), max(0, k)
            for d in (-1, 0, 1):
                x_len = min(m - m_st, n - n_st) + d
                if x_len <= 0:
                    continue
                x = numpy.ones((x_len,), dtype=dtype)
                self._test_setdiag(scipy_a, cupyx_a, x, k)

    @testing.for_dtypes('fdFD')
    def test_setdiag_scalar(self, dtype):
        scipy_a, cupyx_a = self._make_matrix(dtype)
        x = numpy.array(1.0, dtype=dtype)
        m, n = self.shape
        for k in range(-m+1, n):
            self._test_setdiag(scipy_a, cupyx_a, x, k)

    def test_setdiag_invalid(self):
        dtype = 'f'
        scipy_a, cupyx_a = self._make_matrix(dtype)
        x = numpy.array(1.0, dtype=dtype)
        m, n = self.shape
        for k in (-m, n):
            with self.assertRaises(ValueError):
                scipy_a.setdiag(x, k=k)
            with self.assertRaises(ValueError):
                cupyx_a.setdiag(x, k=k)
