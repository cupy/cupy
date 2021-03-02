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
    data = xp.array([0, 1, 3, 2], dtype)
    indices = xp.array([0, 0, 2, 1], 'i')
    indptr = xp.array([0, 1, 2, 3, 4], 'i')
    # 0, 1, 0, 0
    # 0, 0, 0, 2
    # 0, 0, 3, 0
    return sp.csc_matrix((data, indices, indptr), shape=(3, 4))


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
    data = xp.array([2, 1, 3, 4], dtype)
    indices = xp.array([1, 0, 1, 2], 'i')
    indptr = xp.array([0, 0, 1, 4, 4], 'i')
    # 0, 0, 1, 0
    # 0, 2, 3, 0
    # 0, 0, 4, 0
    return sp.csc_matrix((data, indices, indptr), shape=(3, 4))


def _make3(xp, sp, dtype):
    data = xp.array([1, 4, 3, 2, 5], dtype)
    indices = xp.array([0, 3, 1, 1, 3], 'i')
    indptr = xp.array([0, 2, 3, 5], 'i')
    # 1, 0, 0
    # 0, 3, 2
    # 0, 0, 0
    # 4, 0, 5
    return sp.csc_matrix((data, indices, indptr), shape=(4, 3))


def _make_unordered(xp, sp, dtype):
    data = xp.array([1, 2, 3, 4], dtype)
    indices = xp.array([1, 0, 1, 2], 'i')
    indptr = xp.array([0, 0, 0, 2, 4], 'i')
    return sp.csc_matrix((data, indices, indptr), shape=(3, 4))


def _make_duplicate(xp, sp, dtype):
    data = xp.array([1, 4, 3, 0, 2, 5], dtype)
    indices = xp.array([0, 1, 0, 2, 1, 1], 'i')
    indptr = xp.array([0, 3, 4, 6, 6], 'i')
    # 4, 0, 0, 0
    # 4, 0, 7, 0
    # 0, 0, 0, 0
    return sp.csc_matrix((data, indices, indptr), shape=(3, 4))


def _make_empty(xp, sp, dtype):
    data = xp.array([], dtype)
    indices = xp.array([], 'i')
    indptr = xp.array([0, 0, 0, 0, 0], 'i')
    return sp.csc_matrix((data, indices, indptr), shape=(3, 4))


def _make_shape(xp, sp, dtype):
    return sp.csc_matrix((3, 4))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
class TestCscMatrix(unittest.TestCase):

    def setUp(self):
        self.m = _make(cupy, sparse, self.dtype)

    def test_dtype(self):
        assert self.m.dtype == self.dtype

    def test_data(self):
        assert self.m.data.dtype == self.dtype
        testing.assert_array_equal(
            self.m.data, cupy.array([0, 1, 3, 2], self.dtype))

    def test_indices(self):
        assert self.m.indices.dtype == numpy.int32
        testing.assert_array_equal(
            self.m.indices, cupy.array([0, 0, 2, 1], self.dtype))

    def test_indptr(self):
        assert self.m.indptr.dtype == numpy.int32
        testing.assert_array_equal(
            self.m.indptr, cupy.array([0, 1, 2, 3, 4], self.dtype))

    def test_init_copy(self):
        n = sparse.csc_matrix(self.m)
        assert n is not self.m
        cupy.testing.assert_array_equal(n.data, self.m.data)
        cupy.testing.assert_array_equal(n.indices, self.m.indices)
        cupy.testing.assert_array_equal(n.indptr, self.m.indptr)
        assert n.shape == self.m.shape

    def test_init_copy_other_sparse(self):
        n = sparse.csc_matrix(self.m.tocsr())
        cupy.testing.assert_array_equal(n.data, self.m.data)
        cupy.testing.assert_array_equal(n.indices, self.m.indices)
        cupy.testing.assert_array_equal(n.indptr, self.m.indptr)
        assert n.shape == self.m.shape

    @testing.with_requires('scipy')
    def test_init_copy_scipy_sparse(self):
        m = _make(numpy, scipy.sparse, self.dtype)
        n = sparse.csc_matrix(m)
        assert isinstance(n.data, cupy.ndarray)
        assert isinstance(n.indices, cupy.ndarray)
        assert isinstance(n.indptr, cupy.ndarray)
        cupy.testing.assert_array_equal(n.data, m.data)
        cupy.testing.assert_array_equal(n.indices, m.indices)
        cupy.testing.assert_array_equal(n.indptr, m.indptr)
        assert n.shape == m.shape

    @testing.with_requires('scipy')
    def test_init_copy_other_scipy_sparse(self):
        m = _make(numpy, scipy.sparse, self.dtype)
        n = sparse.csc_matrix(m.tocsr())
        assert isinstance(n.data, cupy.ndarray)
        assert isinstance(n.indices, cupy.ndarray)
        assert isinstance(n.indptr, cupy.ndarray)
        cupy.testing.assert_array_equal(n.data, m.data)
        cupy.testing.assert_array_equal(n.indices, m.indices)
        cupy.testing.assert_array_equal(n.indptr, m.indptr)
        assert n.shape == m.shape

    def test_init_dense(self):
        m = cupy.array([[0, 1, 0, 2],
                        [0, 0, 0, 0],
                        [0, 0, 0, 3]], dtype=self.dtype)
        n = sparse.csc_matrix(m)
        assert n.nnz == 3
        assert n.shape == (3, 4)
        cupy.testing.assert_array_equal(n.data, [1, 2, 3])
        cupy.testing.assert_array_equal(n.indices, [0, 0, 2])
        cupy.testing.assert_array_equal(n.indptr, [0, 0, 1, 1, 3])

    def test_init_dense_empty(self):
        m = cupy.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]], dtype=self.dtype)
        n = sparse.csc_matrix(m)
        assert n.nnz == 0
        assert n.shape == (3, 4)
        cupy.testing.assert_array_equal(n.data, [])
        cupy.testing.assert_array_equal(n.indices, [])
        cupy.testing.assert_array_equal(n.indptr, [0, 0, 0, 0, 0])

    def test_init_dense_one_dim(self):
        m = cupy.array([0, 1, 0, 2], dtype=self.dtype)
        n = sparse.csc_matrix(m)
        assert n.nnz == 2
        assert n.shape == (1, 4)
        cupy.testing.assert_array_equal(n.data, [1, 2])
        cupy.testing.assert_array_equal(n.indices, [0, 0])
        cupy.testing.assert_array_equal(n.indptr, [0, 0, 1, 1, 2])

    def test_init_dense_zero_dim(self):
        m = cupy.array(1, dtype=self.dtype)
        n = sparse.csc_matrix(m)
        assert n.nnz == 1
        assert n.shape == (1, 1)
        cupy.testing.assert_array_equal(n.data, [1])
        cupy.testing.assert_array_equal(n.indices, [0])
        cupy.testing.assert_array_equal(n.indptr, [0, 1])

    def test_init_data_row_col(self):
        o = self.m.tocoo()
        n = sparse.csc_matrix((o.data, (o.row, o.col)))
        cupy.testing.assert_array_equal(n.data, self.m.data)
        cupy.testing.assert_array_equal(n.indices, self.m.indices)
        cupy.testing.assert_array_equal(n.indptr, self.m.indptr)
        assert n.shape == self.m.shape

    @testing.with_requires('scipy')
    def test_init_dense_invalid_ndim(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(TypeError):
                m = xp.zeros((1, 1, 1), dtype=self.dtype)
                sp.csc_matrix(m)

    def test_copy(self):
        n = self.m.copy()
        assert isinstance(n, sparse.csc_matrix)
        assert n is not self.m
        assert n.data is not self.m.data
        assert n.indices is not self.m.indices
        assert n.indptr is not self.m.indptr
        cupy.testing.assert_array_equal(n.data, self.m.data)
        cupy.testing.assert_array_equal(n.indices, self.m.indices)
        cupy.testing.assert_array_equal(n.indptr, self.m.indptr)
        assert n.shape == self.m.shape

    def test_shape(self):
        assert self.m.shape == (3, 4)

    def test_ndim(self):
        assert self.m.ndim == 2

    def test_nnz(self):
        assert self.m.nnz == 4

    def test_conj(self):
        n = _make_complex(cupy, sparse, self.dtype)
        cupy.testing.assert_array_equal(n.conj().data, n.data.conj())

    @testing.with_requires('scipy')
    def test_get(self):
        m = self.m.get()
        assert isinstance(m, scipy.sparse.csc_matrix)
        expect = [
            [0, 1, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 3, 0]
        ]
        numpy.testing.assert_allclose(m.toarray(), expect)

    @testing.with_requires('scipy')
    def test_str(self):
        if numpy.dtype(self.dtype).kind == 'f':
            expect = '''  (0, 0)\t0.0
  (0, 1)\t1.0
  (2, 2)\t3.0
  (1, 3)\t2.0'''
        elif numpy.dtype(self.dtype).kind == 'c':
            expect = '''  (0, 0)\t0j
  (0, 1)\t(1+0j)
  (2, 2)\t(3+0j)
  (1, 3)\t(2+0j)'''

        assert str(self.m) == expect

    def test_toarray(self):
        m = self.m.toarray()
        expect = [
            [0, 1, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 3, 0]
        ]
        assert m.flags.c_contiguous
        cupy.testing.assert_allclose(m, expect)

    def test_pickle_roundtrip(self):
        s = _make(cupy, sparse, self.dtype)
        s2 = pickle.loads(pickle.dumps(s))
        assert s._descr.descriptor != s2._descr.descriptor
        assert s.shape == s2.shape
        assert s.dtype == s2.dtype
        if scipy_available:
            assert (s.get() != s2.get()).count_nonzero() == 0


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@testing.with_requires('scipy')
class TestCscMatrixInit(unittest.TestCase):

    def setUp(self):
        self.shape = (3, 4)

    def data(self, xp):
        return xp.array([1, 2, 3, 4], self.dtype)

    def indices(self, xp):
        return xp.array([0, 0, 2, 1], 'i')

    def indptr(self, xp):
        return xp.array([0, 1, 2, 3, 4], 'i')

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_shape_none(self, xp, sp):
        x = sp.csc_matrix(
            (self.data(xp), self.indices(xp), self.indptr(xp)), shape=None)
        assert x.shape == (3, 4)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_dtype(self, xp, sp):
        data = self.data(xp).real.astype('i')
        x = sp.csc_matrix(
            (data, self.indices(xp), self.indptr(xp)), dtype=self.dtype)
        assert x.dtype == self.dtype

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_copy_true(self, xp, sp):
        data = self.data(xp)
        indices = self.indices(xp)
        indptr = self.indptr(xp)
        x = sp.csc_matrix((data, indices, indptr), copy=True)

        assert data is not x.data
        assert indices is not x.indices
        assert indptr is not x.indptr

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_init_with_shape(self, xp, sp):
        s = sp.csc_matrix(self.shape)
        assert s.shape == self.shape
        assert s.dtype == 'd'
        assert s.size == 0
        return s

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_init_with_shape_and_dtype(self, xp, sp):
        s = sp.csc_matrix(self.shape, dtype=self.dtype)
        assert s.shape == self.shape
        assert s.dtype == self.dtype
        assert s.size == 0
        return s

    @testing.numpy_cupy_allclose(sp_name='sp', atol=1e-5)
    def test_intlike_shape(self, xp, sp):
        s = sp.csc_matrix((self.data(xp), self.indices(xp), self.indptr(xp)),
                          shape=(xp.array(self.shape[0]),
                                 xp.int32(self.shape[1])))
        assert isinstance(s.shape[0], int)
        assert isinstance(s.shape[1], int)
        return s

    def test_shape_invalid(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.csc_matrix(
                    (self.data(xp), self.indices(xp), self.indptr(xp)),
                    shape=(2,))

    def test_data_invalid(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.csc_matrix(
                    ('invalid', self.indices(xp), self.indptr(xp)),
                    shape=self.shape)

    def test_data_invalid_ndim(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.csc_matrix(
                    (self.data(xp)[None], self.indices(xp),
                     self.indptr(xp)),
                    shape=self.shape)

    def test_indices_invalid(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.csc_matrix(
                    (self.data(xp), 'invalid', self.indptr(xp)),
                    shape=self.shape)

    def test_indices_invalid_ndim(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.csc_matrix(
                    (self.data(xp), self.indices(xp)[None], self.indptr(xp)),
                    shape=self.shape)

    def test_indptr_invalid(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.csc_matrix(
                    (self.data(xp), self.indices(xp), 'invalid'),
                    shape=self.shape)

    def test_indptr_invalid_ndim(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.csc_matrix(
                    (self.data(xp), self.indices(xp), self.indptr(xp)[None]),
                    shape=self.shape)

    def test_data_indices_different_length(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            data = xp.arange(5, dtype=self.dtype)
            with pytest.raises(ValueError):
                sp.csc_matrix(
                    (data, self.indices(xp), self.indptr(xp)),
                    shape=self.shape)

    def test_indptr_invalid_length(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            indptr = xp.array([0, 1], 'i')
            with pytest.raises(ValueError):
                sp.csc_matrix(
                    (self.data(xp), self.indices(xp), indptr),
                    shape=self.shape)

    def test_unsupported_dtype(self):
        with self.assertRaises(ValueError):
            sparse.csc_matrix(
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
@testing.with_requires('scipy')
class TestCscMatrixScipyComparison(unittest.TestCase):

    @property
    def make(self):
        return globals()[self.make_method]

    def test_len(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(TypeError):
                len(m)

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_asfptype(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.asfptype()

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_toarray(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        a = m.toarray()
        if sp is sparse:
            assert a.flags.c_contiguous
        return a

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray_c_order(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        a = m.toarray(order='C')
        assert a.flags.c_contiguous
        return a

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray_f_order(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        a = m.toarray(order='F')
        assert a.flags.f_contiguous
        return a

    @testing.with_requires('numpy>=1.19')
    def test_toarray_unknown_order(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(ValueError):
                m.toarray(order='#')

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
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
        assert m.data is not n.data
        return n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.tocsc()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc_copy(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = m.tocsc(copy=True)
        assert m.data is not n.data
        assert m.indices is not n.indices
        assert m.indptr is not n.indptr
        return n

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.tocsr()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsr_copy(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = m.tocsr(copy=True)
        assert m.data is not n.data
        assert m.indices is not n.indices
        assert m.indptr is not n.indptr
        return n

    # dot
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.dot(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_dot_numpy_scalar(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.dot(numpy.dtype(self.dtype).type(2.0))

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_dot_csr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return m.dot(x)

    def test_dot_csr_invalid_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            x = sp.csr_matrix((5, 3), dtype=self.dtype)
            with pytest.raises(ValueError):
                m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_dot_csc(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocsc()
        return m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
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

    def test_dot_dense_vector_invalid_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            x = xp.arange(5).astype(self.dtype)
            with pytest.raises(ValueError):
                m.dot(x)

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_dot_dense_matrix(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(8).reshape(4, 2).astype(self.dtype)
        return m.dot(x)

    def test_dot_dense_matrix_invalid_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            x = xp.arange(10).reshape(5, 2).astype(self.dtype)
            with pytest.raises(ValueError):
                m.dot(x)

    def test_dot_dense_ndim3(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
            with pytest.raises(ValueError):
                m.dot(x)

    def test_dot_unsupported(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(TypeError):
                m.dot(None)

    # __add__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add_zero(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m + 0

    def test_add_scalar(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(NotImplementedError):
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

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_add_dense(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return m + n

    # __radd__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_radd_zero(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return 0 + m

    def test_radd_scalar(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(NotImplementedError):
                1 + m

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_radd_dense(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = xp.arange(12).reshape(3, 4)
        return n + m

    # __sub__
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_zero(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m - 0

    def test_sub_scalar(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(NotImplementedError):
                m - 1

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_csr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = _make2(xp, sp, self.dtype)
        return m - n

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

    def test_rsub_scalar(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(NotImplementedError):
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

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_mul_csr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return m * x

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_mul_csc(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype).tocsc()
        return m * x

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
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

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_mul_dense_matrix(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = xp.arange(8).reshape(4, 2).astype(self.dtype)
        return m * x

    def test_mul_dense_ndim3(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
            with pytest.raises(ValueError):
                m * x

    def test_mul_unsupported(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(TypeError):
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

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_rmul_csr(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        x = _make3(xp, sp, self.dtype)
        return x * m

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
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

    def test_rmul_dense_ndim3(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            x = xp.arange(24).reshape(4, 2, 3).astype(self.dtype)
            with pytest.raises(ValueError):
                x * m

    def test_rmul_unsupported(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            # TODO(unno): When a sparse matrix has no element, scipy.sparse
            # does not raise an error.
            if m.nnz == 0:
                continue
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

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_has_canonical_format(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.has_canonical_format

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_has_canonical_format2(self, xp, sp):
        # this test is adopted from SciPy's
        M = sp.csc_matrix((xp.array([2], dtype=self.dtype),
                           xp.array([0]), xp.array([0, 1])))
        assert M.has_canonical_format is True
        return M

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_has_canonical_format3(self, xp, sp):
        # this test is adopted from SciPy's
        indices = xp.array([0, 0])  # contains duplicate
        data = xp.array([1, 1], dtype=self.dtype)
        indptr = xp.array([0, 2])

        M = sp.csc_matrix((data, indices, indptr))
        assert M.has_canonical_format is False

        # set by deduplicating
        M.sum_duplicates()
        assert M.has_canonical_format is True
        assert 1 == len(M.indices)
        return M

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_has_canonical_format4(self, xp, sp):
        # this test is adopted from SciPy's
        indices = xp.array([0, 0])  # contains duplicate
        data = xp.array([1, 1], dtype=self.dtype)
        indptr = xp.array([0, 2])

        M = sp.csc_matrix((data, indices, indptr))
        # set manually (although underlyingly duplicated)
        M.has_canonical_format = True
        assert M.has_canonical_format
        assert 2 == len(M.indices)  # unaffected content

        # ensure deduplication bypassed when has_canonical_format == True
        M.sum_duplicates()
        assert 2 == len(M.indices)  # unaffected content
        return M

    @testing.with_requires('scipy>1.6.0')
    @testing.numpy_cupy_equal(sp_name='sp')
    def test_has_sorted_indices(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.has_sorted_indices

    # TODO(asi1024): Remove test after the fixed version is released.
    # https://github.com/scipy/scipy/pull/13426
    @testing.with_requires('scipy<=1.6.0')
    @testing.numpy_cupy_equal(sp_name='sp')
    def test_has_sorted_indices_for_old_scipy(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return bool(m.has_sorted_indices)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_has_sorted_indices2(self, xp, sp):
        # this test is adopted from SciPy's
        sorted_inds = xp.array([0, 1])
        data = xp.array([1, 1], dtype=self.dtype)
        indptr = xp.array([0, 2])
        M = sp.csc_matrix((data, sorted_inds, indptr))
        assert M.has_sorted_indices
        return M

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_has_sorted_indices3(self, xp, sp):
        # this test is adopted from SciPy's
        sorted_inds = xp.array([0, 1])
        unsorted_inds = xp.array([1, 0])
        data = xp.array([1, 1], dtype=self.dtype)
        indptr = xp.array([0, 2])
        M = sp.csc_matrix((data, unsorted_inds, indptr))
        assert not M.has_sorted_indices

        # set by sorting
        M.sort_indices()
        assert M.has_sorted_indices
        assert (M.indices == sorted_inds).all()
        return M

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_has_sorted_indices4(self, xp, sp):
        # this test is adopted from SciPy's
        unsorted_inds = xp.array([1, 0])
        data = xp.array([1, 1], dtype=self.dtype)
        indptr = xp.array([0, 2])
        M = sp.csc_matrix((data, unsorted_inds, indptr))

        # set manually (although underlyingly unsorted)
        M.has_sorted_indices = True
        assert M.has_sorted_indices
        assert (M.indices == unsorted_inds).all()

        # ensure sort bypassed when has_sorted_indices == True
        M.sort_indices()
        assert (M.indices == unsorted_inds).all()
        return M

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sort_indices(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.sort_indices()
        assert m.has_sorted_indices
        return m

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_sort_indices2(self, xp, sp):
        # 1. this test is adopted from SciPy's.
        # 2. we don't check the contiguity flag because SciPy and CuPy handle
        #    the underlying data differently
        data = xp.arange(5).astype(xp.float32)
        indices = xp.array([7, 2, 1, 5, 4])
        indptr = xp.array([0, 3, 5])
        asp = sp.csc_matrix((data, indices, indptr), shape=(10, 2))
        asp.sort_indices()
        assert (asp.indices == xp.array([1, 2, 7, 4, 5])).all()
        return asp.todense()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sorted_indices(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m = m.sorted_indices()
        assert m.has_sorted_indices
        return m

    def test_sum_tuple_axis(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(TypeError):
                m.sum(axis=(0, 1))

    def test_sum_too_large_axis(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(ValueError):
                m.sum(axis=3)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_duplicates(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        m.sum_duplicates()
        assert m.has_canonical_format
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_transpose(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.transpose()

    def test_transpose_axes_int(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(ValueError):
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
    'dtype': [numpy.float32, numpy.float64],
    'ret_dtype': [None, numpy.float32, numpy.float64],
    'axis': [None, 0, 1, -1, -2],
}))
@testing.with_requires('scipy')
class TestCscMatrixSum(unittest.TestCase):

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
@testing.with_requires('scipy')
class TestCscMatrixScipyCompressed(unittest.TestCase):

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_get_shape(self, xp, sp):
        return _make(xp, sp, self.dtype).get_shape()

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_getnnz(self, xp, sp):
        return _make(xp, sp, self.dtype).getnnz()


@testing.parameterize(*testing.product({
    # TODO(takagi): Test dtypes
    'axis': [None, -2, -1, 0, 1],
    'dense': [False, True],  # means a sparse matrix but all elements filled
}))
@testing.with_requires('scipy>=0.19.0')
class TestCscMatrixScipyCompressedMinMax(unittest.TestCase):

    def _make_data_min(self, xp, sp, dense=False):
        dm_data = testing.shaped_random((10, 20), xp=xp, scale=1.0)
        if not dense:
            dm_data[abs(dm_data) < 0.95] = 0
        return sp.csc_matrix(xp.array(dm_data))

    def _make_data_max(self, xp, sp, dense=False):
        return -self._make_data_min(xp, sp, dense=dense)

    def _make_data_min_explicit(self, xp, sp, axis):
        dm_data = testing.shaped_random((10, 20), xp=xp, scale=1.0)
        if xp is cupy:
            dm_data[dm_data < 0.95] = 0
        else:
            # As SciPy sparse matrix does not have `explicit` parameter, we
            # make SciPy inputs such that SciPy's spmatrix.min(axis=axis)
            # returns the same value as CuPy's spmatrix.min(axis=axis,
            # explicit=True).

            # Put infinity instead of zeros so spmatrix.min(axis=axis) returns
            # the smallest numbers except for zero.
            dm_data[dm_data < 0.95] = numpy.inf

            if axis is None:
                # If all elements in the array are set to infinity, we make it
                # have at least a zero so SciPy's spmatrix.min(axis=None)
                # returns zero.
                if numpy.isinf(dm_data).all():
                    dm_data[0, 0] = 0
            else:
                if axis < 0:
                    axis += 2

                # If all elements in a row/column are set to infinity, we make
                # it have at least a zero so spmatrix.min(axis=axis) returns
                # zero for the row/column.
                mask = numpy.zeros_like(dm_data, dtype=numpy.bool_)
                if axis == 0:
                    rows = dm_data.argmin(axis=0)
                    cols = numpy.arange(20)
                else:
                    rows = numpy.arange(10)
                    cols = dm_data.argmin(axis=1)
                mask[rows, cols] = numpy.isinf(dm_data[rows, cols])
                dm_data[mask] = 0

        return sp.csc_matrix(xp.array(dm_data))

    def _make_data_max_explicit(self, xp, sp, axis):
        return -self._make_data_min_explicit(xp, sp, axis=axis)

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_min(self, xp, sp):
        data = self._make_data_min(xp, sp, dense=self.dense)
        return data.min(axis=self.axis)

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_min_explicit(self, xp, sp):
        data = self._make_data_min_explicit(xp, sp, axis=self.axis)
        if xp is cupy:
            return data.min(axis=self.axis, explicit=True)
        else:
            return data.min(axis=self.axis)

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_max(self, xp, sp):
        data = self._make_data_max(xp, sp, dense=self.dense)
        return data.max(axis=self.axis)

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_max_explicit(self, xp, sp):
        data = self._make_data_max_explicit(xp, sp, axis=self.axis)
        if xp is cupy:
            return data.max(axis=self.axis, explicit=True)
        else:
            return data.max(axis=self.axis)

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_argmin(self, xp, sp):
        # TODO(takagi) Fix axis=None
        if self.axis is None:
            pytest.skip()
        data = self._make_data_min(xp, sp, dense=self.dense)
        return data.argmin(axis=self.axis)

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_argmax(self, xp, sp):
        # TODO(takagi) Fix axis=None
        if self.axis is None:
            pytest.skip()
        data = self._make_data_max(xp, sp, dense=self.dense)
        return data.argmax(axis=self.axis)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@testing.with_requires('scipy')
class TestCscMatrixData(unittest.TestCase):

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
        return (-m)

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

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean_axis_None(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.mean(axis=None)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean_axis_0(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.mean(axis=0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean_axis_1(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.mean(axis=1)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean_axis_negative_1(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.mean(axis=-1)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean_axis_negative_2(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.mean(axis=-2)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'ufunc': [
        'arcsin', 'arcsinh', 'arctan', 'arctanh', 'ceil', 'deg2rad', 'expm1',
        'floor', 'log1p', 'rad2deg', 'rint', 'sign', 'sin', 'sinh', 'sqrt',
        'tan', 'tanh', 'trunc',
    ],
}))
@testing.with_requires('scipy')
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


class TestIsspmatrixCsc(unittest.TestCase):

    def test_csr(self):
        x = sparse.csr_matrix(
            (cupy.array([], 'f'),
             cupy.array([], 'i'),
             cupy.array([0], 'i')),
            shape=(0, 0), dtype='f')
        assert not sparse.isspmatrix_csc(x)

    def test_csc(self):
        x = sparse.csc_matrix(
            (cupy.array([], 'f'),
             cupy.array([], 'i'),
             cupy.array([0], 'i')),
            shape=(0, 0), dtype='f')
        assert sparse.isspmatrix_csc(x)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@testing.with_requires('scipy>=1.4.0')
class TestCsrMatrixGetitem(unittest.TestCase):

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_getitem_int_int(self, xp, sp):
        assert _make(xp, sp, self.dtype)[0, 1] == 1

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_getitem_int_int_not_found(self, xp, sp):
        assert _make(xp, sp, self.dtype)[1, 1] == 0

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_getitem_int_int_negative(self, xp, sp):
        assert _make(xp, sp, self.dtype)[-1, -2] == 3

    def test_getitem_int_int_too_small_row(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(IndexError):
                _make(xp, sp, self.dtype)[-4, 0]

    def test_getitem_int_int_too_large_row(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(IndexError):
                _make(xp, sp, self.dtype)[3, 0]

    def test_getitem_int_int_too_small_col(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(IndexError):
                _make(xp, sp, self.dtype)[0, -5]

    def test_getitem_int_int_too_large_col(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(IndexError):
                _make(xp, sp, self.dtype)[0, 4]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_int(self, xp, sp):
        return _make(xp, sp, self.dtype)[:, 1]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_negative_int(self, xp, sp):
        return _make(xp, sp, self.dtype)[:, -1]

    def test_getitem_int_too_small(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(IndexError):
                _make(xp, sp, self.dtype)[:, -5]

    def test_getitem_int_too_large(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(IndexError):
                _make(xp, sp, self.dtype)[:, 4]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice(self, xp, sp):
        return _make(xp, sp, self.dtype)[:, 1:3]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_negative(self, xp, sp):
        return _make(xp, sp, self.dtype)[:, -2:-1]

    # SciPy prior to 1.4 has bugs where either an IndexError is raised or a
    # segfault occurs instead of returning an empty slice.
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_start_larger_than_stop(self, xp, sp):
        return _make(xp, sp, self.dtype)[:, 3:2]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_rowslice_all(self, xp, sp):
        # This test is adapted from Scipy's CSC tests
        return _make(xp, sp, self.dtype)[slice(None, None, None)]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_rowslice_negative_stop(self, xp, sp):
        # This test is adapted from Scipy's CSC tests
        return _make(xp, sp, self.dtype)[slice(1, -2, 2)]

    def test_getrow(self):

        # This test is adapted from Scipy's CSC tests
        N = 10
        X = testing.shaped_random((N, N), cupy, seed=0)
        X[X > 0.7] = 0
        Xcsc = sparse.csc_matrix(X)

        for i in range(N):
            arr_row = X[i:i + 1, :]
            csc_row = Xcsc.getrow(i)

            assert sparse.isspmatrix_csr(csc_row)
            assert (arr_row == csc_row.toarray()).all()

    def test_getcol(self):
        # This test is adapted from Scipy's CSC tests
        N = 10
        X = testing.shaped_random((N, N), cupy, seed=0)
        X[X > 0.7] = 0
        Xcsc = sparse.csc_matrix(X)

        for i in range(N):
            arr_col = X[:, i:i + 1]
            csc_col = Xcsc.getcol(i)

            assert sparse.isspmatrix_csc(csc_col)
            assert (arr_col == csc_col.toarray()).all()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@testing.with_requires('scipy>=1.4.0')
class TestCsrMatrixGetitem2(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_start_too_small(self, xp, sp):
        return _make(xp, sp, self.dtype)[:, -5:None]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_start_too_large(self, xp, sp):
        return _make(xp, sp, self.dtype)[:, 5:None]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_stop_too_small(self, xp, sp):
        return _make(xp, sp, self.dtype)[:, None:-5]

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_getitem_slice_stop_too_large(self, xp, sp):
        return _make(xp, sp, self.dtype)[:, None:5]
