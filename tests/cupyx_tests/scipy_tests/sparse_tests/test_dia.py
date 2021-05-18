import pickle
import unittest


import numpy
import pytest
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


def _make_complex(xp, sp, dtype):
    data = xp.array([[0, 1, 2], [3, 4, 5]], dtype)
    if dtype in [numpy.complex64, numpy.complex128]:
        data = data - 1j
    offsets = xp.array([0, -1], 'i')
    # 0, 0, 0, 0
    # 3 - 1j, 1 - 1j, 0, 0
    # 0, 4 - 1j, 2 - 1j, 0
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
        assert self.m.dtype == self.dtype

    def test_data(self):
        assert self.m.data.dtype == self.dtype
        testing.assert_array_equal(
            self.m.data, cupy.array([[0, 1, 2], [3, 4, 5]], self.dtype))

    def test_offsets(self):
        assert self.m.offsets.dtype == numpy.int32
        testing.assert_array_equal(
            self.m.offsets, cupy.array([0, -1], self.dtype))

    def test_shape(self):
        assert self.m.shape == (3, 4)

    def test_ndim(self):
        assert self.m.ndim == 2

    def test_nnz(self):
        assert self.m.nnz == 5

    def test_conj(self):
        n = _make_complex(cupy, sparse, self.dtype)
        cupy.testing.assert_array_equal(n.conj().data, n.data.conj())

    def test_conjugate(self):
        n = _make_complex(cupy, sparse, self.dtype)
        cupy.testing.assert_array_equal(n.conjugate().data, n.data.conj())

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
        assert str(self.m) == expect

    def test_toarray(self):
        m = self.m.toarray()
        expect = [
            [0, 0, 0, 0],
            [3, 1, 0, 0],
            [0, 4, 2, 0]
        ]
        assert m.flags.c_contiguous
        cupy.testing.assert_allclose(m, expect)

    def test_pickle_roundtrip(self):
        s = _make(cupy, sparse, self.dtype)
        s2 = pickle.loads(pickle.dumps(s))
        assert s.shape == s2.shape
        assert s.dtype == s2.dtype
        if scipy_available:
            assert (s.get() != s2.get()).count_nonzero() == 0

    def test_diagonal(self):
        testing.assert_array_equal(
            self.m.diagonal(-2), cupy.array([0], self.dtype))
        testing.assert_array_equal(
            self.m.diagonal(-1), cupy.array([3, 4], self.dtype))
        testing.assert_array_equal(
            self.m.diagonal(), cupy.array([0, 1, 2], self.dtype))
        testing.assert_array_equal(
            self.m.diagonal(1), cupy.array([0, 0, 0], self.dtype))
        testing.assert_array_equal(
            self.m.diagonal(2), cupy.array([0, 0], self.dtype))
        testing.assert_array_equal(
            self.m.diagonal(3), cupy.array([0], self.dtype))


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

    def test_shape_none(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.dia_matrix(
                    (self.data(xp), self.offsets(xp)), shape=None)

    def test_scipy_sparse(self):
        s_h = scipy.sparse.dia_matrix((self.data(numpy), self.offsets(numpy)),
                                      shape=self.shape)
        s_d = sparse.dia_matrix(s_h)
        s_h2 = s_d.get()
        assert s_h.shape == s_d.shape
        assert s_h.dtype == s_d.dtype
        assert s_h.shape == s_h2.shape
        assert s_h.dtype == s_h2.dtype
        assert (s_h.data == s_h2.data).all()
        assert (s_h.offsets == s_h2.offsets).all()

    @testing.numpy_cupy_allclose(sp_name='sp', atol=1e-5)
    def test_intlike_shape(self, xp, sp):
        s = sp.dia_matrix((self.data(xp), self.offsets(xp)),
                          shape=(xp.array(self.shape[0]),
                                 xp.int32(self.shape[1])))
        assert isinstance(s.shape[0], int)
        assert isinstance(s.shape[1], int)
        return s

    def test_large_rank_offset(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.dia_matrix(
                    (self.data(xp), self.offsets(xp)[None]), shape=self.shape)

    def test_large_rank_data(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            with pytest.raises(ValueError):
                sp.dia_matrix(
                    (self.data(xp)[None], self.offsets(xp)), shape=self.shape)

    def test_data_offsets_different_size(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            offsets = xp.array([0, -1, 1], 'i')
            with pytest.raises(ValueError):
                sp.dia_matrix(
                    (self.data(xp), offsets), shape=self.shape)

    def test_duplicated_offsets(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            offsets = xp.array([1, 1], 'i')
            with pytest.raises(ValueError):
                sp.dia_matrix(
                    (self.data(xp), offsets), shape=self.shape)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_conj(self, xp, sp):
        n = _make_complex(xp, sp, self.dtype)
        cupy.testing.assert_array_equal(n.conj().data, n.data.conj())


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

    def test_nnz_axis_not_none(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = self.make(xp, sp, self.dtype)
            with pytest.raises(NotImplementedError):
                m.getnnz(axis=0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_A(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.A

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
        m = _make(xp, sp, self.dtype)
        return m.tocsc()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc_copy(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        n = m.tocsc(copy=True)
        assert m.data is not n.data
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
        return n

    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_transpose(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.transpose()

    @testing.with_requires('scipy>=1.5.0')
    def test_diagonal_error(self):
        # Before scipy 1.5.0 dia_matrix diagonal raised
        # `ValueError`, now returns empty array.
        # Check #3469
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            m = _make(xp, sp, self.dtype)
            d = m.diagonal(k=10)
            assert d.size == 0


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
        assert sparse.isspmatrix_dia(x) is True

    def test_csr(self):
        x = sparse.csr_matrix(
            (cupy.array([], 'f'),
             cupy.array([], 'i'),
             cupy.array([0], 'i')),
            shape=(0, 0), dtype='f')
        assert sparse.isspmatrix_dia(x) is False
