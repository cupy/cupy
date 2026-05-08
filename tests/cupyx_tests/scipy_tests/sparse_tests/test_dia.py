from __future__ import annotations

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
from cupy.cuda import driver
from cupy.cuda import runtime
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
        if (runtime.is_hip and self.dtype == numpy.float32
                and driver.get_build_version() == 400):
            pytest.xfail('generated wrong result -- may be buggy?')
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

    @testing.with_requires('scipy>=1.14')
    def test_str(self):
        # The exact DIA __str__ format differs across SciPy versions:
        # SciPy 1.14-1.16 use diagonal-major order; SciPy 1.17 switched
        # to row-major.  CuPy delegates ``__str__`` to ``str(self.get())``
        # so the output automatically tracks the installed SciPy.
        s = str(self.m)
        # Sanity-check the format-, type-, and shape-bearing repr line.
        assert 'DIAgonal' in s
        assert 'sparse matrix' in s
        assert str(self.m.shape) in s
        assert '(2 diagonals)' in s
        # Each stored value must show up exactly once.
        for value in [1.0, 2.0, 3.0, 4.0]:
            tok = (f'{value}' if numpy.dtype(self.dtype).kind == 'f'
                   else f'({int(value)}+0j)')
            assert tok in s, f'missing {tok!r} in {s!r}'
        # Delegation invariant.
        assert s == str(self.m.get())

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

    def test_todia_returns_self(self):
        # Base ``_spbase.todia`` round-trips via CSR which raises
        # NotImplementedError for csr_matrix.todia, so DIA must override.
        assert self.m.todia() is self.m
        assert self.m.todia(copy=True) is not self.m
        cupy.testing.assert_array_equal(
            self.m.todia(copy=True).toarray(), self.m.toarray())

    def test_empty_data_nnz(self):
        # gh-23055: an "empty" DIA buffer (data.shape[1] == 0) with
        # non-empty offsets should report nnz=0, not over-count.
        m = sparse.dia_matrix(
            (cupy.zeros((1, 0), self.dtype), cupy.array([0])),
            shape=(2, 2))
        assert m.nnz == 0

    def test_tocsc_data_wider_than_matrix(self):
        # When the DIA data buffer is wider than the matrix, columns
        # beyond ``num_cols`` lie outside the matrix.  ``tocsc`` must
        # truncate the indptr write rather than crashing on a broadcast
        # mismatch.
        m = sparse.dia_matrix(
            (cupy.ones((1, 5), self.dtype), cupy.array([0])),
            shape=(2, 2))
        c = m.tocsc()
        assert c.shape == (2, 2)
        assert c.nnz == 2
        cupy.testing.assert_array_equal(
            c.toarray(), cupy.eye(2, dtype=self.dtype))


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

    def setUp(self):
        if runtime.is_hip:
            if self.make_method in ('_make_empty',):
                # xcsr2coo could raise HIPSPARSE_STATUS_INVALID_VALUE, maybe
                # because we have a zero matrix (nnz=0)?
                pytest.xfail('may be buggy')

    @property
    def make(self):
        return globals()[self.make_method]

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_nnz_axis(self, xp, sp):
        # CuPy fixes scipy gh-23055 (DIA nnz with empty data buffer)
        # ahead of scipy: bound the diagonal length by the actual data
        # buffer.  scipy < 1.17 over-counts empty DIAs, so the
        # comparison would disagree for the empty variant; skip it.
        if self.make_method == '_make_empty':
            from packaging.version import parse as _v
            if _v(scipy.__version__) < _v('1.17'):
                pytest.skip('scipy < 1.17 over-counts empty DIA nnz')
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

    @testing.with_requires('scipy<1.14')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_A(self, xp, sp):
        m = self.make(xp, sp, self.dtype)
        return m.A

    @testing.with_requires('scipy>=1.16')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_tuple_axis(self, xp, sp):
        m = _make(xp, sp, self.dtype)
        return m.sum(axis=(0, 1))

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

    def setUp(self):
        if runtime.is_hip and self.axis in (0, -2):
            HIP_version = driver.get_build_version()
            if HIP_version < 5_00_00000:
                # internally a temporary CSC matrix is generated and thus
                # causes problems (see test_csc.py)
                pytest.xfail('spmv is buggy (trans=True)')

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
