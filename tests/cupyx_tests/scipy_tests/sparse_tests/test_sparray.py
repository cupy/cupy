"""Tests for sparse array classes (csr_array, csc_array, coo_array, dia_array).
"""
from __future__ import annotations

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


def _make_csr(xp, sp, dtype, *, array=False):
    """3x4 CSR with 4 nonzeros."""
    data = xp.array([0, 1, 2, 3], dtype)
    indices = xp.array([0, 1, 3, 2], 'i')
    indptr = xp.array([0, 2, 3, 4], 'i')
    cls = sp.csr_array if array else sp.csr_matrix
    return cls((data, indices, indptr), shape=(3, 4))


def _make_csr_sq(xp, sp, dtype, *, array=False):
    """3x3 square CSR."""
    data = xp.array([1, 2, 3], dtype)
    indices = xp.array([0, 1, 2], 'i')
    indptr = xp.array([0, 1, 2, 3], 'i')
    cls = sp.csr_array if array else sp.csr_matrix
    return cls((data, indices, indptr), shape=(3, 3))


def _make_csr_sq2(xp, sp, dtype, *, array=False):
    """Another 3x3 square CSR (different values)."""
    data = xp.array([4, 5, 6], dtype)
    indices = xp.array([2, 0, 1], 'i')
    indptr = xp.array([0, 1, 2, 3], 'i')
    cls = sp.csr_array if array else sp.csr_matrix
    return cls((data, indices, indptr), shape=(3, 3))


def _make_for_matmul(xp, sp, dtype, *, array=False):
    """4x3 CSR for matmul with 3x4."""
    data = xp.array([1, 2, 3, 4, 5], dtype)
    indices = xp.array([0, 2, 1, 0, 2], 'i')
    indptr = xp.array([0, 1, 3, 3, 5], 'i')
    cls = sp.csr_array if array else sp.csr_matrix
    return cls((data, indices, indptr), shape=(4, 3))


class TestSparseArrayTypeIdentity:
    """issparse, isspmatrix, isinstance checks for all formats."""

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_array_issparse(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        A = cls((2, 3), dtype=numpy.float64)
        assert sparse.issparse(A)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_array_not_isspmatrix(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        A = cls((2, 3), dtype=numpy.float64)
        assert not sparse.isspmatrix(A)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_array_isinstance_sparray(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        A = cls((2, 3), dtype=numpy.float64)
        assert isinstance(A, sparse.sparray)
        assert not isinstance(A, sparse.spmatrix)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_matrix_isinstance_spmatrix(self, fmt):
        cls = getattr(sparse, f'{fmt}_matrix')
        M = cls((2, 3), dtype=numpy.float64)
        assert isinstance(M, sparse.spmatrix)
        assert not isinstance(M, sparse.sparray)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_matrix_issparse(self, fmt):
        cls = getattr(sparse, f'{fmt}_matrix')
        M = cls((2, 3), dtype=numpy.float64)
        assert sparse.issparse(M)
        assert sparse.isspmatrix(M)

    def test_dense_not_sparse(self):
        assert not sparse.issparse(cupy.array([1, 2]))
        assert not sparse.isspmatrix(cupy.array([1, 2]))

    @testing.with_requires('scipy')
    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_type_system_matches_scipy(self, fmt):
        """CuPy and SciPy type predicates should agree."""
        sp_arr_cls = getattr(scipy.sparse, f'{fmt}_array')
        sp_mat_cls = getattr(scipy.sparse, f'{fmt}_matrix')
        cp_arr_cls = getattr(sparse, f'{fmt}_array')
        cp_mat_cls = getattr(sparse, f'{fmt}_matrix')

        sp_a = sp_arr_cls((2, 3))
        sp_m = sp_mat_cls((2, 3))
        cp_a = cp_arr_cls((2, 3))
        cp_m = cp_mat_cls((2, 3))

        assert sparse.issparse(cp_a) == scipy.sparse.issparse(sp_a)
        assert sparse.issparse(cp_m) == scipy.sparse.issparse(sp_m)
        assert sparse.isspmatrix(cp_a) == scipy.sparse.isspmatrix(sp_a)
        assert sparse.isspmatrix(cp_m) == scipy.sparse.isspmatrix(sp_m)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@testing.with_requires('scipy')
class TestCsrArrayConstruction:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_from_data_indices_indptr(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        assert m.format == 'csr'
        assert isinstance(m, sp.sparray)
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_from_dense(self, xp, sp):
        dense = xp.array([[1, 0, 2], [0, 3, 0]], dtype=self.dtype)
        m = sp.csr_array(dense)
        assert isinstance(m, sp.sparray)
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_from_coo_tuple(self, xp, sp):
        data = xp.array([1, 2, 3], self.dtype)
        row = xp.array([0, 1, 2], 'i')
        col = xp.array([2, 0, 1], 'i')
        m = sp.csr_array((data, (row, col)), shape=(3, 3))
        assert isinstance(m, sp.sparray)
        return m

    def test_empty(self):
        m = sparse.csr_array((3, 4), dtype=numpy.float64)
        assert m.shape == (3, 4)
        assert m.nnz == 0
        assert isinstance(m, sparse.sparray)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCsrArrayStarIsElementwise:
    """Verify that * is element-wise for csr_array (matching scipy.sparse)."""

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_sparse(self, xp, sp):
        """array * array should be element-wise."""
        a = _make_csr_sq(xp, sp, self.dtype, array=True)
        b = _make_csr_sq(xp, sp, self.dtype, array=True)
        return a * b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_scalar(self, xp, sp):
        """array * scalar should be scalar multiplication."""
        a = _make_csr(xp, sp, self.dtype, array=True)
        return a * self.dtype(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rstar_scalar(self, xp, sp):
        """scalar * array should be scalar multiplication."""
        a = _make_csr(xp, sp, self.dtype, array=True)
        return self.dtype(3.0) * a


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCsrArrayMatmul:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_matmul_sparse(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        b = _make_for_matmul(xp, sp, self.dtype, array=True)
        return a @ b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_matmul_dense_vector(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        x = xp.arange(4).astype(self.dtype)
        return a @ x

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_matmul_dense_matrix(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        x = xp.arange(8).reshape(4, 2).astype(self.dtype)
        return a @ x

    def test_matmul_scalar_raises(self):
        a = _make_csr_sq(cupy, sparse, numpy.float64, array=True)
        with pytest.raises(ValueError):
            a @ 5.0


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCsrArrayPower:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_elementwise(self, xp, sp):
        """array ** n should be element-wise, not matrix power."""
        a = _make_csr_sq(xp, sp, self.dtype, array=True)
        return a ** 2

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_matches_dense(self, xp, sp):
        """Verify ** gives same result as dense element-wise power."""
        a = _make_csr_sq(xp, sp, self.dtype, array=True)
        result_sparse = (a ** 2).toarray()
        result_dense = a.toarray() ** 2
        xp.testing.assert_allclose(result_sparse, result_dense)
        return result_sparse


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCsrMatrixStarIsMatmul:
    """Verify that * is still matmul for csr_matrix (unchanged)."""

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_matmul(self, xp, sp):
        """matrix * matrix should be matmul."""
        a = _make_csr(xp, sp, self.dtype)
        b = _make_for_matmul(xp, sp, self.dtype)
        return a * b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_scalar(self, xp, sp):
        """matrix * scalar should still work."""
        a = _make_csr(xp, sp, self.dtype)
        return a * self.dtype(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_pow_matrix_power(self, xp, sp):
        """matrix ** n should be matrix power."""
        a = _make_csr_sq(xp, sp, self.dtype)
        return a ** 2

    def test_power_zero_raises(self):
        """Array ** 0 raises NotImplementedError (would densify)."""
        a = _make_csr_sq(cupy, sparse, numpy.float64, array=True)
        with pytest.raises(NotImplementedError):
            a ** 0


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
class TestCsrArrayTypePreservation:
    """Operations on csr_array should return csr_array (not csr_matrix)."""

    def _check_array(self, result):
        assert isinstance(result, sparse.sparray), (
            f'Expected sparray, got {type(result).__name__}')
        assert not isinstance(result, sparse.spmatrix)

    def test_star(self):
        a = _make_csr_sq(cupy, sparse, self.dtype, array=True)
        b = _make_csr_sq(cupy, sparse, self.dtype, array=True)
        self._check_array(a * b)

    def test_matmul(self):
        a = _make_csr_sq(cupy, sparse, self.dtype, array=True)
        b = _make_csr_sq2(cupy, sparse, self.dtype, array=True)
        self._check_array(a @ b)

    def test_add(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        b = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(a + b)

    def test_sub(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        b = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(a - b)

    def test_neg(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(-a)

    def test_scalar_mul(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(a * self.dtype(2.0))

    def test_pow(self):
        a = _make_csr_sq(cupy, sparse, self.dtype, array=True)
        self._check_array(a ** 2)

    def test_copy(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(a.copy())

    def test_abs(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(abs(a))

    def test_T(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        result = a.T
        assert isinstance(result, sparse.sparray)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCsrArrayConversions:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        result = m.tocsc()
        assert isinstance(result, sp.sparray)
        assert result.format == 'csc'
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocoo(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        result = m.tocoo()
        assert isinstance(result, sp.sparray)
        assert result.format == 'coo'
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        return m.toarray()

    def test_tocsr_returns_self(self):
        m = _make_csr(cupy, sparse, numpy.float64, array=True)
        assert m.tocsr() is m

    def test_tocsr_copy(self):
        m = _make_csr(cupy, sparse, numpy.float64, array=True)
        n = m.tocsr(copy=True)
        assert n is not m
        assert isinstance(n, sparse.sparray)


@testing.with_requires('scipy')
class TestCsrArrayGet:

    def test_array_get_returns_scipy_array(self):
        m = _make_csr(cupy, sparse, numpy.float64, array=True)
        sp_m = m.get()
        assert isinstance(sp_m, scipy.sparse.csr_array)
        assert isinstance(sp_m, scipy.sparse.sparray)

    def test_matrix_get_returns_scipy_matrix(self):
        m = _make_csr(cupy, sparse, numpy.float64, array=False)
        sp_m = m.get()
        assert isinstance(sp_m, scipy.sparse.csr_matrix)
        assert isinstance(sp_m, scipy.sparse.spmatrix)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_array_get_values(self, xp, sp):
        m = _make_csr(xp, sp, numpy.float64, array=True)
        return m.toarray()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@testing.with_requires('scipy')
class TestCsrArrayArithmeticSciPyComparison:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        b = _make_csr(xp, sp, self.dtype, array=True)
        return a + b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        b = _make_csr(xp, sp, self.dtype, array=True)
        return (a - b).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_neg(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        return (-a).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_elementwise(self, xp, sp):
        a = _make_csr_sq(xp, sp, self.dtype, array=True)
        b = _make_csr_sq(xp, sp, self.dtype, array=True)
        return a * b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_scalar(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        return a * self.dtype(2.5)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_matmul(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        b = _make_for_matmul(xp, sp, self.dtype, array=True)
        return a @ b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_elementwise(self, xp, sp):
        a = _make_csr_sq(xp, sp, self.dtype, array=True)
        return a ** 2

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_abs(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        return abs(a)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_transpose(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        return a.T

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_conj(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        return a.conj()


class TestCsrArrayRemovedMethods:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.arr = sparse.csr_array(
            (cupy.array([1.0]), cupy.array([0], dtype='i'),
             cupy.array([0, 1], dtype='i')), shape=(1, 2))

    def test_no_A(self):
        with pytest.raises(AttributeError):
            self.arr.A

    def test_no_H(self):
        with pytest.raises(AttributeError):
            self.arr.H

    # NOTE: getrow/getcol are inherited from _csr_base in CuPy and work
    # on both arrays and matrices, unlike SciPy where they're matrix-only.

    def test_no_getH(self):
        with pytest.raises(AttributeError):
            self.arr.getH()

    def test_no_asfptype(self):
        with pytest.raises(AttributeError):
            self.arr.asfptype()

    def test_no_getformat(self):
        with pytest.raises(AttributeError):
            self.arr.getformat()

    def test_no_getmaxprint(self):
        with pytest.raises(AttributeError):
            self.arr.getmaxprint()

    def test_no_shape_setter(self):
        with pytest.raises(AttributeError):
            self.arr.shape = (2, 1)

    def test_has_shape(self):
        assert self.arr.shape == (1, 2)

    def test_has_nnz(self):
        assert self.arr.nnz == 1

    def test_has_format(self):
        assert self.arr.format == 'csr'

    def test_has_ndim(self):
        assert self.arr.ndim == 2


class TestCsrMatrixLegacyMethods:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.mat = sparse.csr_matrix(
            (cupy.array([1.0]), cupy.array([0], dtype='i'),
             cupy.array([0, 1], dtype='i')), shape=(1, 2))

    def test_has_A(self):
        result = self.mat.A
        assert isinstance(result, cupy.ndarray)

    def test_has_H(self):
        result = self.mat.H
        assert sparse.issparse(result)

    def test_has_getH(self):
        result = self.mat.getH()
        assert sparse.issparse(result)

    def test_has_asfptype(self):
        result = self.mat.asfptype()
        assert sparse.issparse(result)

    def test_has_getformat(self):
        assert self.mat.getformat() == 'csr'

    def test_has_shape_setter(self):
        # shape setter exists on matrices (even if reshape is no-op here)
        self.mat.shape = (1, 2)


# Index dtype policy: arrays preserve int64, matrices may downcast

@testing.with_requires('scipy')
class TestCsrArrayIndexDtype:

    def test_array_preserves_int64(self):
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        A = sparse.csr_array((data, indices, indptr), shape=(3, 3))
        assert A.indices.dtype == cupy.int64
        assert A.indptr.dtype == cupy.int64

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_array_int64_matches_scipy(self, xp, sp):
        data = xp.array([1.0, 2.0, 3.0])
        indices = xp.array([0, 1, 2], dtype='int64')
        indptr = xp.array([0, 1, 2, 3], dtype='int64')
        A = sp.csr_array((data, indices, indptr), shape=(3, 3))
        return A.indices.dtype == 'int64'

    def test_matrix_may_downcast(self):
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        M = sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        # matrix may downcast small int64 values to int32
        assert M.indices.dtype in (cupy.int32, cupy.int64)

    def test_coo_array_preserves_int64(self):
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        A = sparse.coo_array((data, (row, col)), shape=(3, 3))
        assert A.row.dtype == cupy.int64
        assert A.col.dtype == cupy.int64


# CSC/COO array conversion type preservation

class TestNonCsrArrayConversions:

    def test_coo_array_tocsr_type(self):
        A = sparse.coo_array(
            (cupy.array([1.0]), (cupy.array([0], dtype='i'),
             cupy.array([0], dtype='i'))), shape=(2, 2))
        B = A.tocsr()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'csr'

    def test_coo_array_tocsc_type(self):
        A = sparse.coo_array(
            (cupy.array([1.0]), (cupy.array([0], dtype='i'),
             cupy.array([0], dtype='i'))), shape=(2, 2))
        B = A.tocsc()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'csc'

    def test_csc_array_tocsr_type(self):
        data = cupy.array([1.0, 2.0], dtype='d')
        indices = cupy.array([0, 1], dtype='i')
        indptr = cupy.array([0, 1, 2], dtype='i')
        A = sparse.csc_array((data, indices, indptr), shape=(2, 2))
        B = A.tocsr()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'csr'

    def test_csc_array_tocoo_type(self):
        data = cupy.array([1.0, 2.0], dtype='d')
        indices = cupy.array([0, 1], dtype='i')
        indptr = cupy.array([0, 1, 2], dtype='i')
        A = sparse.csc_array((data, indices, indptr), shape=(2, 2))
        B = A.tocoo()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'coo'

    def test_csc_array_transpose_type(self):
        data = cupy.array([1.0, 2.0], dtype='d')
        indices = cupy.array([0, 1], dtype='i')
        indptr = cupy.array([0, 1, 2], dtype='i')
        A = sparse.csc_array((data, indices, indptr), shape=(2, 2))
        AT = A.T
        assert isinstance(AT, sparse.sparray)


# Construction functions

class TestConstructionFunctions:

    def test_eye_array_exists(self):
        A = sparse.eye_array(3)
        assert isinstance(A, sparse.sparray)
        assert A.shape == (3, 3)

    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_eye_array_values(self, xp, sp):
        return sp.eye_array(4, k=1, dtype='d').toarray()

    def test_eye_array_format(self):
        A = sparse.eye_array(3, format='csc')
        assert isinstance(A, sparse.sparray)
        assert A.format == 'csc'

    def test_diags_array(self):
        A = sparse.diags_array([1, 2, 3])
        assert isinstance(A, sparse.sparray)
        assert A.shape == (3, 3)

    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_diags_array_values(self, xp, sp):
        return sp.diags_array([1, 2, 3], offsets=0).toarray()

    def test_random_array(self):
        A = sparse.random_array((10, 10), density=0.5)
        assert isinstance(A, sparse.sparray)
        assert A.shape == (10, 10)

    def test_random_array_format(self):
        A = sparse.random_array((5, 5), format='csr')
        assert isinstance(A, sparse.sparray)
        assert A.format == 'csr'


# Type-aware construction: kron, hstack, vstack, tril, triu

class TestTypeAwareConstruct:

    @pytest.fixture
    def arr_pair(self):
        d = cupy.array([[1, 0], [0, 2]], dtype='d')
        return sparse.csr_array(d), sparse.csr_array(d)

    @pytest.fixture
    def mat_pair(self):
        d = cupy.array([[1, 0], [0, 2]], dtype='d')
        return sparse.csr_matrix(d), sparse.csr_matrix(d)

    def test_hstack_arrays(self, arr_pair):
        result = sparse.hstack(list(arr_pair))
        assert isinstance(result, sparse.sparray)

    def test_hstack_matrices(self, mat_pair):
        result = sparse.hstack(list(mat_pair))
        assert isinstance(result, sparse.spmatrix)

    def test_vstack_arrays(self, arr_pair):
        result = sparse.vstack(list(arr_pair))
        assert isinstance(result, sparse.sparray)

    def test_vstack_matrices(self, mat_pair):
        result = sparse.vstack(list(mat_pair))
        assert isinstance(result, sparse.spmatrix)

    def test_kron_arrays(self, arr_pair):
        result = sparse.kron(*arr_pair)
        assert isinstance(result, sparse.sparray)

    def test_kron_matrices(self, mat_pair):
        result = sparse.kron(*mat_pair)
        assert isinstance(result, sparse.spmatrix)

    def test_tril_array(self, arr_pair):
        result = sparse.tril(arr_pair[0])
        assert isinstance(result, sparse.sparray)

    def test_tril_matrix(self, mat_pair):
        result = sparse.tril(mat_pair[0])
        assert isinstance(result, sparse.spmatrix)

    def test_triu_array(self, arr_pair):
        result = sparse.triu(arr_pair[0])
        assert isinstance(result, sparse.sparray)

    def test_triu_matrix(self, mat_pair):
        result = sparse.triu(mat_pair[0])
        assert isinstance(result, sparse.spmatrix)


# CSC array arithmetic

@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCscArrayArithmetic:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add(self, xp, sp):
        data = xp.array([1, 2, 3], self.dtype)
        indices = xp.array([0, 1, 2], 'i')
        indptr = xp.array([0, 1, 2, 3], 'i')
        a = sp.csc_array((data, indices, indptr), shape=(3, 3))
        b = sp.csc_array((data, indices, indptr), shape=(3, 3))
        return a + b

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_sub(self, xp, sp):
        data = xp.array([1, 2, 3], self.dtype)
        indices = xp.array([0, 1, 2], 'i')
        indptr = xp.array([0, 1, 2, 3], 'i')
        a = sp.csc_array((data, indices, indptr), shape=(3, 3))
        b = sp.csc_array((data, indices, indptr), shape=(3, 3))
        return (a - b).toarray()

    def test_add_preserves_type(self):
        data = cupy.array([1, 2, 3], numpy.float64)
        indices = cupy.array([0, 1, 2], 'i')
        indptr = cupy.array([0, 1, 2, 3], 'i')
        a = sparse.csc_array((data, indices, indptr), shape=(3, 3))
        b = sparse.csc_array((data, indices, indptr), shape=(3, 3))
        result = a + b
        assert isinstance(result, sparse.sparray)


# Cross-format multiply

@testing.with_requires('scipy')
class TestCrossFormatMultiply:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_coo_star_coo(self, xp, sp):
        """COO * COO element-wise should work."""
        data = xp.array([1, 2, 3], numpy.float64)
        row = xp.array([0, 1, 2], 'i')
        col = xp.array([0, 1, 2], 'i')
        a = sp.coo_array((data, (row, col)), shape=(3, 3))
        b = sp.coo_array((data, (row, col)), shape=(3, 3))
        return (a * b).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_csc_star_csc(self, xp, sp):
        """CSC * CSC element-wise should work."""
        data = xp.array([1, 2, 3], numpy.float64)
        indices = xp.array([0, 1, 2], 'i')
        indptr = xp.array([0, 1, 2, 3], 'i')
        a = sp.csc_array((data, indices, indptr), shape=(3, 3))
        b = sp.csc_array((data, indices, indptr), shape=(3, 3))
        return (a * b).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_csr_star_coo(self, xp, sp):
        """CSR * COO cross-format multiply should work."""
        data = xp.array([1, 2, 3], numpy.float64)
        indices = xp.array([0, 1, 2], 'i')
        indptr = xp.array([0, 1, 2, 3], 'i')
        a = sp.csr_array((data, indices, indptr), shape=(3, 3))
        row = xp.array([0, 1, 2], 'i')
        col = xp.array([0, 1, 2], 'i')
        b = sp.coo_array((data, (row, col)), shape=(3, 3))
        return (a * b).toarray()


# Reduction 1D shaping

@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestArrayReductions:

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_sum_axis0_ndim(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        result = m.sum(axis=0)
        return result.ndim

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_sum_axis1_ndim(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        result = m.sum(axis=1)
        return result.ndim

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_axis0_values(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        return m.sum(axis=0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_axis1_values(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        return m.sum(axis=1)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean_axis0(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        return m.mean(axis=0)

    def test_matrix_sum_stays_2d(self):
        """Matrix sum(axis=0) should still be 2D."""
        m = _make_csr(cupy, sparse, numpy.float64, array=False)
        result = m.sum(axis=0)
        assert result.ndim == 2


# DIA array

class TestDiaArrayBasic:

    def test_construction(self):
        data = cupy.array([[1, 2, 3]], dtype=numpy.float64)
        offsets = cupy.array([0])
        A = sparse.dia_array((data, offsets), shape=(3, 3))
        assert isinstance(A, sparse.sparray)
        assert A.format == 'dia'

    def test_tocsr(self):
        data = cupy.array([[1, 2, 3]], dtype=numpy.float64)
        offsets = cupy.array([0])
        A = sparse.dia_array((data, offsets), shape=(3, 3))
        B = A.tocsr()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'csr'

    def test_tocsc(self):
        data = cupy.array([[1, 2, 3]], dtype=numpy.float64)
        offsets = cupy.array([0])
        A = sparse.dia_array((data, offsets), shape=(3, 3))
        B = A.tocsc()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'csc'

    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray_matches_scipy(self, xp, sp):
        data = xp.array([[1, 2, 3]], dtype=numpy.float64)
        offsets = xp.array([0])
        A = sp.dia_array((data, offsets), shape=(3, 3))
        return A.toarray()


# LinearOperator from array

class TestLinearOperatorFromArray:

    def test_aslinearoperator_csr_array(self):
        from cupyx.scipy.sparse.linalg import aslinearoperator
        m = _make_csr_sq(cupy, sparse, numpy.float64, array=True)
        op = aslinearoperator(m)
        v = cupy.ones(3, dtype=numpy.float64)
        result = op @ v
        expected = m @ v
        cupy.testing.assert_allclose(result, expected)


# Linalg solvers accept arrays

class TestSpsolveArray:

    def test_spsolve_csr_array(self):
        from cupyx.scipy.sparse.linalg import spsolve
        n = 8
        A_dense = cupy.zeros((n, n), dtype=numpy.float64)
        A_dense[cupy.arange(n), cupy.arange(n)] = 4
        A_dense[cupy.arange(n - 1), cupy.arange(1, n)] = 1
        A_dense[cupy.arange(1, n), cupy.arange(n - 1)] = 1
        A = sparse.csr_array(A_dense)
        b = cupy.arange(1, n + 1, dtype=numpy.float64)
        x = spsolve(A, b)
        cupy.testing.assert_allclose(A @ x, b, rtol=1e-10)

    def test_spsolve_csr_matrix(self):
        from cupyx.scipy.sparse.linalg import spsolve
        n = 8
        A_dense = cupy.zeros((n, n), dtype=numpy.float64)
        A_dense[cupy.arange(n), cupy.arange(n)] = 4
        A_dense[cupy.arange(n - 1), cupy.arange(1, n)] = 1
        A_dense[cupy.arange(1, n), cupy.arange(n - 1)] = 1
        M = sparse.csr_matrix(A_dense)
        b = cupy.arange(1, n + 1, dtype=numpy.float64)
        x = spsolve(M, b)
        cupy.testing.assert_allclose(M * x, b, rtol=1e-10)
