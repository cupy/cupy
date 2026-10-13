"""Scipy-parity tests for the full set of supported sparse value dtypes.

``cupyx.scipy.sparse`` accepts the same value dtypes as scipy: ``bool``,
every signed/unsigned integer width, ``float32``/``float64`` and
``complex64``/``complex128``.  cuSPARSE itself only handles float/complex, so
bool and integer data route through pure-CuPy fallbacks; these tests pin the
fallbacks to scipy across construction, conversion, arithmetic, matmul,
reductions and indexing (including integer overflow, which must wrap).
"""
from __future__ import annotations

import functools
import operator

import numpy
import pytest
try:
    import scipy.sparse  # noqa: F401
except ImportError:
    pass

import cupy
from cupy import testing
from cupyx import cusparse
from cupyx.scipy import sparse

# ==/<=/>= (and out-of-range </<=/!=) comparisons on a sparse matrix
# intentionally emit SparseEfficiencyWarning; these tests check the comparison
# RESULT, not the warning, so silence it module-wide.
pytestmark = pytest.mark.filterwarnings(
    'ignore::cupyx.scipy.sparse.SparseEfficiencyWarning')

# These tests pin value/dtype parity, not memory layout; cupy and scipy
# differ in the C/F contiguity of ``toarray`` for some formats (e.g. CSC),
# which is irrelevant here, so skip the contiguity check.
_allclose = functools.partial(
    testing.numpy_cupy_allclose, sp_name='sp', contiguous_check=False)
_array_equal = functools.partial(
    testing.numpy_cupy_array_equal, sp_name='sp')


INT_DTYPES = [
    numpy.int8, numpy.int16, numpy.int32, numpy.int64,
    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
]
# Every value dtype scipy stores that the GPU can represent.
ALL_VALUE_DTYPES = (
    [numpy.bool_] + INT_DTYPES + [numpy.float32, numpy.float64,
                                  numpy.complex64, numpy.complex128]
)
COMPRESSED_FORMATS = ('csr', 'csc', 'coo')

# 16-bit floats cupy supports beyond scipy (scipy rejects both).  They have
# no scipy oracle, so ``TestNarrowFloatVsDense`` checks them against cupy's
# dense arrays instead.  bfloat16 needs the optional ``ml_dtypes`` package.
NARROW_FLOAT_DTYPES = [numpy.dtype(numpy.float16)]
try:
    import ml_dtypes as _ml_dtypes
    _bfloat16 = numpy.dtype(_ml_dtypes.bfloat16)
    NARROW_FLOAT_DTYPES.append(_bfloat16)
except ImportError:
    _bfloat16 = None
requires_bfloat16 = pytest.mark.skipif(
    _bfloat16 is None, reason='ml_dtypes (bfloat16) not installed')

# A representative subset of integer widths for the scipy-parity helpers.
_scipy_dtypes = [numpy.int8, numpy.int32, numpy.int64, numpy.uint8]


def _dense_a(xp, dtype):
    """A fixed 4x5 array with a mix of zeros (small values, no overflow)."""
    a = xp.array([[3, 0, 1, 0, 2],
                  [0, 5, 0, 4, 0],
                  [7, 0, 0, 0, 6],
                  [0, 8, 9, 0, 0]])
    return a.astype(dtype)


def _dense_b(xp, dtype):
    """A second fixed 4x5 array; sparsity distinct from ``_dense_a``."""
    b = xp.array([[0, 2, 0, 0, 1],
                  [4, 0, 3, 0, 0],
                  [0, 0, 5, 6, 0],
                  [1, 0, 0, 2, 3]])
    return b.astype(dtype)


def _square(xp, dtype):
    """A fixed 4x4 array (for matmul / power)."""
    a = xp.array([[2, 0, 1, 0],
                  [0, 3, 0, 1],
                  [1, 0, 2, 0],
                  [0, 1, 0, 2]])
    return a.astype(dtype)


def _make(sp, xp, dense, fmt):
    return getattr(sp, f'{fmt}_array')(dense)


@testing.parameterize(*testing.product({
    'dtype': ALL_VALUE_DTYPES,
    'fmt': COMPRESSED_FORMATS,
}))
class TestConstructionConversion:

    @_allclose()
    def test_toarray_roundtrip(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).toarray()

    @_allclose()
    def test_tocsr(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).tocsr()\
            .toarray()

    @_allclose()
    def test_tocsc(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).tocsc()\
            .toarray()

    @_allclose()
    def test_tocoo(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).tocoo()\
            .toarray()

    @_allclose()
    def test_full_roundtrip(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), self.fmt)
        return a.tocsr().tocsc().tocoo().tocsr().toarray()

    @_allclose()
    def test_copy(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).copy()\
            .toarray()

    def test_dtype_preserved(self):
        a = _make(sparse, cupy, _dense_a(cupy, self.dtype), self.fmt)
        assert a.dtype == self.dtype
        assert a.toarray().dtype == self.dtype


@testing.parameterize(*testing.product({
    'dtype': ALL_VALUE_DTYPES,
    'fmt': COMPRESSED_FORMATS,
}))
class TestArithmetic:

    @_allclose()
    def test_add(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), self.fmt)
        b = _make(sp, xp, _dense_b(xp, self.dtype), self.fmt)
        return (a + b).toarray()

    @_allclose()
    def test_multiply_elementwise(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), self.fmt)
        b = _make(sp, xp, _dense_b(xp, self.dtype), self.fmt)
        return a.multiply(b).toarray()

    @_allclose()
    def test_multiply_dense(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), self.fmt)
        return a.multiply(_dense_b(xp, self.dtype)).toarray()

    @_allclose()
    def test_scalar_mul(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), self.fmt)
        return (a * 3).toarray()

    @_allclose()
    def test_maximum(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), self.fmt)
        b = _make(sp, xp, _dense_b(xp, self.dtype), self.fmt)
        return a.maximum(b).toarray()

    @_allclose()
    def test_minimum(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), self.fmt)
        b = _make(sp, xp, _dense_b(xp, self.dtype), self.fmt)
        return a.minimum(b).toarray()

    @_allclose()
    def test_matmul_sparse(self, xp, sp):
        a = _make(sp, xp, _square(xp, self.dtype), self.fmt)
        b = _make(sp, xp, _square(xp, self.dtype), self.fmt)
        return (a @ b).toarray()


@testing.parameterize(*testing.product({
    'dtype': ALL_VALUE_DTYPES,
    'fmt': COMPRESSED_FORMATS,
}))
class TestSubtraction:

    @_allclose()
    def test_sub(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), self.fmt)
        b = _make(sp, xp, _dense_b(xp, self.dtype), self.fmt)
        return (a - b).toarray()


@testing.parameterize(*testing.product({
    'fmt': COMPRESSED_FORMATS,
}))
class TestBoolSubtraction:
    """``bool - bool`` matches scipy.sparse (which supports it, unlike numpy
    dense): it is computed in int8 so a coalesced ``True - True`` cancels to
    ``False`` rather than being OR-ed to ``True``."""

    @_allclose()
    def test_bool_sub_cancels(self, xp, sp):
        # (0, 0) is True in both operands -> True - True -> False.
        a = _make(sp, xp, xp.array([[True, True], [False, True]]), self.fmt)
        b = _make(sp, xp, xp.array([[True, False], [True, True]]), self.fmt)
        return (a - b).toarray()


@testing.parameterize(*testing.product({
    'dtype': ALL_VALUE_DTYPES,
    'op': ['lt', 'le', 'gt', 'ge', 'eq', 'ne'],
}))
class TestComparisons:

    @_allclose()
    def test_sparse_sparse(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), 'csr')
        b = _make(sp, xp, _dense_b(xp, self.dtype), 'csr')
        return getattr(operator, self.op)(a, b).toarray()


@testing.parameterize(*testing.product({
    'dtype': INT_DTYPES + [numpy.bool_],
}))
class TestMatmulDense:

    @_allclose()
    def test_sparse_at_vector(self, xp, sp):
        a = _make(sp, xp, _square(xp, self.dtype), 'csr')
        v = xp.arange(4).astype(self.dtype)
        return a @ v

    @_allclose()
    def test_sparse_at_matrix(self, xp, sp):
        a = _make(sp, xp, _square(xp, self.dtype), 'csr')
        m = _square(xp, self.dtype).T.copy()
        return a @ m

    @_allclose()
    def test_csc_at_matrix(self, xp, sp):
        a = _make(sp, xp, _square(xp, self.dtype), 'csc')
        m = _square(xp, self.dtype).T.copy()
        return a @ m

    @_allclose()
    def test_dense_at_sparse(self, xp, sp):
        a = _make(sp, xp, _square(xp, self.dtype), 'csr')
        m = _square(xp, self.dtype).T.copy()
        return m @ a


@testing.parameterize(*testing.product({
    'dtype': INT_DTYPES,
    'fmt': COMPRESSED_FORMATS,
}))
class TestReductionsSum:

    @_allclose()
    def test_sum_none(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).sum()

    @_allclose()
    def test_sum_axis0(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).sum(axis=0)

    @_allclose()
    def test_sum_axis1(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).sum(axis=1)

    @_allclose()
    def test_mean_axis0(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).mean(axis=0)


@testing.parameterize(*testing.product({
    'dtype': INT_DTYPES,
    # cupy exposes min/max/argmin/argmax only on the compressed formats
    # (a pre-existing gap for COO, independent of value dtype).
    'fmt': ('csr', 'csc'),
}))
class TestReductionsMinMax:

    @_allclose()
    def test_min_axis0(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), self.fmt)
        return a.min(axis=0).toarray()

    @_allclose()
    def test_max_axis1(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), self.fmt)
        return a.max(axis=1).toarray()

    @_array_equal()
    def test_argmin_axis0(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).argmin(axis=0)

    @_array_equal()
    def test_argmax_axis1(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).argmax(axis=1)


class TestMinMaxLargeInt64:
    """64-bit integer magnitudes exceed float64's exact range (2**53)."""

    @_allclose()
    def test_int64_max_axis(self, xp, sp):
        base = 2 ** 53
        dense = xp.array([[base + 1, 0, base + 5],
                          [0, base + 3, 0]], dtype=xp.int64)
        return sp.csr_array(dense).max(axis=0).toarray()

    @_allclose()
    def test_uint64_max_axis(self, xp, sp):
        base = 2 ** 63
        dense = xp.array([[base + 1, 0, base + 5],
                          [0, base + 3, 0]], dtype=xp.uint64)
        return sp.csr_array(dense).max(axis=0).toarray()


@testing.parameterize(*testing.product({
    'dtype': ALL_VALUE_DTYPES,
}))
class TestIndexing:

    @_allclose()
    def test_row_slice(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), 'csr')[1:3].toarray()

    @_allclose()
    def test_row_fancy(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), 'csr')
        return a[[0, 2, 1]].toarray()

    @_allclose()
    def test_col_fancy(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), 'csr')
        return a[:, [4, 1, 0]].toarray()

    @_allclose()
    def test_scalar_getitem(self, xp, sp):
        # Return the raw scalar so complex dtypes work too (``int()`` cannot
        # cast a complex scalar); position (2, 4) holds a stored nonzero.
        a = _make(sp, xp, _dense_a(xp, self.dtype), 'csr')
        return a[2, 4]

    @_allclose()
    def test_diagonal(self, xp, sp):
        return _make(sp, xp, _square(xp, self.dtype), 'csr').diagonal()

    @_allclose()
    def test_transpose(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), 'csr').T.toarray()


@testing.parameterize(*testing.product({
    'dtype': INT_DTYPES,
}))
class TestUnary:

    @_allclose()
    def test_abs(self, xp, sp):
        return abs(_make(sp, xp, _dense_a(xp, self.dtype), 'csr')).toarray()

    @_allclose()
    def test_neg_signed(self, xp, sp):
        if self.dtype(0).dtype.kind != 'i':
            pytest.skip('negation only well-defined for signed integers')
        return (-_make(sp, xp, _dense_a(xp, self.dtype), 'csr')).toarray()


@testing.parameterize(*testing.product({
    'dtype': [numpy.int8, numpy.int64, numpy.uint8],
    'ddtype': [numpy.float32, numpy.float64, numpy.int64],
}))
class TestDivision:

    @_allclose()
    def test_div_dense(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), 'csr')
        d = _dense_b(xp, self.ddtype) + 1  # avoid divide-by-zero
        return (a / d).toarray()

    @_allclose()
    def test_div_scalar(self, xp, sp):
        a = _make(sp, xp, _dense_a(xp, self.dtype), 'csr')
        return (a / 3).toarray()


@testing.parameterize(*testing.product({
    'idtype': [numpy.int8, numpy.int16, numpy.uint8],
}))
class TestIntegerOverflowWraps:
    """Integer results must wrap on overflow, matching numpy/scipy."""

    @_array_equal()
    def test_add_overflow(self, xp, sp):
        info = numpy.iinfo(self.idtype)
        big = info.max
        dense = xp.array([[big, 0], [0, big]]).astype(self.idtype)
        a = sp.csr_array(dense)
        return (a + a).toarray()

    @_array_equal()
    def test_matmul_overflow(self, xp, sp):
        info = numpy.iinfo(self.idtype)
        v = info.max // 2 + 1
        dense = xp.array([[v, v], [0, v]]).astype(self.idtype)
        a = sp.csr_array(dense)
        return (a @ a).toarray()

    @_array_equal()
    def test_scalar_mul_overflow(self, xp, sp):
        info = numpy.iinfo(self.idtype)
        dense = xp.array([[info.max, 0], [0, info.max]]).astype(self.idtype)
        return (sp.csr_array(dense) * 3).toarray()


@testing.parameterize(*testing.product({
    'fmt': COMPRESSED_FORMATS,
}))
class TestDtypeGating:

    def test_longdouble_rejected(self):
        # Extended precision cannot be stored on the GPU.
        if numpy.dtype(numpy.longdouble).itemsize <= 8:
            pytest.skip('longdouble aliases float64 on this platform')
        a = getattr(sparse, f'{self.fmt}_array')(
            cupy.eye(3, dtype=numpy.float64))
        with pytest.raises(ValueError):
            a.astype(numpy.longdouble)

    def test_astype_to_int_ok(self):
        a = getattr(sparse, f'{self.fmt}_array')(
            cupy.eye(3, dtype=numpy.float64))
        assert a.astype(numpy.int16).dtype == numpy.int16

    def test_astype_to_float16_ok(self):
        a = getattr(sparse, f'{self.fmt}_array')(
            cupy.eye(3, dtype=numpy.float64))
        assert a.astype(numpy.float16).dtype == numpy.float16


@testing.parameterize(*testing.product({
    'dtype': ALL_VALUE_DTYPES,
}))
class TestRandomArray:

    def test_random_dtype_and_nonzero(self):
        r = sparse.random_array((30, 30), density=0.3, dtype=self.dtype,
                                rng=cupy.random.RandomState(0))
        assert r.dtype == self.dtype
        assert r.nnz > 0
        data = r.data.get()
        kind = numpy.dtype(self.dtype).kind
        if kind == 'i':
            # Full-range sampling should produce some negative values.
            assert (data < 0).any()
        elif kind == 'c':
            assert (data.imag != 0).any()


class TestRandomDtypeGating:

    def test_float16_accepted(self):
        r = sparse.random_array((6, 6), density=0.5, dtype=numpy.float16)
        assert r.dtype == numpy.float16
        assert r.nnz > 0

    def test_longdouble_rejected(self):
        if numpy.dtype(numpy.longdouble).itemsize <= 8:
            pytest.skip('longdouble aliases float64 on this platform')
        with pytest.raises(ValueError):
            sparse.random_array((5, 5), dtype=numpy.longdouble)


@testing.parameterize(*testing.product({
    'dtype': ALL_VALUE_DTYPES + NARROW_FLOAT_DTYPES,
    'fmt': ('csr', 'csc'),
}))
class TestEliminateZeros:
    """``eliminate_zeros`` must work for every dtype.  One pure-CuPy
    mask-and-rebuild path serves all dtypes: cuSPARSE's ``csr2csr_compress``
    prunes by a signed ``value > tol`` (which drops negatives, not just
    zeros) and is int32-only, so it is not used."""

    def test_drops_stored_zero(self):
        a = getattr(sparse, f'{self.fmt}_array')(_dense_a(cupy, self.dtype))
        a.data[0] = 0            # force one explicitly-stored zero
        n_before = int(a.nnz)
        a.eliminate_zeros()
        assert int(a.nnz) == n_before - 1
        assert not bool((a.data == 0).any())


def _narrow_check(sp_res, dn_res, *, dtype_match=True, rtol=4e-2):
    """Compare a sparse result to a cupy dense oracle in float32.

    The pure-CuPy fallbacks accumulate 16-bit floats in float32, so results
    are compared after widening; a modest tolerance covers 16-bit rounding.
    """
    sr = sp_res.toarray() if hasattr(sp_res, 'toarray') else sp_res
    sr = cupy.asarray(sr)
    dn = cupy.asarray(dn_res)
    if dtype_match:
        assert sr.dtype == dn.dtype, f'{sr.dtype} != {dn.dtype}'
    testing.assert_allclose(sr.astype(cupy.float32), dn.astype(cupy.float32),
                            rtol=rtol, atol=1e-2)


@testing.parameterize(*testing.product({
    'dtype': NARROW_FLOAT_DTYPES,
    'fmt': COMPRESSED_FORMATS,
}))
class TestNarrowFloatVsDense:
    """float16/bfloat16 have no scipy equivalent, so check against cupy dense.

    Covers the ops whose fallbacks route 16-bit floats through float32
    (arithmetic, matmul, reductions); dense uses the same values.
    """

    def _pair(self):
        da = _dense_a(cupy, self.dtype)
        db = _dense_b(cupy, self.dtype)
        make = getattr(sparse, f'{self.fmt}_array')
        return make(da), make(db), da, db

    def test_toarray(self):
        A, _, da, _ = self._pair()
        _narrow_check(A, da)

    def test_add(self):
        A, B, da, db = self._pair()
        _narrow_check(A + B, da + db)

    def test_multiply(self):
        A, B, da, db = self._pair()
        _narrow_check(A.multiply(B), da * db)

    def test_scalar_mul(self):
        A, _, da, _ = self._pair()
        _narrow_check(A * 3, da * self.dtype.type(3))

    def test_matmul_sparse(self):
        d = _square(cupy, self.dtype)
        A = getattr(sparse, f'{self.fmt}_array')(d)
        _narrow_check(A @ A, d @ d)

    def test_matmul_dense(self):
        d = _square(cupy, self.dtype)
        A = getattr(sparse, f'{self.fmt}_array')(d)
        _narrow_check(A @ d, d @ d)

    def test_sum_axis0(self):
        A, _, da, _ = self._pair()
        _narrow_check(A.sum(axis=0), da.sum(axis=0))

    def test_sum_none(self):
        A, _, da, _ = self._pair()
        _narrow_check(A.sum(), da.sum())

    def test_mean_axis0(self):
        A, _, da, _ = self._pair()
        _narrow_check(A.mean(axis=0), da.mean(axis=0))

    def test_dtype_preserved(self):
        A, _, _, _ = self._pair()
        assert A.dtype == self.dtype
        assert A.tocsc().dtype == self.dtype


@testing.parameterize(*testing.product({
    'dtype': NARROW_FLOAT_DTYPES,
}))
class TestNarrowFloatReductionsIndexing:
    """min/max/argmin and indexing for 16-bit floats (CSR), vs cupy dense."""

    def _A(self):
        da = _dense_a(cupy, self.dtype)
        return sparse.csr_array(da), da

    def test_min_axis0(self):
        A, da = self._A()
        _narrow_check(A.min(axis=0), da.min(axis=0))

    def test_max_axis1(self):
        A, da = self._A()
        _narrow_check(A.max(axis=1), da.max(axis=1))

    def test_argmin_axis0(self):
        A, da = self._A()
        _narrow_check(A.argmin(axis=0), da.argmin(axis=0), dtype_match=False)

    def test_argmax_axis1(self):
        A, da = self._A()
        _narrow_check(A.argmax(axis=1), da.argmax(axis=1), dtype_match=False)

    def test_col_fancy(self):
        A, da = self._A()
        _narrow_check(A[:, [0, 2]], da[:, [0, 2]])

    def test_row_fancy(self):
        A, da = self._A()
        _narrow_check(A[[0, 2, 1]], da[[0, 2, 1]])

    def test_row_slice(self):
        A, da = self._A()
        _narrow_check(A[1:3], da[1:3])

    def test_scalar_getitem(self):
        A, da = self._A()
        _narrow_check(A[2, 4], da[2, 4])

    def test_transpose(self):
        A, da = self._A()
        _narrow_check(A.T, da.T)


class TestValueSemanticsRegressions:
    """Value-semantics edge cases the happy-path tests do not cover."""

    @pytest.mark.parametrize('dtype', INT_DTYPES)
    def test_matmul_dimension_mismatch_raises(self, dtype):
        # The pure-CuPy int matmul must raise (not silently clamp the gather)
        # on a length mismatch, like the float path and scipy.
        a = sparse.csr_array(
            cupy.asarray(numpy.array([[1, 0, 2, 0], [0, 3, 0, 0]], dtype)))
        with pytest.raises(ValueError):
            a @ cupy.arange(3).astype(dtype)          # needs length 4
        with pytest.raises(ValueError):
            a @ cupy.arange(3 * 2).reshape(3, 2).astype(dtype)

    @pytest.mark.parametrize('op', ['argmin', 'argmax'])
    def test_complex_argreduce_axis(self, op):
        # Complex arg-reduction along an axis must not crash; it reduces on
        # the real part (like complex min/max).  With distinct real parts per
        # column that agrees with scipy's lexicographic result.
        m = numpy.array([[1 + 2j, 0, 3], [0, 5 - 1j, 0], [4 + 0j, 0, 2 - 3j]])
        g = getattr(sparse.csr_array(cupy.asarray(m)), op)(axis=0)
        s = numpy.asarray(getattr(scipy.sparse.csr_array(m), op)(axis=0))
        testing.assert_array_equal(cupy.asarray(g).ravel().get(), s.ravel())

    def test_complex_argreduce_reduces_on_real_part(self):
        # cupy compares complex by real part (consistent with min/max), which
        # diverges from scipy's lexicographic tie-break.  Pin it so a future
        # "parity" change is a conscious decision, not an accidental break.
        c = cupy.array([[1 + 2j, 1 + 9j, 0]], dtype=cupy.complex64)
        a = sparse.csr_array(c)
        # Real parts tie at 1; the first max-real wins (not the larger imag).
        assert int(a.argmax(axis=1).get().ravel()[0]) == 0
        assert int(a.argmin(axis=1).get().ravel()[0]) == 2  # the implicit 0

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int32, numpy.int64])
    def test_toarray_dedups_despite_stale_flag(self, dtype):
        # A plain scatter densify must still sum duplicates even when
        # ``has_canonical_format`` is (wrongly) True.  Build the CSR directly
        # (not via ``asformat``, which would coalesce first).
        a = sparse.csr_array(
            (cupy.array([3, 4]).astype(dtype),
             cupy.array([0, 0], numpy.int32),
             cupy.array([0, 2], numpy.int32)), shape=(1, 2))
        a.has_canonical_format = True         # lie: (0, 0) is duplicated
        testing.assert_array_equal(
            a.toarray().get(), numpy.array([[7, 0]], dtype))

    @pytest.mark.parametrize('target', [numpy.bool_, numpy.int8, numpy.int64])
    def test_power_explicit_dtype(self, target):
        # ``power(n, dtype=)`` must not crash on a widening target (bool**n
        # -> int); the explicit-dtype branch is out-of-place like the default.
        a = sparse.csr_array(
            cupy.asarray(numpy.array([[2, 0], [0, 3]], numpy.int32)))
        r = a.power(2, dtype=target)
        assert r.nnz == 2

    def test_uint_spgemm_negative_alpha(self):
        # Uint operand scaling by -1 must wrap, not raise OverflowError.
        a = sparse.csr_matrix(
            cupy.asarray(numpy.array([[1, 0], [0, 2]], numpy.uint8)))
        r = cusparse.spgemm(a, a, alpha=-1)
        # (-1) * diag(1, 4) wraps in uint8: 255, 252
        testing.assert_array_equal(
            r.toarray().get(), numpy.array([[255, 0], [0, 252]], numpy.uint8))

    @_allclose()
    @pytest.mark.parametrize('op', ['lt', 'le', 'gt', 'ge', 'eq', 'ne'])
    @pytest.mark.parametrize('scalar', [300, 2.5, 2, -1])
    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int64, numpy.uint8])
    @pytest.mark.parametrize('fmt', COMPRESSED_FORMATS)
    def test_int_scalar_comparison(self, xp, sp, fmt, dtype, op, scalar):
        # Comparing an integer matrix to an out-of-range (300) or fractional
        # (2.5) scalar must match scipy -- not overflow (casting 300 into
        # int8) or truncate (2.5 -> 2).  The promotion is format- and
        # width-independent, so pin it across csr/csc/coo and several ints.
        a = _make(sp, xp, _dense_a(xp, dtype), fmt)
        return getattr(operator, op)(a, scalar).toarray()

    @pytest.mark.parametrize('dtype', INT_DTYPES)
    def test_dia_int_sum_excludes_padding(self, dtype):
        # DIA ``data`` holds off-matrix padding; sum(axis=None) must count
        # only the stored values (matches scipy).
        data = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(dtype)
        offsets = numpy.array([0, -1])
        g = sparse.dia_array((cupy.asarray(data), cupy.asarray(offsets)),
                             shape=(4, 4))
        s = scipy.sparse.dia_array((data, offsets), shape=(4, 4))
        assert int(g.sum()) == int(s.sum())

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int64, numpy.bool_])
    def test_spgemm_preserves_int32_indices(self, dtype):
        # Integer/bool sparse@sparse must keep int32 indices like the float
        # path (the pure-CuPy fallback normalises to int64 internally).
        a = sparse.csr_array(
            cupy.asarray(_dense_a(numpy, dtype)[:, :4]))  # 4x4, int32 indices
        assert (a @ a.T).indices.dtype == numpy.int32

    @pytest.mark.parametrize('k', [255, 256, 257, 512])
    def test_bool_spgemm_overlap_multiple_of_256(self, k):
        # bool sparse@sparse is a logical any-over-products: a cell is True
        # iff the operands overlap at any position.  The pure-CuPy fallback
        # must keep bool operands as bool -- an int8 carrier wraps 256 True
        # products to int8(0), a spurious False whenever the shared inner
        # dimension is a multiple of 256.
        a = numpy.ones((1, k), dtype=bool)
        b = numpy.ones((k, 1), dtype=bool)
        got = (sparse.csr_array(cupy.asarray(a))
               @ sparse.csr_array(cupy.asarray(b))).toarray()
        exp = (scipy.sparse.csr_array(a) @ scipy.sparse.csr_array(b)).toarray()
        assert bool(cupy.asnumpy(got)[0, 0])  # True for every k >= 1
        testing.assert_array_equal(cupy.asnumpy(got), exp)

    @pytest.mark.parametrize('dtype', NARROW_FLOAT_DTYPES)
    def test_str_of_narrow_float(self, dtype):
        # scipy.sparse cannot represent float16/bfloat16, so ``str`` must
        # fall back to ``repr`` instead of raising ValueError.
        a = sparse.csr_array(cupy.eye(3, dtype=dtype))
        assert isinstance(str(a), str)
        assert '{}'.format(a)

    @pytest.mark.parametrize('dtype', NARROW_FLOAT_DTYPES)
    def test_narrow_float_division_matches_dense_ufunc(self, dtype):
        # 16-bit floats have no scipy oracle, so division dtype must match the
        # cupy dense *ufunc* -- not numpy.result_type, which is inconsistent
        # for bfloat16 (``result_type(bf16, 2.0)`` is float64 but the ufunc
        # ``bf16 / 2.0`` is float32).  Scalar and dense-array divisor paths.
        dense = _dense_a(cupy, dtype)
        a = sparse.csr_array(dense)
        divisors = [(2.0, 'scalar'), (cupy.full(a.shape, 2, dtype=dtype),
                                      'dense')]
        for divisor, label in divisors:
            got = a / divisor
            exp = dense / divisor
            assert got.dtype == exp.dtype, f'{dtype}/{label}'
            testing.assert_allclose(got.toarray().astype(cupy.float32),
                                    exp.astype(cupy.float32), atol=1e-2)

    @pytest.mark.parametrize('fmt', COMPRESSED_FORMATS)
    def test_bool_sum_is_exact_count(self, fmt):
        # ``bool.sum`` counts the True entries as int64 (scipy/numpy), not the
        # any-nonzero collapse a bool-matmul would give.  bool reduces through
        # the deduplicated float64 carrier, so pin the count and dtype on a
        # canonical array across every axis and format.
        dense = numpy.array([[True, True, False], [False, True, True]])
        a = getattr(sparse, f'{fmt}_array')(cupy.asarray(dense))
        s = getattr(scipy.sparse, f'{fmt}_array')(dense)
        assert int(a.sum()) == int(s.sum()) == 4
        for axis in (0, 1):
            got, exp = a.sum(axis=axis), s.sum(axis=axis)
            assert got.dtype == numpy.int64
            testing.assert_array_equal(
                cupy.asnumpy(got).ravel(), numpy.asarray(exp).ravel())

    def test_bool_power_0d_cupy_exponent(self):
        # ``power`` accepts a 0-D cupy array exponent (``isscalarlike``); the
        # bool branch must read it to host for numpy's per-n carrier rather
        # than compute ``numpy ** device-array`` (which raises TypeError).
        a = sparse.csr_array(
            cupy.asarray(numpy.array([[True, False], [True, True]])))
        assert a.power(cupy.asarray(2)).dtype == numpy.int64
        assert a.power(2).dtype == numpy.int8   # python int -> scipy's int8

    def test_truediv_by_longdouble_scalar(self):
        # ``matrix / longdouble(2)`` promotes to float128, which the GPU
        # cannot store; fall back to float64 instead of crashing in
        # ``reciprocal`` with "Wrong type (float128)".
        dense = numpy.array([[1., 2.], [0., 4.]])
        a = sparse.csr_array(cupy.asarray(dense))
        r = a / numpy.longdouble(2)
        assert r.dtype == numpy.float64
        testing.assert_allclose(r.toarray().get(), dense / 2.0)

    def test_truediv_by_clongdouble_keeps_imag(self):
        # A clongdouble divisor promotes to complex256 (unstorable); the
        # fallback must keep the *complex* kind (-> complex128) rather than
        # collapse to float64 and silently drop the imaginary part.
        dense = numpy.array([[1., 2.], [0., 4.]], numpy.complex128)
        a = sparse.csr_array(cupy.asarray(dense))
        r = a / numpy.clongdouble(2 + 1j)
        assert r.dtype == numpy.complex128
        testing.assert_allclose(
            r.toarray().get(), dense / numpy.complex128(2 + 1j))

    @pytest.mark.parametrize('op', ['min', 'max', 'argmin', 'argmax'])
    @pytest.mark.parametrize('axis', [(0, 1), (0,), (1,)])
    def test_tuple_axis_reduction(self, op, axis):
        # scipy accepts a tuple ``axis`` for 2-D reductions (a length-2 tuple
        # spanning both axes is a full reduction).  min/max/argmin/argmax
        # collapse it the same way sum/mean do.
        dense = _dense_a(numpy, numpy.float64)
        g = getattr(sparse.csr_array(cupy.asarray(dense)), op)(axis=axis)
        s = getattr(scipy.sparse.csr_array(dense), op)(axis=axis)
        gv = g.toarray().get() if sparse.issparse(g) else cupy.asnumpy(g)
        sv = s.toarray() if scipy.sparse.issparse(s) else numpy.asarray(s)
        testing.assert_array_equal(
            numpy.ravel(gv), numpy.ravel(sv))

    def test_bfloat16_mixed_int_elementwise(self):
        # bfloat16 mixed with a >= 16-bit int has no numpy promotion, but
        # dense cupy resolves it via ufunc loops (-> float64 here); the
        # elementwise sparse ops must match dense, while matmul rejects the
        # mix exactly like dense.
        ml = pytest.importorskip('ml_dtypes')
        bf16 = ml.bfloat16
        da = numpy.array([[1, 2], [0, 4]]).astype(bf16)
        db = numpy.array([[1, 0], [3, 4]]).astype(numpy.int32)
        A, B = sparse.csr_array(cupy.asarray(da)), sparse.csr_array(
            cupy.asarray(db))
        Ad, Bd = cupy.asarray(da), cupy.asarray(db)
        for sfn, dfn in [(lambda: A + B, lambda: Ad + Bd),
                         (lambda: A.multiply(B), lambda: Ad * Bd),
                         (lambda: A.maximum(B), lambda: cupy.maximum(Ad, Bd))]:
            s, d = sfn(), dfn()
            assert s.dtype == d.dtype
            testing.assert_array_equal(
                s.toarray().get().astype(numpy.float64),
                cupy.asnumpy(d).astype(numpy.float64))
        with pytest.raises(numpy.exceptions.DTypePromotionError):
            A @ B

    def test_bfloat16_scalar_maximum_minimum_and_imul(self):
        # ``maximum``/``minimum`` and in-place ``*=`` with a scalar take the
        # ``isscalarlike`` promotion branches (``_csr._maximum_minimum`` and
        # ``_data._scalar_op_dtype``).  A bare ``numpy.result_type`` mishandles
        # bfloat16 two ways: it RAISES for a typed numpy int (``bf16 + int16``
        # has no abstract promotion) and OVER-PROMOTES a Python float
        # (``result_type(bf16, 2.0)`` -> float64, where the ufunc gives
        # float32).  Dense cupy resolves both through its ufunc loops, so the
        # sparse ops must match dense -- neither raise nor widen;
        # ``_sputils.promote_scalar_data_type`` probes the actual ufunc for the
        # bfloat16 case.  Array operands and Python-int scalars dodge both
        # quirks, so only a scalar operand exercises this.
        ml = pytest.importorskip('ml_dtypes')
        bf16 = ml.bfloat16
        bf = numpy.dtype(bf16)
        # Both failure modes of the bare promotion:
        with pytest.raises(numpy.exceptions.DTypePromotionError):
            cupy.result_type(bf, numpy.int16(1))       # typed int -> raises
        assert cupy.result_type(bf, 2.0) == numpy.float64  # py-float -> wider

        Ad = cupy.asarray(
            numpy.array([[3, 0, 1], [0, 5, 0], [7, 0, 6]]).astype(bf16))
        # Both a typed numpy int and a Python float must match dense cupy in
        # dtype and value.
        for scalar in (numpy.int16(2), 2.0):
            # maximum / minimum -- via ``_csr._maximum_minimum``.
            for name, op in [('maximum', cupy.maximum),
                             ('minimum', cupy.minimum)]:
                got = getattr(sparse.csr_array(Ad), name)(scalar)
                exp = op(Ad, scalar)           # float32 (dense parity)
                assert got.dtype == exp.dtype, (name, scalar)
                testing.assert_array_equal(
                    got.toarray().get().astype(numpy.float64),
                    cupy.asnumpy(exp).astype(numpy.float64))
            # in-place ``*=`` -- via ``_data._scalar_op_dtype``.
            A = sparse.csr_array(Ad)
            A *= scalar
            exp = Ad * scalar                  # float32 (dense parity)
            assert A.dtype == exp.dtype, scalar
            testing.assert_array_equal(
                A.toarray().get().astype(numpy.float64),
                cupy.asnumpy(exp).astype(numpy.float64))

    def test_float16_matmul_int_dense(self):
        # float16 sparse @ integer dense promotes to float16 (dense parity)
        # and runs through the native cuSPARSE f16 (data) / f32 (compute)
        # mixed path rather than upcasting the whole operand.
        da = (numpy.arange(6).reshape(2, 3) * 0.5).astype(numpy.float16)
        A = sparse.csr_array(cupy.asarray(da))
        x = cupy.array([1, 2, 3], numpy.int8)
        r = A @ x
        ref = cupy.asarray(da) @ x
        assert r.dtype == ref.dtype == numpy.float16
        testing.assert_allclose(r.get(), ref.get(), atol=1e-2)


@testing.parameterize(*testing.product({
    # Integer ``power`` keeps its width and wraps, exactly like scipy.
    # (Float/complex power is unchanged by this work; bool is covered below,
    # where cupy intentionally follows its own dense-array dtype idiom.)
    'dtype': INT_DTYPES,
    'fmt': COMPRESSED_FORMATS,
}))
class TestPower:

    @_array_equal()
    def test_power_2(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).power(2)\
            .toarray()

    @_array_equal()
    def test_power_3(self, xp, sp):
        return _make(sp, xp, _dense_a(xp, self.dtype), self.fmt).power(3)\
            .toarray()


@testing.parameterize(*testing.product({
    'fmt': COMPRESSED_FORMATS,
    'n': [2, 3, 4],
}))
class TestBoolPower:
    """Bool ``power`` matches scipy/numpy's exponent-dependent result dtype
    (int8 at n == 2, int64 for n >= 3), not cupy dense's uniform int64 -- as a
    scipy-compatible sparse type it follows the scipy oracle for this
    scipy-supported dtype."""

    def test_matches_scipy(self):
        dense = _dense_a(numpy, numpy.bool_)
        a = getattr(sparse, f'{self.fmt}_array')(cupy.asarray(dense))
        result = a.power(self.n)
        # numpy's own bool**n dtype is the scipy oracle (int8 @ 2, int64 @ >=3)
        assert result.dtype == (numpy.ones(1, numpy.bool_) ** self.n).dtype
        testing.assert_array_equal(result.toarray().get(), dense ** self.n)


def _dia(sp, xp, dtype):
    """A 4x4 DIA built from (data, offsets) (cupy has no dense->DIA)."""
    data = xp.array([[1, 2, 3, 4],
                     [5, 6, 7, 8]]).astype(dtype)
    offsets = xp.array([0, -1])
    return sp.dia_array((data, offsets), shape=(4, 4))


@testing.parameterize(*testing.product({
    'dtype': ALL_VALUE_DTYPES,
}))
class TestDia:

    @_allclose()
    def test_toarray(self, xp, sp):
        return _dia(sp, xp, self.dtype).toarray()

    @_allclose()
    def test_tocsr(self, xp, sp):
        return _dia(sp, xp, self.dtype).tocsr().toarray()

    @_allclose()
    def test_tocoo(self, xp, sp):
        return _dia(sp, xp, self.dtype).tocoo().toarray()

    def test_dtype_preserved(self):
        assert _dia(sparse, cupy, self.dtype).dtype == self.dtype


@testing.parameterize(*testing.product({
    'dtype': [numpy.bool_] + INT_DTYPES,
    'fmt': ('csr', 'coo'),
}))
class TestOneDimensional:

    def _vec(self, sp, xp):
        return getattr(sp, f'{self.fmt}_array')(
            xp.array([0, 3, 0, 5, 0, 7]).astype(self.dtype))

    @_allclose()
    def test_toarray(self, xp, sp):
        return self._vec(sp, xp).toarray()

    @_allclose()
    def test_sum(self, xp, sp):
        return self._vec(sp, xp).sum()

    @_allclose()
    def test_add(self, xp, sp):
        v = self._vec(sp, xp)
        return (v + v).toarray()

    def test_ndim_and_shape(self):
        v = self._vec(sparse, cupy)
        assert v.ndim == 1
        assert v.shape == (6,)
        assert v.dtype == self.dtype


@testing.parameterize(*testing.product({
    'dtype': [numpy.int64, numpy.uint64],
}))
class TestExactInt64PastThreshold:
    """64-bit integer ops must stay bit-exact past 2**53 (float64's exact
    range) -- a scatter/carrier, never a float compute type.  Pins the
    behaviour against scipy across sum, sparse@dense, and sparse@sparse."""

    def _mat(self):
        base = 2 ** 53
        m = numpy.array([[base + 1, 0, base + 3],
                         [0, base + 5, 0],
                         [base + 7, 0, base + 2]], dtype=self.dtype)
        return m

    def test_sum_none(self):
        m = self._mat()
        g = int(sparse.csr_array(cupy.asarray(m)).sum())
        s = int(scipy.sparse.csr_array(m).sum())
        assert g == s == int(m.sum(dtype=self.dtype))

    def test_matmul_dense(self):
        m = self._mat()
        v = numpy.array([1, 1, 1], dtype=self.dtype)
        g = (sparse.csr_array(cupy.asarray(m)) @ cupy.asarray(v)).get()
        s = numpy.asarray(scipy.sparse.csr_array(m) @ v)
        testing.assert_array_equal(g, s)
        testing.assert_array_equal(g, m @ v)   # exact, not float64-rounded

    def test_matmul_sparse(self):
        # diagonal so the product entries are single (2**53+k)**2-ish terms
        d = numpy.diag(numpy.array(
            [2 ** 53 + 1, 2 ** 53 + 3], dtype=self.dtype))
        g = (sparse.csr_array(cupy.asarray(d))
             @ sparse.csr_array(cupy.asarray(d))).toarray().get()
        s = (scipy.sparse.csr_array(d) @ scipy.sparse.csr_array(d)).toarray()
        testing.assert_array_equal(g, s)


@testing.with_requires('scipy')
class TestConstructionHelpersScipyParity:

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_hstack_vstack(self, xp, sp, dtype):
        a = sp.csr_matrix(xp.array([[1, 0], [0, 2]], dtype=dtype))
        b = sp.csr_matrix(xp.array([[3, 0], [0, 4]], dtype=dtype))
        h = sp.hstack([a, b])
        v = sp.vstack([a, b])
        return h.toarray(), v.toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_kron(self, xp, sp, dtype):
        a = sp.csr_matrix(xp.array([[1, 0], [0, 2]], dtype=dtype))
        return sp.kron(a, a).toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_eye_identity_tril(self, xp, sp, dtype):
        e = sp.eye(3, dtype=dtype)
        i = sp.identity(3, dtype=dtype)
        t = sp.tril(sp.csr_matrix(
            xp.array([[1, 2], [3, 4]], dtype=dtype)))
        return e.toarray(), i.toarray(), t.toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_block_diag(self, xp, sp, dtype):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype)
        b = xp.array([[5]], dtype=dtype)
        return sp.block_diag([a, b]).toarray()

    @pytest.mark.parametrize('dtype',
                             [numpy.int8, numpy.uint64, numpy.float16,
                              numpy.bool_])
    def test_random(self, dtype):
        m = sparse.random(30, 40, density=0.1, dtype=dtype)
        assert m.dtype == dtype
        assert m.nnz == 120


class TestValueDtypeRegressions:

    @pytest.mark.parametrize('fmt', ['csr', 'csc'])
    @pytest.mark.parametrize(
        'dtype', [numpy.float64, numpy.float32, numpy.complex128,
                  numpy.int32, numpy.int64])
    def test_eliminate_zeros_keeps_negative_values(self, dtype, fmt):
        # cuSPARSE's csr2csr_compress prunes by a signed ``value > tol``, so
        # the old float/complex path silently dropped *negative* stored values
        # along with the zeros.  The pure-CuPy ``data != 0`` mask is exact.
        # Build with explicit stored zeros so the mask-and-rebuild runs, then
        # match scipy across csr and csc (csc delegates to the csr path).
        import scipy.sparse
        data = numpy.array([3, -1, 0, -2, 0, 5], dtype)
        indices = numpy.array([0, 1, 2, 0, 1, 2], numpy.int32)
        indptr = numpy.array([0, 3, 6], numpy.int32)
        g = getattr(sparse.csr_matrix(
            (cupy.asarray(data), cupy.asarray(indices),
             cupy.asarray(indptr)), shape=(2, 3)), f'to{fmt}')()
        s = getattr(scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(2, 3)), f'to{fmt}')()
        g.eliminate_zeros()
        s.eliminate_zeros()
        testing.assert_array_equal(cupy.asnumpy(g.toarray()), s.toarray())
        assert int(g.nnz) == int(s.nnz) == 4            # the two zeros dropped
        assert (cupy.asnumpy(g.toarray()).real < 0).any()   # negatives survive

    @requires_bfloat16
    @pytest.mark.parametrize('container', ['csr_matrix', 'csc_matrix'])
    @pytest.mark.parametrize('dense_dtype', [numpy.int32, numpy.int64])
    def test_multiply_by_dense_bfloat16_int(self, container, dense_dtype):
        # ``multiply`` of a bfloat16 matrix by an integer dense array resolves
        # the mix through the dense CuPy ufunc (float64 here) and matches both
        # dtype and values -- ``numpy.promote_types`` cannot resolve
        # bfloat16 vs a >= 16-bit int.
        d = cupy.array([[1, 0, 2], [0, 3, 0]], dtype=_bfloat16)
        m = getattr(sparse, container)(d)
        dense = cupy.array([[2, 2, 2], [3, 3, 3]], dtype=dense_dtype)
        # numpy.promote_types(bfloat16, int32/int64) raises
        # DTypePromotionError (a TypeError subclass); dense CuPy is the oracle.
        with pytest.raises(TypeError):
            numpy.promote_types(_bfloat16, numpy.dtype(dense_dtype))
        oracle = d * dense
        r = m.multiply(dense)
        assert numpy.dtype(r.dtype) == numpy.dtype(oracle.dtype)
        got = r.toarray() if sparse.issparse(r) else r
        cupy.testing.assert_array_equal(cupy.asarray(got), oracle)

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_densetosparse_integer_direct(self, dtype, fmt):
        # ``cusparse.denseToSparse`` converts a non-float dense array via a
        # pure-CuPy fallback (cuSPARSE denseToSparse is float/complex only).
        if not cusparse.check_availability('denseToSparse'):
            pytest.skip('denseToSparse unavailable on this platform')
        d = cupy.array([[1, 0, 2, 0], [0, 3, 0, 0], [4, 0, 0, 5]],
                       dtype=dtype)
        y = cusparse.denseToSparse(d, format=fmt)
        assert y.format == fmt
        assert numpy.dtype(y.dtype) == numpy.dtype(dtype)
        cupy.testing.assert_array_equal(y.toarray(), d)

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    def test_csr_matrix_from_dense_integer_roundtrip(self, dtype):
        # An integer dense array constructs a sparse matrix and round-trips
        # back to the same dense array.
        d = cupy.array([[1, 0, 2, 0], [0, 3, 0, 0], [4, 0, 0, 5]],
                       dtype=dtype)
        m = sparse.csr_matrix(d)
        assert numpy.dtype(m.dtype) == numpy.dtype(dtype)
        cupy.testing.assert_array_equal(m.toarray(), d)

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @pytest.mark.parametrize('fa', ['csr_matrix', 'csc_matrix'])
    @pytest.mark.parametrize('fb', ['csr_matrix', 'csc_matrix'])
    def test_integer_spgemm_exact(self, dtype, fa, fb):
        # An integer ``sparse @ sparse`` returns the exact integer product
        # (via the pure-CuPy SpGEMM fallback, so it also works where the
        # cuSPARSE gemm is float-only).
        a_dense = numpy.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=dtype)
        b_dense = numpy.array([[0, 1, 0], [2, 0, 0], [0, 0, 3]], dtype=dtype)
        # No entry overflows the narrowest dtype here, so the numpy matmul is
        # an exact oracle (and matches scipy's wraparound semantics).
        oracle = (a_dense @ b_dense).astype(dtype)
        a = getattr(sparse, fa)(cupy.asarray(a_dense))
        b = getattr(sparse, fb)(cupy.asarray(b_dense))
        c = a @ b
        assert numpy.dtype(c.dtype) == numpy.dtype(dtype)
        cupy.testing.assert_array_equal(cupy.asnumpy(c.toarray()), oracle)

    @requires_bfloat16
    def test_asfptype_preserves_bfloat16(self):
        # asfptype() keeps a bfloat16 matrix as bfloat16 (a float), rather
        # than upcasting to float32: bfloat16's numpy kind is 'V', so the base
        # ``kind == 'f'`` test misses it; ``_sputils.is_float_dtype`` treats
        # it as the float it is (matching dense CuPy, whose bf16 is a float).
        m = sparse.csr_matrix(
            cupy.array([[1, 0, 2], [0, 3, 0]], dtype=_bfloat16))
        assert numpy.dtype(m.asfptype().dtype) == numpy.dtype(_bfloat16)
        # An integer matrix still promotes to a floating dtype.
        mi = sparse.csr_matrix(
            cupy.array([[1, 0], [0, 2]], dtype=numpy.int32))
        assert numpy.dtype(mi.asfptype().dtype).kind == 'f'

    @pytest.mark.parametrize('op', ['maximum', 'minimum'])
    @pytest.mark.parametrize('dtype', INT_DTYPES)
    def test_maximum_minimum_out_of_range_scalar_raises(self, op, dtype):
        # A Python int that does not fit the matrix dtype must raise, not
        # wrap: value-based promotion keeps ``result_type(int8, 300)`` at
        # int8, so a plain cast would silently turn 300 into 44.  numpy,
        # dense CuPy and scipy all raise OverflowError here.
        info = numpy.iinfo(dtype)
        dense = numpy.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
        m = sparse.csr_matrix(cupy.asarray(dense))
        for scalar in (int(info.max) + 1, int(info.min) - 1):
            with pytest.raises(OverflowError):
                getattr(m, op)(scalar)

    @pytest.mark.parametrize('other_dtype', [numpy.complex64,
                                             numpy.complex128])
    @pytest.mark.parametrize('dtype', [numpy.bool_, numpy.int8, numpy.int64,
                                       numpy.complex64])
    @pytest.mark.parametrize('dense_operand', [False, True])
    def test_multiply_mixed_complex(self, dtype, other_dtype, dense_operand):
        # ``multiply`` shares one promoted dtype between the sparse and the
        # other operand: the kernel has no ``integer * thrust::complex`` (nor
        # ``complex<float> * complex<double>``) operator, so both must be cast
        # before launch.
        a = numpy.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
        b = numpy.array([[2, 1, 0], [0, 4, 5]], dtype=other_dtype)
        oracle = a * b
        m = sparse.csr_matrix(cupy.asarray(a))
        other = cupy.asarray(b) if dense_operand else sparse.csr_matrix(
            cupy.asarray(b))
        r = m.multiply(other)
        got = r.toarray() if sparse.issparse(r) else r
        assert numpy.dtype(got.dtype) == numpy.dtype(oracle.dtype)
        cupy.testing.assert_array_equal(cupy.asnumpy(got), oracle)

    @pytest.mark.parametrize('dtype', [numpy.bool_, numpy.int8, numpy.float32])
    @pytest.mark.parametrize('target', [numpy.bool_, numpy.int8,
                                        numpy.float32])
    def test_power_honours_requested_dtype(self, dtype, target):
        # ``power(n, dtype=...)`` must return the requested dtype.  ``bool``
        # is the one target whose ``x ** n`` promotes (numpy gives int8 at
        # n == 2), so it needs the same carrier the ``dtype is None`` path
        # derives.
        dense = numpy.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
        m = sparse.csr_matrix(cupy.asarray(dense))
        expected = (dense.astype(target) ** 2).dtype
        assert numpy.dtype(m.power(2, dtype=target).dtype) == expected

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @pytest.mark.parametrize('fmt', COMPRESSED_FORMATS)
    def test_frobenius_norm_does_not_wrap(self, dtype, fmt):
        # The Frobenius norm squares the stored values; squaring in the
        # matrix's own narrow integer dtype would wrap (int8 100**2 -> 16).
        import cupyx.scipy.sparse.linalg as cupy_spl
        import scipy.sparse.linalg as scipy_spl
        dense = (numpy.array([[3, 0, -4], [0, 1, 0]]) * 10).astype(dtype)
        g = cupy_spl.norm(_make(sparse, cupy, cupy.asarray(dense), fmt))
        s = scipy_spl.norm(_make(scipy.sparse, numpy, dense, fmt))
        assert numpy.dtype(g.dtype) == numpy.asarray(s).dtype
        testing.assert_allclose(cupy.asnumpy(g), numpy.asarray(s))

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    def test_coo_scalar_getitem_keeps_dtype(self, dtype):
        # ``coo[i, j]`` sums the stored values at the coordinate; it must
        # accumulate in -- and return -- the matrix dtype, so duplicates wrap
        # exactly as ``toarray()`` and the CSR path do.
        vals = numpy.array([100, 100], dtype=dtype)
        row = numpy.array([0, 0])
        col = numpy.array([1, 1])
        g = sparse.coo_array(
            (cupy.asarray(vals), (cupy.asarray(row), cupy.asarray(col))),
            shape=(2, 2))[0, 1]
        # scipy's ``coo_array`` is not subscriptable (element access is a CSR
        # operation), so take the oracle through ``tocsr`` -- it coalesces the
        # duplicate at the storage width just like cupy's COO getitem.
        s = numpy.asarray(scipy.sparse.coo_array(
            (vals, (row, col)), shape=(2, 2)).tocsr()[0, 1])
        assert numpy.dtype(g.dtype) == s.dtype
        testing.assert_array_equal(cupy.asnumpy(g), s)


# ---------------------------------------------------------------
# Coverage-gap tests: mutating/utility methods, matrix_power, 64-bit
# axis-wise exact reductions, the chunked scatter matmul, all-width
# integer overflow, 1-D ops, mixed int+float native routing, dtype-gate
# edges, kronsum/triu/find, unsorted-COO alignment, arg-reduce edges,
# the complex axis min/max known bug (xfail), narrow-float float32
# staging, and scipy interop.
# ---------------------------------------------------------------


def _big_int(xp, dtype):
    """3x3 with magnitudes past float64's exact range (2**53 / 2**63)."""
    base = 2 ** 63 if numpy.dtype(dtype).kind == 'u' else 2 ** 60
    return xp.array([[base + 1, 0, base + 5],
                     [0, base + 3, 0],
                     [base + 7, 0, base + 9]], dtype=dtype)


@testing.with_requires('scipy')
class TestMutatingUtilityMethods:
    """``sum_duplicates``/``sort_indices``/``setdiag``/``count_nonzero`` --
    mutating and utility methods the happy-path suite never exercises."""

    @pytest.mark.parametrize('dtype', INT_DTYPES)
    @_array_equal()
    def test_coo_sum_duplicates(self, xp, sp, dtype):
        # Duplicate (0, 0) coords accumulate in the storage width: 100 + 100
        # wraps to -56 for int8 (numpy modular overflow), matching scipy.
        data = xp.array([100, 100, 5], dtype=dtype)
        row = xp.array([0, 0, 1], 'i')
        col = xp.array([0, 0, 1], 'i')
        m = sp.coo_array((data, (row, col)), shape=(2, 2))
        m.sum_duplicates()
        assert m.dtype == dtype
        return m.toarray()

    def test_coo_sum_duplicates_int8_wraps_explicit(self):
        data = cupy.array([100, 100], dtype=numpy.int8)
        row = cupy.array([0, 0], 'i')
        col = cupy.array([0, 0], 'i')
        m = sparse.coo_array((data, (row, col)), shape=(1, 1))
        m.sum_duplicates()
        assert m.dtype == numpy.int8
        assert int(m.toarray()[0, 0]) == -56  # 200 - 256

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @_array_equal()
    def test_sort_indices(self, xp, sp, dtype):
        # sort_indices must gather ``data`` along with ``indices`` (not just
        # reorder the indices), so the stored values stay put.
        data = xp.array([2, 1], dtype=dtype)
        indices = xp.array([2, 0], 'i')
        indptr = xp.array([0, 2], 'i')
        m = sp.csr_array((data, indices, indptr), shape=(1, 3))
        m.sort_indices()
        assert m.has_sorted_indices
        return m.toarray()

    @pytest.mark.parametrize('dtype', [numpy.int16, numpy.int64, numpy.uint8])
    @pytest.mark.parametrize('fmt', ['csr', 'csc'])
    @_array_equal()
    def test_setdiag_preserves_dtype_values(self, xp, sp, dtype, fmt):
        # setdiag on a NON-bool dtype must keep the dtype and write the given
        # diagonal values (bool CSR setdiag is a known scipy TypeError).
        m = _make(sp, xp, _dense_a(xp, dtype), fmt)
        m.setdiag(xp.array([11, 12, 13, 14], dtype=dtype))
        assert m.dtype == dtype
        return m.diagonal(), m.toarray()

    @pytest.mark.parametrize('dtype', [numpy.int16, numpy.int64, numpy.uint8])
    @pytest.mark.parametrize('axis', [0, 1])
    @_array_equal()
    def test_count_nonzero_axis(self, xp, sp, dtype, axis):
        # The counts are what matter for value-dtype support; scipy's own
        # count array width varies by axis, so normalize to int64.
        d = xp.array([[1, 0, 2], [0, 0, 0], [3, 0, 4]], dtype=dtype)
        return xp.asarray(
            sp.csr_array(d).count_nonzero(axis=axis)).astype(numpy.int64)


@testing.with_requires('scipy')
class TestMatrixPower:
    """``matrix_power`` / ``A ** k`` across ints and bool, vs scipy; plus a
    64-bit case that stays exact past 2**53."""

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int64, numpy.uint8,
                                       numpy.bool_])
    @_array_equal()
    def test_matrix_power(self, xp, sp, dtype):
        from cupyx.scipy.sparse.linalg import matrix_power as cp_mp
        if xp is numpy:
            from scipy.sparse.linalg import matrix_power as mp
        else:
            mp = cp_mp
        d = xp.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=dtype)
        r = mp(sp.csr_matrix(d), 3)
        assert r.dtype == dtype
        return r.toarray()

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int64, numpy.uint8,
                                       numpy.bool_])
    @_array_equal()
    def test_pow_operator(self, xp, sp, dtype):
        d = xp.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=dtype)
        return (sp.csr_matrix(d) ** 3).toarray()

    def test_matrix_power_int64_exact_past_2_53(self):
        from cupyx.scipy.sparse.linalg import matrix_power
        k = 4 * 10 ** 15 + 1        # 3*k = 1.2e16 + 3, odd and > 2**53
        d = numpy.array([[1, k], [0, 1]], dtype=numpy.int64)
        m = sparse.csr_matrix(cupy.asarray(d))
        r = matrix_power(m, 3)      # [[1, 3k], [0, 1]]
        assert int(r.toarray()[0, 1]) == 3 * k
        numpy.testing.assert_array_equal(
            r.toarray().get(), numpy.linalg.matrix_power(d, 3))


@testing.with_requires('scipy')
class TestExactInt64AxisReductions:
    """64-bit axis-wise reductions must reduce in an integer accumulator, not
    a float64 carrier, so they stay bit-exact past 2**53 / 2**63 (scipy)."""

    @pytest.mark.parametrize('dtype', [numpy.int64, numpy.uint64])
    @pytest.mark.parametrize('fmt', COMPRESSED_FORMATS)
    @pytest.mark.parametrize('axis', [0, 1])
    @_array_equal()
    def test_sum_axis(self, xp, sp, dtype, fmt, axis):
        m = _make(sp, xp, _big_int(xp, dtype), fmt)
        return xp.asarray(m.sum(axis=axis))

    @pytest.mark.parametrize('dtype', [numpy.int64, numpy.uint64])
    @pytest.mark.parametrize('fmt', ['csr', 'csc'])
    @pytest.mark.parametrize('axis', [0, 1])
    @_array_equal()
    def test_min_axis(self, xp, sp, dtype, fmt, axis):
        m = _make(sp, xp, _big_int(xp, dtype), fmt)
        return m.min(axis=axis).toarray()

    @pytest.mark.parametrize('dtype', [numpy.int64, numpy.uint64])
    @pytest.mark.parametrize('op', ['argmax', 'argmin'])
    @pytest.mark.parametrize('axis', [0, 1])
    @_array_equal()
    def test_arg_axis(self, xp, sp, dtype, op, axis):
        m = _make(sp, xp, _big_int(xp, dtype), 'csr')
        return xp.asarray(getattr(m, op)(axis=axis))


@testing.with_requires('scipy')
class TestChunkedScatterMatmul:
    """The integer scatter matmul materialises an (nnz, N) product, so a wide
    dense operand is processed in column blocks bounded by free memory.  Force
    the chunk loop (report ~no free memory) and confirm the blocked
    accumulation is still exact past 2**53 -- a float64 fallback would
    round."""

    def test_wide_operand_chunks_and_stays_exact(self, monkeypatch):
        import scipy.sparse
        d = numpy.array([[2 ** 60 + 1, 0, 3],
                         [0, 5, 0],
                         [7, 0, 2 ** 60]], dtype=numpy.int64)
        x = numpy.arange(1, 13, dtype=numpy.int64).reshape(3, 4)
        ref = numpy.asarray(scipy.sparse.csr_matrix(d) @ x)
        ag = sparse.csr_matrix(cupy.asarray(d))
        xg = cupy.asarray(x)

        cupy.get_default_memory_pool().free_all_blocks()

        class _FakePool:
            def free_bytes(self):
                return 0
        # Report a tiny free amount from BOTH the driver probe and the pool's
        # free blocks so ``block`` collapses to a single dense column.
        monkeypatch.setattr(cupy.cuda.runtime, 'memGetInfo', lambda: (1, 1))
        monkeypatch.setattr(cusparse._cupy, 'get_default_memory_pool',
                            lambda: _FakePool())
        got = cupy.asnumpy(cusparse._cupy_csr_dense_matmul(ag, xg))
        numpy.testing.assert_array_equal(got, ref)
        # A float64 compute would round 2**60 + 1 down to 2**60.
        assert got[0, 0] == 2 ** 60 + 28 > 2 ** 53


@testing.with_requires('scipy')
class TestIntegerOverflowAllWidths:
    """Integer results wrap modularly on overflow (numpy/scipy) at every
    width -- not just the 8/16-bit cases the happy-path suite pins."""

    @pytest.mark.parametrize('dtype', [numpy.int32, numpy.int64,
                                       numpy.uint32, numpy.uint64])
    @_array_equal()
    def test_add_overflow(self, xp, sp, dtype):
        hi = int(numpy.iinfo(dtype).max)
        dense = xp.array([[hi, 0, hi // 2], [0, hi - 3, 0]], dtype=dtype)
        a = _make(sp, xp, dense, 'csr')
        return (a + a).toarray()

    @pytest.mark.parametrize('dtype', [numpy.int32, numpy.uint32])
    @_array_equal()
    def test_matmul_dense_overflow(self, xp, sp, dtype):
        # sparse @ dense whose true product overflows must wrap, not saturate
        # (the exact 64-bit carrier wraps on the down-cast).
        k = 40
        val = int(numpy.sqrt(int(numpy.iinfo(dtype).max)))  # k*val**2 >> max
        a = sp.csr_array(xp.full((1, k), val, dtype=dtype))
        b = xp.full((k, 1), val, dtype=dtype)
        return a @ b


@testing.with_requires('scipy')
class TestOneDimensionalOps:
    """1-D sparse arrays: reductions, indexing, and 1-D @ 2-D matmul."""

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @pytest.mark.parametrize(
        'reduction', ['sum', 'mean', 'max', 'min', 'argmax'])
    @_allclose()
    def test_reductions(self, xp, sp, dtype, reduction):
        v = sp.csr_array(xp.array([100, 0, 100, 50, 0, 60], dtype=dtype))
        return xp.asarray(getattr(v, reduction)())

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @_array_equal()
    def test_indexing(self, xp, sp, dtype):
        v = sp.csr_array(xp.array([1, 0, 2, 3, 0, 4], dtype=dtype))
        cols = xp.array([0, 2, 5])
        return (xp.asarray(v[2]),
                v[1:4].toarray(),
                v[cols].toarray())

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @_array_equal()
    def test_matmul_2d_dense(self, xp, sp, dtype):
        # 1-D sparse @ 2-D dense -> 1-D dense.
        v = sp.csr_array(xp.array([1, 0, 2, 3], dtype=dtype))
        m = xp.array([[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 0, 6]],
                     dtype=dtype)
        return xp.asarray(v @ m)


@testing.with_requires('scipy')
class TestMixedIntFloatNativePath:
    """int + float32 promotes and routes through the native cuSPARSE path;
    the promoted result must match scipy in dtype and value."""

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @_allclose()
    def test_add_int_float32_sparse(self, xp, sp, dtype):
        a = _make(sp, xp, _dense_a(xp, dtype), 'csr')
        b = sp.csr_array(xp.eye(4, 5, dtype=numpy.float32))
        return (a + b).toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @_allclose()
    def test_spmv_int_float32_dense(self, xp, sp, dtype):
        a = _make(sp, xp, _dense_a(xp, dtype), 'csr')
        v = xp.arange(5, dtype=numpy.float32)
        return a @ v


class TestDtypeGatingEdges:
    """The dtype gate rejects storable-looking-but-unsupported dtypes by
    identity (not by single-char code, which aliases)."""

    def test_reject_ml_dtypes_float8_int4_uint4(self):
        # numpy.dtype(ml_dtypes.float8_e4m3b11fnuz).char == 'L' collides with
        # uint64, so a char-only gate would wrongly accept it; identity is the
        # correct discriminator.
        ml_dtypes = pytest.importorskip('ml_dtypes')
        from cupyx.scipy.sparse import _sputils
        assert numpy.dtype(ml_dtypes.float8_e4m3b11fnuz).char == 'L'
        names = ['float8_e4m3b11fnuz', 'float8_e5m2']
        for extra in ('int4', 'uint4'):
            if hasattr(ml_dtypes, extra):
                names.append(extra)
        for name in names:
            dt = numpy.dtype(getattr(ml_dtypes, name))
            assert not _sputils.is_sparse_data_dtype(dt), name
            with pytest.raises(ValueError):
                _sputils.check_data_dtype(dt)

    def test_reject_clongdouble(self):
        from cupyx.scipy.sparse import _sputils
        if numpy.dtype(numpy.clongdouble).itemsize <= 16:
            pytest.skip('clongdouble not extended on this platform')
        assert not _sputils.is_sparse_data_dtype(numpy.clongdouble)
        with pytest.raises(ValueError):
            _sputils.check_data_dtype(numpy.clongdouble)
        a = sparse.csr_array(cupy.eye(3, dtype=numpy.complex128))
        with pytest.raises((ValueError, TypeError)):
            a.astype(numpy.clongdouble)

    def test_bool_neg_raises(self):
        m = sparse.csr_array(
            cupy.asarray(numpy.array([[True, False], [False, True]])))
        with pytest.raises((NotImplementedError, TypeError)):
            -m


@testing.with_requires('scipy')
class TestConstructionHelpers:
    """``kronsum``/``triu``/``find`` preserve the value dtype (scipy)."""

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @_array_equal()
    def test_kronsum(self, xp, sp, dtype):
        a = _make(sp, xp, _square(xp, dtype), 'csr')
        r = sp.kronsum(a, a)
        assert r.dtype == dtype
        return r.toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @_array_equal()
    def test_triu(self, xp, sp, dtype):
        m = _make(sp, xp, _dense_a(xp, dtype), 'csr')
        r = sp.triu(m, k=1)
        assert r.dtype == dtype
        return r.toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    def test_find_preserves_dtype(self, dtype):
        import scipy.sparse
        dense = _dense_a(numpy, dtype)
        i, j, v = sparse.find(sparse.csr_array(cupy.asarray(dense)))
        assert v.dtype == dtype
        # find's ordering is unspecified, so reconstruct dense (order-free).
        got = numpy.zeros_like(dense)
        got[cupy.asnumpy(i), cupy.asnumpy(j)] = cupy.asnumpy(v)
        si, sj, sv = scipy.sparse.find(scipy.sparse.csr_array(dense))
        assert sv.dtype == dtype
        exp = numpy.zeros_like(dense)
        exp[si, sj] = sv
        numpy.testing.assert_array_equal(got, exp)


@testing.with_requires('scipy')
class TestUnsortedCooRegression:
    """coosort must permute ``data`` along with the coordinates; an
    explicitly stored value at a coordinate that moves during the sort must
    stay aligned through ``tocsr``/``tocsc`` (the bool data-gather bug)."""

    @pytest.mark.parametrize('to', ['tocsr', 'tocsc'])
    @_array_equal()
    def test_bool_explicit_false_stays_aligned(self, xp, sp, to):
        # An explicit stored ``False`` at (0, 2) moves during the column sort;
        # a skipped data gather would misalign it with its coordinates.
        row = xp.array([2, 0, 1], 'i')
        col = xp.array([0, 2, 1], 'i')
        data = xp.array([True, False, True])
        m = sp.coo_array((data, (row, col)), shape=(3, 3))
        return getattr(m, to)().toarray()

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int64])
    @pytest.mark.parametrize('to', ['tocsr', 'tocsc'])
    @_array_equal()
    def test_int_unsorted_stays_aligned(self, xp, sp, dtype, to):
        row = xp.array([2, 0, 1], 'i')
        col = xp.array([0, 2, 1], 'i')
        data = xp.array([1, 2, 3], dtype=dtype)
        m = sp.coo_array((data, (row, col)), shape=(3, 3))
        return getattr(m, to)().toarray()


@testing.with_requires('scipy')
class TestArgReduceEdges:
    """Arg-reduction edges: bool with an all-false row/column, and complex
    with nonzero imaginary parts (reduces on the real part, like min/max)."""

    @pytest.mark.parametrize('op', ['argmax', 'argmin'])
    @pytest.mark.parametrize('axis', [0, 1])
    @_array_equal()
    def test_bool_all_false_line(self, xp, sp, op, axis):
        # Row 2 and (for axis) columns are all-False; must match scipy.
        d = xp.array([[False, True, False],
                      [True, False, True],
                      [False, False, False]], dtype=xp.bool_)
        return xp.asarray(getattr(sp.csr_array(d), op)(axis=axis))

    @pytest.mark.parametrize('dtype', [numpy.complex64, numpy.complex128])
    @pytest.mark.parametrize('op', ['argmax', 'argmin'])
    @pytest.mark.parametrize('axis', [0, 1])
    @_array_equal()
    def test_complex_nonzero_imag(self, xp, sp, dtype, op, axis):
        d = xp.array([[1 + 9j, 0, 3 + 1j],
                      [0, 2 + 0j, 0],
                      [4 + 2j, 0, 5 - 7j]], dtype=dtype)
        return xp.asarray(getattr(sp.csr_array(d), op)(axis=axis))


@testing.with_requires('scipy')
class TestComplexAxisMinMaxKnownBug:
    """KNOWN BUG: complex ``max``/``min`` along an axis drops the imaginary
    part of the winning entry (cupy returns e.g. 5+0j where scipy returns
    5-1j).  The no-axis reduction is correct, so only the axis form is
    xfailed."""

    _M = numpy.array([[5 - 1j, -2 + 3j, 0],
                      [0, 0, 4 + 2j],
                      [3 + 7j, -6 - 5j, 1 - 9j]], dtype=numpy.complex128)

    def _pair(self):
        import scipy.sparse
        return (sparse.csr_array(cupy.asarray(self._M)),
                scipy.sparse.csr_array(self._M))

    @pytest.mark.xfail(strict=True,
                       reason='complex max(axis) drops the imaginary part')
    def test_max_axis0_keeps_imag(self):
        g, s = self._pair()
        numpy.testing.assert_array_equal(
            g.max(axis=0).toarray().ravel().get(),
            numpy.asarray(s.max(axis=0).toarray()).ravel())

    @pytest.mark.xfail(strict=True,
                       reason='complex min(axis) drops the imaginary part')
    def test_min_axis0_keeps_imag(self):
        g, s = self._pair()
        numpy.testing.assert_array_equal(
            g.min(axis=0).toarray().ravel().get(),
            numpy.asarray(s.min(axis=0).toarray()).ravel())

    def test_noaxis_max_min_correct(self):
        # The full (no-axis) reduction keeps the imaginary part (not xfailed).
        g, s = self._pair()
        assert complex(g.max()) == complex(s.max())
        assert complex(g.min()) == complex(s.min())


class TestNarrowFloatAccumulation:
    """16-bit-float sparse@sparse over a few-hundred inner dimension must stay
    near 16-bit epsilon of a float64 oracle -- proving float32 staging, not a
    uniform 16-bit accumulation (which would compound to ~1e-2)."""

    def test_float16_spgemm_stages_float32(self):
        rng = numpy.random.default_rng(0)
        k = 400
        a_d = rng.random((6, k)).astype(numpy.float16)
        b_d = rng.random((k, 6)).astype(numpy.float16)
        a = sparse.csr_array(cupy.asarray(a_d))
        b = sparse.csr_array(cupy.asarray(b_d))
        got = (a @ b).toarray().get().astype(numpy.float64)
        ref = a_d.astype(numpy.float64) @ b_d.astype(numpy.float64)
        rel = numpy.max(numpy.abs(got - ref) / (numpy.abs(ref) + 1e-9))
        assert rel < 5e-3, rel

    @requires_bfloat16
    def test_bfloat16_spgemm_stages_float32(self):
        rng = numpy.random.default_rng(0)
        k = 200
        a_d = rng.random((6, k)).astype(numpy.float32).astype(_bfloat16)
        b_d = rng.random((k, 6)).astype(numpy.float32).astype(_bfloat16)
        a = sparse.csr_array(cupy.asarray(a_d))
        b = sparse.csr_array(cupy.asarray(b_d))
        got = (a @ b).toarray().astype(numpy.float64).get()
        ref = a_d.astype(numpy.float64) @ b_d.astype(numpy.float64)
        rel = numpy.max(numpy.abs(got - ref) / (numpy.abs(ref) + 1e-9))
        assert rel < 5e-2, rel


@testing.with_requires('scipy')
class TestScipyInterop:
    """Construct a cupyx matrix FROM a scipy.sparse object and ``.get()`` it
    back; integer dtype and values must survive the round trip."""

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    def test_scipy_roundtrip(self, dtype):
        import scipy.sparse
        d = numpy.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
        s = scipy.sparse.csr_matrix(d)
        m = sparse.csr_matrix(s)
        assert m.dtype == dtype
        numpy.testing.assert_array_equal(m.toarray().get(), d)
        back = m.get()
        assert back.dtype == dtype
        numpy.testing.assert_array_equal(back.toarray(), d)
