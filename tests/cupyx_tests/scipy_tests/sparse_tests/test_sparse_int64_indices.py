from __future__ import annotations

import warnings

import numpy
import pytest

import cupy
from cupy import testing
from cupyx import cusparse
from cupyx.scipy import sparse


# Index value that exceeds INT32_MAX (= 2**31 - 1), forcing int64.
_LARGE = 2**31 + 1


class TestInt64Construction:
    """Index dtype is chosen by get_index_dtype, not forced to int32.

    The key change: constructors call get_index_dtype(check_contents=True)
    instead of unconditionally casting to int32.  Small values still produce
    int32 (no regression); large values now produce int64.
    """

    def test_csr_large_col_index_uses_int64(self):
        # Column index > INT32_MAX → both indices and indptr use int64.
        # CSR invariant: indices and indptr always share the same dtype.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 1))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        assert int(m.indices[1]) == _LARGE

    def test_csc_large_row_index_uses_int64(self):
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csc_matrix((data, indices, indptr), shape=(_LARGE + 1, 2))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        assert int(m.indices[1]) == _LARGE

    def test_coo_large_col_index_uses_int64(self):
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, _LARGE], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(2, _LARGE + 1))
        assert m.row.dtype == cupy.int64
        assert m.col.dtype == cupy.int64
        assert int(m.col[1]) == _LARGE

    def test_csr_int64_with_small_values_stays_int32(self):
        # get_index_dtype(check_contents=True) downcasts int64 arrays when
        # all values fit in int32.  This is the scipy-compatible behavior.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 5], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 10))
        assert m.indices.dtype == cupy.int32
        assert m.indptr.dtype == cupy.int32

    def test_coo_int64_with_small_values_stays_int32(self):
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, 5], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)))
        assert m.row.dtype == cupy.int32
        assert m.col.dtype == cupy.int32

    def test_empty_csr_large_shape_uses_int64(self):
        # Shape-only construction: max(shape) > INT32_MAX → int64 index arrays,
        # even when there are no stored elements.
        m = sparse.csr_matrix((2, _LARGE + 1))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64

    def test_empty_coo_large_shape_uses_int64(self):
        m = sparse.coo_matrix((2, _LARGE + 1))
        assert m.row.dtype == cupy.int64
        assert m.col.dtype == cupy.int64

    @testing.with_requires('scipy')
    def test_from_scipy_csr_preserves_int64(self):
        # scipy uses its own get_index_dtype; CuPy must trust and preserve it.
        import scipy.sparse
        data = numpy.array([1.0, 2.0])
        indices = numpy.array([0, _LARGE], dtype=numpy.int64)
        indptr = numpy.array([0, 1, 2], dtype=numpy.int64)
        sp = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))
        m = sparse.csr_matrix(sp)
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        assert int(m.indices[1]) == _LARGE


class TestInt64FormatConversion:
    """Pure-CuPy fallbacks for tocoo / tocsr / tocsc with int64.

    cuSPARSE's xcsr2coo / xcoo2csr / csr2cscEx2 only accept int32 pointers.
    The fallbacks use searchsorted (for indptr expansion) and unique+scatter
    (for indptr construction), avoiding the 2×large-allocation OOM of bincount.
    """

    def _make_int64_csr(self):
        """2-row CSR: row 0 → col 0, row 1 → col _LARGE."""
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))

    def _make_int64_coo(self):
        """2-entry COO: (0,0,1.0) and (1,_LARGE,2.0)."""
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, _LARGE], dtype=cupy.int64)
        return sparse.coo_matrix(
            (data, (row, col)), shape=(2, _LARGE + 1))

    def test_csr_tocoo_int64(self):
        # Earlier in development, xcsr2coo read int64 indptr
        # as int32, producing silently wrong row indices
        # (e.g. row=[1,0] instead of [0,1]).
        coo = self._make_int64_csr().tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.col.dtype == cupy.int64
        assert int(coo.row[0]) == 0
        assert int(coo.row[1]) == 1
        assert int(coo.col[0]) == 0
        assert int(coo.col[1]) == _LARGE

    def test_coo_tocsr_int64(self):
        csr = self._make_int64_coo().tocsr()
        assert csr.indices.dtype == cupy.int64
        assert csr.indptr.dtype == cupy.int64
        assert int(csr.indptr[0]) == 0
        assert int(csr.indptr[1]) == 1
        assert int(csr.indptr[2]) == 2
        assert int(csr.indices[1]) == _LARGE

    def test_csc_tocoo_int64(self):
        # csc2coo has its own searchsorted fallback, separate from csr2coo.
        # CSC: col 0 → row 0, col 1 → row _LARGE.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)  # row indices
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)     # 1 nnz per column
        csc = sparse.csc_matrix((data, indices, indptr), shape=(_LARGE + 1, 2))
        coo = csc.tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.col.dtype == cupy.int64
        assert int(coo.row[0]) == 0
        assert int(coo.row[1]) == _LARGE
        assert int(coo.col[0]) == 0
        assert int(coo.col[1]) == 1

    def test_csr_tocsc_int32_regression(self):
        # The int32 tocsc path must remain correct as we make changes.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 2, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 2, 3], dtype=cupy.int32)
        csr = sparse.csr_matrix((data, indices, indptr), shape=(2, 3))
        csc = csr.tocsc()
        assert csc.indices.dtype == cupy.int32
        assert csc.indptr.dtype == cupy.int32
        testing.assert_array_equal(csc.toarray(), csr.toarray())

    @testing.slow
    def test_csr_tocsc_int64(self):
        # _cupy_csr2csc_int64 uses unique+scatter for the CSC indptr.
        # This test requires ~17 GB for the CSC indptr; skipped if OOM.
        n = numpy.iinfo(numpy.int32).max + 3  # INT32_MAX + 3 ≈ 2.15 B
        try:
            data = cupy.array([1.0, 2.0])
            indices = cupy.array([0, n - 1], dtype=cupy.int64)
            indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
            csr = sparse.csr_matrix((data, indices, indptr), shape=(2, n))
            csc = csr.tocsc()
        except cupy.cuda.memory.OutOfMemoryError:
            pytest.skip('not enough GPU memory')
        assert csc.indices.dtype == cupy.int64
        assert csc.indptr.dtype == cupy.int64
        # Row indices in CSC: col 0 holds row 0, col n-1 holds row 1.
        assert int(csc.indices[0]) == 0
        assert int(csc.indices[1]) == 1


class TestInt64Sort:
    """Lexsort-based fallbacks for csrsort, cscsort, sum_duplicates.

    cuSPARSE xcsrsort / xcscsort / xcoosort accept only int32 index pointers.
    """

    def test_csr_sort_indices_int64(self):
        # csrsort int64 path: expand indptr via searchsorted, then lexsort.
        # Row 0 has columns [_LARGE+1, _LARGE] (deliberately unsorted).
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([_LARGE + 1, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 2))
        assert not m.has_sorted_indices

        m.sort_indices()

        assert m.has_sorted_indices
        assert m.indices.dtype == cupy.int64
        assert int(m.indices[0]) == _LARGE       # smaller col first
        assert int(m.indices[1]) == _LARGE + 1
        assert float(m.data[0]) == 2.0           # data reordered with indices
        assert float(m.data[1]) == 1.0

    def test_csc_sort_indices_int64(self):
        # cscsort int64 path: column 0 has rows [_LARGE+1, _LARGE] (unsorted).
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([_LARGE + 1, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 2], dtype=cupy.int64)  # 2 nnz in col 0
        m = sparse.csc_matrix(
            (data, indices, indptr), shape=(_LARGE + 2, 2))
        assert not m.has_sorted_indices

        m.sort_indices()

        assert m.has_sorted_indices
        assert m.indices.dtype == cupy.int64
        assert int(m.indices[0]) == _LARGE       # smaller row first
        assert int(m.indices[1]) == _LARGE + 1
        assert float(m.data[0]) == 2.0
        assert float(m.data[1]) == 1.0

    def test_coo_sum_duplicates_int64(self):
        # Previously, the ElementwiseKernel declared
        # 'int32 src_col', silently truncating int64 col
        # values > INT32_MAX to their low 32 bits.
        # Two duplicate entries: both at (row=0, col=_LARGE).
        data = cupy.array([2.0, 3.0])
        row = cupy.array([0, 0], dtype=cupy.int64)
        col = cupy.array([_LARGE, _LARGE], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(1, _LARGE + 1))

        m.sum_duplicates()

        assert m.nnz == 1
        assert m.col.dtype == cupy.int64
        assert int(m.col[0]) == _LARGE  # must not be truncated to int32
        assert float(m.data[0]) == pytest.approx(5.0)

    def test_csr_sort_indices_int32_regression(self):
        # The int32 csrsort path must remain correct.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([2, 0, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 3], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, 3))
        m.sort_indices()
        assert m.has_sorted_indices
        testing.assert_array_equal(m.indices, cupy.array([0, 1, 2]))
        testing.assert_array_equal(m.data, cupy.array([2.0, 3.0, 1.0]))


class TestInt64ArithmeticFallback:
    """Sparse addition with int64 indices — pure-CuPy fallback.

    csrgeam2 routes int64 inputs to _cupy_csrgeam_int64 *before* checking
    cuSPARSE availability, so the path works on any CUDA version.

    The fallback: expand indptr→row via searchsorted, concatenate COO entries
    from both matrices, call sum_duplicates() to merge overlapping positions,
    then convert back to CSR.  Index dtype is
    numpy.result_type(a.indices.dtype, b.indices.dtype) throughout.

    Note: cusparseSpGEAM (the Generic API path) is absent from all public
    cuSPARSE releases through 12.7.9, so the pure-CuPy fallback is always
    active on current installations.
    """

    # Shape has _LARGE+2 columns so a column index of _LARGE is valid and
    # forces int64.  Only 2 rows, so indptr has 3 elements (cheap).
    _shape = (2, _LARGE + 2)

    def _make_int64_csr(self, col, value=1.0):
        """Single-entry CSR: row 0 has one nonzero at (0, col)."""
        data = cupy.array([value])
        indices = cupy.array([col], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        return sparse.csr_matrix((data, indices, indptr), shape=self._shape)

    def test_add_int64_preserves_dtype(self):
        # Both operands have int64 indices; the result must too.
        # (If the fallback accidentally truncated to int32, int(indices[1])
        # would silently wrap and give the wrong column.)
        a = self._make_int64_csr(0)
        b = self._make_int64_csr(_LARGE)
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.indptr.dtype == cupy.int64
        assert c.nnz == 2

    def test_add_int64_values_correct(self):
        # Values at column positions 0 and _LARGE are preserved after addition.
        # After sort_indices(), col 0 is always at position 0 and col _LARGE
        # at position 1, so direct array access is safe.
        a = self._make_int64_csr(0, value=3.0)
        b = self._make_int64_csr(_LARGE, value=7.0)
        c = (a + b)
        c.sort_indices()
        assert c.nnz == 2
        assert int(c.indices[0]) == 0
        assert int(c.indices[1]) == _LARGE
        assert float(c.data[0]) == pytest.approx(3.0)
        assert float(c.data[1]) == pytest.approx(7.0)

    def test_add_int64_overlapping_entries_summed(self):
        # When A and B share a (row, col) position, the fallback concatenates
        # both entries into a COO and relies on sum_duplicates() to merge them.
        a = self._make_int64_csr(_LARGE, value=2.0)
        b = self._make_int64_csr(_LARGE, value=5.0)
        c = a + b
        assert c.nnz == 1
        assert c.indices.dtype == cupy.int64
        assert int(c.indices[0]) == _LARGE
        assert float(c.data[0]) == pytest.approx(7.0)

    def test_add_int64_alpha_beta_scaling(self):
        # _cupy_csrgeam_int64 scales a.data by alpha and b.data by beta before
        # concatenation.  Verify through the direct cusparse.csrgeam2 interface
        # since the __add__ operator always uses alpha=1, beta=1.
        # (This test caught a bug where _numpy.array(alpha, ...) returned a
        # 0-d ndarray that CuPy's __mul__ rejected with TypeError.)
        a = self._make_int64_csr(0, value=1.0)
        b = self._make_int64_csr(_LARGE, value=1.0)
        c = cusparse.csrgeam2(a, b, alpha=3.0, beta=4.0)
        c.sort_indices()
        assert c.nnz == 2
        # col 0 → alpha*1.0 = 3.0;  col _LARGE → beta*1.0 = 4.0.
        assert float(c.data[0]) == pytest.approx(3.0)
        assert float(c.data[1]) == pytest.approx(4.0)

    def test_spgeam_int64_fallback(self):
        # cusparse.spgeam() routes int64 directly to
        # _cupy_csrgeam_int64 when cusparseSpGEAM is unavailable
        # (absent from all public releases <=12.7.9).
        a = self._make_int64_csr(0, value=1.0)
        b = self._make_int64_csr(_LARGE, value=2.0)
        c = cusparse.spgeam(a, b)
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 2

    def test_add_int64_multirow(self):
        # The searchsorted(indptr[1:], arange(nnz)) expansion must assign
        # each nonzero to the correct row.  a has entries in both rows;
        # b has an entry only in row 0 that overlaps a's row-0 entry.
        a = sparse.csr_matrix(
            (cupy.array([1.0, 2.0]),
             cupy.array([0, _LARGE], dtype=cupy.int64),
             cupy.array([0, 1, 2], dtype=cupy.int64)),   # 1 nnz per row
            shape=self._shape)
        b = self._make_int64_csr(_LARGE, value=3.0)   # row 0 → col _LARGE
        c = a + b
        c.sort_indices()
        assert c.nnz == 3
        # Row 0: cols 0 and _LARGE (2 entries).  Row 1: col _LARGE (1 entry).
        assert int(c.indptr[1]) == 2
        assert int(c.indptr[2]) == 3
        # Row 1's entry must retain the exact int64 column value.
        assert int(c.indices[2]) == _LARGE
        assert float(c.data[2]) == pytest.approx(2.0)

    def test_add_mixed_dtype_int32_plus_int64_promotes(self):
        # idx_dtype = numpy.result_type(int32, int64) == int64.
        # The int32 matrix has small column values;
        # the int64 matrix has _LARGE.
        # The result must use int64 to represent _LARGE.
        data = cupy.array([1.0])
        a = sparse.csr_matrix(
            (data, cupy.array([5], dtype=cupy.int32),
             cupy.array([0, 1, 1], dtype=cupy.int32)),
            shape=self._shape)
        b = self._make_int64_csr(_LARGE, value=2.0)
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 2

    def test_add_int64_empty_operand(self):
        # When one matrix has nnz=0, _cupy_csrgeam_int64 enters the
        # `a_rows = cupy.empty(0, idx_dtype)` branch.  The result equals
        # the non-empty matrix.
        a = self._make_int64_csr(_LARGE)
        b = sparse.csr_matrix(
            (cupy.empty(0, cupy.float64),
             cupy.empty(0, cupy.int64),
             cupy.zeros(3, cupy.int64)),
            shape=self._shape)
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 1
        assert int(c.indices[0]) == _LARGE

    def test_add_int32_regression(self):
        # int32 + int32 must continue to use the cuSPARSE (csrgeam2) path and
        # return int32 indices with correct values.
        data = cupy.array([1.0, 2.0])
        a = sparse.csr_matrix(
            (data[:1], cupy.array([0], dtype=cupy.int32),
             cupy.array([0, 1, 1], dtype=cupy.int32)),
            shape=(2, 4))
        b = sparse.csr_matrix(
            (data[1:], cupy.array([3], dtype=cupy.int32),
             cupy.array([0, 0, 1], dtype=cupy.int32)),
            shape=(2, 4))
        c = a + b
        assert c.indices.dtype == cupy.int32
        assert c.nnz == 2
        testing.assert_array_equal(c.toarray(), a.toarray() + b.toarray())


class TestInt64ScalarIndex:
    """Scalar index m[i, j] with int64 column/row > INT32_MAX.

    _compress_getitem_kern previously typed 'minor' as int32, silently
    truncating large column values so the equality check always failed and
    the lookup returned 0.  The fix changes int32 minor to S minor, matching
    the dtype of the ind (column/row index) array.
    """

    def _make_int64_csr(self, col=_LARGE, value=5.0):
        """Single nonzero at (row=0, col=col)."""
        data = cupy.array([value])
        indices = cupy.array([col], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))

    def test_csr_scalar_index_large_col_returns_value(self):
        # m[0, _LARGE] must return the stored value, not 0.
        # Previously, int32 minor truncated _LARGE to a negative int32,
        # ind == minor was always False, and m[0, _LARGE] silently returned 0.
        m = self._make_int64_csr()
        result = m[0, _LARGE]
        assert float(result) == pytest.approx(5.0)

    def test_csr_scalar_index_absent_large_col_returns_zero(self):
        # Absence (structural zero) at a large column must return 0.
        # stored at _LARGE-1, not _LARGE
        m = self._make_int64_csr(col=_LARGE - 1)
        result = m[0, _LARGE]
        assert float(result) == pytest.approx(0.0)

    def test_csr_scalar_index_small_col_int32_regression(self):
        # int32 matrix: scalar index must remain correct.
        data = cupy.array([7.0])
        indices = cupy.array([3], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 10))
        assert float(m[0, 3]) == pytest.approx(7.0)
        assert float(m[0, 4]) == pytest.approx(0.0)

    def test_csc_scalar_index_large_row_returns_value(self):
        # CSC m[row, col] uses the same kernel with minor = target row.
        # Verify the fix works for CSC too.
        data = cupy.array([3.0])
        indices = cupy.array([_LARGE], dtype=cupy.int64)  # row index
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)  # 1 nnz in col 0
        m = sparse.csc_matrix(
            (data, indices, indptr), shape=(_LARGE + 1, 2))
        assert float(m[_LARGE, 0]) == pytest.approx(3.0)
        assert float(m[_LARGE - 1, 0]) == pytest.approx(0.0)

    def test_csr_complex_scalar_index_large_col(self):
        # Complex variant uses _compress_getitem_complex_kern, same int32 fix.
        data = cupy.array([2.0 + 3.0j])
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))
        result = complex(m[0, _LARGE])
        assert result.real == pytest.approx(2.0)
        assert result.imag == pytest.approx(3.0)

    def test_csr_scalar_index_multiple_rows(self):
        # Matrix with one nonzero per row; index into both rows.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))
        assert float(m[0, 0]) == pytest.approx(1.0)
        assert float(m[1, _LARGE]) == pytest.approx(2.0)
        assert float(m[0, _LARGE]) == pytest.approx(0.0)
        assert float(m[1, 0]) == pytest.approx(0.0)


class TestInt64FancyRowIndex:
    """Fancy row index m[[r1, r2], :] with int64 column indices > INT32_MAX.

    Two int64 fixes work together here:
    - _csr_row_index_ker: all int32 parameters changed to I so int64 column
      values are not truncated in the output matrix.
    - _csr_indptr_to_coo_rows: xcsr2coo is int32-only; int64 path uses
      searchsorted to expand indptr to per-nnz row assignments.
    """

    _shape = (4, _LARGE + 2)

    def _make_int64_csr(self):
        """4-row CSR: row 0 → col 0 (val=1), row 2 → col _LARGE (val=2),
        rows 1 and 3 are empty."""
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1, 2, 2], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)

    def test_fancy_row_result_has_int64_indices(self):
        m = self._make_int64_csr()
        sub = m[[0, 2], :]
        assert sub.indices.dtype == cupy.int64
        assert sub.indptr.dtype == cupy.int64

    def test_fancy_row_large_col_preserved(self):
        # Previously, _csr_row_index_ker wrote Bj as int32, truncating
        # _LARGE to its low 32 bits (wrong negative int).
        m = self._make_int64_csr()
        sub = m[[0, 2], :]
        assert sub.nnz == 2
        assert int(sub.indices[0]) == 0
        assert int(sub.indices[1]) == _LARGE

    def test_fancy_row_values_correct(self):
        m = self._make_int64_csr()
        sub = m[[0, 2], :]
        assert float(sub.data[0]) == pytest.approx(1.0)
        assert float(sub.data[1]) == pytest.approx(2.0)

    def test_fancy_row_reverse_order(self):
        # Rows requested in reverse order: result should have reversed rows.
        m = self._make_int64_csr()
        sub = m[[2, 0], :]
        assert sub.nnz == 2
        assert int(sub.indices[0]) == _LARGE
        assert int(sub.indices[1]) == 0
        assert float(sub.data[0]) == pytest.approx(2.0)
        assert float(sub.data[1]) == pytest.approx(1.0)

    def test_fancy_row_single_row(self):
        m = self._make_int64_csr()
        sub = m[[2], :]
        assert sub.nnz == 1
        assert int(sub.indices[0]) == _LARGE
        assert sub.indices.dtype == cupy.int64

    def test_fancy_row_empty_rows_selected(self):
        # Selecting only empty rows should produce an empty matrix with
        # correct int64 dtypes.
        m = self._make_int64_csr()
        sub = m[[1, 3], :]
        assert sub.nnz == 0
        assert sub.indices.dtype == cupy.int64
        assert sub.indptr.dtype == cupy.int64

    def test_fancy_row_indptr_correct(self):
        # indptr[r+1] - indptr[r] must equal the nnz for each selected row.
        m = self._make_int64_csr()
        sub = m[[0, 1, 2, 3], :]  # all rows
        assert int(sub.indptr[0]) == 0
        assert int(sub.indptr[1]) == 1   # row 0 has 1 nnz
        assert int(sub.indptr[2]) == 1   # row 1 has 0 nnz
        assert int(sub.indptr[3]) == 2   # row 2 has 1 nnz
        assert int(sub.indptr[4]) == 2   # row 3 has 0 nnz

    def test_fancy_row_int32_regression(self):
        # int32 path must remain correct.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 2, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(3, 5))
        sub = m[[2, 0], :]
        assert sub.indices.dtype == cupy.int32
        assert int(sub.indices[0]) == 1  # row 2 has col 1
        assert int(sub.indices[1]) == 0  # row 0 has col 0
        assert float(sub.data[0]) == pytest.approx(3.0)
        assert float(sub.data[1]) == pytest.approx(1.0)

    def test_fancy_row_dtype_not_demoted_when_values_small(self):
        # int64 dtype is preserved even when all values fit in int32.
        # The dtype is a property of the matrix, not the values.
        data = cupy.array([1.0])
        indices = cupy.array([5], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 1))
        sub = m[[0], :]
        assert sub.indices.dtype == cupy.int64


class TestInt64Argmax:
    """argmax/argmin along an axis with int64 column indices > INT32_MAX.

    _argmax_argmin_code previously used int* for the indices and indptr
    slice arrays, silently truncating int64 column values.  We added a
    TI template parameter so all index-typed values use the correct dtype.
    """

    def _make_int64_csr(self, nrows=3):
        """CSR matrix with one nonzero per row:
          row 0 → (col=0, val=1.0)
          row 1 → (col=_LARGE, val=2.0)   ← argmax column for float comparison
          row 2 → (col=_LARGE//2, val=1.5)
        """
        data = cupy.array([1.0, 2.0, 1.5])[:nrows]
        cols = [0, _LARGE, _LARGE // 2]
        indices = cupy.array(cols[:nrows], dtype=cupy.int64)
        indptr = cupy.arange(nrows + 1, dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(nrows, _LARGE + 1))

    def test_csr_argmax_axis1_large_col(self):
        # argmax(axis=1) must return the correct int64 column index.
        # Previously, int* indices truncated _LARGE, returning wrong col.
        m = self._make_int64_csr()
        result = m.argmax(axis=1)
        # Row 0: max at col 0.  Row 1: max at col _LARGE.
        # Row 2: max at col _LARGE//2.
        assert int(result[1, 0]) == _LARGE
        assert int(result[0, 0]) == 0

    def test_csr_argmin_axis1_large_col(self):
        # argmin(axis=1) where the minimum is a negative
        # value at a large column.
        # Row has two nonzeros: col 1 → 1.0, col _LARGE → -3.0.
        # Implicit zeros at other cols are 0.0; -3.0 is the global min.
        data = cupy.array([1.0, -3.0])
        indices = cupy.array([1, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(1, _LARGE + 1))
        result = m.argmin(axis=1)
        assert int(result[0, 0]) == _LARGE

    def test_csr_argmax_axis1_result_dtype(self):
        # Result dtype is int (default out dtype), not affected by index dtype.
        m = self._make_int64_csr()
        result = m.argmax(axis=1)
        assert result.dtype in (cupy.int32, cupy.int64, cupy.intp)

    def test_csc_argmax_axis0_large_row(self):
        # CSC argmax(axis=0) finds the row of the max per column.
        # With a large row index, TI=int64 must be used.
        data = cupy.array([3.0, 1.0])
        indices = cupy.array([_LARGE, 0], dtype=cupy.int64)  # row indices
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)     # 1 nnz per col
        m = sparse.csc_matrix(
            (data, indices, indptr), shape=(_LARGE + 1, 2))
        result = m.argmax(axis=0)
        # Col 0: max is at row _LARGE.  Col 1: max is at row 0.
        assert int(result[0, 0]) == _LARGE
        assert int(result[0, 1]) == 0

    def test_csr_argmax_no_axis_flat_index(self):
        # argmax() with no axis returns a flat index.  This path goes through
        # COO conversion (int64-aware), not _arg_minor_reduce — so it worked
        # before too.  Include as a regression guard.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        ncols = _LARGE + 1
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, ncols))
        flat = int(m.argmax())
        r = flat // ncols
        c = flat % ncols
        assert r == 1
        assert c == _LARGE

    def test_csr_argmax_axis1_int32_regression(self):
        # int32 matrix: argmax must remain correct.
        data = cupy.array([1.0, 5.0, 3.0])
        indices = cupy.array([0, 2, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(3, 5))
        result = m.argmax(axis=1)
        assert int(result[0, 0]) == 0   # row 0: only col 0
        assert int(result[1, 0]) == 2   # row 1: only col 2
        assert int(result[2, 0]) == 1   # row 2: only col 1

    def test_csr_argmax_axis1_multiple_large_cols(self):
        # Multiple rows, each with the argmax at a large col.
        # Both rows have a nonzero at a large column.
        data = cupy.array([1.0, 2.0, 0.5, 3.0])
        indices = cupy.array([0, _LARGE, 0, _LARGE + 1], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 4], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 2))
        result = m.argmax(axis=1)
        assert int(result[0, 0]) == _LARGE     # max(1.0, 2.0) → col _LARGE
        assert int(result[1, 0]) == _LARGE + 1  # max(0.5, 3.0) → col _LARGE+1


class TestInt64EliminateZeros:
    """eliminate_zeros with int64 indices uses a pure-CuPy fallback.

    csr2csr_compress is int32-only (Legacy API). For int64 matrices we use
    boolean masking + searchsorted + unique/scatter.
    """

    def test_eliminate_zeros_removes_explicit_zeros(self):
        # Row 0: [1.0@0, 0.0@_LARGE, 2.0@_LARGE+1];
        # after: [1.0@0, 2.0@_LARGE+1]
        data = cupy.array([1.0, 0.0, 2.0])
        indices = cupy.array([0, _LARGE, _LARGE + 1], dtype=cupy.int64)
        indptr = cupy.array([0, 3, 3], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 2))
        m.eliminate_zeros()
        assert m.nnz == 2
        assert m.indices.dtype == cupy.int64
        testing.assert_array_equal(m.indices, cupy.array([0, _LARGE + 1],
                                                         dtype=cupy.int64))
        testing.assert_array_equal(m.data, cupy.array([1.0, 2.0]))
        testing.assert_array_equal(m.indptr,
                                   cupy.array([0, 2, 2], dtype=cupy.int64))

    def test_eliminate_zeros_all_nonzero_is_noop(self):
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        m.eliminate_zeros()
        assert m.nnz == 2
        assert m.indices.dtype == cupy.int64

    def test_eliminate_zeros_all_zero(self):
        data = cupy.array([0.0, 0.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        m.eliminate_zeros()
        assert m.nnz == 0
        assert m.indices.dtype == cupy.int64
        testing.assert_array_equal(m.indptr,
                                   cupy.array([0, 0], dtype=cupy.int64))

    def test_eliminate_zeros_int32_regression(self):
        # int32 path (csr2csr_compress) must still work.
        data = cupy.array([1.0, 0.0, 2.0])
        indices = cupy.array([0, 3, 5], dtype=cupy.int32)
        indptr = cupy.array([0, 3, 3], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 10))
        m.eliminate_zeros()
        assert m.nnz == 2
        assert m.indices.dtype == cupy.int32


class TestInt64Multiply:
    """Element-wise multiply for int64 matrices.

    cupy_multiply_by_dense and cupy_multiply_by_csr_step1/step2 previously
    had int32 shape parameters; they now use I (long long for int64 matrices).
    """

    def test_multiply_dense_broadcast_int64(self):
        # (1, _LARGE+1) sparse * (1, 1) dense — broadcasting.
        data = cupy.array([2.0, 3.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        sp = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        dn = cupy.full((1, 1), 4.0)
        result = sp.multiply(dn)
        assert result.nnz == 2
        assert result.indices.dtype == cupy.int64
        assert abs(float(result[0, 0]) - 8.0) < 1e-9
        assert abs(float(result[0, _LARGE]) - 12.0) < 1e-9

    def test_multiply_csr_int64(self):
        # element-wise sparse * sparse, same sparsity pattern.
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        a = sparse.csr_matrix(
            (cupy.array([2.0, 3.0]), indices, indptr),
            shape=(1, _LARGE + 1))
        b = sparse.csr_matrix(
            (cupy.array([4.0, 5.0]), indices.copy(), indptr.copy()),
            shape=(1, _LARGE + 1))
        result = a.multiply(b)
        assert result.nnz == 2
        assert result.indices.dtype == cupy.int64
        assert abs(float(result[0, 0]) - 8.0) < 1e-9
        assert abs(float(result[0, _LARGE]) - 15.0) < 1e-9

    def test_multiply_dense_int32_regression(self):
        # int32 matrix must still work (int32 shape params, no overflow risk).
        sp = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        dn = cupy.full((3, 3), 2.0)
        result = sp.multiply(dn)
        testing.assert_array_almost_equal(result.toarray(),
                                          2.0 * cupy.eye(3))

    def test_multiply_csr_int32_regression(self):
        # int32 × int32 sparse element-wise.
        a = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        b = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        result = a.multiply(b)
        assert result.indices.dtype == cupy.int32
        testing.assert_array_almost_equal(result.toarray(), cupy.eye(3))


class TestInt64Diagonal:
    """diagonal() for int64 matrices.

    _cupy_csr_diagonal previously had int32 rows/cols; now uses I.
    For a matrix with shape (few_rows, large_cols), diagonal() is practical
    (output has few_rows elements, no OOM).
    """

    def test_diagonal_int64_large_cols(self):
        # (2, _LARGE+1) matrix — diagonal is 2 elements.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 1], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 1))
        d = m.diagonal()
        assert d.shape == (2,)
        testing.assert_array_almost_equal(d, cupy.array([1.0, 2.0]))

    def test_diagonal_int64_absent_returns_zero(self):
        # Diagonal element at (1,1) is absent → 0.0.
        data = cupy.array([1.0])
        indices = cupy.array([0], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 1))
        d = m.diagonal()
        assert abs(float(d[0]) - 1.0) < 1e-9
        assert abs(float(d[1]) - 0.0) < 1e-9

    def test_diagonal_int32_regression(self):
        # int32 path must still work.
        m = sparse.csr_matrix(cupy.eye(4, dtype=cupy.float64))
        d = m.diagonal()
        testing.assert_array_almost_equal(d, cupy.ones(4))


class TestInt64MinMaxReduction:
    """max/min axis-reductions with int64 index dtype.

    _max_min_reduction_code previously used int* for indptr slices and a plain
    int32 'length' parameter (= shape[axis]).  For int64 matrices with ncols >
    INT32_MAX, passing shape[1] to int32 raised OverflowError at kernel launch.
    The fix: RawModule + name_expressions templated on TI, dispatched by
    get_typename(self.indptr.dtype); shape param passed as idx_dtype.type(N).

    Design note: axis=1 (reduce over columns) on a CSR matrix sends
    length = shape[1] directly to the kernel — the critical int64 path.
    axis=0 converts to CSC first, then sends length = shape[0].
    Both paths exercise the same TI template, just with different shapes.
    """

    def _make_int64_csr(self):
        # 2 × (_LARGE+1) CSR — shape forces int64 index dtype.
        # Row 0: col 0 → 1.0, col 2 → 3.0
        # Row 1: col 1 → 2.0, col 2 → -1.0
        data = cupy.array([1.0, 3.0, 2.0, -1.0])
        indices = cupy.array([0, 2, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 4], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))

    def test_max_axis1_int64(self):
        # Previously raised OverflowError: shape[1] = _LARGE+1 overflowed int32
        # in the kernel launch argument.  Now uses idx_dtype.type(N) = int64.
        m = self._make_int64_csr()
        assert m.indices.dtype == cupy.int64
        result = m.max(axis=1).toarray()
        # Row 0: max(1.0, 3.0, implicit zeros) = 3.0
        # Row 1: max(2.0, -1.0, implicit zeros) = 2.0
        assert float(result[0, 0]) == pytest.approx(3.0)
        assert float(result[1, 0]) == pytest.approx(2.0)

    def test_min_axis1_int64(self):
        m = self._make_int64_csr()
        assert m.indices.dtype == cupy.int64
        result = m.min(axis=1).toarray()
        # Row 0: min(1.0, 3.0, implicit zeros) = 0.0 (implicit zeros dominate)
        # Row 1: min(2.0, -1.0, implicit zeros) = -1.0
        assert float(result[0, 0]) == pytest.approx(0.0)
        assert float(result[1, 0]) == pytest.approx(-1.0)

    def test_max_axis0_int64(self):
        # axis=0 on CSC: _minor_reduce receives length = shape[0] = _LARGE+1.
        # Construct (_LARGE+1, 2) CSC directly — indptr has 3 elements (tiny).
        # Col 0: rows 0,1 → 1.0, -2.0   Col 1: row 0 → 3.0
        data = cupy.array([1.0, -2.0, 3.0])
        indices = cupy.array([0, 1, 0], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 3], dtype=cupy.int64)
        m = sparse.csc_matrix((data, indices, indptr), shape=(_LARGE + 1, 2))
        assert m.indices.dtype == cupy.int64
        result = m.max(axis=0).toarray()
        # Col 0: max(1.0, -2.0, implicit zeros) = 1.0
        # Col 1: max(3.0, implicit zeros) = 3.0
        assert float(result[0, 0]) == pytest.approx(1.0)
        assert float(result[0, 1]) == pytest.approx(3.0)

    def test_max_axis1_int32_regression(self):
        # int32 path must still work after the RawKernel → RawModule change.
        data = cupy.array([1.0, 3.0, 2.0, -1.0])
        indices = cupy.array([0, 2, 1, 2], dtype=cupy.int32)
        indptr = cupy.array([0, 2, 4], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 5))
        result = m.max(axis=1).toarray()
        assert float(result[0, 0]) == pytest.approx(3.0)
        assert float(result[1, 0]) == pytest.approx(2.0)


class TestInt64Toarray:
    """toarray() for int64 matrices.

    _cupy_csr2dense previously had int32 M, N shape
    parameters; for matrices where shape[1] > INT32_MAX,
    these caused OverflowError at the Python layer before
    the kernel was launched (numpy.int32(N) overflows for
    N > INT32_MAX). Now they use I (idx_dtype.type(N)),
    matching the index dtype.

    Practical constraint: a (1, _LARGE+1) dense output requires ~17 GB, so
    these tests use a try/except OOM guard and skip on 16 GB hardware.
    The int32 regression test always runs.
    """

    @testing.slow
    def test_toarray_int64_no_overflow_error(self):
        # Before fix: OverflowError at numpy.int32(_LARGE+1) in kernel args.
        # After fix: either succeeds (≥17 GB GPU) or OOMs gracefully.
        data = cupy.array([1.0])
        indices = cupy.array([0], dtype=cupy.int64)
        indptr = cupy.array([0, 1], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        try:
            arr = m.toarray()
            assert arr.shape == (1, _LARGE + 1)
            assert float(arr[0, 0]) == pytest.approx(1.0)
        except cupy.cuda.memory.OutOfMemoryError:
            pytest.skip('not enough GPU memory for dense output')

    @testing.slow
    def test_toarray_order_f_no_overflow_error(self):
        # order='F' calls _cupy_csr2dense with row_major=False.
        data = cupy.array([1.0])
        indices = cupy.array([0], dtype=cupy.int64)
        indptr = cupy.array([0, 1], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 1))
        try:
            arr = m.toarray(order='F')
            assert arr.shape == (1, _LARGE + 1)
            assert float(arr[0, 0]) == pytest.approx(1.0)
        except cupy.cuda.memory.OutOfMemoryError:
            pytest.skip('not enough GPU memory for dense output')

    def test_toarray_int32_regression(self):
        # int32 path must continue to work after the M, N parameter change.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 2], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 3))
        arr = m.toarray()
        expected = cupy.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        testing.assert_array_almost_equal(arr, expected)


class TestInt64SpGEMM:
    """int64 sparse matrix multiplication via the pure-CuPy fallback.

    cuSPARSE spgemm returns CUSPARSE_STATUS_NOT_SUPPORTED for int64 index
    matrices despite advertising int64 support via the Generic API.
    The pure-CuPy fallback (product-expansion + sum_duplicates) is invoked
    automatically when either input has int64 indices.  The int32 cuSPARSE
    path is completely unchanged.
    """

    def test_spgemm_int64_large_col_value(self):
        # A(2,1) @ B(1, 2^32) = C(2, 2^32) with one nnz at col _LARGE.
        data = cupy.array([1.0])
        a_indices = cupy.array([0], dtype=cupy.int64)
        b_indices = cupy.array([_LARGE], dtype=cupy.int64)
        a_indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        b_indptr = cupy.array([0, 1], dtype=cupy.int64)
        a = sparse.csr_matrix(
            (data, a_indices, a_indptr), shape=(2, 1))
        b = sparse.csr_matrix(
            (data, b_indices, b_indptr), shape=(1, _LARGE + 1))
        c = a @ b
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 1
        assert int(c.indices[0]) == _LARGE
        assert float(c.data[0]) == pytest.approx(1.0)

    def test_spgemm_int64_sum_duplicate_products(self):
        # Two A entries in the same row → same output col → values summed.
        a_data = cupy.array([2.0, 3.0])
        a_indices = cupy.array([0, 1], dtype=cupy.int64)
        a_indptr = cupy.array([0, 2], dtype=cupy.int64)
        a = sparse.csr_matrix((a_data, a_indices, a_indptr), shape=(1, 2))
        b_data = cupy.array([5.0, 7.0])
        b_indices = cupy.array([_LARGE, _LARGE], dtype=cupy.int64)
        b_indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        b = sparse.csr_matrix(
            (b_data, b_indices, b_indptr), shape=(2, _LARGE + 1))
        c = a @ b
        # c[0, _LARGE] = 2*5 + 3*7 = 31
        assert c.nnz == 1
        assert float(c.data[0]) == pytest.approx(31.0)

    def test_spgemm_int64_zero_products(self):
        # A nonzero at col 0, but B row 0 is empty → product is all-zeros.
        a_data = cupy.array([1.0])
        a_indices = cupy.array([0], dtype=cupy.int64)
        a_indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        a = sparse.csr_matrix((a_data, a_indices, a_indptr), shape=(2, 3))
        b_data = cupy.array([1.0])
        b_indices = cupy.array([_LARGE], dtype=cupy.int64)
        b_indptr = cupy.array([0, 0, 1, 1], dtype=cupy.int64)  # row 0 empty
        b = sparse.csr_matrix(
            (b_data, b_indices, b_indptr), shape=(3, _LARGE + 1))
        c = a @ b
        assert c.nnz == 0
        assert c.shape == (2, _LARGE + 1)

    def test_spgemm_int32_regression(self):
        # int32 path must still use cuSPARSE and return int32 indices.
        if not cusparse.check_availability('spgemm'):
            pytest.skip('spgemm is not available')
        a = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        b = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        c = a @ b
        assert c.indices.dtype == cupy.int32
        testing.assert_array_almost_equal(c.toarray(), cupy.eye(3))

    def test_spgemm_int64_alpha_coefficient(self):
        # alpha scales all output values: c[i,j] = alpha * sum_k a[i,k]*b[k,j].
        a_data = cupy.array([2.0])
        a_indices = cupy.array([0], dtype=cupy.int64)
        a_indptr = cupy.array([0, 1], dtype=cupy.int64)
        a = sparse.csr_matrix((a_data, a_indices, a_indptr), shape=(1, 1))
        b_data = cupy.array([3.0])
        b_indices = cupy.array([_LARGE], dtype=cupy.int64)
        b_indptr = cupy.array([0, 1], dtype=cupy.int64)
        b = sparse.csr_matrix(
            (b_data, b_indices, b_indptr),
            shape=(1, _LARGE + 1))
        c = cusparse.spgemm(a, b, alpha=5.0)
        # c[0, _LARGE] = 5.0 * 2.0 * 3.0 = 30.0
        assert c.nnz == 1
        assert int(c.indices[0]) == _LARGE
        assert float(c.data[0]) == pytest.approx(30.0)

    def test_spgemm_mixed_index_dtypes(self):
        # A has int32 indices, B has int64 indices (large col value).
        # The int64 path is triggered by b.indices.dtype and handles
        # the mixed case by casting A's indices to int64 internally.
        a_data = cupy.array([4.0])
        a_indices = cupy.array([0], dtype=cupy.int32)
        a_indptr = cupy.array([0, 1], dtype=cupy.int32)
        a = sparse.csr_matrix((a_data, a_indices, a_indptr), shape=(1, 1))
        b_data = cupy.array([6.0])
        b_indices = cupy.array([_LARGE], dtype=cupy.int64)
        b_indptr = cupy.array([0, 1], dtype=cupy.int64)
        b = sparse.csr_matrix(
            (b_data, b_indices, b_indptr),
            shape=(1, _LARGE + 1))
        c = a @ b
        assert c.nnz == 1
        assert c.indices.dtype == cupy.int64
        assert float(c.data[0]) == pytest.approx(24.0)

    def test_spgemm_int64_float32_data(self):
        # float32 data arrays must remain float32 after int64 SpGEMM.
        a_data = cupy.array([2.0], dtype=cupy.float32)
        a_indices = cupy.array([0], dtype=cupy.int64)
        a_indptr = cupy.array([0, 1], dtype=cupy.int64)
        a = sparse.csr_matrix((a_data, a_indices, a_indptr), shape=(1, 1))
        b_data = cupy.array([3.0], dtype=cupy.float32)
        b_indices = cupy.array([_LARGE], dtype=cupy.int64)
        b_indptr = cupy.array([0, 1], dtype=cupy.int64)
        b = sparse.csr_matrix(
            (b_data, b_indices, b_indptr),
            shape=(1, _LARGE + 1))
        c = a @ b
        assert c.dtype == cupy.float32
        assert c.nnz == 1
        assert float(c.data[0]) == pytest.approx(6.0)

    def test_spgemm_int64_multiple_output_rows(self):
        # A(3,1) @ B(1, _LARGE+1): each A row produces one nonzero.
        # Verifies the row-index expansion (cumsum+searchsorted) works for
        # multiple rows, not just row 0.
        a_data = cupy.array([1.0, 2.0, 3.0])
        a_indices = cupy.array([0, 0, 0], dtype=cupy.int64)
        a_indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        a = sparse.csr_matrix((a_data, a_indices, a_indptr), shape=(3, 1))
        b_data = cupy.array([10.0])
        b_indices = cupy.array([_LARGE], dtype=cupy.int64)
        b_indptr = cupy.array([0, 1], dtype=cupy.int64)
        b = sparse.csr_matrix(
            (b_data, b_indices, b_indptr),
            shape=(1, _LARGE + 1))
        c = a @ b
        assert c.shape == (3, _LARGE + 1)
        assert c.nnz == 3
        # CSR stores rows in order; each row has one nonzero at col _LARGE.
        expected = cupy.array([10.0, 20.0, 30.0])
        testing.assert_array_almost_equal(c.data, expected)


class TestInt64SpGEMMCscDispatch:
    """csr * csc, csc * csr, and csc * csc with int64 indices.

    These paths previously fell through to csrgemm/csrgemm2, which are
    int32-only.  ``__mul__`` now detects int64 operands and routes
    through ``cusparse.spgemm`` (Generic API on CUDA 13.0+; pure-CuPy
    ``_cupy_spgemm_int64`` fallback otherwise).
    """

    def _diag_csr_int64(self, vals):
        n = len(vals)
        return sparse.csr_matrix._from_parts(
            cupy.asarray(vals, dtype=cupy.float64),
            cupy.arange(n, dtype=cupy.int64),
            cupy.arange(n + 1, dtype=cupy.int64),
            (n, n))

    def _diag_csc_int64(self, vals):
        n = len(vals)
        return sparse.csc_matrix._from_parts(
            cupy.asarray(vals, dtype=cupy.float64),
            cupy.arange(n, dtype=cupy.int64),
            cupy.arange(n + 1, dtype=cupy.int64),
            (n, n))

    def test_csr_times_csc_int64(self):
        # csr_int64 @ csc_int64 used to fall to csrgemm (int32-only)
        # via __mul__'s csc branch; now routes through spgemm.
        a = self._diag_csr_int64([1.0, 2.0, 3.0])
        b = self._diag_csc_int64([4.0, 5.0, 6.0])
        c = a @ b
        assert c.indices.dtype == cupy.int64
        testing.assert_array_almost_equal(
            c.toarray(), cupy.diag(cupy.array([4.0, 10.0, 18.0])))

    def test_csc_times_csr_int64(self):
        a = self._diag_csc_int64([1.0, 2.0, 3.0])
        b = self._diag_csr_int64([4.0, 5.0, 6.0])
        c = a @ b
        assert c.indices.dtype == cupy.int64
        testing.assert_array_almost_equal(
            c.toarray(), cupy.diag(cupy.array([4.0, 10.0, 18.0])))

    def test_csc_times_csc_int64(self):
        # Both operands CSC with int64 indices.  This case used to
        # fall through to the csrgemm path (int32-only) on most CUDA
        # builds; now routes through spgemm.
        a = self._diag_csc_int64([1.0, 2.0, 3.0])
        b = self._diag_csc_int64([4.0, 5.0, 6.0])
        c = a @ b
        assert c.indices.dtype == cupy.int64
        testing.assert_array_almost_equal(
            c.toarray(), cupy.diag(cupy.array([4.0, 10.0, 18.0])))

    def test_csc_times_csc_mixed_dtypes(self):
        # int32 csc * int64 csc should also route through spgemm
        # (mixed-dtype detection picks int64).
        a32 = sparse.csc_matrix._from_parts(
            cupy.array([1.0, 2.0, 3.0]),
            cupy.arange(3, dtype=cupy.int32),
            cupy.arange(4, dtype=cupy.int32),
            (3, 3))
        b64 = self._diag_csc_int64([4.0, 5.0, 6.0])
        c = a32 @ b64
        assert c.indices.dtype == cupy.int64
        testing.assert_array_almost_equal(
            c.toarray(), cupy.diag(cupy.array([4.0, 10.0, 18.0])))


class TestInt64DtypePreservation:
    """_with_data and construction bypass preserve index dtype.

    Before, five independent check_contents=True barriers silently
    downcasted int64 indices to int32 when the index values happened to fit
    in int32.  This affected copy(), abs(), neg(), scalar multiply, astype(),
    vstack(), hstack(), bmat(), and tocsr() on COO matrices.

    All tests here use int64 index arrays whose values fit in int32, which
    is the scenario that was broken.  This is distinct from large-value int64
    (> INT32_MAX) which was already working.
    """

    def test_csr_copy_preserves_int64_small_values(self):
        # _with_data bypass: copy() must not downcast int64 indices
        # whose values happen to fit in int32.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        # Force int64 by constructing without check_contents downcast.
        m.indices = indices
        m.indptr = indptr
        c = m.copy()
        assert c.indices.dtype == cupy.int64
        assert c.indptr.dtype == cupy.int64
        testing.assert_array_almost_equal(c.toarray(), m.toarray())

    def test_csr_abs_preserves_int64_small_values(self):
        # _with_data bypass: abs() must not downcast int64 indices.
        data = cupy.array([-1.0, 2.0, -3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        m.indices = indices
        m.indptr = indptr
        result = abs(m)
        assert result.indices.dtype == cupy.int64
        assert result.indptr.dtype == cupy.int64
        expected = cupy.array([[1.0, 0.0, 0.0],
                               [0.0, 2.0, 0.0],
                               [0.0, 0.0, 3.0]])
        testing.assert_array_almost_equal(result.toarray(), expected)

    def test_csr_neg_preserves_int64_small_values(self):
        # _with_data bypass: negation must not downcast int64 indices.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        m.indices = indices
        m.indptr = indptr
        result = -m
        assert result.indices.dtype == cupy.int64
        assert result.indptr.dtype == cupy.int64
        expected = cupy.array([[-1.0, 0.0, 0.0],
                               [0.0, -2.0, 0.0],
                               [0.0, 0.0, -3.0]])
        testing.assert_array_almost_equal(result.toarray(), expected)

    def test_csr_scalar_multiply_preserves_int64_small_values(self):
        # _with_data bypass: scalar multiply must not downcast int64 indices.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        m.indices = indices
        m.indptr = indptr
        result = m * 2.0
        assert result.indices.dtype == cupy.int64
        assert result.indptr.dtype == cupy.int64
        expected = cupy.array([[2.0, 0.0, 0.0],
                               [0.0, 4.0, 0.0],
                               [0.0, 0.0, 6.0]])
        testing.assert_array_almost_equal(result.toarray(), expected)

    def test_csr_astype_preserves_int64_small_values(self):
        # _with_data bypass: astype() must not downcast int64 indices.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        m.indices = indices
        m.indptr = indptr
        result = m.astype(cupy.float32)
        assert result.indices.dtype == cupy.int64
        assert result.indptr.dtype == cupy.int64
        assert result.data.dtype == cupy.float32

    def test_csc_copy_preserves_int64_small_values(self):
        # _with_data bypass applies to CSC via self.__class__: copy() must
        # not downcast int64 indices for CSC matrices.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csc_matrix((data, indices, indptr), shape=(3, 3))
        m.indices = indices
        m.indptr = indptr
        c = m.copy()
        assert c.indices.dtype == cupy.int64
        assert c.indptr.dtype == cupy.int64
        testing.assert_array_almost_equal(c.toarray(), m.toarray())

    def test_coo_copy_preserves_int64_small_values(self):
        # COO _with_data bypass: copy() must not downcast int64 row/col
        # arrays whose values fit in int32.
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
        # Overwrite with explicit int64 arrays (constructor may downcast).
        m.row = row
        m.col = col
        c = m.copy()
        assert c.row.dtype == cupy.int64
        assert c.col.dtype == cupy.int64
        testing.assert_array_almost_equal(c.toarray(), m.toarray())

    def test_coo_has_canonical_format_preserved_by_copy(self):
        # COO _with_data must propagate has_canonical_format because it is
        # a structural property of the (row, col) pattern, not of data values.
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
        m.row = row
        m.col = col
        m.has_canonical_format = True
        c = m.copy()
        assert c.has_canonical_format is True

        m2 = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
        m2.row = row
        m2.col = col
        m2.has_canonical_format = False
        c2 = m2.copy()
        assert c2.has_canonical_format is False

    def test_csr_transpose_preserves_int64_small_values(self):
        # transpose() used the public constructor, which ran
        # check_contents=True and downcast small int64 to int32.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True)
        mt = m.T
        assert mt.indices.dtype == cupy.int64
        assert mt.indptr.dtype == cupy.int64
        testing.assert_array_equal(mt.toarray(), m.toarray().T)

    def test_csc_transpose_preserves_int64_small_values(self):
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csc_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True)
        mt = m.T
        assert mt.indices.dtype == cupy.int64
        assert mt.indptr.dtype == cupy.int64
        testing.assert_array_equal(mt.toarray(), m.toarray().T)

    def test_coo_transpose_preserves_int64_small_values(self):
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.coo_matrix._from_parts(
            data, row, col, shape=(3, 3),
            has_canonical_format=True)
        mt = m.T
        assert mt.row.dtype == cupy.int64
        assert mt.col.dtype == cupy.int64
        testing.assert_array_equal(mt.toarray(), m.toarray().T)

    def test_coo_dot_scalar_preserves_int64_small_values(self):
        # COO.dot(scalar) used the public constructor, downcasting.
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.coo_matrix._from_parts(
            data, row, col, shape=(3, 3),
            has_canonical_format=True)
        result = m.dot(2.0)
        assert result.row.dtype == cupy.int64
        assert result.col.dtype == cupy.int64
        testing.assert_array_equal(
            result.toarray(), m.toarray() * 2.0)

    def test_csr_max_axis_preserves_dtype(self):
        # _min_or_max_axis returned float64 when input was float32
        # because the reduction promotes dtype and the old code
        # didn't cast back.
        data = cupy.array([1.0, 2.0, 3.0], dtype=cupy.float32)
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True)
        result = m.max(axis=0)
        assert result.dtype == cupy.float32

    def test_csr_getrow_preserves_int64_small_values(self):
        # _major_slice used the public constructor, which downcast.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True)
        r = m.getrow(1)
        assert r.indices.dtype == cupy.int64
        assert float(r[0, 1]) == 2.0

    def test_csr_getcol_preserves_int64_small_values(self):
        # _minor_slice used the public constructor, which downcast.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True)
        r = m.getcol(0)
        assert r.indices.dtype == cupy.int64

    def test_csr_slice_preserves_int64_small_values(self):
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True)
        r = m[0:2, :]
        assert r.indices.dtype == cupy.int64
        r2 = m[:, 0:2]
        assert r2.indices.dtype == cupy.int64

    def test_csr_fancy_row_preserves_int64_small_values(self):
        # _major_index_fancy used the public constructor.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True)
        r = m[[0, 2], :]
        assert r.indices.dtype == cupy.int64

    def test_csr_comparison_preserves_int64_small_values(self):
        # binopt_csr used the public constructor for the result.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True,
            has_sorted_indices=True)
        r = m != 0
        assert r.indices.dtype == cupy.int64

    def test_csr_matmul_preserves_int64_small_values(self):
        # _cupy_spgemm_int64 used the public COO constructor.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True,
            has_sorted_indices=True)
        r = m @ m
        assert r.indices.dtype == cupy.int64

    def test_coo_reshape_preserves_int64_small_values(self):
        # COO reshape used the public constructor.
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.coo_matrix._from_parts(
            data, row, col, shape=(3, 3),
            has_canonical_format=True)
        r = m.reshape((1, 9))
        assert r.row.dtype == cupy.int64
        assert r.col.dtype == cupy.int64

    def test_csr_truediv_preserves_int64_small_values(self):
        # multiply_by_scalar used the public constructor.
        data = cupy.array([2.0, 4.0, 6.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True)
        r = m / 2.0
        assert r.indices.dtype == cupy.int64
        testing.assert_array_almost_equal(
            r.toarray(), cupy.diag(cupy.array([1., 2., 3.])))

    def test_csr_multiply_csr_preserves_int64_small_values(self):
        # multiply_by_csr used the public constructor.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True,
            has_sorted_indices=True)
        r = m.multiply(m)
        assert r.indices.dtype == cupy.int64

    def test_csr_fancy_col_preserves_int64_small_values(self):
        # _minor_index_fancy_sorted recomputed dtype from output
        # shape instead of propagating the input's index dtype.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 3),
            has_canonical_format=True,
            has_sorted_indices=True)
        r = m[:, cupy.array([0, 2])]
        assert r.indices.dtype == cupy.int64

    def test_csr_fancy_col_empty_preserves_int64(self):
        # _minor_index_fancy empty case used the public constructor.
        m = sparse.csr_matrix._from_parts(
            cupy.empty(0, 'f'),
            cupy.empty(0, cupy.int64),
            cupy.zeros(4, cupy.int64),
            shape=(3, 3))
        r = m[:, [0, 2]]
        assert r.indices.dtype == cupy.int64

    def test_csr_fancy_col_zero_match_preserves_int64(self):
        # _minor_index_fancy_sorted zero-nnz case used the public
        # constructor.
        data = cupy.array([1., 2., 3.])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(
            data, indices, indptr, shape=(3, 5),
            has_canonical_format=True,
            has_sorted_indices=True)
        r = m[:, cupy.array([3, 4])]
        assert r.indices.dtype == cupy.int64
        assert r.nnz == 0

    def test_vstack_preserves_explicitly_set_int64(self):
        # _compressed_sparse_stack bypass: vstack must not downcast int64
        # indices set explicitly on input matrices.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 1], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 2))
        m.indices = indices
        m.indptr = indptr
        result = sparse.vstack([m, m])
        assert result.indices.dtype == cupy.int64
        assert result.indptr.dtype == cupy.int64
        expected = cupy.array([[1.0, 0.0],
                               [0.0, 2.0],
                               [1.0, 0.0],
                               [0.0, 2.0]])
        testing.assert_array_almost_equal(result.toarray(), expected)

    def test_hstack_index_dtype_matches_max_dim(self):
        # bmat uses max(shape) for index dtype, matching scipy.
        # Shape (33000, 66000): max dim = 66000, fits int32.
        nrows = 33000
        ncols = 66000
        half = ncols // 2
        a = sparse.csr_matrix(
            (cupy.array([1.0]), cupy.array([0]), cupy.array([0, 1])),
            shape=(1, half))
        a = sparse.vstack([a, sparse.csr_matrix((nrows - 1, half))])
        b = sparse.csr_matrix(
            (cupy.array([2.0]), cupy.array([0]), cupy.array([0, 1])),
            shape=(1, half))
        b = sparse.vstack([b, sparse.csr_matrix((nrows - 1, half))])
        result = sparse.hstack([a, b], format='csr')
        assert result.shape == (nrows, ncols)
        assert result.indices.dtype == cupy.int32

    def test_coo2csr_preserves_int64(self):
        # cusparse.coo2csr must preserve int64 row/col dtype even when
        # values fit in int32.
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
        m.row = row
        m.col = col
        result = cusparse.coo2csr(m)
        assert result.indices.dtype == cupy.int64
        assert result.indptr.dtype == cupy.int64
        expected = cupy.diag(cupy.array([1.0, 2.0, 3.0]))
        testing.assert_array_almost_equal(result.toarray(), expected)

    def test_coo2csc_preserves_int64(self):
        # cusparse.coo2csc must preserve int64 row/col dtype even when
        # values fit in int32.
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
        m.row = row
        m.col = col
        result = cusparse.coo2csc(m)
        assert result.indices.dtype == cupy.int64
        assert result.indptr.dtype == cupy.int64
        expected = cupy.diag(cupy.array([1.0, 2.0, 3.0]))
        testing.assert_array_almost_equal(result.toarray(), expected)

    def test_tocsr_on_int64_coo_preserves_dtype(self):
        # Full chain: COO.tocsr() must preserve int64 through coo2csr.
        # This exercises the path: _coo.tocsr() → cusparse.coo2csr() →
        # csr_matrix bypass.
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
        m.row = row
        m.col = col
        result = m.tocsr()
        assert result.indices.dtype == cupy.int64
        assert result.indptr.dtype == cupy.int64
        expected = cupy.diag(cupy.array([1.0, 2.0, 3.0]))
        testing.assert_array_almost_equal(result.toarray(), expected)

    def test_int32_copy_regression(self):
        # Regression guard: copy() on an int32 CSR matrix must still
        # produce int32 indices after the _with_data bypass.
        m = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        assert m.indices.dtype == cupy.int32
        assert m.indptr.dtype == cupy.int32
        c = m.copy()
        assert c.indices.dtype == cupy.int32
        assert c.indptr.dtype == cupy.int32
        testing.assert_array_almost_equal(c.toarray(), cupy.eye(3))

    def test_vstack_int32_regression(self):
        # Regression guard: vstack on int32 matrices must still produce
        # int32 indices after the _compressed_sparse_stack bypass.
        m = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        assert m.indices.dtype == cupy.int32
        result = sparse.vstack([m, m])
        assert result.indices.dtype == cupy.int32
        assert result.indptr.dtype == cupy.int32
        expected = cupy.zeros((6, 3))
        expected[:3, :3] = cupy.eye(3)
        expected[3:, :3] = cupy.eye(3)
        testing.assert_array_almost_equal(result.toarray(), expected)


class TestInt64FancyMinorIndex:
    """Fancy minor-axis indexing via _minor_index_fancy_sorted for int64.

    Previously, _minor_index_fancy() used a histogram kernel that
    allocated O(N) memory (col_counts = zeros(N)).  For int64 matrices N can
    exceed INT32_MAX, causing OOM (e.g. N = 2**32 → 16 GB).  The kernels also
    used const int* parameters, silently truncating index values > INT32_MAX.

    The new _minor_index_fancy_sorted routes int64 matrices through
    argsort + searchsorted instead of the histogram, with O(nnz + n_idx)
    space (no N-sized buffer).

    For CSR: the minor axis is columns → `m[:, [col1, col2]]`.
    For CSC: the minor axis is rows    → `m[[row1, row2], :]`.
    """

    def _make_int64_csr_2row(self):
        """2-row CSR: row 0 → col 0 (val=1.0), row 1 → col _LARGE (val=2.0)."""
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))

    def test_csr_fancy_col_large_index_selects_correct_row(self):
        # Selecting col 0 from a 2-row matrix where row 1 has a large col
        # index.  Before the fix, the histogram kernel would OOM (O(N)
        # allocation for N = _LARGE+1 cols) or truncate the int64 index.
        # The source matrix must have int64 indices (the routing condition).
        m = self._make_int64_csr_2row()
        assert m.indices.dtype == cupy.int64
        sub = m[:, [0]]
        assert sub.nnz == 1
        # indptr must show row 0 has the entry, row 1 does not.
        assert int(sub.indptr[1]) == 1
        assert int(sub.indptr[2]) == 1
        assert float(sub.data[0]) == pytest.approx(1.0)

    def test_csr_fancy_col_value_at_large_index(self):
        # Selecting col _LARGE — previously the histogram kernel silently
        # truncated _LARGE to its low 32 bits, producing a wrong column match
        # and returning 0.0 instead of the stored value.
        m = self._make_int64_csr_2row()
        sub = m[:, [_LARGE]]
        assert sub.nnz == 1
        # Row 1 has the entry at col _LARGE; row 0 does not.
        assert int(sub.indptr[1]) == 0
        assert int(sub.indptr[2]) == 1
        assert float(sub.data[0]) == pytest.approx(2.0)

    def test_csr_fancy_col_multiple_columns(self):
        # Select three columns spanning small and large index space.
        # Row 0: col 0→1.0, col _LARGE→2.0; Row 1: col 5→3.0.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, _LARGE, 5], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))
        sub = m[:, [0, 5, _LARGE]]
        # Output shape (2, 3), nnz=3.
        assert sub.shape == (2, 3)
        assert sub.nnz == 3
        # Row 0: two entries (cols 0 and _LARGE → output positions 0 and 2).
        assert int(sub.indptr[1]) == 2
        # Row 1: one entry (col 5 → output position 1).
        assert int(sub.indptr[2]) == 3
        # Sort-based path always returns has_sorted_indices=True.
        assert sub.has_sorted_indices

    def test_csr_fancy_col_duplicate_request(self):
        # Request the same large column twice: row 1 must appear twice in
        # output.  The histogram approach would have OOM; the sorted approach
        # handles duplicates via the lo/hi searchsorted range.
        # Matrix: row 0→col 0, row 1→col _LARGE.
        m = self._make_int64_csr_2row()
        sub = m[:, [_LARGE, _LARGE]]
        # Output shape (2, 2): two copies of col _LARGE.
        assert sub.shape == (2, 2)
        # Row 0 has no entry at _LARGE → 0 nnz; row 1 has it twice.
        assert int(sub.indptr[1]) == 0
        assert int(sub.indptr[2]) == 2
        assert sub.nnz == 2
        assert float(sub.data[0]) == pytest.approx(2.0)
        assert float(sub.data[1]) == pytest.approx(2.0)

    def test_csr_fancy_col_absent_column(self):
        # Requesting a column that has no stored entries → empty output.
        # The sort-based path reaches total_nnz==0 and returns an empty matrix.
        m = self._make_int64_csr_2row()
        assert m.indices.dtype == cupy.int64
        sub = m[:, [_LARGE - 1]]
        assert sub.nnz == 0
        assert sub.shape == (2, 1)

    def test_csc_fancy_row_large_index(self):
        # CSC: minor axis is rows.  m[[_LARGE], :] triggers the same
        # _minor_index_fancy_sorted path.
        # CSC shape (_LARGE+1, 2): col 0→row 0 (val=1.0), col 1→row _LARGE.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)  # row indices
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)     # 1 nnz per col
        m = sparse.csc_matrix(
            (data, indices, indptr), shape=(_LARGE + 1, 2))
        sub = m[[_LARGE], :]
        assert sub.shape == (1, 2)
        assert sub.nnz == 1
        assert float(sub.data[0]) == pytest.approx(2.0)

    def test_csr_fancy_col_complex_data(self):
        # Complex128 values flow through the same code path; verify real/imag.
        data = cupy.array([1.0 + 2.0j], dtype=cupy.complex128)
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))
        sub = m[:, [_LARGE]]
        assert sub.nnz == 1
        val = complex(sub.data[0])
        assert val.real == pytest.approx(1.0)
        assert val.imag == pytest.approx(2.0)

    def test_csr_fancy_col_matches_int32_kernel(self):
        # Verify sort-based path agrees with the int32 histogram kernel.
        # Use shape (3, 10) so toarray() is safe.  Build an int64 matrix with
        # small-value indices by constructing via the bypass pattern
        # (set .indices/.indptr directly to prevent check_contents downcast).
        data = cupy.array([1.0, 2.0, 3.0, 4.0])
        indices32 = cupy.array([0, 3, 5, 9], dtype=cupy.int32)
        indptr32 = cupy.array([0, 1, 3, 4], dtype=cupy.int32)
        m32 = sparse.csr_matrix(
            (data, indices32, indptr32), shape=(3, 10))
        assert m32.indices.dtype == cupy.int32

        # Force int64 by bypassing the constructor downcast.
        m64 = sparse.csr_matrix((3, 10), dtype=cupy.float64)
        m64.data = data.copy()
        m64.indices = cupy.array([0, 3, 5, 9], dtype=cupy.int64)
        m64.indptr = cupy.array([0, 1, 3, 4], dtype=cupy.int64)
        assert m64.indices.dtype == cupy.int64

        cols = [0, 5, 9]
        sub32 = m32[:, cols]
        sub64 = m64[:, cols]
        # Both must produce identical dense output.
        assert numpy.allclose(sub32.toarray().get(), sub64.toarray().get())

    def test_csr_fancy_col_unsorted_source(self):
        # Source matrix has unsorted indices within a row.  The sort-based
        # path must still produce correct results with has_sorted_indices=True
        # on the output, regardless of the source order.
        # Row 0: col _LARGE→1.0 then col 0→2.0 (deliberately reversed order).
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([_LARGE, 0], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(1, _LARGE + 1))
        assert not m.has_sorted_indices

        sub = m[:, [0, _LARGE]]
        assert sub.has_sorted_indices
        assert sub.nnz == 2
        # Output col 0 → output index 0 → value 2.0.
        assert int(sub.indices[0]) == 0
        assert float(sub.data[0]) == pytest.approx(2.0)
        # Output col _LARGE → output index 1 → value 1.0.
        assert int(sub.indices[1]) == 1
        assert float(sub.data[1]) == pytest.approx(1.0)

    def test_csr_fancy_col_int32_regression(self):
        # int32 matrix must NOT take the sort-based path; it uses the
        # histogram kernel and retains int32 indices.  Values must be correct.
        data = cupy.array([5.0, 7.0])
        indices = cupy.array([0, 3], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 5))
        sub = m[:, [0, 3]]
        assert sub.indices.dtype == cupy.int32
        assert sub.nnz == 2
        expected = cupy.array([[5.0, 0.0], [0.0, 7.0]])
        testing.assert_array_equal(sub.toarray(), expected)


class TestInt64ConversionPreservation:
    """Format conversions must preserve explicit int64 index dtype.

    A second class of check_contents=True barriers existed in format-conversion
    code paths: csr2coo, csc2coo, _cupy_csr2csc_int64, _cupy_csc2csr_int64,
    _cupy_csrgeam_int64, and empty-matrix early-returns in coo.tocsr/tocsc.

    All tests here use int64 index arrays whose values fit in int32 (the
    "explicit precision" use case).  Large-value int64 matrices (> INT32_MAX)
    are covered by sparse_tests/test_02_format_conversion.py.
    """

    def _make_csr_int64(self, data, row_ind, col_ind, shape):
        """Build a CSR with int64 indices whose values fit in int32."""
        d = cupy.array(data, dtype=cupy.float64)
        r = cupy.array(row_ind, dtype=cupy.int32)
        c = cupy.array(col_ind, dtype=cupy.int32)
        m = sparse.csr_matrix((d, (r, c)), shape=shape)
        m.indices = m.indices.astype(cupy.int64)
        m.indptr = m.indptr.astype(cupy.int64)
        return m

    def test_csr_tocoo_small_values_preserves_int64(self):
        # csr2coo must preserve int64 indices even when values fit in int32.
        # Before the fix, the tuple-2 COO constructor applied
        # check_contents=True and silently downcasted to int32.
        m = self._make_csr_int64([1.0, 2.0], [0, 1], [1, 2], (3, 3))
        coo = m.tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.col.dtype == cupy.int64
        testing.assert_array_almost_equal(coo.toarray(), m.toarray())

    def test_csc_tocoo_small_values_preserves_int64(self):
        # csc2coo must preserve int64 indices even when values fit in int32.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 2, 2], dtype=cupy.int32)
        mc = sparse.csc_matrix((data, indices, indptr), shape=(3, 3))
        mc.indices = mc.indices.astype(cupy.int64)
        mc.indptr = mc.indptr.astype(cupy.int64)
        coo = mc.tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.col.dtype == cupy.int64

    def test_coo_tocsr_empty_preserves_int64(self):
        # Empty COO with int64 row/col → tocsr() must return int64 CSR.
        # Before the fix, the early-return used csr_matrix(shape) which
        # ignores self.row.dtype and uses get_index_dtype(maxval=...) → int32.
        c = sparse.coo_matrix((3, 3), dtype=cupy.float64)
        c.row = c.row.astype(cupy.int64)
        c.col = c.col.astype(cupy.int64)
        csr = c.tocsr()
        assert csr.indices.dtype == cupy.int64
        assert csr.indptr.dtype == cupy.int64
        assert csr.nnz == 0

    def test_coo_tocsc_empty_preserves_int64(self):
        # Symmetric to test_coo_tocsr_empty_preserves_int64 for CSC.
        c = sparse.coo_matrix((3, 3), dtype=cupy.float64)
        c.row = c.row.astype(cupy.int64)
        c.col = c.col.astype(cupy.int64)
        csc = c.tocsc()
        assert csc.indices.dtype == cupy.int64
        assert csc.indptr.dtype == cupy.int64
        assert csc.nnz == 0

    def test_csr_sum_duplicates_preserves_int64(self):
        # sum_duplicates on an int64 CSR must not silently downcast indices.
        # Before the fix, sum_duplicates called tocoo() → csr2coo returned
        # int32 COO → coo.asformat('csr') returned int32 CSR.
        m = self._make_csr_int64([1.0, 2.0], [0, 1], [1, 2], (3, 3))
        m.has_canonical_format = False
        m.sum_duplicates()
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64

    def test_csr_tocsc_small_values_preserves_int64(self):
        # _cupy_csr2csc_int64 must return CSC with int64 indices even for
        # small shapes and small index values (nnz > 0 path).
        m = self._make_csr_int64([1.0, 2.0, 3.0], [0, 0, 2], [0, 2, 1], (3, 3))
        csc = m.tocsc()
        assert csc.indices.dtype == cupy.int64
        assert csc.indptr.dtype == cupy.int64
        testing.assert_array_almost_equal(csc.toarray(), m.toarray())

    def test_csr_tocsc_empty_small_shape_preserves_int64(self):
        # _cupy_csr2csc_int64 nnz=0 path must preserve int64 indices.
        m = self._make_csr_int64([], [], [], (3, 3))
        csc = m.tocsc()
        assert csc.indices.dtype == cupy.int64
        assert csc.indptr.dtype == cupy.int64
        assert csc.nnz == 0

    def test_csc_tocsr_empty_small_shape_preserves_int64(self):
        # _cupy_csc2csr_int64 nnz=0 path must preserve int64 indices.
        mc = sparse.csc_matrix((3, 3), dtype=cupy.float64)
        mc.indices = mc.indices.astype(cupy.int64)
        mc.indptr = mc.indptr.astype(cupy.int64)
        csr = mc.tocsr()
        assert csr.indices.dtype == cupy.int64
        assert csr.indptr.dtype == cupy.int64
        assert csr.nnz == 0

    def test_csc_tocsr_small_values_preserves_int64(self):
        # _cupy_csc2csr_int64 must return CSR with int64 indices (nnz > 0).
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 2], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 1, 2], dtype=cupy.int32)
        mc = sparse.csc_matrix((data, indices, indptr), shape=(3, 3))
        mc.indices = mc.indices.astype(cupy.int64)
        mc.indptr = mc.indptr.astype(cupy.int64)
        csr = mc.tocsr()
        assert csr.indices.dtype == cupy.int64
        assert csr.indptr.dtype == cupy.int64
        testing.assert_array_almost_equal(csr.toarray(), mc.toarray())

    def test_add_int64_small_values_preserves_dtype(self):
        # csrgeam2 (sparse + sparse) result must have int64 indices when both
        # inputs are int64.  Before the fix, the intermediate COO was
        # downcasted to int32 by the tuple-2 constructor.
        a = self._make_csr_int64([1.0], [0], [1], (3, 3))
        b = self._make_csr_int64([2.0], [1], [2], (3, 3))
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.indptr.dtype == cupy.int64
        expected = cupy.array([[0., 1., 0.], [0., 0., 2.], [0., 0., 0.]])
        testing.assert_array_almost_equal(c.toarray(), expected)

    def test_csr_round_trip_preserves_int64(self):
        # CSR → CSC → CSR round-trip must preserve int64 throughout.
        m = self._make_csr_int64([5.0, 3.0], [0, 2], [2, 0], (3, 3))
        rt = m.tocsc().tocsr()
        assert rt.indices.dtype == cupy.int64
        assert rt.indptr.dtype == cupy.int64
        testing.assert_array_almost_equal(rt.toarray(), m.toarray())

    def test_csc_from_csr_has_canonical_format(self):
        # _cupy_csr2csc_int64 uses object.__new__ and leaves
        # _has_canonical_format unset.  The property getter must compute it
        # lazily via kernel rather than raising AttributeError.  The lexsort
        # in _cupy_csr2csc_int64 produces sorted, duplicate-free output, so
        # the result should be canonical.
        m = self._make_csr_int64([1.0, 2.0], [0, 1], [2, 0], (3, 3))
        csc = m.tocsc()
        assert csc.has_canonical_format
        assert csc.has_sorted_indices

    def test_csc_from_csr_copy_preserves_int64(self):
        # copy() on an object.__new__-constructed CSC must preserve int64
        # dtype.  The sparse-from-sparse constructor path uses
        # idx_dtype = indices.dtype (no check_contents), so this works.
        m = self._make_csr_int64([5.0], [1], [2], (3, 3))
        csc = m.tocsc()
        csc_copy = csc.copy()
        assert csc_copy.indices.dtype == cupy.int64
        assert csc_copy.indptr.dtype == cupy.int64
        testing.assert_array_almost_equal(csc_copy.toarray(), csc.toarray())

    def test_csc_tocoo_after_csr2csc_preserves_int64(self):
        # Converting an int64 CSC (returned by _cupy_csr2csc_int64) back to
        # COO via csc2coo must preserve int64 row/col dtype.
        m = self._make_csr_int64([3.0, 7.0], [0, 2], [1, 0], (3, 3))
        coo = m.tocsc().tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.col.dtype == cupy.int64
        testing.assert_array_almost_equal(coo.toarray(), m.toarray())

    def test_empty_coo_tocsc_tocsr_chain_preserves_int64(self):
        # Empty COO with int64 row/col → tocsc (Pattern B) → tocsr
        # (_cupy_csc2csr_int64 nnz=0, Pattern C).  Both conversions must
        # preserve int64 index dtype.
        c = sparse.coo_matrix((3, 3), dtype=cupy.float64)
        c.row = c.row.astype(cupy.int64)
        c.col = c.col.astype(cupy.int64)
        csc = c.tocsc()
        assert csc.indices.dtype == cupy.int64
        assert csc.indptr.dtype == cupy.int64
        csr = csc.tocsr()
        assert csr.indices.dtype == cupy.int64
        assert csr.indptr.dtype == cupy.int64
        assert csr.nnz == 0


class TestInt64RealImag:
    """The .real and .imag properties preserve int64 index dtype.

    These properties use _with_data(self.data.real / .imag), which
    goes through the _with_data bypass that constructs via the
    shape-only constructor to avoid check_contents=True downcast.
    """

    _shape = (2, _LARGE + 2)

    def _make_int64_csr(self, value=1.0+2.0j):
        data = cupy.array([value])
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)

    def test_real_preserves_int64(self):
        m = self._make_int64_csr(1.0+2.0j)
        r = m.real
        assert r.indices.dtype == cupy.int64
        assert r.indptr.dtype == cupy.int64
        assert int(r.indices[0]) == _LARGE
        assert float(r.data[0]) == pytest.approx(1.0)

    def test_imag_preserves_int64(self):
        m = self._make_int64_csr(1.0+2.0j)
        im = m.imag
        assert im.indices.dtype == cupy.int64
        assert im.indptr.dtype == cupy.int64
        assert int(im.indices[0]) == _LARGE
        assert float(im.data[0]) == pytest.approx(2.0)

    def test_real_float_is_identity(self):
        # .real on a real-valued matrix returns an identical matrix.
        m = self._make_int64_csr(5.0+0j)
        r = m.real
        assert r.indices.dtype == cupy.int64
        assert r.nnz == m.nnz
        testing.assert_array_almost_equal(r.data, cupy.array([5.0]))

    def test_imag_float_is_zero(self):
        # .imag on a real-valued matrix returns a zero matrix.
        data = cupy.array([5.0])
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)
        im = m.imag
        assert im.indices.dtype == cupy.int64
        assert float(im.data.sum()) == pytest.approx(0.0)

    def test_coo_real_preserves_int64(self):
        data = cupy.array([3.0+4.0j])
        row = cupy.array([0], dtype=cupy.int64)
        col = cupy.array([_LARGE], dtype=cupy.int64)
        m = sparse.coo_matrix(
            (data, (row, col)), shape=self._shape)
        r = m.real
        assert r.row.dtype == cupy.int64
        assert r.col.dtype == cupy.int64
        assert float(r.data[0]) == pytest.approx(3.0)

    def test_csc_imag_preserves_int64(self):
        data = cupy.array([3.0+4.0j])
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csc_matrix(
            (data, indices, indptr), shape=(_LARGE + 2, 2))
        im = m.imag
        assert im.indices.dtype == cupy.int64
        assert im.indptr.dtype == cupy.int64
        assert float(im.data[0]) == pytest.approx(4.0)

    def test_real_small_values_preserves_int64(self):
        # .real must preserve int64 even when index values fit in
        # int32.  Uses the force-assign pattern to bypass downcast.
        data = cupy.array([1.0+2.0j, 3.0+4.0j])
        indices = cupy.array([0, 1], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, 3))
        m.indices = indices
        m.indptr = indptr
        r = m.real
        assert r.indices.dtype == cupy.int64
        assert r.indptr.dtype == cupy.int64


class TestInt64Trace:
    """trace() works on int64 sparse matrices.

    trace(offset) delegates to diagonal(k=offset).sum().  diagonal()
    uses the _cupy_csr_diagonal kernel, which is int64-aware (uses
    template parameter I for index arrays).
    """

    _shape = (2, _LARGE + 2)

    def test_trace_int64_off_diagonal(self):
        # Entry at (0, _LARGE) is off the main diagonal → trace = 0.
        data = cupy.array([7.0])
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)
        assert float(m.trace()) == pytest.approx(0.0)

    def test_trace_int64_on_diagonal(self):
        # m[1,1] = 5.0 is on the main diagonal → trace = 5.0.
        data = cupy.array([5.0])
        indices = cupy.array([1], dtype=cupy.int64)
        indptr = cupy.array([0, 0, 1], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)
        assert float(m.trace()) == pytest.approx(5.0)

    def test_trace_coo_int64(self):
        data = cupy.array([3.0, 7.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, 1], dtype=cupy.int64)
        m = sparse.coo_matrix(
            (data, (row, col)), shape=self._shape)
        assert float(m.trace()) == pytest.approx(10.0)

    def test_trace_offset_int64(self):
        # m[0, 1] = 9.0 is on the k=1 superdiagonal.
        data = cupy.array([9.0])
        indices = cupy.array([1], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)
        assert float(m.trace(offset=1)) == pytest.approx(9.0)
        assert float(m.trace(offset=0)) == pytest.approx(0.0)

    def test_trace_empty_int64(self):
        m = sparse.csr_matrix((2, _LARGE + 1))
        assert float(m.trace()) == pytest.approx(0.0)


class TestInt64Subtraction:
    """Sparse subtraction (A - B) with int64 indices.

    Uses the same _cupy_csrgeam_int64 fallback as addition (beta=-1).
    """

    _shape = (2, _LARGE + 2)

    def _make_int64_csr(self, col, value=1.0):
        data = cupy.array([value])
        indices = cupy.array([col], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)

    def test_sub_int64_disjoint(self):
        a = self._make_int64_csr(0, value=5.0)
        b = self._make_int64_csr(_LARGE, value=3.0)
        c = a - b
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 2
        c.sort_indices()
        assert float(c.data[0]) == pytest.approx(5.0)
        assert float(c.data[1]) == pytest.approx(-3.0)

    def test_sub_int64_overlapping(self):
        a = self._make_int64_csr(_LARGE, value=10.0)
        b = self._make_int64_csr(_LARGE, value=3.0)
        c = a - b
        assert c.nnz == 1
        assert int(c.indices[0]) == _LARGE
        assert float(c.data[0]) == pytest.approx(7.0)

    def test_sub_int64_self_is_zero(self):
        # A - A should produce all-zero entries; eliminate_zeros
        # removes them.
        a = self._make_int64_csr(_LARGE, value=5.0)
        c = a - a
        c.eliminate_zeros()
        assert c.nnz == 0


class TestInt64Transpose:
    """Transpose (.T) preserves int64 index dtype.

    CSR.T → CSC via _cupy_csr2csc_int64 (lexsort).
    CSC.T → CSR via _cupy_csc2csr_int64 (lexsort).
    COO.T swaps row and col arrays.
    """

    _shape = (2, _LARGE + 2)

    def test_csr_T_preserves_int64(self):
        data = cupy.array([5.0])
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)
        mt = m.T
        assert mt.indices.dtype == cupy.int64
        assert mt.indptr.dtype == cupy.int64
        assert mt.shape == (self._shape[1], self._shape[0])
        assert mt.nnz == 1

    def test_csc_T_preserves_int64(self):
        data = cupy.array([2.0])
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csc_matrix(
            (data, indices, indptr), shape=(_LARGE + 2, 2))
        mt = m.T
        assert mt.indices.dtype == cupy.int64
        assert mt.indptr.dtype == cupy.int64
        assert mt.shape == (2, _LARGE + 2)

    def test_coo_T_preserves_int64(self):
        data = cupy.array([3.0])
        row = cupy.array([0], dtype=cupy.int64)
        col = cupy.array([_LARGE], dtype=cupy.int64)
        m = sparse.coo_matrix(
            (data, (row, col)), shape=(1, _LARGE + 1))
        mt = m.T
        assert mt.row.dtype == cupy.int64
        assert mt.col.dtype == cupy.int64
        assert mt.shape == (_LARGE + 1, 1)
        assert int(mt.row[0]) == _LARGE
        assert int(mt.col[0]) == 0

    def test_csr_T_round_trip(self):
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)
        rt = m.T.T
        # Can't use toarray() (OOM for _LARGE columns); compare
        # sparse structure directly.
        assert rt.nnz == m.nnz
        rt.sort_indices()
        m.sort_indices()
        testing.assert_array_equal(rt.indices, m.indices)
        testing.assert_array_equal(rt.indptr, m.indptr)
        testing.assert_array_almost_equal(rt.data, m.data)


class TestInt64UnaryAndScalarOps:
    """Unary ops and scalar arithmetic preserve int64 index dtype.

    These all go through _with_data(), which uses the shape-only
    constructor bypass.  Tests use large index values to confirm the
    bypass works correctly when indices exceed INT32_MAX.
    """

    _shape = (2, _LARGE + 2)

    def _make_int64_csr(self, value=5.0):
        data = cupy.array([value])
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)

    def test_truediv_scalar_preserves_int64(self):
        m = self._make_int64_csr(6.0)
        r = m / 2.0
        assert r.indices.dtype == cupy.int64
        assert int(r.indices[0]) == _LARGE
        assert float(r.data[0]) == pytest.approx(3.0)

    def test_power_preserves_int64(self):
        m = self._make_int64_csr(3.0)
        r = m.power(2)
        assert r.indices.dtype == cupy.int64
        assert int(r.indices[0]) == _LARGE
        assert float(r.data[0]) == pytest.approx(9.0)

    def test_conj_preserves_int64(self):
        data = cupy.array([1.0+2.0j])
        indices = cupy.array([_LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)
        c = m.conj()
        assert c.indices.dtype == cupy.int64
        assert int(c.indices[0]) == _LARGE
        assert complex(c.data[0]).imag == pytest.approx(-2.0)

    def test_abs_large_col_preserves_int64(self):
        m = self._make_int64_csr(-7.0)
        a = abs(m)
        assert a.indices.dtype == cupy.int64
        assert int(a.indices[0]) == _LARGE
        assert float(a.data[0]) == pytest.approx(7.0)

    def test_neg_large_col_preserves_int64(self):
        m = self._make_int64_csr(3.0)
        n = -m
        assert n.indices.dtype == cupy.int64
        assert int(n.indices[0]) == _LARGE
        assert float(n.data[0]) == pytest.approx(-3.0)


class TestInt64LinalgGuards:
    """spsolve and csrilu02 reject int64 indices with ValueError.

    cuSPARSE csrlsvqr and csrilu02 are int32-only at the CUDA level.
    The guards raise ValueError with a message suggesting int32 cast.
    """

    def _make_int64_identity(self, n=2):
        """Small n x n identity with int64 indices."""
        m = sparse.csr_matrix((n, n), dtype=cupy.float64)
        m.data = cupy.ones(n, dtype=cupy.float64)
        m.indices = cupy.arange(n, dtype=cupy.int64)
        m.indptr = cupy.arange(n + 1, dtype=cupy.int64)
        m._shape = (n, n)
        return m

    def test_spsolve_int64_raises(self):
        from cupyx.scipy.sparse import linalg
        A = self._make_int64_identity()
        b = cupy.array([1.0, 2.0])
        with pytest.raises(ValueError, match='int64'):
            linalg.spsolve(A, b)

    def test_csrilu02_int64_raises(self):
        A = self._make_int64_identity()
        with pytest.raises(ValueError, match='int64'):
            cusparse.csrilu02(A)

    def test_csrsm2_int64_raises(self):
        A = self._make_int64_identity()
        b = cupy.eye(2, dtype=cupy.float64)
        # csrsm2 may be unavailable (removed in CUDA 12.x) or may
        # reject int64 with ValueError.  Either error is acceptable.
        with pytest.raises((ValueError, RuntimeError)):
            cusparse.csrsm2(A, b)


class TestInt64EmptyEdgeCases:
    """Edge cases with nnz=0 and int64 indices."""

    def test_empty_csr_int64_tocoo(self):
        # Small row count, large col count to avoid indptr OOM.
        m = sparse.csr_matrix((2, _LARGE + 1))
        assert m.indices.dtype == cupy.int64
        coo = m.tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.nnz == 0

    def test_empty_coo_int64_tocsr(self):
        m = sparse.coo_matrix((2, _LARGE + 1))
        assert m.row.dtype == cupy.int64
        csr = m.tocsr()
        assert csr.indices.dtype == cupy.int64
        assert csr.nnz == 0

    def test_add_empty_int64(self):
        a = sparse.csr_matrix((2, _LARGE + 1))
        b = sparse.csr_matrix((2, _LARGE + 1))
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 0

    def test_sort_indices_already_sorted_int64(self):
        data = cupy.array([1.0, 2.0])
        indices = cupy.array(
            [_LARGE, _LARGE + 1], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(1, _LARGE + 2))
        m.sort_indices()
        assert m.has_sorted_indices
        assert int(m.indices[0]) == _LARGE
        assert int(m.indices[1]) == _LARGE + 1

    def test_with_indices_dtype_preserves_sorted(self):
        # _with_indices_dtype must propagate has_sorted_indices.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int32)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, 3))
        m.has_sorted_indices = True
        promoted = cusparse._with_indices_dtype(m, cupy.int64)
        assert promoted.indices.dtype == cupy.int64
        assert promoted.indptr.dtype == cupy.int64
        assert promoted.has_sorted_indices


class TestInt64SumAxis:
    """sum(axis=0) and sum(axis=1) with int64 indices.

    sum(axis) works via dot(ones(...)): axis=0 creates ones(nrows),
    axis=1 creates ones(ncols).  The tests use shapes that keep the
    ones-vector small to avoid OOM.
    """

    def test_sum_axis0_large_cols(self):
        # Verify sum(axis=0) works with int64 indices.
        # Uses small shape with forced int64 to avoid the ~17 GB dense
        # result that a truly _LARGE shape would produce.
        data = cupy.array([3.0, 7.0])
        indices = cupy.array([0, 4], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, 6))
        m.indices = indices  # force int64 (bypass check_contents)
        m.indptr = indptr
        s = m.sum(axis=0)
        assert float(s[0, 0]) == pytest.approx(3.0)
        assert float(s[0, 4]) == pytest.approx(7.0)

    def test_sum_axis1_large_cols(self):
        # shape=(2, _LARGE+2): axis=1 creates ones(_LARGE+2) — OOM.
        # Use small-value int64 matrix to keep ncols small.
        data = cupy.array([3.0, 7.0])
        indices = cupy.array([0, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, 5))
        m.indices = indices  # force int64
        m.indptr = indptr
        s = m.sum(axis=1)
        assert float(s[0, 0]) == pytest.approx(3.0)
        assert float(s[1, 0]) == pytest.approx(7.0)

    def test_sum_no_axis(self):
        # sum() allocates ones(ncols); use small shape with int64.
        data = cupy.array([3.0, 7.0])
        indices = cupy.array([0, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, 5))
        m.indices = indices  # force int64
        m.indptr = indptr
        assert float(m.sum()) == pytest.approx(10.0)

    def test_sum_axis0_int32_regression(self):
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int32)
        indptr = cupy.array([0, 2, 3], dtype=cupy.int32)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, 3))
        s = m.sum(axis=0)
        testing.assert_array_almost_equal(
            s, cupy.array([[1.0, 2.0, 3.0]]))


class TestInt64EliminateZerosToarray:
    """eliminate_zeros preserves dense representation.

    After eliminate_zeros, toarray() must give the same result as
    before (the structural zeros are removed but the dense view
    is unchanged).  Uses small shapes to allow toarray().
    """

    def test_eliminate_zeros_toarray_matches(self):
        # Small matrix with explicit zeros; compare dense before/after.
        data = cupy.array([1.0, 0.0, 2.0, 0.0])
        indices = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 4], dtype=cupy.int64)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, 5))
        m.indices = indices  # force int64
        m.indptr = indptr
        dense_before = m.toarray().copy()
        m.eliminate_zeros()
        testing.assert_array_equal(m.toarray(), dense_before)
        assert m.nnz == 2

    def test_eliminate_zeros_toarray_int32_regression(self):
        data = cupy.array([1.0, 0.0, 2.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int32)
        indptr = cupy.array([0, 2, 3], dtype=cupy.int32)
        m = sparse.csr_matrix(
            (data, indices, indptr), shape=(2, 3))
        dense_before = m.toarray().copy()
        m.eliminate_zeros()
        testing.assert_array_equal(m.toarray(), dense_before)
        assert m.nnz == 2


class TestInt64SumDuplicatesEmpty:
    """sum_duplicates on empty COO with int64 indices."""

    def test_empty_coo_sum_duplicates(self):
        m = sparse.coo_matrix((2, _LARGE + 1))
        assert m.row.dtype == cupy.int64
        m.sum_duplicates()
        assert m.nnz == 0
        assert m.row.dtype == cupy.int64
        assert m.col.dtype == cupy.int64
        assert m.has_canonical_format


class TestInt64SetitemInsert:
    """Inserting new entries into int64 sparse matrices via __setitem__.

    _insert_many uses cupy.add.at on an int64 target array to count
    row insertions.  cupy.add.at supports int64 natively (CuPy's
    atomics.cuh provides the long-long atomicAdd overload).
    """

    _shape = (2, _LARGE + 2)

    def test_insert_new_entry_csr(self):
        m = sparse.csr_matrix._from_parts(
            cupy.array([1.0]),
            cupy.array([_LARGE], dtype=cupy.int64),
            cupy.array([0, 1, 1], dtype=cupy.int64),
            self._shape)
        m[1, 0] = 42.0
        assert m.nnz == 2
        assert float(m[1, 0]) == pytest.approx(42.0)
        assert float(m[0, _LARGE]) == pytest.approx(1.0)

    def test_insert_new_entry_csc(self):
        m = sparse.csc_matrix._from_parts(
            cupy.array([1.0]),
            cupy.array([_LARGE], dtype=cupy.int64),
            cupy.array([0, 1, 1], dtype=cupy.int64),
            (_LARGE + 2, 2))
        m[0, 1] = 42.0
        assert m.nnz == 2
        assert m.indices.dtype == cupy.int64
        assert float(m[0, 1]) == pytest.approx(42.0)


class TestInt64Setdiag:
    """setdiag on int64 sparse matrices with large diagonal offsets.

    setdiag creates a temporary CSR with indices for the diagonal,
    which must use the same index dtype as self.
    """

    _shape = (2, _LARGE + 2)

    def test_setdiag_large_positive_offset(self):
        # k=_LARGE: set m[0, _LARGE].
        m = sparse.csr_matrix._from_parts(
            cupy.array([1.0]),
            cupy.array([0], dtype=cupy.int64),
            cupy.array([0, 1, 1], dtype=cupy.int64),
            self._shape)
        m.setdiag(cupy.array([99.0]), k=_LARGE)
        assert float(m[0, _LARGE]) == pytest.approx(99.0)
        assert float(m[0, 0]) == pytest.approx(1.0)


class TestInt64Binopt:
    """Element-wise comparisons and maximum/minimum on int64 CSR.

    binopt_csr kernels use int32 shape/nnz params and int* locals
    that must be templated to I for int64 support.
    """

    _shape = (2, _LARGE + 2)

    def _make_pair(self, v1=1.0, v2=2.0):
        a = sparse.csr_matrix._from_parts(
            cupy.array([v1]),
            cupy.array([_LARGE], dtype=cupy.int64),
            cupy.array([0, 1, 1], dtype=cupy.int64),
            self._shape)
        b = sparse.csr_matrix._from_parts(
            cupy.array([v2]),
            cupy.array([_LARGE], dtype=cupy.int64),
            cupy.array([0, 1, 1], dtype=cupy.int64),
            self._shape)
        return a, b

    def test_ne_int64(self):
        a, b = self._make_pair(1.0, 2.0)
        c = a != b
        assert c.nnz == 1
        assert c.indices.dtype == cupy.int64

    def test_maximum_int64(self):
        a, b = self._make_pair(1.0, 5.0)
        c = a.maximum(b)
        assert c.indices.dtype == cupy.int64
        assert float(c[0, _LARGE]) == pytest.approx(5.0)

    def test_ne_mixed_int32_int64(self):
        # One operand int64, the other int32 (small shape).
        a = sparse.csr_matrix._from_parts(
            cupy.array([1.0, 2.0]),
            cupy.array([0, 1], dtype=cupy.int64),
            cupy.array([0, 1, 2], dtype=cupy.int64),
            (2, 3))
        b = sparse.csr_matrix._from_parts(
            cupy.array([1.0, 9.0]),
            cupy.array([0, 1], dtype=cupy.int32),
            cupy.array([0, 1, 2], dtype=cupy.int32),
            (2, 3))
        c = a != b
        assert c.nnz == 1


class TestInt64Dense2csrGuard:
    """dense2csr with int64 kernels for large bool matrices."""

    @testing.slow
    def test_large_bool_dense_to_csr(self):
        n = numpy.iinfo(numpy.int32).max // 2 + 1
        # Requires ~19 GB: 2*n bytes (dense) + (2*n+1)*8 (info array)
        mem_free = cupy.cuda.runtime.memGetInfo()[0]
        if mem_free < 20 * (1 << 30):
            pytest.skip('insufficient GPU memory (~20 GB needed)')
        a = cupy.zeros((2, n), dtype=bool)
        a[0, 0] = True
        m = sparse.csr_matrix(a)
        assert m.indices.dtype == cupy.int64
        assert m.nnz == 1
        assert int(m.indices[0]) == 0


class TestInt64DiaConversion:
    """DIA → CSR/CSC conversion with large shape.

    DIA tocsc hardcodes int32 for kernel params and output arrays.
    For shapes > INT32_MAX, the conversion must produce int64.
    """

    def test_dia_tocsr_large_shape(self):
        # Shape (_LARGE+1, 2): max(shape) > INT32_MAX triggers int64.
        # tocsc() produces a 2-column CSC (indptr has 3 entries — cheap).
        # A full tocsr() would create _LARGE+1 row CSR (17 GB indptr),
        # so verify the int64 chain via tocsc→tocoo instead.
        data = cupy.ones((1, 2))
        offsets = cupy.array([0], dtype=cupy.int32)
        m = sparse.dia_matrix(
            (data, offsets), shape=(_LARGE + 1, 2))
        coo = m.tocsc().tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.col.dtype == cupy.int64
        assert coo.nnz == 2
        assert int(coo.row[0]) == 0
        assert int(coo.row[1]) == 1

    def test_dia_tocsc_large_shape(self):
        # Shape (_LARGE+1, 2): tocsc() produces a 2-column CSC
        # (indptr has 3 entries — cheap) while still exercising the
        # int64 kernel path (idx_dtype chosen by max(shape) > INT32_MAX).
        data = cupy.ones((1, 2))
        offsets = cupy.array([0], dtype=cupy.int32)
        m = sparse.dia_matrix(
            (data, offsets), shape=(_LARGE + 1, 2))
        csc = m.tocsc()
        assert csc.indices.dtype == cupy.int64
        assert csc.indptr.dtype == cupy.int64
        assert csc.nnz == 2


class TestInt64CoosortCanonicalSkip:
    """coosort skips lexsort for canonical int64 COO (row-sorted).

    tocsr() calls coosort(x, 'r') after sum_duplicates().  If the
    COO is already canonical, coosort can skip the O(nnz log nnz)
    lexsort entirely.
    """

    def test_canonical_coo_tocsr_preserves_data(self):
        # Canonical COO → tocsr should produce correct CSR.
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 1], dtype=cupy.int64)
        col = cupy.array([_LARGE, 0, _LARGE], dtype=cupy.int64)
        m = sparse.coo_matrix._from_parts(
            data, row, col, (2, _LARGE + 1))
        m.has_canonical_format = True
        csr = m.tocsr()
        assert csr.nnz == 3
        assert float(csr[0, _LARGE]) == pytest.approx(1.0)
        assert float(csr[1, 0]) == pytest.approx(2.0)
        assert float(csr[1, _LARGE]) == pytest.approx(3.0)


class TestInt64DiaNnz:
    """DIA getnnz (nnz property) with int64 offsets."""

    def test_dia_nnz_large_shape(self):
        data = cupy.ones((1, 2))
        offsets = cupy.array([0], dtype=cupy.int32)
        m = sparse.dia_matrix(
            (data, offsets), shape=(2, _LARGE + 1))
        assert m.nnz == 2


class TestInt64RandomSparse:
    """random() with large shapes uses O(k) memory, not O(m*n)."""

    def test_random_large_shape_int64(self):
        m = sparse.random(_LARGE, 2, density=1e-9, format='coo')
        assert m.row.dtype == cupy.int64
        assert m.shape == (_LARGE, 2)

    def test_random_large_mn(self):
        # m*n = 10^12 — old code would OOM allocating 8 TB
        m = sparse.random(10**6, 10**6, density=1e-9, format='coo')
        assert m.nnz > 0
        assert m.shape == (10**6, 10**6)

    def test_random_small_shape_int32(self):
        # Small shapes must still produce int32 indices
        m = sparse.random(100, 200, density=0.1, format='coo')
        assert m.row.dtype == cupy.int32
        assert m.col.dtype == cupy.int32

    def test_random_seed_reproducibility(self):
        rs1 = cupy.random.RandomState(42)
        rs2 = cupy.random.RandomState(42)
        m1 = sparse.random(50, 50, density=0.1, random_state=rs1)
        m2 = sparse.random(50, 50, density=0.1, random_state=rs2)
        cupy.testing.assert_array_equal(m1.toarray(), m2.toarray())

    def test_random_density_zero(self):
        m = sparse.random(100, 100, density=0.0, format='coo')
        assert m.nnz == 0

    def test_random_formats(self):
        for fmt in ('coo', 'csr', 'csc'):
            m = sparse.random(50, 50, density=0.1, format=fmt)
            assert m.format == fmt
            assert m.nnz > 0

    def test_random_large_shape_csr(self):
        # COO uses int64 (flat mn > INT32_MAX), but CSR indices/indptr
        # are int32 because the individual dimensions fit in int32.
        m = sparse.random(10**6, 10**6, density=1e-9, format='csr')
        assert m.indices.dtype == cupy.int32
        assert m.indptr.dtype == cupy.int32
        assert m.nnz > 0


# ===================================================================
# Regression tests for CUDA 13.0 followup and deep-review fixes.
#
# Each test demonstrates a specific bug that existed before the fix.
# They should fail if the corresponding fix is reverted.
# ===================================================================


def _small_int64_csr(values=None, shape=(3, 3)):
    """Diagonal CSR with int64 indices whose values fit in int32."""
    if values is None:
        values = [1., 2., 3.]
    n = len(values)
    return sparse.csr_matrix._from_parts(
        cupy.array(values, dtype='f'),
        cupy.array(list(range(n)), dtype=cupy.int64),
        cupy.array(list(range(n + 1)), dtype=cupy.int64),
        shape=shape,
        has_canonical_format=True,
        has_sorted_indices=True)


class TestInt64FollowupOpsPreserveDtype:
    """Operations that used the public constructor (which calls
    check_contents=True) would silently downcast int64 indices to
    int32 when values fit.  Fixed by using _from_parts."""

    def test_multiply_scalar_truediv_csr_dense(self):
        m = _small_int64_csr()
        for r in [m * 2.0, m / 2.0, m.multiply(m),
                  m.multiply(cupy.ones((3, 3), dtype='f'))]:
            assert r.indices.dtype == cupy.int64

    def test_maximum_minimum(self):
        m = _small_int64_csr([1., 5., 3.])
        # Sparse paths: maximum(<=0), minimum(>=0).
        assert m.maximum(-1.0).indices.dtype == cupy.int64
        assert m.minimum(10.0).indices.dtype == cupy.int64
        # Original fix broke float32 → float64 promotion.
        assert m.maximum(-0.5).dtype == numpy.float32

    def test_setdiag(self):
        m = _small_int64_csr()
        m.setdiag(cupy.array([10., 20., 30.], dtype='f'))
        assert m.indices.dtype == cupy.int64

    def test_spgemm(self):
        a = sparse.random(50, 50, density=0.1, format='csr')
        a = sparse.csr_matrix._from_parts(
            a.data, a.indices.astype(cupy.int64),
            a.indptr.astype(cupy.int64), a.shape,
            has_canonical_format=True,
            has_sorted_indices=True)
        assert (a @ a).indices.dtype == cupy.int64

    def test_hstack_bmat(self):
        a = _small_int64_csr([1., 2.], shape=(2, 2))
        # hstack goes through bmat's slow COO path, which converted
        # inputs via the public COO constructor (downcasting int64).
        h = sparse.hstack([a, a]).tocsr()
        assert h.indices.dtype == cupy.int64
        b = sparse.bmat([[a, a], [a, a]]).tocsr()
        assert b.indices.dtype == cupy.int64
        # int32 inputs must stay int32 (no spurious upgrade).
        a32 = sparse.csr_matrix(cupy.eye(2))
        h32 = sparse.hstack([a32, a32])
        idx = h32.row if hasattr(h32, 'row') else h32.indices
        assert idx.dtype == cupy.int32

    def test_bmat_with_dense_blocks(self):
        # bmat's blocks_flat was stale after COO conversion, causing
        # AttributeError: 'ndarray' has no attribute 'nnz'.
        d = cupy.eye(2, dtype='f')
        a = sparse.csr_matrix(cupy.eye(2, dtype='f'))
        r = sparse.bmat([[a, d]]).tocsr()
        assert r.shape == (2, 4)
        assert r.nnz == 4


class TestInt64FollowupScalarComparison:
    """Scalar comparison created a (1,1) CSR and broadcast it to the
    full shape via binopt_csr, allocating O(m*n) temporaries.  The
    fast path filters stored entries directly when op(0, scalar) is
    False."""

    def test_fast_path_correctness(self):
        m = _small_int64_csr([1., 5., 3.])
        assert (m != 0).nnz == 3
        assert (m == 5).nnz == 1
        assert (m > 2).nnz == 2

    def test_matches_scipy(self):
        import scipy.sparse
        dense = numpy.array([[1., 0., 3.], [0., -2., 0.]])
        sp_m = scipy.sparse.csr_matrix(dense)
        cp_m = sparse.csr_matrix(cupy.array(dense))
        for scalar in [0, 1, -1]:
            for op in ['__eq__', '__ne__', '__gt__', '__lt__']:
                sp_r = getattr(sp_m, op)(scalar).toarray()
                cp_r = getattr(cp_m, op)(scalar).toarray().get()
                assert numpy.array_equal(sp_r, cp_r), \
                    f'{op}({scalar}) mismatch'

    def test_large_shape_no_oom(self):
        # Without the fast path these would each allocate ~34 GB.
        m = sparse.csr_matrix._from_parts(
            cupy.array([5.0]),
            cupy.array([_LARGE], dtype=cupy.int64),
            cupy.array([0, 1, 1], dtype=cupy.int64),
            (2, _LARGE + 2), has_canonical_format=True)
        assert (m != 0).nnz == 1
        assert (m > 3).nnz == 1
        assert (m == 1.0).nnz == 0


class TestInt64FollowupSlicingFlags:
    """_major_slice did not copy indptr on copy=True and lost
    sort/canonical flags.  _minor_index_fancy_sorted did not set
    has_canonical_format."""

    def test_major_slice_copy_indptr(self):
        m = _small_int64_csr()
        s = m[0:1, :].copy()
        assert s.indptr.data.ptr != m.indptr.data.ptr

    def test_major_slice_propagates_flags(self):
        m = _small_int64_csr()
        s = m[0:1, :]
        assert s._has_sorted_indices is True
        assert s._has_canonical_format is True

    def test_fancy_col_values_correct(self):
        m = _small_int64_csr()
        r = m[:, cupy.array([0, 2])]
        # Values must be correct regardless of sort order
        cupy.testing.assert_array_equal(
            r.toarray(),
            m.toarray()[:, [0, 2]])

    def test_empty_fancy_col_preserves_int64(self):
        m = _small_int64_csr(shape=(3, 5))
        r = m[:, cupy.array([3, 4])]
        assert r.indices.dtype == cupy.int64
        assert r.nnz == 0


class TestInt64FollowupCumsum:
    """cupy.cumsum works for arrays > 2^31 elements after #9867."""

    def test_cumsum_int64_basic(self):
        for dtype in [cupy.int32, cupy.int64]:
            a = cupy.array([0, 1, 0, 2, 0, 3], dtype=dtype)
            cupy.cumsum(a, out=a)
            cupy.testing.assert_array_equal(
                a, cupy.array([0, 1, 1, 3, 3, 6]))

    def test_coo_sum_duplicates_uses_safe_cumsum(self):
        data = cupy.array([1., 2., 3., 4.])
        row = cupy.array([0, 0, 1, 1], dtype=cupy.int64)
        col = cupy.array([0, 0, 1, 1], dtype=cupy.int64)
        m = sparse.coo_matrix._from_parts(data, row, col, (2, 2))
        m.sum_duplicates()
        assert m.nnz == 2
        cupy.testing.assert_array_equal(
            m.toarray(), cupy.array([[3., 0.], [0., 7.]]))


class TestInt64FollowupAssertToValueError:
    """Bare asserts in cusparse.py would be hidden by python -O.
    Now they raise ValueError/TypeError."""

    def test_raises_valueerror_not_assertionerror(self):
        m = sparse.csr_matrix(cupy.eye(2, dtype=cupy.float64))
        m._has_canonical_format = False
        with pytest.raises(ValueError, match='canonical'):
            cusparse.spmv(m, cupy.ones(2, dtype=cupy.float64))
        m._has_canonical_format = True
        with pytest.raises(ValueError, match='2-D'):
            cusparse.spmm(m, cupy.ones(2, dtype=cupy.float64))


# ===================================================================
# Int64 coverage: atomic ops, fallback paths, operations
# ===================================================================

def _make_int64_csr(m, n, density=0.2, dtype=numpy.float64):
    a = sparse.random(m, n, density=density, format='csr', dtype=dtype)
    return sparse.csr_matrix._from_parts(
        a.data, a.indices.astype(cupy.int64),
        a.indptr.astype(cupy.int64), a.shape)


def _make_int64_csc(m, n, density=0.2, dtype=numpy.float64):
    a = sparse.random(m, n, density=density, format='csc', dtype=dtype)
    return sparse.csc_matrix._from_parts(
        a.data, a.indices.astype(cupy.int64),
        a.indptr.astype(cupy.int64), a.shape)


def _make_int64_coo(m, n, density=0.2, dtype=numpy.float64):
    a = sparse.random(m, n, density=density, format='coo', dtype=dtype)
    return sparse.coo_matrix._from_parts(
        a.data, a.row.astype(cupy.int64),
        a.col.astype(cupy.int64), a.shape)


class TestInt64AtomicOps:
    """Regression: add.at/maximum.at/minimum.at int64 type guards."""

    def test_scatter_ops_int64(self):
        idx = cupy.array([0, 1, 2, 0], dtype=cupy.int32)

        # add.at (atomics.cuh provides long long overload)
        x = cupy.zeros(5, dtype=cupy.int64)
        cupy.add.at(x, idx, cupy.int64(1))
        assert int(x[0]) == 2 and int(x[1]) == 1

        # add.at with large values beyond int32 range
        x2 = cupy.zeros(3, dtype=cupy.int64)
        big = numpy.int64(2**40)
        cupy.add.at(x2, cupy.array([0, 0], dtype=cupy.int32),
                    cupy.int64(big))
        assert int(x2[0]) == 2 * big

        # maximum.at (CUDA native atomicMax(long long*) since sm_35)
        y = cupy.zeros(5, dtype=cupy.int64)
        cupy.maximum.at(y, idx[:3], cupy.int64(5))
        cupy.testing.assert_array_equal(y[:3], cupy.full(3, 5, cupy.int64))

        # minimum.at (CUDA native atomicMin(long long*) since sm_35)
        z = cupy.full(5, 100, dtype=cupy.int64)
        cupy.minimum.at(z, idx[:3], cupy.int64(-1))
        cupy.testing.assert_array_equal(z[:3], cupy.full(3, -1, cupy.int64))


class TestInt64ScalarComparison:
    """Scalar comparison paths in _csr.py:_comparison."""

    def test_fast_path(self):
        """op(0, scalar) is False: sparse result via mask filtering."""
        m = _make_int64_csr(20, 30, density=0.3)
        ref = m.toarray()

        # mask.all() branch: random values in (0,1), all > 0
        r = m > 0
        assert r.indices.dtype == cupy.int64
        cupy.testing.assert_array_equal(r.toarray(), ref > 0)

        # partial mask branch: only some entries pass
        threshold = float(cupy.median(m.data))
        r2 = m > threshold
        cupy.testing.assert_array_equal(r2.toarray(), ref > threshold)

    def test_slow_path_and_nan(self):
        """op(0, scalar) is True: dense result. Also NaN edge cases."""
        m = _make_int64_csr(10, 10, density=0.2)
        ref = m.toarray()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
            # < 1.0: op(0, 1.0) is True → dense expansion
            r = m < 1.0
            assert r.indices.dtype == cupy.int64
            cupy.testing.assert_array_equal(r.toarray(), ref < 1.0)
            # != 0: slow path
            r2 = m != 0
            cupy.testing.assert_array_equal(r2.toarray(), ref != 0)
            # NaN: != returns all-True, == returns all-False
            assert (m != numpy.nan).nnz == 100
        assert (m == numpy.nan).nnz == 0


class TestInt64EliminateZerosInt64Path:
    """eliminate_zeros int64 fallback path in _csr.py."""

    def test_eliminate_zeros(self):
        m = _make_int64_csr(50, 50, density=0.3)
        ref_nnz = m.nnz
        # Partial zeros
        m.data[:10] = 0
        m.eliminate_zeros()
        assert m.nnz == ref_nnz - 10
        assert m.indices.dtype == cupy.int64
        assert (m.data != 0).all()
        # All zeros
        m.data[:] = 0
        m.eliminate_zeros()
        assert m.nnz == 0
        assert m.indptr.dtype == cupy.int64
        # No zeros (no-op)
        m2 = _make_int64_csr(20, 20, density=0.2)
        old_nnz = m2.nnz
        m2.eliminate_zeros()
        assert m2.nnz == old_nnz


class TestInt64CsrgeamFallback:
    """CSR addition fallback (_cupy_csrgeam_int64)."""

    def test_addition_and_mixed_dtypes(self):
        a = _make_int64_csr(40, 50, density=0.2)
        b = _make_int64_csr(40, 50, density=0.2)
        # int64 + int64
        cupy.testing.assert_allclose((a + b).toarray(),
                                     a.toarray() + b.toarray())
        assert (a + b).indices.dtype == cupy.int64
        # int64 - int64
        cupy.testing.assert_allclose((a - b).toarray(),
                                     a.toarray() - b.toarray())
        # mixed int32 + int64 → int64
        a32 = sparse.random(40, 50, density=0.2, format='csr')
        c = a32 + b
        assert c.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(c.toarray(),
                                     a32.toarray() + b.toarray())


class TestInt64TransposeCompressed:
    """Pure-CuPy CSR↔CSC transpose for int64."""

    def test_csr_csc_roundtrip(self):
        a = _make_int64_csr(40, 60, density=0.2)
        ref = a.toarray()
        # CSR → CSC
        csc = a.tocsc()
        assert csc.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(csc.toarray(), ref)
        # CSC → CSR roundtrip
        csr2 = csc.tocsr()
        assert csr2.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(csr2.toarray(), ref)
        # Empty matrix
        e = sparse.csr_matrix._from_parts(
            cupy.empty(0, dtype=cupy.float64),
            cupy.empty(0, dtype=cupy.int64),
            cupy.zeros(11, dtype=cupy.int64), (10, 20))
        assert e.tocsc().nnz == 0
        assert e.tocsc().indices.dtype == cupy.int64


class TestInt64SpgemmFallback:
    """Pure-CuPy sort-merge SpGEMM fallback."""

    def test_fallback_vs_native(self):
        """Fallback gives same result as native cuSPARSE, with alpha."""
        a = _make_int64_csr(30, 30, density=0.3)
        b = _make_int64_csr(30, 30, density=0.3)
        ref = a.toarray() @ b.toarray()
        # Basic
        c = cusparse._cupy_spgemm_int64(a, b, alpha=1)
        assert c.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(c.toarray(), ref, atol=1e-6)
        # With alpha scaling
        c3 = cusparse._cupy_spgemm_int64(a, b, alpha=3.0)
        cupy.testing.assert_allclose(c3.toarray(), 3.0 * ref, atol=1e-6)
        # Matches native dispatch
        native = a @ b
        cupy.testing.assert_allclose(c.toarray(), native.toarray(),
                                     atol=1e-10)

    def test_empty_result(self):
        """No overlapping column/row → zero products."""
        a = sparse.csr_matrix._from_parts(
            cupy.array([1.0]), cupy.array([0], dtype=cupy.int64),
            cupy.array([0, 1, 1], dtype=cupy.int64), (2, 3))
        b = sparse.csr_matrix._from_parts(
            cupy.array([1.0]), cupy.array([0], dtype=cupy.int64),
            cupy.array([0, 0, 0, 1], dtype=cupy.int64), (3, 2))
        assert cusparse._cupy_spgemm_int64(a, b, alpha=1).nnz == 0


class TestInt64SortFunctions:
    """csrsort, cscsort, coosort with int64 indices."""

    def _shuffle_compressed(self, m):
        for i in range(m.indptr.size - 1):
            s, e = int(m.indptr[i]), int(m.indptr[i+1])
            if e - s > 1:
                perm = cupy.random.permutation(e - s) + s
                m.indices[s:e] = m.indices[perm]
                m.data[s:e] = m.data[perm]

    def test_sort_all_formats(self):
        # CSR
        m = _make_int64_csr(30, 40, density=0.3)
        ref = m.toarray()
        self._shuffle_compressed(m)
        cusparse.csrsort(m)
        cupy.testing.assert_allclose(m.toarray(), ref)
        # CSC
        m2 = _make_int64_csc(40, 30, density=0.3)
        ref2 = m2.toarray()
        self._shuffle_compressed(m2)
        cusparse.cscsort(m2)
        cupy.testing.assert_allclose(m2.toarray(), ref2)
        # COO
        m3 = _make_int64_coo(30, 30, density=0.3)
        ref3 = m3.toarray()
        perm = cupy.random.permutation(m3.nnz)
        m3.row[:] = m3.row[perm]
        m3.col[:] = m3.col[perm]
        m3.data[:] = m3.data[perm]
        cusparse.coosort(m3)
        cupy.testing.assert_allclose(m3.toarray(), ref3)


class TestInt64RealImagFormats:
    """.real/.imag preserve int64 indices across formats and dtypes."""

    def test_real_imag(self):
        # CSR complex
        data = cupy.array([1+2j, 3+4j, 5+0j], dtype=cupy.complex128)
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        m = sparse.csr_matrix._from_parts(data, indices, indptr, (3, 3))
        assert m.real.indices.dtype == cupy.int64
        assert m.imag.indices.dtype == cupy.int64
        cupy.testing.assert_array_equal(m.real.data, cupy.array([1, 3, 5]))
        cupy.testing.assert_array_equal(m.imag.data, cupy.array([2, 4, 0]))
        # CSR float: .real is identity
        m2 = _make_int64_csr(10, 10, density=0.3)
        assert m2.real.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(m2.real.toarray(), m2.toarray())
        # COO complex
        coo = sparse.coo_matrix._from_parts(
            cupy.array([1+2j], dtype=cupy.complex64),
            cupy.array([0], dtype=cupy.int64),
            cupy.array([1], dtype=cupy.int64), (2, 2))
        assert coo.real.row.dtype == cupy.int64
        assert coo.real.dtype == cupy.float32


class TestInt64Conversions:
    """Format conversions preserve int64."""

    def test_conversions(self):
        # COO → CSR
        coo = _make_int64_coo(30, 40, density=0.3)
        ref = coo.toarray()
        csr = coo.tocsr()
        assert csr.indptr.dtype == cupy.int64
        cupy.testing.assert_allclose(csr.toarray(), ref)
        # CSR → COO
        coo2 = csr.tocoo()
        assert coo2.row.dtype == cupy.int64
        cupy.testing.assert_allclose(coo2.toarray(), ref)
        # CSC → COO
        csc = _make_int64_csc(30, 40, density=0.3)
        coo3 = csc.tocoo()
        assert coo3.row.dtype == cupy.int64
        cupy.testing.assert_allclose(coo3.toarray(), csc.toarray())


class TestInt64Bmat:
    """bmat preserves int64 with sparse, mixed, and dense blocks."""

    def test_bmat(self):
        a = _make_int64_csr(10, 10, density=0.2)
        b = _make_int64_csr(10, 15, density=0.2)
        # Horizontal concat
        c = sparse.bmat([[a, b]])
        assert c.row.dtype == cupy.int64
        cupy.testing.assert_allclose(
            c.toarray(),
            cupy.concatenate([a.toarray(), b.toarray()], axis=1))
        # Mixed int32 + int64 → int64
        a32 = sparse.random(10, 10, density=0.2, format='csr')
        assert sparse.bmat([[a32, b]]).row.dtype == cupy.int64
        # Dense block alongside sparse int64
        d = cupy.eye(10, dtype=cupy.float64)
        r = sparse.bmat([[a, d], [d, a]])
        assert r.shape == (20, 20)
        ref = cupy.concatenate([
            cupy.concatenate([a.toarray(), d], axis=1),
            cupy.concatenate([d, a.toarray()], axis=1),
        ], axis=0)
        cupy.testing.assert_allclose(r.toarray(), ref)


class TestInt64Helpers:
    """cusparse.py helpers and int32-only guards."""

    def test_helpers_and_guards(self):
        # _check_int32_indices raises on int64
        m = _make_int64_csr(5, 5, density=0.5)
        with pytest.raises(ValueError, match='int32-only'):
            cusparse._check_int32_indices(m, 'test_func')

        # _with_indices_dtype: upcast int32→int64
        m32 = sparse.random(10, 10, density=0.3, format='csr')
        m64 = cusparse._with_indices_dtype(m32, cupy.int64)
        assert m64.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(m64.toarray(), m32.toarray())

        # _with_indices_dtype: no-op when already int64
        assert cusparse._with_indices_dtype(m, cupy.int64) is m

        # spsolve rejects int64
        eye64 = _make_int64_csr(10, 10, density=0.5)
        eye64 = eye64 + 5.0 * sparse.eye(10, dtype=cupy.float64,
                                         format='csr')
        eye64.indices = eye64.indices.astype(cupy.int64)
        eye64.indptr = eye64.indptr.astype(cupy.int64)
        with pytest.raises(ValueError, match='int64'):
            from cupyx.scipy.sparse.linalg import spsolve
            spsolve(eye64, cupy.ones(10))


class TestInt64Operations:
    """Int64 index preservation across common sparse operations."""

    def test_setitem(self):
        m = _make_int64_csr(20, 20, density=0.3)
        m[0, 0] = 99.0
        assert m.indices.dtype == cupy.int64
        assert float(m[0, 0]) == 99.0
        # New entry (sparsity structure change)
        m2 = sparse.csr_matrix._from_parts(
            cupy.array([1.0]), cupy.array([0], dtype=cupy.int64),
            cupy.array([0, 1, 1, 1], dtype=cupy.int64), (3, 3))
        m2[1, 1] = 5.0
        assert m2.indices.dtype == cupy.int64
        assert float(m2[1, 1]) == 5.0

    def test_setdiag(self):
        m = _make_int64_csr(20, 20, density=0.2)
        m.setdiag(cupy.ones(20, dtype=cupy.float64))
        assert m.indices.dtype == cupy.int64
        cupy.testing.assert_array_equal(m.diagonal(), cupy.ones(20))

    def test_copy_abs_neg(self):
        m = _make_int64_csr(10, 10, density=0.3)
        m.data[:5] = -m.data[:5]
        for r in (m.copy(), abs(m), -m):
            assert r.indices.dtype == cupy.int64
        cupy.testing.assert_allclose((-m).toarray(), -(m.toarray()))

    def test_slicing_and_fancy_indexing(self):
        m = _make_int64_csr(50, 60, density=0.2)
        ref = m.toarray()
        # Major slice
        r = m[10:30]
        assert r.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(r.toarray(), ref[10:30])
        # Minor slice
        r2 = m[:, 10:30]
        assert r2.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(r2.toarray(), ref[:, 10:30])
        # Fancy row indexing
        idx = [1, 5, 10, 49]
        r3 = m[idx]
        assert r3.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(r3.toarray(), ref[idx])

    def test_multiply(self):
        m = _make_int64_csr(20, 25, density=0.3)
        ref = m.toarray()
        # Scalar
        assert (m * 3.0).indices.dtype == cupy.int64
        cupy.testing.assert_allclose((m * 3.0).toarray(), ref * 3.0)
        # Dense element-wise
        d = cupy.random.random((20, 25))
        r = m.multiply(d)
        assert r.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(r.toarray(), cupy.multiply(ref, d))

    def test_min_max_axis(self):
        m = _make_int64_csr(30, 40, density=0.2)
        ref = cupy.asarray(m.toarray())
        cupy.testing.assert_allclose(m.max(axis=1).toarray(),
                                     ref.max(axis=1, keepdims=True))
        cupy.testing.assert_allclose(m.min(axis=0).toarray(),
                                     ref.min(axis=0, keepdims=True))

    def test_transpose(self):
        m = _make_int64_csr(30, 40, density=0.2)
        t = m.T
        assert isinstance(t, sparse.csc_matrix)
        assert t.indices.dtype == cupy.int64
        cupy.testing.assert_allclose(t.toarray(), m.toarray().T)


class TestInt64Regressions:
    """Regression tests for specific bugs found during int64 work."""

    def test_coo_transpose_canonical(self):
        """COO transpose must NOT propagate has_canonical_format
        (swapping row/col destroys lexicographic order)."""
        m = _make_int64_coo(10, 10, density=0.3)
        m.has_canonical_format = True
        assert not m.T.has_canonical_format

    def test_dia_nnz_empty_data(self):
        """DIA nnz is bounded by the actual data buffer length.

        Matches scipy 1.17 (gh-23055): an empty data array
        (``data.shape[1] == 0``) gives nnz == 0 even when offsets
        indicate diagonals exist.
        """
        data = cupy.array([[]], dtype=cupy.float32)
        offsets = cupy.array([0], dtype=cupy.int32)
        m = sparse.dia_matrix((data, offsets), shape=(3, 4))
        assert m.nnz == 0
        # Non-empty data shorter than the diagonal still works:
        data = cupy.array([[1.0, 2.0]], dtype=cupy.float32)
        m2 = sparse.dia_matrix((data, offsets), shape=(3, 4))
        assert m2.nnz == 2


class TestFromPartsValidation:
    """_from_parts is internal but must enforce its declared contract
    so a buggy caller fails loudly rather than producing a corrupt matrix."""

    def test_rejects_mismatched_index_dtypes(self):
        data = cupy.array([1.0, 2.0])
        with pytest.raises(ValueError, match='same dtype'):
            sparse.csr_matrix._from_parts(
                data,
                cupy.array([0, 1], dtype=cupy.int32),
                cupy.array([0, 1, 2], dtype=cupy.int64),
                (2, 2))

    def test_rejects_canonical_with_unsorted(self):
        data = cupy.array([1.0])
        idx = cupy.array([0], dtype=cupy.int32)
        ptr = cupy.array([0, 1, 1], dtype=cupy.int32)
        with pytest.raises(ValueError, match='canonical'):
            sparse.csr_matrix._from_parts(
                data, idx, ptr, (2, 2),
                has_canonical_format=True,
                has_sorted_indices=False)

    def test_rejects_data_indices_length_mismatch(self):
        with pytest.raises(ValueError, match='same length'):
            sparse.csr_matrix._from_parts(
                cupy.array([1.0, 2.0]),
                cupy.array([0], dtype=cupy.int32),
                cupy.array([0, 1], dtype=cupy.int32),
                (1, 2))

    def test_rejects_indptr_length_mismatch_csr(self):
        # shape[0]=3 requires len(indptr)=4; we pass len(indptr)=2.
        with pytest.raises(ValueError, match='major axis'):
            sparse.csr_matrix._from_parts(
                cupy.array([1.0]),
                cupy.array([0], dtype=cupy.int32),
                cupy.array([0, 1], dtype=cupy.int32),
                (3, 2))

    def test_rejects_indptr_length_mismatch_csc(self):
        # CSC major axis is shape[1]; we pass len(indptr)=2 instead of 4.
        with pytest.raises(ValueError, match='major axis'):
            sparse.csc_matrix._from_parts(
                cupy.array([1.0]),
                cupy.array([0], dtype=cupy.int32),
                cupy.array([0, 1], dtype=cupy.int32),
                (2, 3))

    def test_canonical_implies_sorted(self):
        data = cupy.array([1.0])
        idx = cupy.array([0], dtype=cupy.int32)
        ptr = cupy.array([0, 1, 1], dtype=cupy.int32)
        # has_canonical_format=True alone should also set _has_sorted_indices.
        m = sparse.csr_matrix._from_parts(
            data, idx, ptr, (2, 2), has_canonical_format=True)
        assert m._has_canonical_format is True
        assert m._has_sorted_indices is True


class TestBmatIndexDtypeDetection:
    """bmat must detect int64 from any input format, including DIA
    (which stores its index information in .offsets, not .indices/.row)."""

    def test_bmat_detects_int64_dia_offsets(self):
        # The DIA constructor downcasts to int32 when shape fits int32
        # (matching scipy), so we force int64 offsets via direct
        # attribute assignment to simulate a DIA whose int64 dtype
        # cannot be inferred from shape alone.  This is the only way
        # to test the _has_int64 detection branch in isolation: with a
        # shape > INT32_MAX, the result would be int64 regardless of
        # whether bmat detects the DIA's offsets dtype, because
        # get_index_dtype(maxval=max(shape)-1) would also return int64.
        data = cupy.ones((1, 3), dtype=cupy.float64)
        offsets = cupy.array([0], dtype=cupy.int32)
        d = sparse.dia_matrix((data, offsets), shape=(3, 3))
        d.offsets = cupy.array([0], dtype=cupy.int64)
        csr = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        assert csr.indices.dtype == cupy.int32
        result = sparse.bmat([[d, csr]])
        # The int64 from DIA's offsets should propagate.
        assert result.row.dtype == cupy.int64
        assert result.col.dtype == cupy.int64

    @testing.slow
    def test_bmat_int64_dia_natural_large_shape(self):
        # End-to-end check with a legitimately int64 DIA (shape large
        # enough that the constructor preserves int64 offsets without
        # any force-assign).  data shape (1, 1) keeps the input small,
        # but bmat routes through CSC -> COO via ``_indptr_to_coo``,
        # which materialises an ``arange(num_rows)`` of int64 — that's
        # ~17 GB for shape ``(2**31 + 1, 1)``.  Marked ``slow`` so the
        # default CI run excludes it; OOM-skip for hosts under 17 GB.
        data = cupy.ones((1, 1), dtype=cupy.float64)
        offsets = cupy.array([0], dtype=cupy.int32)
        d = sparse.dia_matrix((data, offsets), shape=(2**31 + 1, 1))
        assert d.offsets.dtype == cupy.int64
        try:
            result = sparse.bmat([[d]])
        except cupy.cuda.memory.OutOfMemoryError:
            pytest.skip('not enough GPU memory for arange(2**31+1)')
        assert result.row.dtype == cupy.int64
        assert result.shape == (2**31 + 1, 1)
        assert result.nnz == 1

    def test_bmat_all_int32_stays_int32(self):
        csr = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        result = sparse.bmat([[csr, csr]])
        assert result.row.dtype == cupy.int32


class TestAddAtNegativeInt64:
    """cupy.add.at / maximum.at / minimum.at on signed int64 arrays.

    CUDA's atomicAdd has no native long-long overload, but CuPy's
    atomics.cuh provides one via reinterpret_cast (bit-exact for
    two's-complement addition).  atomicMax/atomicMin for signed int64
    are provided natively by CUDA on sm_50+."""

    def test_add_at_negative_int64(self):
        v = cupy.array([0, -100, 50], dtype=cupy.int64)
        idx = cupy.array([0, 1, 2, 0, 1])
        vals = cupy.array([-5, -10, 100, 3, 200], dtype=cupy.int64)
        cupy.add.at(v, idx, vals)
        cupy.testing.assert_array_equal(
            v, cupy.array([-5 + 3, -100 - 10 + 200, 50 + 100],
                          dtype=cupy.int64))

    def test_maximum_at_negative_int64(self):
        v = cupy.array([10, -5, 100, -1000], dtype=cupy.int64)
        idx = cupy.array([0, 1, 2, 3, 0, 1, 2, 3])
        vals = cupy.array([20, -100, -50, -2000, -50, 5, 1000, 0],
                          dtype=cupy.int64)
        cupy.maximum.at(v, idx, vals)
        cupy.testing.assert_array_equal(
            v, cupy.array([20, 5, 1000, 0], dtype=cupy.int64))

    def test_minimum_at_negative_int64(self):
        v = cupy.array([10, -5, 100, -1000], dtype=cupy.int64)
        idx = cupy.array([0, 1, 2, 3, 0, 1, 2, 3])
        vals = cupy.array([20, -100, -50, -2000, -50, 5, 1000, 0],
                          dtype=cupy.int64)
        cupy.minimum.at(v, idx, vals)
        cupy.testing.assert_array_equal(
            v, cupy.array([-50, -100, -50, -2000], dtype=cupy.int64))


class TestSpsolveTriangularGuard:
    """spsolve_triangular must raise a clear error (with the user-facing
    function name) when given int64 indices on the csrsm2 path."""

    def test_int64_csr_raises_on_old_cuda(self):
        # The csrsm2 path is only reached when spsm is unavailable.  On
        # CUDA 12+ spsm is always available and supports int64, so this
        # test is a no-op in practice.  We still verify the dispatch by
        # forcing csrsm2 routing only when spsm is unavailable.
        if cusparse.check_availability('spsm'):
            pytest.skip('spsm is available; csrsm2 path not exercised')
        from cupyx.scipy.sparse.linalg import spsolve_triangular
        idx = cupy.array([0], dtype=cupy.int64)
        ptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        a = sparse.csr_matrix._from_parts(
            cupy.array([1.0]), idx, ptr, (2, 2),
            has_canonical_format=True)
        b = cupy.array([1.0, 2.0])
        with pytest.raises(ValueError, match='spsolve_triangular'):
            spsolve_triangular(a, b)


class TestDiaGetnnzAccumulator:
    """DIA getnnz uses an int64 accumulator so the sum across diagonals
    cannot overflow even when offsets dtype is int32."""

    def test_getnnz_with_int32_offsets(self):
        # Small DIA, smoke test that the int64-accumulator change didn't
        # break the int32-offsets path.
        data = cupy.ones((3, 5), dtype=cupy.float64)
        offsets = cupy.array([-1, 0, 1], dtype=cupy.int32)
        m = sparse.dia_matrix((data, offsets), shape=(5, 5))
        assert isinstance(m.getnnz(), int)
        assert m.getnnz() == 4 + 5 + 4

    def test_getnnz_returns_python_int(self):
        # Ensure type(int) regardless of the kernel output dtype.
        data = cupy.ones((1, 3), dtype=cupy.float64)
        offsets = cupy.array([0], dtype=cupy.int32)
        m = sparse.dia_matrix((data, offsets), shape=(3, 3))
        n = m.getnnz()
        assert type(n) is int
        assert n == 3


class TestIndptrToCooSearchsorted:
    """``_indptr_to_coo`` switches to a searchsorted-based formula
    when the major axis dwarfs ``nnz``.  The motivating case is a
    ``(2, 2**31+5)`` CSC produced by transposing a wide-sparse CSR
    with one stored entry: the ``cupy.repeat(arange(major), ...)``
    formula would otherwise allocate ``arange(2**31+5)`` -- 17 GB.
    """

    def test_repeat_path_for_typical_matrices(self):
        # Below the 16K-row threshold the legacy ``repeat`` path runs
        # without any extra D2H sync.
        indptr = cupy.array([0, 2, 3, 5], dtype=cupy.int64)
        result = cusparse._indptr_to_coo(indptr)
        cupy.testing.assert_array_equal(
            result, cupy.array([0, 0, 1, 2, 2], dtype=cupy.int64))
        assert result.dtype == cupy.int64

    def test_searchsorted_path_for_tall_sparse(self):
        # nrows > 16K and nrows > 4*nnz: triggers searchsorted.
        # Output must match the repeat-formula reference exactly.
        nrows = 2**16
        indptr = cupy.zeros(nrows + 1, dtype=cupy.int64)
        # Three nnz at rows 100, 1000, 50000.
        indptr[101:1001] = 1
        indptr[1001:50001] = 2
        indptr[50001:] = 3
        result = cusparse._indptr_to_coo(indptr)
        cupy.testing.assert_array_equal(
            result, cupy.array([100, 1000, 50000], dtype=cupy.int64))

    def test_searchsorted_path_pathological_tall_csc(self):
        # The motivating case: nnz=1 in a moderately tall matrix.
        BIG_N = 1 << 18  # 256K rows
        indptr = cupy.zeros(BIG_N + 1, dtype=cupy.int64)
        indptr[BIG_N:] = 1  # single nnz at the end
        result = cusparse._indptr_to_coo(indptr)
        assert result.size == 1
        assert int(result[0]) == BIG_N - 1


class TestLsqrInt32Guard:
    """``lsqr`` calls cuSOLVER's ``csrlsvqr`` which is int32-only;
    without an explicit guard, int64 indices were silently
    reinterpreted, producing NaN/Inf garbage.  Mirror the guard
    already present in ``spsolve``.
    """

    def test_int64_csr_raises(self):
        from cupy.cuda import runtime
        if runtime.is_hip:
            pytest.skip('HIP does not support lsqr')
        from cupyx.scipy.sparse.linalg import lsqr
        idx = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        ptr = cupy.array([0, 1, 2, 3, 4], dtype=cupy.int64)
        a = sparse.csr_matrix._from_parts(
            cupy.array([2.0, 2.0, 2.0, 2.0]), idx, ptr, (4, 4))
        b = cupy.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match='lsqr'):
            lsqr(a, b)

    def test_int32_csr_works(self):
        # Regression: the int32 happy path is unchanged.
        from cupy.cuda import runtime
        if runtime.is_hip:
            pytest.skip('HIP does not support lsqr')
        from cupyx.scipy.sparse.linalg import lsqr
        a = sparse.eye(4, format='csr', dtype=cupy.float64) * 2.0
        b = cupy.array([1.0, 2.0, 3.0, 4.0])
        x, *_ = lsqr(a, b)
        cupy.testing.assert_allclose(x, [0.5, 1.0, 1.5, 2.0])


class TestMultiplyMixedInt32Int64:
    """``csr.multiply(csr)`` previously rejected mixed int32/int64
    operands with a kernel template-type-mismatch error
    (``Type is mismatched. B_INDPTR int32 int64 I``).  Promote both
    operands to a common dtype so the kernel template parameter ``I``
    is consistent.
    """

    def test_int32_multiply_int64(self):
        a = sparse.csr_matrix(cupy.array([[1.0, 2.0], [3.0, 4.0]]))
        idx = cupy.array([0, 1], dtype=cupy.int64)
        ptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        b = sparse.csr_matrix._from_parts(
            cupy.array([2.0, 5.0]), idx, ptr, (2, 2))
        c = a.multiply(b)
        assert c.indices.dtype == cupy.int64
        cupy.testing.assert_array_equal(
            c.toarray(), cupy.array([[2.0, 0.0], [0.0, 20.0]]))

    def test_int64_multiply_int32(self):
        idx = cupy.array([0, 1], dtype=cupy.int64)
        ptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        a = sparse.csr_matrix._from_parts(
            cupy.array([2.0, 5.0]), idx, ptr, (2, 2))
        b = sparse.csr_matrix(cupy.array([[1.0, 2.0], [3.0, 4.0]]))
        c = a.multiply(b)
        assert c.indices.dtype == cupy.int64
        cupy.testing.assert_array_equal(
            c.toarray(), cupy.array([[2.0, 0.0], [0.0, 20.0]]))

    def test_int32_only_unchanged(self):
        a = sparse.csr_matrix(cupy.array([[1.0, 2.0], [3.0, 4.0]]))
        b = sparse.csr_matrix(cupy.array([[1.0, 0.0], [0.0, 1.0]]))
        c = a.multiply(b)
        assert c.indices.dtype == cupy.int32
