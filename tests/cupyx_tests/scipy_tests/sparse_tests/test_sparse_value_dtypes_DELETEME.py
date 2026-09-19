"""TEMPORARY -- DELETE BEFORE MERGING.

Extra sparse value-dtype tests carried over from two of the three exploratory
branches (0a and 0b) that developed this feature in parallel; the third
branch's tests are all already in the permanent file.  The permanent,
de-duplicated coverage lives in ``test_sparse_value_dtypes.py``; this file
keeps the *rest* of those branches' tests around so they run in CI and are
available during review as a cross-check.  It is expected to be
**removed before the pull request is merged** -- do not add new tests here and
do not rely on it.  If a test here proves worth keeping, port it into
``test_sparse_value_dtypes.py`` instead.
"""
from __future__ import annotations
import cupyx.scipy.sparse
from cupy import testing
import cupy
import pytest
import numpy

# ==/<=/>= (and out-of-range) comparisons intentionally emit
# SparseEfficiencyWarning; these tests check the RESULT, so silence it here as
# the permanent file does.
pytestmark = pytest.mark.filterwarnings(
    'ignore::cupyx.scipy.sparse.SparseEfficiencyWarning')

# --- cross-branch portability shim (added for cross-pollination review) ---
# The three dtype-support branches (0a/0b/0c) named a few private _sputils
# helpers differently. This shim aliases the MISSING names to whatever the
# branch under test provides, so this (foreign-branch) test file runs on any
# of them. It only ADDS missing attributes; it never overwrites existing ones,
# so it cannot change the branch's own behavior.
from cupyx.scipy.sparse import _sputils as _xsp  # noqa: E402


def _xalias(dst, *srcs):
    if not hasattr(_xsp, dst):
        for s in srcs:
            if hasattr(_xsp, s):
                setattr(_xsp, dst, getattr(_xsp, s))
                return


_xalias("check_data_dtype", "check_sparse_data_dtype")
_xalias("check_sparse_data_dtype", "check_data_dtype")
_xalias("is_bfloat16", "is_extra_float_dtype")
_xalias("is_extra_float_dtype", "is_bfloat16")
if not hasattr(_xsp, "bfloat16"):
    _xbf = getattr(_xsp, "_bfloat16", None)
    if _xbf is None:
        _xbf = next(iter(getattr(_xsp, "_extra_float_dtypes", ()) or ()), None)
    _xsp.bfloat16 = _xbf
# --- end shim ---


try:
    import ml_dtypes
    _bfloat16 = numpy.dtype(ml_dtypes.bfloat16)
except ImportError:
    _bfloat16 = None

# bfloat16 is a CuPy extension available only through the optional
# ``ml_dtypes`` package; sparse support for it is likewise optional.
requires_bfloat16 = pytest.mark.skipif(
    _bfloat16 is None, reason='ml_dtypes (bfloat16) not installed')


_int_dtypes = [
    numpy.int8, numpy.int16, numpy.int32, numpy.int64,
    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
]
_new_dtypes = _int_dtypes + [numpy.float16]
# Representative subset for expensive parametrizations: one sub-32-bit
# signed (no atomicAdd), one 32-bit, one 64-bit, one unsigned, float16.
_key_dtypes = [numpy.int8, numpy.int32, numpy.int64,
               numpy.uint8, numpy.float16]
_scipy_dtypes = [numpy.int8, numpy.int32, numpy.int64, numpy.uint8]
_containers = ['csr_matrix', 'csc_matrix', 'coo_matrix',
               'csr_array', 'csc_array', 'coo_array']


def _make_csr(sp, xp, dtype):
    data = xp.array([1, 2, 3, 4, 5, 6], dtype=dtype)
    indices = xp.array([0, 3, 1, 4, 2, 5], dtype=numpy.int32)
    indptr = xp.array([0, 2, 2, 4, 5, 6], dtype=numpy.int32)
    return sp.csr_matrix((data, indices, indptr), shape=(5, 6))


def test_float16_str_and_print_do_not_crash():
    # scipy has no float16 host kernels, so ``str(self.get())`` raises
    # ValueError; ``__str__`` must degrade to ``repr`` rather than crash
    # ``print()`` / ``format()`` (``repr`` already works without scipy).
    m = cupyx.scipy.sparse.csr_matrix(
        cupy.array([[1, 0, 2], [0, 3, 0]], dtype='float16'))
    assert 'sparse' in str(m)
    assert 'sparse' in '{}'.format(m)
    assert 'sparse' in repr(m)


class TestConstruction:

    @pytest.mark.parametrize('dtype', _new_dtypes)
    def test_from_dense_all_formats(self, dtype):
        d = cupy.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
        for cls in (cupyx.scipy.sparse.csr_matrix,
                    cupyx.scipy.sparse.csc_matrix,
                    cupyx.scipy.sparse.coo_matrix,
                    cupyx.scipy.sparse.csr_array,
                    cupyx.scipy.sparse.csc_array,
                    cupyx.scipy.sparse.coo_array):
            m = cls(d)
            assert m.dtype == dtype
            cupy.testing.assert_array_equal(m.toarray(), d)

    @pytest.mark.parametrize('dtype', _new_dtypes)
    def test_empty_shape_and_copy(self, dtype):
        m = cupyx.scipy.sparse.csr_matrix((3, 4), dtype=dtype)
        assert m.dtype == dtype
        assert m.nnz == 0
        m2 = _make_csr(cupyx.scipy.sparse, cupy, dtype).copy()
        assert m2.dtype == dtype

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_from_coo_2tuple(self, dtype):
        data = cupy.array([2, 7], dtype=dtype)
        row = cupy.array([0, 1], dtype=numpy.int32)
        col = cupy.array([1, 0], dtype=numpy.int32)
        m = cupyx.scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
        assert m.dtype == dtype
        cupy.testing.assert_array_equal(
            m.toarray(), cupy.array([[0, 2], [7, 0]], dtype=dtype))

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_dia_from_parts(self, dtype):
        data = cupy.array([[1, 2, 3]], dtype=dtype)
        offsets = cupy.array([0])
        m = cupyx.scipy.sparse.dia_matrix((data, offsets), shape=(3, 3))
        assert m.dtype == dtype
        cupy.testing.assert_array_equal(
            m.toarray(), cupy.diag(cupy.array([1, 2, 3], dtype=dtype)))
        cupy.testing.assert_array_equal(
            m.tocsr().toarray(), m.toarray())
        assert m.tocsr().dtype == dtype

    @testing.with_requires('scipy')
    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    def test_scipy_roundtrip(self, dtype):
        import scipy.sparse
        d = numpy.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
        s = scipy.sparse.csr_matrix(d)
        m = cupyx.scipy.sparse.csr_matrix(s)
        assert m.dtype == dtype
        numpy.testing.assert_array_equal(m.toarray().get(), d)
        back = m.get()
        assert back.dtype == dtype
        numpy.testing.assert_array_equal(back.toarray(), d)


class TestConversion:

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_format_roundtrips(self, dtype):
        m = _make_csr(cupyx.scipy.sparse, cupy, dtype)
        expected = m.toarray()
        assert expected.dtype == dtype
        for other in (m.tocsc(), m.tocoo(), m.tocsc().tocsr(),
                      m.tocoo().tocsc(), m.T.T):
            assert other.dtype == dtype
            cupy.testing.assert_array_equal(other.toarray(), expected)

    @pytest.mark.parametrize('dtype', _key_dtypes)
    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_toarray_order(self, dtype, order):
        m = _make_csr(cupyx.scipy.sparse, cupy, dtype)
        out = m.toarray(order=order)
        assert out.flags['%s_CONTIGUOUS' % order]
        cupy.testing.assert_array_equal(out, m.toarray())

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_csc_toarray(self, dtype):
        d = cupy.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
        m = cupyx.scipy.sparse.csc_matrix(d)
        cupy.testing.assert_array_equal(m.toarray(), d)

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_toarray_sums_duplicates(self, dtype):
        # Non-canonical input: duplicate entries must sum on densify
        # (the direct-write kernels for 8/16-bit ints canonicalize a
        # copy first).
        data = cupy.array([1, 2, 3], dtype=dtype)
        row = cupy.array([0, 0, 1], dtype=numpy.int32)
        col = cupy.array([1, 1, 0], dtype=numpy.int32)
        m = cupyx.scipy.sparse.coo_matrix(
            (data, (row, col)), shape=(2, 2)).tocsr()
        cupy.testing.assert_array_equal(
            m.toarray(), cupy.array([[0, 3], [3, 0]], dtype=dtype))

    @testing.with_requires('scipy')
    @pytest.mark.parametrize('to', ['tocsr', 'tocsc'])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_unsorted_bool_coo_keeps_data_aligned(self, xp, sp, to):
        # Regression: coosort's cuSPARSE path skipped the bool data
        # gather, so an explicitly stored ``False`` ended up misaligned
        # from its (permuted) coordinates.  With an explicit False at
        # (0, 2), tocsr/tocsc must still match toarray / scipy.
        row = xp.array([2, 0, 1], 'i')
        col = xp.array([0, 2, 1], 'i')
        data = xp.array([True, False, True])
        m = sp.coo_matrix((data, (row, col)), shape=(3, 3))
        return getattr(m, to)().toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_unsorted_coo_convert_matches_toarray(self, dtype):
        row = cupy.array([2, 0, 1], dtype=numpy.int32)
        col = cupy.array([0, 2, 1], dtype=numpy.int32)
        data = cupy.array([1, 2, 3], dtype=dtype)
        m = cupyx.scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 3))
        cupy.testing.assert_array_equal(m.tocsc().toarray(), m.toarray())
        cupy.testing.assert_array_equal(m.tocsr().toarray(), m.toarray())


class TestIndexing:

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_scalar_getitem(self, dtype):
        m = _make_csr(cupyx.scipy.sparse, cupy, dtype)
        assert m[0, 0] == 1
        assert m[0, 0].dtype == dtype
        assert m[1, 0] == 0
        assert m[3, 2] == 5

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_slices(self, dtype):
        m = _make_csr(cupyx.scipy.sparse, cupy, dtype)
        dense = m.toarray()
        for sub, ref in (
                (m[0:2, :], dense[:2, :]),
                (m[:, 1:4], dense[:, 1:4]),
                (m[::2, :], dense[::2, :]),
        ):
            assert sub.dtype == dtype
            cupy.testing.assert_array_equal(sub.toarray(), ref)

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_fancy(self, dtype):
        m = _make_csr(cupyx.scipy.sparse, cupy, dtype)
        dense = m.toarray()
        rows = cupy.array([0, 2, 4])
        cols = cupy.array([0, 3, 5])
        sub_r = m[rows, :]
        sub_c = m[:, cols]
        assert sub_r.dtype == dtype
        assert sub_c.dtype == dtype
        cupy.testing.assert_array_equal(
            sub_r.toarray(), dense[[0, 2, 4], :])
        cupy.testing.assert_array_equal(
            sub_c.toarray(), dense[:, [0, 3, 5]])

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_setitem_and_setdiag(self, dtype):
        m = _make_csr(cupyx.scipy.sparse, cupy, dtype)
        ref = m.toarray()
        with pytest.warns(cupyx.scipy.sparse.SparseEfficiencyWarning):
            m[1, 1] = 9
        ref[1, 1] = 9
        cupy.testing.assert_array_equal(m.toarray(), ref)
        m2 = _make_csr(cupyx.scipy.sparse, cupy, dtype)
        m2.setdiag(cupy.array([7, 8], dtype=dtype))
        ref2 = _make_csr(cupyx.scipy.sparse, cupy, dtype).toarray()
        ref2[0, 0] = 7
        ref2[1, 1] = 8
        cupy.testing.assert_array_equal(m2.toarray(), ref2)


@testing.with_requires('scipy')
class TestArithmeticScipyParity:
    """Value & dtype parity with scipy (dtype checked by the decorator)."""

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add(self, xp, sp, dtype):
        a = _make_csr(sp, xp, dtype)
        b = sp.csr_matrix(xp.eye(5, 6, dtype=dtype))
        return (a + b).toarray()

    @pytest.mark.parametrize('dtype',
                             [numpy.int8, numpy.int32, numpy.int64])
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub(self, xp, sp, dtype):
        a = _make_csr(sp, xp, dtype)
        b = sp.csr_matrix(xp.eye(5, 6, dtype=dtype))
        return (a - b).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub_bool(self, xp, sp):
        # scipy computes bool subtraction in C bool arithmetic
        # (True - True -> False, False - True -> True); numpy raises.
        a = sp.csr_matrix(xp.array([[True, False], [False, True]]))
        b = sp.csr_matrix(xp.array([[True, True], [False, False]]))
        return (a - b).toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_scalar_multiply(self, xp, sp, dtype):
        return (_make_csr(sp, xp, dtype) * 2).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_bool_scalar_multiply_promotes(self, xp, sp):
        # bool * 2 -> int64 under numpy promotion (scipy does data * 2).
        a = sp.csr_matrix(xp.array([[True, False], [False, True]]))
        return (a * 2).toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes + [numpy.uint64])
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_truediv_scalar(self, xp, sp, dtype):
        # int / scalar -> float64 for every integer width (scipy rule).
        return (_make_csr(sp, xp, dtype) / 2).toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_multiply_sparse(self, xp, sp, dtype):
        a = _make_csr(sp, xp, dtype)
        b = sp.csr_matrix(xp.eye(5, 6, dtype=dtype) * 2)
        return a.multiply(b).toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_maximum(self, xp, sp, dtype):
        a = _make_csr(sp, xp, dtype)
        b = sp.csr_matrix(xp.eye(5, 6, dtype=dtype) * 4)
        return a.maximum(b).toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power(self, xp, sp, dtype):
        return _make_csr(sp, xp, dtype).power(2).toarray()

    @pytest.mark.parametrize('dtype',
                             [numpy.int8, numpy.int32, numpy.int64])
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_neg_abs(self, xp, sp, dtype):
        m = _make_csr(sp, xp, dtype)
        return abs(-m).toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_comparison(self, xp, sp, dtype):
        a = _make_csr(sp, xp, dtype)
        b = sp.csr_matrix(xp.eye(5, 6, dtype=dtype) * 3)
        return (a != b).toarray()


@testing.with_requires('scipy')
class TestMatmulScipyParity:

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_spmv(self, xp, sp, dtype):
        m = _make_csr(sp, xp, dtype)
        v = xp.arange(6, dtype=dtype)
        return m @ v

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_spmm(self, xp, sp, dtype):
        # contiguous_check=False: cupy's SpMM returns F-order (as for
        # float dtypes); scipy returns C-order.
        m = _make_csr(sp, xp, dtype)
        b = xp.arange(12, dtype=dtype).reshape(6, 2)
        return m @ b

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_spgemm(self, xp, sp, dtype):
        m = _make_csr(sp, xp, dtype)
        b = sp.csr_matrix(xp.eye(6, 4, dtype=dtype) * 2)
        return (m @ b).toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_csc_matmul(self, xp, sp, dtype):
        m = _make_csr(sp, xp, dtype).tocsc()
        b = sp.csc_matrix(xp.eye(6, 4, dtype=dtype) * 2)
        return (m @ b).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_bool_spgemm(self, xp, sp):
        a = sp.csr_matrix(xp.array([[True, False], [True, True]]))
        return (a @ a).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mixed_int_float(self, xp, sp):
        a = sp.csr_matrix(
            xp.array([[1, 0, 2], [0, 3, 0]], dtype=numpy.int8))
        v = xp.array([1., 2., 3.], dtype=numpy.float32)
        return a @ v

    def test_spgemm_preserves_int32_indices(self):
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[1, 0], [0, 2]], dtype=numpy.int32))
        c = m @ m
        assert c.indices.dtype == numpy.int32
        assert c.indptr.dtype == numpy.int32

    def test_spgemm_int_exact(self):
        # The pure-CuPy integer SpGEMM computes in the integer dtype:
        # values beyond float64's 2**53 mantissa stay exact.
        big = numpy.int64(2) ** 60 + 3
        a = cupyx.scipy.sparse.csr_matrix(
            (cupy.array([big, 1], dtype=numpy.int64),
             cupy.array([0, 1], dtype=numpy.int32),
             cupy.array([0, 1, 2], dtype=numpy.int32)),
            shape=(2, 2))
        eye = cupyx.scipy.sparse.csr_matrix(
            cupy.eye(2, dtype=numpy.int64))
        c = a @ eye
        assert int(c[0, 0]) == int(big)


@testing.with_requires('scipy')
class TestReductionsScipyParity:

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @pytest.mark.parametrize('axis', [None, 0, 1])
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum(self, xp, sp, dtype, axis):
        # int8 -> int64, uint8 -> uint64 accumulators (get_sum_dtype).
        return _make_csr(sp, xp, dtype).sum(axis=axis)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_bool_sum(self, xp, sp):
        # Regression: bool.sum() used to raise TypeError from cuSPARSE.
        a = sp.csr_matrix(xp.array([[True, False], [True, True]]))
        return a.sum()

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_dia_int64_sum_excludes_padding(self, xp, sp):
        # DIA ``data`` holds unused off-matrix diagonal slots; the total
        # sum must count only real entries (route through COO), and the
        # axis sums must stay exact past 2**53.
        big = 2 ** 60
        data = xp.array([[big + 1, big + 2, big + 3],
                         [big + 4, big + 5, big + 6]], dtype=numpy.int64)
        offsets = xp.array([0, -1], 'i')
        m = sp.dia_matrix((data, offsets), shape=(3, 3))
        return (xp.asarray(m.sum()),
                xp.asarray(m.sum(axis=0)).ravel(),
                xp.asarray(m.sum(axis=1)).ravel())

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    @pytest.mark.parametrize('array', [False, True])
    @pytest.mark.parametrize('axis', [None, 0, 1])
    def test_int64_sum_exact_past_2_53(self, fmt, array, axis):
        # Summing int64 through the float64 matmul path rounds past
        # 2**53; the segmented-cumsum path stays exact (scipy parity).
        import scipy.sparse
        big = 2 ** 60
        dense = numpy.array([[big + 7, 0, big + 3],
                             [0, big + 1, big + 5]], dtype=numpy.int64)
        cons = getattr(cupyx.scipy.sparse,
                       f'{fmt}_array' if array else f'{fmt}_matrix')
        scons = getattr(scipy.sparse,
                        f'{fmt}_array' if array else f'{fmt}_matrix')
        m = cons(cupy.asarray(dense))
        sm = scons(dense)
        got = m.sum(axis=axis)
        exp = sm.sum(axis=axis)
        if axis is None:
            assert int(got) == int(exp)
        else:
            numpy.testing.assert_array_equal(
                cupy.asnumpy(got).ravel(), numpy.asarray(exp).ravel())

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @pytest.mark.parametrize('axis', [None, 0, 1])
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean(self, xp, sp, dtype, axis):
        return _make_csr(sp, xp, dtype).mean(axis=axis)

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    def test_max_min_dtype_preserved(self, dtype):
        m = _make_csr(cupyx.scipy.sparse, cupy, dtype)
        assert m.max().dtype == dtype
        assert int(m.max()) == 6
        assert m.min().dtype == dtype
        assert int(m.min()) == 0
        assert m.max(axis=1).dtype == dtype
        assert m.min(axis=0).dtype == dtype

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_argmax_axis(self, xp, sp, dtype):
        return xp.asarray(_make_csr(sp, xp, dtype).argmax(axis=1))

    def test_int64_reductions_exact_past_2_53(self):
        # int64 max/min/argmax must not round through float64.
        big = 2 ** 60
        vals = cupy.array([big + 7, big + 3, -(big + 5)],
                          dtype=numpy.int64)
        m = cupyx.scipy.sparse.csr_matrix(
            (vals, cupy.array([0, 1, 2], dtype=numpy.int32),
             cupy.array([0, 3], dtype=numpy.int32)), shape=(1, 3))
        assert int(m.max()) == big + 7
        assert int(m.min()) == -(big + 5)
        assert int(m.max(axis=1).toarray()[0, 0]) == big + 7
        assert int(m.argmax(axis=1)[0, 0]) == 0
        assert int(m.argmin(axis=1)[0, 0]) == 2

    def test_uint64_max_exact_past_2_53(self):
        big = numpy.uint64(2 ** 63 + 9)
        m = cupyx.scipy.sparse.csr_matrix(
            (cupy.array([big, 1], dtype=numpy.uint64),
             cupy.array([0, 1], dtype=numpy.int32),
             cupy.array([0, 2], dtype=numpy.int32)), shape=(1, 2))
        assert int(m.max()) == int(big)
        assert int(m.max(axis=1).toarray()[0, 0]) == int(big)


class TestSortEliminateDuplicates:

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_sort_indices(self, dtype):
        data = cupy.array([2, 1], dtype=dtype)
        indices = cupy.array([2, 0], dtype=numpy.int32)
        indptr = cupy.array([0, 2], dtype=numpy.int32)
        m = cupyx.scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(1, 3))
        before = m.toarray().copy()
        m.sort_indices()
        assert m.has_sorted_indices
        cupy.testing.assert_array_equal(m.toarray(), before)

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_eliminate_zeros(self, dtype):
        data = cupy.array([1, 0, 3], dtype=dtype)
        indices = cupy.array([0, 1, 2], dtype=numpy.int32)
        indptr = cupy.array([0, 3], dtype=numpy.int32)
        m = cupyx.scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(1, 3))
        m.eliminate_zeros()
        assert m.nnz == 2
        assert m.dtype == dtype

    @pytest.mark.parametrize('dtype', _new_dtypes)
    def test_coo_sum_duplicates(self, dtype):
        data = cupy.array([3, 4, 5], dtype=dtype)
        row = cupy.array([0, 0, 1], dtype=numpy.int32)
        col = cupy.array([0, 0, 1], dtype=numpy.int32)
        m = cupyx.scipy.sparse.coo_matrix(
            (data, (row, col)), shape=(2, 2))
        m.sum_duplicates()
        assert m.dtype == dtype
        assert m.nnz == 2
        assert int(m.toarray()[0, 0]) == 7

    def test_coo_sum_duplicates_int8_wraparound(self):
        # Accumulation matches numpy's modular overflow.
        data = cupy.array([100, 100], dtype=numpy.int8)
        row = cupy.array([0, 0], dtype=numpy.int32)
        col = cupy.array([0, 0], dtype=numpy.int32)
        m = cupyx.scipy.sparse.coo_matrix(
            (data, (row, col)), shape=(1, 1))
        m.sum_duplicates()
        # 100 + 100 = 200 wraps to 200 - 256 = -56 in int8.
        assert int(m.toarray()[0, 0]) == -56


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
        m = cupyx.scipy.sparse.random(30, 40, density=0.1, dtype=dtype)
        assert m.dtype == dtype
        assert m.nnz == 120


class TestSparrayAnd1D:

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_csr_array_elementwise_mul(self, dtype):
        d = cupy.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
        a = cupyx.scipy.sparse.csr_array(d)
        r = a * a
        assert r.dtype == dtype
        cupy.testing.assert_array_equal(r.toarray(), d * d)

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_coo_array_1d(self, dtype):
        v = cupy.array([1, 0, 2, 3], dtype=dtype)
        a = cupyx.scipy.sparse.coo_array(v)
        assert a.dtype == dtype
        assert a.ndim == 1
        cupy.testing.assert_array_equal(a.toarray(), v)
        csr = a.tocsr()
        assert csr.dtype == dtype
        cupy.testing.assert_array_equal(csr.toarray(), v)
        assert int(csr.sum()) == 6
        r = csr + csr
        assert r.dtype == dtype
        cupy.testing.assert_array_equal(r.toarray(), v + v)

    @pytest.mark.parametrize('dtype', _key_dtypes)
    def test_1d_reductions(self, dtype):
        v = cupy.array([1, 0, 2, 3], dtype=dtype)
        a = cupyx.scipy.sparse.coo_array(v).tocsr()
        assert int(a.max()) == 3
        assert a.max().dtype == dtype
        assert int(a.argmax()) == 3


class TestFloat16:
    """float16 is a CuPy extension (scipy rejects it); test natively."""

    def test_scipy_rejects_but_cupy_accepts(self):
        d = [[1, 0], [0, 2]]
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.array(d, dtype=numpy.float16))
        assert m.dtype == numpy.float16

    def test_native_spmv(self):
        d = cupy.array([[1, 0, 2], [0, 3, 0]], dtype=numpy.float16)
        m = cupyx.scipy.sparse.csr_matrix(d)
        v = cupy.array([1, 2, 3], dtype=numpy.float16)
        result = m @ v
        assert result.dtype == numpy.float16
        cupy.testing.assert_allclose(result, d @ v, rtol=1e-2)

    def test_native_spmm(self):
        d = cupy.array([[1, 0, 2], [0, 3, 0]], dtype=numpy.float16)
        rhs = cupy.array([[1, 0], [0, 1], [2, 0]], dtype=numpy.float16)
        m = cupyx.scipy.sparse.csr_matrix(d)
        result = m @ rhs
        assert result.dtype == numpy.float16
        cupy.testing.assert_allclose(result, d @ rhs, rtol=5e-2)

    def test_native_spgemm(self):
        d1 = cupy.array([[1, 0, 2], [0, 3, 0]], dtype=numpy.float16)
        d2 = cupy.array([[1, 0], [0, 1], [2, 0]], dtype=numpy.float16)
        a = cupyx.scipy.sparse.csr_matrix(d1)
        b = cupyx.scipy.sparse.csr_matrix(d2)
        c = a @ b
        assert c.dtype == numpy.float16
        cupy.testing.assert_allclose(c.toarray(), d1 @ d2, rtol=5e-2)

    def test_spgemm_heavy_accumulation_accuracy(self):
        # float16 SpGEMM stages through float32 (accumulates in float32,
        # narrows once), so a long inner contraction stays near float16
        # epsilon.  Uniform-float16 accumulation would compound to ~1e-3.
        rng = numpy.random.default_rng(0)
        k = 400
        a_d = rng.random((6, k)).astype(numpy.float16)
        b_d = rng.random((k, 6)).astype(numpy.float16)
        a = cupyx.scipy.sparse.csr_matrix(cupy.asarray(a_d))
        b = cupyx.scipy.sparse.csr_matrix(cupy.asarray(b_d))
        got = (a @ b).toarray().get().astype(numpy.float64)
        ref = a_d.astype(numpy.float64) @ b_d.astype(numpy.float64)
        rel = numpy.max(numpy.abs(got - ref) / (numpy.abs(ref) + 1e-9))
        assert rel < 5e-3

    def test_toarray_sums_duplicates_via_atomicadd(self):
        # float16 densify uses atomicAdd (no forced canonicalization);
        # duplicate (row, col) entries must sum.
        data = cupy.array([1.5, 2.5, 3.0], dtype=numpy.float16)
        row = cupy.array([0, 0, 1], dtype=numpy.int32)
        col = cupy.array([0, 0, 1], dtype=numpy.int32)
        m = cupyx.scipy.sparse.coo_matrix(
            (data, (row, col)), shape=(2, 2)).tocsr()
        cupy.testing.assert_array_equal(
            m.toarray(), cupy.array([[4.0, 0.0], [0.0, 3.0]],
                                    dtype=numpy.float16))
        cupy.testing.assert_array_equal(
            m.toarray(order='F'), m.toarray(order='C'))

    def test_mixed_f16_f32_promotes(self):
        d = cupy.array([[1, 0, 2], [0, 3, 0]], dtype=numpy.float16)
        m = cupyx.scipy.sparse.csr_matrix(d)
        v = cupy.array([1, 2, 3], dtype=numpy.float32)
        result = m @ v
        assert result.dtype == numpy.float32
        cupy.testing.assert_allclose(result, d.astype('f') @ v, rtol=1e-6)

    def test_add_and_reductions(self):
        d = cupy.array([[1, 0, 2], [0, 3, 0]], dtype=numpy.float16)
        m = cupyx.scipy.sparse.csr_matrix(d)
        r = m + m
        assert r.dtype == numpy.float16
        cupy.testing.assert_array_equal(r.toarray(), d + d)
        assert m.sum().dtype == numpy.float16
        assert float(m.sum()) == 6.0
        assert m.max().dtype == numpy.float16
        assert float(m.max()) == 3.0
        assert m.mean(axis=1).dtype == numpy.float16

    def test_truediv_keeps_float16(self):
        # scipy rejects float16, so dense CuPy is the oracle:
        # ``float16 / scalar -> float16`` (not float64).
        d = cupy.array([[4, 0], [0, 6]], dtype=numpy.float16)
        m = cupyx.scipy.sparse.csr_matrix(d)
        r = m / 2
        assert r.dtype == numpy.float16
        cupy.testing.assert_allclose(r.toarray(), d / 2)


class TestBoolRegressions:

    def test_bool_sum_dtype(self):
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[True, False], [True, True]]))
        s = m.sum()
        assert int(s) == 3
        assert s.dtype == numpy.int64
        s0 = m.sum(axis=0)
        assert s0.dtype == numpy.int64

    def test_bool_matmul_float_vector(self):
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[True, False], [False, True]]))
        v = cupy.array([1.0, 2.0], dtype=numpy.float32)
        cupy.testing.assert_array_equal(m @ v, cupy.array([1.0, 2.0]))

    def test_bool_matmul_bool_vector(self):
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[True, False], [False, True]]))
        v = cupy.array([True, True])
        r = m @ v
        assert r.dtype == numpy.bool_
        cupy.testing.assert_array_equal(r, cupy.array([True, True]))


@testing.with_requires('scipy')
class TestValueSemanticsRegressions:
    """Edge cases where a wrong dtype cast would crash or wrap silently."""

    @pytest.mark.parametrize('dtype', [numpy.uint8, numpy.uint16,
                                       numpy.uint32, numpy.uint64])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_uint_subtraction_wraps(self, xp, sp, dtype):
        # ``uint - uint`` scales the subtrahend by -1; the coefficient
        # must wrap modularly (``-1 -> max``) rather than raise
        # OverflowError under NEP 50.
        a = sp.csr_matrix(xp.array([[1, 0], [0, 5]], dtype=dtype))
        b = sp.csr_matrix(xp.array([[3, 0], [0, 2]], dtype=dtype))
        return (a - b).toarray()

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.uint8, numpy.int64])
    def test_truediv_by_exotic_scalar_falls_back_to_float64(self, dtype):
        # A scalar whose promotion is unstorable (e.g. numpy.longdouble)
        # must fall back to float64 rather than trying to build an
        # unstorable reciprocal dtype (which cupy.reciprocal rejects).
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[4, 0], [0, 6]], dtype=dtype))
        r = m / numpy.longdouble(2)
        assert r.dtype == numpy.float64
        cupy.testing.assert_array_equal(
            r.toarray(), cupy.array([[2.0, 0.0], [0.0, 3.0]]))

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.uint8])
    @pytest.mark.parametrize('op', ['gt', 'lt', 'ge', 'le', 'eq', 'ne'])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_out_of_range_scalar_comparison(self, xp, sp, dtype, op):
        # A scalar outside the matrix dtype's range must compare by
        # magnitude, not wrap into range (int8 vs 300 -> all False for >).
        import operator
        m = sp.csr_matrix(xp.array([[1, 0, 100], [0, 50, 0]], dtype=dtype))
        with numpy.errstate(over='ignore'):
            return getattr(operator, op)(m, 300).toarray()

    def test_bool_power_matches_scipy_per_n(self):
        # numpy/scipy give ``bool ** n`` a per-exponent dtype (int8 at
        # n==2, int64 at n>=3).  bool is a scipy-supported dtype, so scipy
        # is the oracle -- match it exactly at every n (CuPy dense's
        # int64-all would diverge from both scipy and numpy at n==2).
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[True, False], [True, True]]))
        assert m.power(2).dtype == numpy.int8
        assert m.power(3).dtype == numpy.int64
        cupy.testing.assert_array_equal(
            m.power(2).toarray(),
            cupy.array([[1, 0], [1, 1]], dtype=numpy.int8))

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_int_truediv_by_dense_is_float64(self, xp, sp, dtype):
        m = _make_csr(sp, xp, dtype)
        div = (xp.arange(1, 31, dtype=dtype).reshape(5, 6))
        r = m / div
        assert r.dtype == numpy.float64
        return r.toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_mixed_int_float_add(self, xp, sp, dtype):
        # int + float32 promotes and uses the native cuSPARSE path; the
        # decorator checks the promoted dtype matches scipy (float32 for
        # int8/uint8, float64 for int32/int64 under numpy promotion).
        a = _make_csr(sp, xp, dtype)
        b = sp.csr_matrix(xp.eye(5, 6, dtype=numpy.float32))
        return (a + b).toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mixed_int_float_spmv(self, xp, sp, dtype):
        a = _make_csr(sp, xp, dtype)
        v = xp.arange(6, dtype=numpy.float32)
        return a @ v

    @pytest.mark.parametrize('container',
                             ['csr_matrix', 'csc_matrix', 'coo_matrix',
                              'csr_array', 'csc_array', 'coo_array'])
    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.uint8])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_all_containers_roundtrip(self, xp, sp, dtype, container):
        dense = xp.array([[0, 1, 0, 2], [3, 0, 0, 0], [0, 4, 5, 0]],
                         dtype=dtype)
        m = getattr(sp, container)(dense)
        assert m.dtype == dtype
        return m.toarray()


@testing.with_requires('scipy')
class TestMergedApiValueDtypes:
    """New API surface from sparray-step3 must also honor value dtypes:
    ``matrix_power`` / ``**``, 1-D array indexing/reductions/arithmetic,
    and ``count_nonzero(axis)``.  These build on the dtype-general matmul,
    indexing, and reduction primitives, so they should already work --
    these tests lock that in."""

    @pytest.mark.parametrize('dtype', _scipy_dtypes + [numpy.bool_])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_matrix_power(self, xp, sp, dtype):
        from cupyx.scipy.sparse.linalg import matrix_power as cp_mp
        if xp is numpy:
            from scipy.sparse.linalg import matrix_power as mp
        else:
            mp = cp_mp
        d = xp.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=dtype)
        m = sp.csr_matrix(d)
        r = mp(m, 3)
        assert r.dtype == dtype
        return r.toarray()

    @pytest.mark.parametrize('dtype', _scipy_dtypes + [numpy.bool_])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_matrix_pow_operator(self, xp, sp, dtype):
        d = xp.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=dtype)
        return (sp.csr_matrix(d) ** 3).toarray()

    def test_matrix_power_float16(self):
        from cupyx.scipy.sparse.linalg import matrix_power
        d = cupy.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]],
                       dtype=numpy.float16)
        r = matrix_power(cupyx.scipy.sparse.csr_matrix(d), 3)
        assert r.dtype == numpy.float16
        ref = numpy.linalg.matrix_power(d.get().astype('f4'), 3)
        cupy.testing.assert_allclose(r.toarray(), ref, rtol=1e-2)

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @pytest.mark.parametrize('axis', [0, 1])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_count_nonzero_axis(self, xp, sp, dtype, axis):
        # The counts are what matter for value-dtype support; the count
        # array's own int width (int32/int64) is a separate, value-dtype-
        # independent quirk (scipy even differs by axis), so normalize it.
        d = xp.array([[1, 0, 2], [0, 0, 0], [3, 0, 4]], dtype=dtype)
        return xp.asarray(
            sp.csr_matrix(d).count_nonzero(axis=axis)).astype(numpy.int64)

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @pytest.mark.parametrize('reduction',
                             ['sum', 'mean', 'max', 'min', 'argmax'])
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_array_reductions(self, xp, sp, dtype, reduction):
        # 1-D array reductions must match scipy dtype+value (int8.sum ->
        # int64, not an int8 overflow).
        v = sp.csr_array(xp.array([100, 0, 100, 50, 0, 60], dtype=dtype))
        return xp.asarray(getattr(v, reduction)())

    def test_1d_int64_sum_exact_past_2_53(self):
        big = 2 ** 60
        v = cupyx.scipy.sparse.csr_array(
            cupy.array([big + 7, 0, big + 3], dtype=numpy.int64))
        assert int(v.sum()) == 2 * big + 10

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_1d_indexing(self, xp, sp, dtype):
        v = sp.csr_array(xp.array([1, 0, 2, 3, 0, 4], dtype=dtype))
        cols = xp.array([0, 2, 5])
        return (xp.asarray(v[2]),
                v[1:4].toarray(),
                v[cols].toarray())

    def test_1d_indexing_float16(self):
        # scipy rejects float16, so compare against dense (CuPy extension).
        dense = cupy.array([1, 0, 2, 3, 0, 4], dtype=numpy.float16)
        v = cupyx.scipy.sparse.csr_array(dense)
        assert v[2] == dense[2]
        cupy.testing.assert_array_equal(v[1:4].toarray(), dense[1:4])
        cols = cupy.array([0, 2, 5])
        cupy.testing.assert_array_equal(v[cols].toarray(), dense[cols])

    @pytest.mark.parametrize('dtype', _scipy_dtypes)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_1d_arithmetic(self, xp, sp, dtype):
        a = sp.csr_array(xp.array([1, 0, 2, 3], dtype=dtype))
        b = sp.csr_array(xp.array([0, 1, 2, 0], dtype=dtype))
        return ((a + b).toarray(), (a * b).toarray(),
                (a * 2).toarray(), xp.asarray(a @ b))


@requires_bfloat16
class TestBfloat16:
    """bfloat16 is a CuPy extension (via ml_dtypes); scipy rejects it, so
    these compare against dense cupy.  Its numpy ``kind`` is ``'V'`` and
    its char ``'E'`` collides with complex-half, so sparse detects it by
    name.  Native cuSPARSE bf16 needs sm_80+ (silently zeros on older
    GPUs), so matmul always upcasts to float32."""

    def _make(self, container='csr_matrix'):
        dense = cupy.array([[0, 1, 0, 2], [3, 0, 0, 0], [0, 4, 5, 0]],
                           dtype='float32').astype(_bfloat16)
        return getattr(cupyx.scipy.sparse, container)(dense), dense

    def test_gate_accepts_by_name(self):
        m, dense = self._make()
        assert m.dtype == _bfloat16
        assert cupyx.scipy.sparse._sputils.is_sparse_data_dtype(_bfloat16)
        assert _bfloat16 in cupyx.scipy.sparse._sputils.supported_dtypes

    @pytest.mark.parametrize('container', _containers)
    def test_construct_roundtrip(self, container):
        m, dense = self._make(container)
        assert m.dtype == _bfloat16
        cupy.testing.assert_array_equal(m.toarray(), dense)
        cupy.testing.assert_array_equal(
            m.tocsc().tocoo().tocsr().toarray(), dense)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_csc_toarray_orders(self, order):
        m, dense = self._make('csc_matrix')
        cupy.testing.assert_array_equal(m.toarray(order=order), dense)

    def test_sum_duplicates(self):
        # bfloat16 has kind 'V', so it must be routed to the atomicAdd
        # accumulate path by name (else duplicates would leave zeros).
        data = cupy.array([1.5, 2.5, 3.0], dtype='float32').astype(_bfloat16)
        row = cupy.array([0, 0, 1], 'i')
        col = cupy.array([0, 0, 1], 'i')
        m = cupyx.scipy.sparse.coo_matrix((data, (row, col)), shape=(2, 2))
        m.sum_duplicates()
        assert m.dtype == _bfloat16
        assert float(m.toarray()[0, 0]) == 4.0

    def test_arithmetic(self):
        m, dense = self._make()
        cupy.testing.assert_array_equal((m + m).toarray(), dense + dense)
        cupy.testing.assert_array_equal((m * 2).toarray(),
                                        (dense.astype('f4') * 2))
        cupy.testing.assert_array_equal(m.multiply(m).toarray(),
                                        (dense.astype('f4') ** 2)
                                        * (dense.astype('f4') != 0))
        # bf16 / int keeps bf16 (matches dense CuPy; scipy has no oracle).
        assert (m / 2).dtype == _bfloat16

    def test_matmul_upcasts_float32(self):
        m, dense = self._make()
        v = cupy.array([1, 2, 0, 3], dtype='float32').astype(_bfloat16)
        r = m @ v
        assert r.dtype == _bfloat16
        cupy.testing.assert_allclose(
            r.astype('f4'), dense.astype('f4') @ v.astype('f4'), rtol=1e-2)
        p = m @ m.T
        assert p.dtype == _bfloat16
        cupy.testing.assert_allclose(
            p.toarray().astype('f4'),
            dense.astype('f4') @ dense.T.astype('f4'), rtol=1e-2)

    def test_spgemm_f32_staged_accuracy(self):
        # bfloat16 SpGEMM stages through float32 (accumulates in float32,
        # narrows once) -- a heavy contraction stays near bf16 epsilon
        # rather than compounding an 8-bit-mantissa accumulation.
        rng = numpy.random.default_rng(0)
        k = 400
        a_d = rng.random((6, k)).astype('float32').astype(_bfloat16)
        b_d = rng.random((k, 6)).astype('float32').astype(_bfloat16)
        a = cupyx.scipy.sparse.csr_matrix(cupy.asarray(a_d))
        b = cupyx.scipy.sparse.csr_matrix(cupy.asarray(b_d))
        got = (a @ b).toarray().astype('float64').get()
        ref = a_d.astype('float64') @ b_d.astype('float64')
        rel = numpy.max(numpy.abs(got - ref) / (numpy.abs(ref) + 1e-9))
        assert rel < 5e-2

    def test_reductions(self):
        m, dense = self._make()
        assert m.sum().dtype == _bfloat16
        assert float(m.sum()) == float(dense.astype('f4').sum())
        assert m.max().dtype == _bfloat16
        assert float(m.max()) == float(dense.astype('f4').max())
        # mean keeps bfloat16, matching dense CuPy (rule 3: dense is the
        # oracle for the scipy-rejected 16-bit floats; consistent with
        # float16.mean -> float16), rounded to bf16's low precision.
        assert m.mean().dtype == _bfloat16
        cupy.testing.assert_allclose(
            float(m.mean().astype('f4')),
            float(dense.astype('f4').mean()), rtol=1e-1)
        cupy.testing.assert_array_equal(
            m.argmax(axis=1),
            dense.astype('f4').argmax(axis=1)[:, None])

    def test_fancy_indexing(self):
        m, dense = self._make()
        cols = cupy.array([0, 2, 3])
        cupy.testing.assert_array_equal(m[:, cols].toarray(), dense[:, cols])
        rows = cupy.array([0, 2])
        cupy.testing.assert_array_equal(m[rows, :].toarray(), dense[rows, :])

    def test_scalar_getitem_and_slices(self):
        m, dense = self._make()
        assert m[1, 0] == dense[1, 0]
        cupy.testing.assert_array_equal(m[0:2, :].toarray(), dense[0:2, :])

    def test_stack_and_eye_preserve_dtype(self):
        m, dense = self._make()
        assert cupyx.scipy.sparse.hstack([m, m]).dtype == _bfloat16
        assert cupyx.scipy.sparse.vstack([m, m]).dtype == _bfloat16
        assert cupyx.scipy.sparse.eye(
            3, dtype=_bfloat16, format='csr').dtype == _bfloat16

    def test_random(self):
        m = cupyx.scipy.sparse.random(20, 20, density=0.3, dtype=_bfloat16)
        assert m.dtype == _bfloat16
        assert m.nnz == 120

    def test_matrix_power(self):
        from cupyx.scipy.sparse.linalg import matrix_power
        d = cupy.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]],
                       dtype='float32').astype(_bfloat16)
        m = cupyx.scipy.sparse.csr_matrix(d)
        r = matrix_power(m, 3)
        assert r.dtype == _bfloat16
        cupy.testing.assert_allclose(
            r.toarray().astype('f4'),
            numpy.linalg.matrix_power(d.get().astype('f4'), 3), rtol=1e-2)
        assert (m ** 3).dtype == _bfloat16

    def test_1d_array(self):
        dense = cupy.array([1, 0, 2, 3, 0, 4],
                           dtype='float32').astype(_bfloat16)
        v = cupyx.scipy.sparse.csr_array(dense)
        assert v.dtype == _bfloat16
        cupy.testing.assert_array_equal(v.toarray(), dense)
        assert float(v.sum()) == 10.0
        assert float(v.max()) == 4.0
        assert float(v[2]) == 2.0
        cupy.testing.assert_array_equal(v[1:4].toarray(), dense[1:4])
        cupy.testing.assert_array_equal(
            v[cupy.array([0, 2, 4])].toarray(), dense[cupy.array([0, 2, 4])])
        cupy.testing.assert_array_equal((v + v).toarray(), dense + dense)
        assert float(v @ v) == 30.0

    def test_comparison_int_scalar(self):
        # bfloat16's numpy kind is 'V', so _comparison must route it
        # like a float (cast the scalar to bf16, matching dense CuPy);
        # otherwise the scalar stays int and numpy.promote_types(
        # bfloat16, int64) crashes binopt_csr.  s=2 exercises the fast
        # path (op(0, 2) False, e.g. '>') and the slow O(m*n) path
        # (op(0, 2) True, e.g. '<') across the six operators.
        dense = cupy.array([[1, 0, 3], [0, 2, 0]],
                           dtype='float32').astype(_bfloat16)
        ops = [lambda a, s: a < s, lambda a, s: a > s,
               lambda a, s: a <= s, lambda a, s: a >= s,
               lambda a, s: a == s, lambda a, s: a != s]
        for name in ('csr_matrix', 'csc_matrix', 'coo_matrix'):
            m = getattr(cupyx.scipy.sparse, name)(dense)
            for op in ops:
                for s in (2, 0):
                    cupy.testing.assert_array_equal(
                        op(m, s).toarray(), op(dense, s))

    def test_str_and_print_do_not_crash(self):
        # bfloat16, like float16, has no scipy host kernels, so
        # ``str(self.get())`` raises ValueError; ``__str__`` falls back
        # to ``repr`` instead of crashing ``print()`` / ``format()``.
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[1, 0, 2], [0, 3, 0]],
                       dtype='float32').astype(_bfloat16))
        assert 'sparse' in str(m)
        assert 'sparse' in '{}'.format(m)

    @pytest.mark.parametrize('other', ['int16', 'int32', 'int64',
                                       'uint32', 'float16'])
    @pytest.mark.parametrize('fmt', ['csr_matrix', 'csc_matrix',
                                     'coo_matrix'])
    def test_mixed_dtype_elementwise_matches_dense(self, other, fmt):
        # numpy cannot promote bfloat16 with a >=16-bit int or float16, but
        # dense CuPy resolves it via ufunc loops (bf16+int32 -> float64);
        # sparse +, *, comparison and maximum must match dense, not raise.
        odt = numpy.float16 if other == 'float16' else numpy.dtype(other)
        db = cupy.array([[1, 0, 2], [0, 3, 0]],
                        dtype='float32').astype(_bfloat16)
        do = cupy.array([[1, 0, 1], [0, 2, 0]], dtype='float32').astype(odt)
        sp_cls = getattr(cupyx.scipy.sparse, fmt)
        mb, mo = sp_cls(db), sp_cls(do)
        assert (mb + mo).dtype == (db + do).dtype
        cupy.testing.assert_array_equal((mb + mo).toarray(), db + do)
        assert mb.multiply(mo).dtype == (db * do).dtype
        cupy.testing.assert_array_equal(
            mb.multiply(mo).toarray(), db * do)
        assert (mb < mo).dtype == numpy.bool_
        assert mb.maximum(mo).dtype == cupy.maximum(db, do).dtype

    def test_matmul_and_concat_with_int_raise(self):
        # dense CuPy also raises for bf16 @ int and
        # concatenate([bf16, int]); only elementwise ops promote.
        mb = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[1, 0, 2], [0, 3, 0]], dtype='f4').astype(_bfloat16))
        mi = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[1, 0, 1], [0, 1, 0]], dtype='i4'))
        with pytest.raises(numpy.exceptions.DTypePromotionError):
            mb @ cupyx.scipy.sparse.csr_matrix(
                cupy.array([[1, 0], [0, 1], [1, 0]], dtype='i4'))
        with pytest.raises(numpy.exceptions.DTypePromotionError):
            mb @ cupy.ones((3, 2), dtype='i4')
        with pytest.raises(numpy.exceptions.DTypePromotionError):
            cupyx.scipy.sparse.hstack([mb, mi])


@testing.with_requires('scipy')
class TestExactIntegerMatmul:
    # cuSPARSE spmv/spmm have no integer compute type; the pure-CuPy scatter
    # must match scipy's modular integer matmul exactly -- a float64 upcast
    # would lose precision past 2**53 and saturate (not wrap) on
    # out-of-range integer casts.

    @pytest.mark.parametrize('dtype', _int_dtypes + [numpy.bool_])
    @pytest.mark.parametrize('fmt', ['csr_matrix', 'csc_matrix'])
    def test_matches_scipy(self, dtype, fmt):
        import scipy.sparse
        rng = numpy.random.default_rng(numpy.dtype(dtype).itemsize + len(fmt))
        d = rng.integers(0, 7, size=(4, 5)).astype(dtype)
        d[rng.random((4, 5)) < 0.4] = 0
        xv = rng.integers(0, 7, size=5).astype(dtype)
        xm = rng.integers(0, 7, size=(5, 3)).astype(dtype)
        m = getattr(cupyx.scipy.sparse, fmt)(cupy.asarray(d))
        sm = getattr(scipy.sparse, fmt)(d)
        numpy.testing.assert_array_equal(
            (m @ cupy.asarray(xv)).get(), numpy.asarray(sm @ xv).ravel())
        numpy.testing.assert_array_equal(
            (m @ cupy.asarray(xm)).get(), numpy.asarray(sm @ xm))

    def test_int64_exact_past_2_53(self):
        import scipy.sparse
        base = 2 ** 53
        d = numpy.array([[base + 1, 0, base + 3],
                         [0, base + 5, 0],
                         [base + 7, 0, base + 2]], dtype=numpy.int64)
        x = numpy.array([1, 1, 1], dtype=numpy.int64)
        got = (cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
               @ cupy.asarray(x)).get()
        numpy.testing.assert_array_equal(
            got, numpy.asarray(scipy.sparse.csr_matrix(d) @ x).ravel())

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int32,
                                       numpy.uint8, numpy.uint32])
    def test_overflow_wraps_like_scipy(self, dtype):
        import scipy.sparse
        big = dtype(numpy.iinfo(dtype).max)
        d = numpy.array([[big, big]], dtype=dtype)
        x = numpy.array([2, 2], dtype=dtype)
        got = (cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
               @ cupy.asarray(x)).get()
        numpy.testing.assert_array_equal(
            got, numpy.asarray(scipy.sparse.csr_matrix(d) @ x).ravel())

    def test_wide_operand_chunks_and_stays_exact(self, monkeypatch):
        # The scatter materialises an (nnz, N) product; a wide dense
        # operand is processed in column blocks so it cannot OOM.  Force
        # multi-chunk (block == 1) by reporting ~no free memory and confirm
        # the chunked accumulation is still exact past 2**53 -- unlike a
        # float64 fallback, which would round.
        import scipy.sparse
        cupy.get_default_memory_pool().free_all_blocks()
        monkeypatch.setattr(cupy.cuda.runtime, 'memGetInfo', lambda: (1, 1))
        d = numpy.array([[2**60 + 1, 0, 3], [0, 5, 0], [7, 0, 2**60]],
                        dtype=numpy.int64)
        x = numpy.arange(1, 13, dtype=numpy.int64).reshape(3, 4)
        got = (cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
               @ cupy.asarray(x)).get()
        numpy.testing.assert_array_equal(
            got, numpy.asarray(scipy.sparse.csr_matrix(d) @ x))


class TestUnsupportedDtypeRejected:

    @pytest.mark.parametrize('name', ['float8_e4m3b11fnuz', 'float8_e5m2'])
    def test_float8_rejected(self, name):
        ml = pytest.importorskip('ml_dtypes')
        from cupyx.scipy.sparse import _sputils
        dt = numpy.dtype(getattr(ml, name))
        # float8_e4m3b11fnuz shares dtype char 'L' with uint64, so a char
        # test would wrongly accept it; the width-based check rejects it.
        assert not _sputils.is_sparse_data_dtype(dt)
        with pytest.raises(ValueError):
            _sputils.check_data_dtype(dt)

    def test_longdouble_rejected(self):
        from cupyx.scipy.sparse import _sputils
        assert not _sputils.is_sparse_data_dtype(numpy.longdouble)
        with pytest.raises(ValueError):
            _sputils.check_data_dtype(numpy.longdouble)


@testing.with_requires('scipy')
class TestComplexArgReduceAxis:
    # Axis-wise argmin/argmax on a complex matrix must not crash (the
    # arg-reduction kernel is instantiated only for real types); reduce on
    # the real part, matching min/max.

    @pytest.mark.parametrize('fmt', ['csr_matrix', 'csc_matrix'])
    @pytest.mark.parametrize('meth', ['argmax', 'argmin'])
    @pytest.mark.parametrize('axis', [0, 1])
    def test_complex_argreduce_axis(self, fmt, meth, axis):
        import scipy.sparse
        d = numpy.array([[0, 2 + 1j, 0, 5],
                         [3 - 2j, 0, 0, 0],
                         [0, 0, 7 + 9j, 1j]], dtype=numpy.complex128)
        m = getattr(cupyx.scipy.sparse, fmt)(cupy.asarray(d))
        sm = getattr(scipy.sparse, fmt)(d)
        got = cupy.asnumpy(getattr(m, meth)(axis=axis)).ravel()
        exp = numpy.asarray(getattr(sm, meth)(axis=axis)).ravel()
        numpy.testing.assert_array_equal(got, exp)


class TestRound2Fixes:
    # Regressions for the round-2 correctness fixes.

    def test_truediv_clongdouble_keeps_complex(self):
        # Dividing a complex matrix by a clongdouble scalar promotes to
        # the unstorable complex256; the fallback must stay complex128
        # (float64 would silently drop the imaginary part).
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[2 + 0j, 0], [0, 4 + 0j]], dtype='complex128'))
        r = m / numpy.clongdouble(2 + 1j)
        assert r.dtype == numpy.complex128
        cupy.testing.assert_allclose(
            r.toarray(), cupy.asarray([[2, 0], [0, 4]], dtype='complex128')
            / (2 + 1j))

    def test_truediv_float16_stays_float16(self):
        # scipy rejects float16, so dense CuPy is the oracle:
        # ``float16 / 2.0 -> float16`` (not float64).
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[1, 0, 2], [0, 3, 0]], dtype='float16'))
        assert (m / 2.0).dtype == numpy.float16
        # float32 still promotes to float64 (scipy parity, unchanged).
        m32 = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[1, 0, 2]], dtype='float32'))
        assert (m32 / 2.0).dtype == numpy.float64

    @pytest.mark.parametrize('n', [2, 3, 4])
    def test_bool_power_matches_scipy_per_n(self, n):
        # bool ** n -> numpy/scipy's per-exponent carrier (int8 @ n==2,
        # int64 @ n>=3), derived by construction, not cupy dense's
        # int64-all.  scipy supports bool, so scipy is the oracle.
        m = cupyx.scipy.sparse.csr_array(
            cupy.array([[True, False], [True, True]]))
        r = m.power(n)
        assert r.dtype == (numpy.ones(1, bool) ** n).dtype
        cupy.testing.assert_array_equal(
            r.toarray(), m.toarray().astype(numpy.int64) ** n)

    @testing.with_requires('scipy')
    @pytest.mark.parametrize('meth', ['min', 'max', 'argmin', 'argmax'])
    @pytest.mark.parametrize('axis', [(0, 1), (1,), (0,)])
    def test_tuple_axis_min_max(self, meth, axis):
        import scipy.sparse
        d = numpy.array([[0, 3, 0], [5, 0, -1], [0, 2, 0]], dtype='float64')
        m = cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
        sm = scipy.sparse.csr_matrix(d)
        got = getattr(m, meth)(axis=axis)
        exp = getattr(sm, meth)(axis=axis)
        got = cupy.asnumpy(got.toarray() if hasattr(got, 'toarray') else got)
        exp = numpy.asarray(exp.toarray() if hasattr(exp, 'toarray') else exp)
        numpy.testing.assert_array_equal(got.ravel(), exp.ravel())

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int16,
                                       numpy.uint8, numpy.uint16])
    def test_toarray_dedups_stale_canonical_flag(self, dtype):
        # The 8/16-bit direct-write densify must sum duplicates even when
        # has_canonical_format is (wrongly) True, matching scipy.
        m = cupyx.scipy.sparse.csr_matrix(
            (cupy.array([3, 7], dtype=dtype), cupy.array([1, 1], dtype='i4'),
             cupy.array([0, 2], dtype='i4')), shape=(1, 3))
        m.has_canonical_format = True  # stale: duplicates at (0, 1)
        cupy.testing.assert_array_equal(
            m.toarray(), cupy.array([[0, 10, 0]], dtype=dtype))


class TestRound3Fixes:
    # Regressions for the round-3 correctness fixes.

    @pytest.mark.parametrize('exp', [2, 3])
    def test_bool_power_accepts_0d_cupy_exponent(self, exp):
        # ``isscalarlike`` accepts a 0-D cupy array as the exponent; the
        # per-n carrier ``numpy.ones(1, bool) ** n`` would crash on a
        # device array, so ``n`` is read to host first.  A 0-D array's
        # dtype (int64) makes numpy promote to int64 -- faithful to scipy,
        # which gives int64 for a ``np.int64`` exponent.
        m = cupyx.scipy.sparse.csr_array(
            cupy.array([[True, False], [True, True]]))
        r = m.power(cupy.asarray(exp))
        assert r.dtype == numpy.int64
        cupy.testing.assert_array_equal(
            r.toarray(), m.toarray().astype(numpy.int64) ** exp)

    @requires_bfloat16
    @pytest.mark.parametrize('scalar,expect', [
        (2.0, 'float32'), (2, 'bfloat16'), (1 + 1j, 'complex64')])
    def test_bf16_scalar_division_matches_dense_ufunc(self, scalar, expect):
        # scipy rejects bfloat16, so the dense ufunc is the oracle -- and
        # it differs from result_type (bf16/2.0 is float32 via the ufunc,
        # float64 via result_type).
        d = cupy.array([[1, 0, 2], [0, 3, 0]], dtype='float32').astype(
            _bfloat16)
        m = cupyx.scipy.sparse.csr_matrix(d)
        assert str((m / scalar).dtype) == expect
        assert (m / scalar).dtype == (d / scalar).dtype

    def test_float16_scalar_division_stays_float16(self):
        d = cupy.array([[1, 0, 2]], dtype='float16')
        m = cupyx.scipy.sparse.csr_matrix(d)
        assert (m / 2.0).dtype == numpy.float16
        assert (m / 2.0).dtype == (d / 2.0).dtype

    @requires_bfloat16
    def test_bf16_mixed_division_promotes_not_raises(self):
        # A bfloat16 sparse divided by a wider dense/sparse operand should
        # promote like dense (float), not raise DTypePromotionError.
        d = cupy.array([[1, 0, 2], [0, 3, 0]], dtype='float32').astype(
            _bfloat16)
        m = cupyx.scipy.sparse.csr_matrix(d)
        r = m / cupy.ones((2, 3), dtype='float16')
        assert r.dtype == (d / cupy.ones(3, dtype='float16')).dtype  # f32
        # wider int dense -> float64 (no raise)
        assert (m / cupy.ones((2, 3), dtype='int32')).dtype == numpy.float64
        # sparse divisor: no raise
        m.__truediv__(cupyx.scipy.sparse.csr_matrix(
            cupy.ones((2, 3), dtype='int32')))


@testing.with_requires('scipy')
class TestExactReductionCoverage:
    # Coverage the round-4 comparison flagged as missing in 0a.

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    @pytest.mark.parametrize('axis', [None, 0, 1])
    def test_uint64_sum_exact_past_2_53(self, fmt, axis):
        # uint64 sums past 2**53 must stay exact (segmented cumsum), not
        # round through the float64 matmul path.
        import scipy.sparse
        big = 2 ** 60
        dense = numpy.array([[big + 7, 0, big + 3],
                             [0, big + 1, 0]], dtype=numpy.uint64)
        m = getattr(cupyx.scipy.sparse, f'{fmt}_matrix')(cupy.asarray(dense))
        sm = getattr(scipy.sparse, f'{fmt}_matrix')(dense)
        got = m.sum(axis=axis)
        exp = sm.sum(axis=axis)
        if axis is None:
            assert int(got) == int(exp)
        else:
            numpy.testing.assert_array_equal(
                cupy.asnumpy(got).ravel(), numpy.asarray(exp).ravel())

    @pytest.mark.parametrize('meth', ['argmax', 'argmin'])
    @pytest.mark.parametrize('axis', [0, 1])
    def test_bool_argreduce_axis_matches_scipy(self, meth, axis):
        # bool is a value dtype; its argmax/argmin (per axis) must match
        # scipy, including an all-False row/column.
        import scipy.sparse
        d = numpy.array([[False, True, False],
                         [True, False, True],
                         [False, False, False]])
        m = cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
        sm = scipy.sparse.csr_matrix(d)
        numpy.testing.assert_array_equal(
            cupy.asnumpy(getattr(m, meth)(axis=axis)).ravel(),
            numpy.asarray(getattr(sm, meth)(axis=axis)).ravel())


# ---- 0b session's tests below (colliding list-helpers renamed to _b) ----

_int_dtypes_b = [numpy.int8, numpy.uint8, numpy.int16, numpy.uint16,
                 numpy.int32, numpy.uint32, numpy.int64, numpy.uint64]
# Representative subset for expensive parametrizations: smallest signed
# (no atomicAdd overload), smallest unsigned, and 64-bit.
_key_dtypes_b = [numpy.int8, numpy.uint8, numpy.int64]

_containers_b = ['csr_matrix', 'csc_matrix', 'coo_matrix',
                 'csr_array', 'csc_array', 'coo_array']


def _make(xp, sp, dtype, container='csr_matrix'):
    dense = xp.array([[0, 1, 0, 2],
                      [3, 0, 0, 0],
                      [0, 4, 5, 0]], dtype=dtype)
    return getattr(sp, container)(dense)


@testing.with_requires('scipy>=1.12')
class TestValueDtypeConstruction:

    @pytest.mark.parametrize('container', _containers_b)
    @pytest.mark.parametrize('dtype', _int_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_from_dense(self, xp, sp, dtype, container):
        m = _make(xp, sp, dtype, container)
        assert m.dtype == dtype
        return m.toarray()

    @pytest.mark.parametrize('dtype', _int_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_from_tuple(self, xp, sp, dtype):
        data = xp.array([1, 2, 3], dtype=dtype)
        indices = xp.array([0, 2, 1], 'i')
        indptr = xp.array([0, 2, 3], 'i')
        return sp.csr_matrix((data, indices, indptr), shape=(2, 3)).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    def test_empty_shape(self, dtype):
        m = cupyx.scipy.sparse.csr_matrix((3, 4), dtype=dtype)
        assert m.dtype == dtype
        assert m.nnz == 0

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    def test_scipy_roundtrip(self, dtype):
        scipy_sparse = pytest.importorskip('scipy.sparse')
        d = numpy.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
        m = cupyx.scipy.sparse.csr_matrix(scipy_sparse.csr_matrix(d))
        assert m.dtype == dtype
        back = m.get()
        assert back.dtype == dtype
        numpy.testing.assert_array_equal(back.toarray(), d)

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_dia(self, xp, sp, dtype):
        data = xp.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        offsets = xp.array([0, -1], 'i')
        m = sp.dia_matrix((data, offsets), shape=(3, 3))
        assert m.dtype == dtype
        return m.toarray()

    def test_unsupported_dtype_message(self):
        d = cupy.zeros((2, 2), dtype=numpy.float32)
        with pytest.raises(ValueError, match='does not support dtype'):
            cupyx.scipy.sparse.csr_matrix(d, dtype='U3')
        with pytest.raises(ValueError, match='does not support dtype'):
            cupyx.scipy.sparse.coo_matrix(d, dtype='U3')

    def test_ml_dtypes_collisions_rejected(self):
        # ml_dtypes registers custom numpy dtypes whose single-char codes
        # collide with standard ones -- e.g.
        # numpy.dtype(ml_dtypes.float8_e4m3b11fnuz).char == 'L', colliding
        # with uint64.  The gate must reject the unsupported ones by dtype
        # *identity*, not char (which would mis-accept the float8).
        # (bfloat16 is separately supported -- see TestValueDtypeBfloat16.)
        ml_dtypes = pytest.importorskip('ml_dtypes')
        from cupyx.scipy.sparse import _sputils
        for name in ('float8_e4m3b11fnuz', 'float8_e5m2',
                     'float8_e4m3', 'int4', 'uint4'):
            dt = numpy.dtype(getattr(ml_dtypes, name))
            assert not _sputils.is_sparse_data_dtype(dt), name
            with pytest.raises(ValueError, match='does not support dtype'):
                _sputils.check_data_dtype(dt)


@testing.with_requires('scipy>=1.12')
class TestValueDtypeConversion:

    @pytest.mark.parametrize('container', _containers_b)
    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_format_roundtrips(self, xp, sp, dtype, container):
        m = _make(xp, sp, dtype, container)
        assert m.tocsc().dtype == dtype
        return (m.tocsc().tocoo().tocsr().toarray(),
                m.tocsr().toarray(),
                m.T.toarray())

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_csc_toarray_orders(self, xp, sp, dtype):
        # csc.toarray for non-float dtypes goes through the CSR kernel.
        m = _make(xp, sp, dtype, 'csc_matrix')
        return m.toarray(order='C'), m.toarray(order='F')

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_unsorted_coo_tocsc(self, xp, sp, dtype):
        # Regression: coosort must permute data along with coordinates
        # (the bool path used to skip the data gather entirely).
        row = xp.array([2, 0, 1], 'i')
        col = xp.array([0, 2, 1], 'i')
        data = xp.array([1, 2, 3], dtype=dtype)
        m = sp.coo_matrix((data, (row, col)), shape=(3, 3))
        return m.tocsc().toarray()

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_unsorted_bool_coo_tocsc(self, xp, sp):
        # An explicit stored ``False`` at a coordinate that moves during
        # the column sort: if the data gather is skipped (the old bool
        # cuSPARSE coosort path), the False misaligns with its coords
        # and the wrong cell reads False.
        row = xp.array([2, 0, 1], 'i')
        col = xp.array([0, 2, 1], 'i')
        data = xp.array([True, False, True])
        m = sp.coo_matrix((data, (row, col)), shape=(3, 3))
        return m.tocsc().toarray(), m.toarray()

    @pytest.mark.parametrize('dtype', _int_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_sum_duplicates(self, xp, sp, dtype):
        # 100 + 100 wraps to -56 for int8 (matching numpy overflow).
        data = xp.array([100, 100, 5], dtype=dtype)
        row = xp.array([0, 0, 1], 'i')
        col = xp.array([0, 0, 1], 'i')
        m = sp.coo_matrix((data, (row, col)), shape=(2, 2))
        m.sum_duplicates()
        assert m.dtype == dtype
        return m.toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_eliminate_zeros(self, xp, sp, dtype):
        data = xp.array([1, 0, 3], dtype=dtype)
        indices = xp.array([0, 1, 2], 'i')
        indptr = xp.array([0, 3], 'i')
        m = sp.csr_matrix((data, indices, indptr), shape=(1, 3))
        m.eliminate_zeros()
        assert m.nnz == 2
        return m.toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_sort_indices(self, xp, sp, dtype):
        data = xp.array([2, 1], dtype=dtype)
        indices = xp.array([2, 0], 'i')
        indptr = xp.array([0, 2], 'i')
        m = sp.csr_matrix((data, indices, indptr), shape=(1, 3))
        m.sort_indices()
        assert m.has_sorted_indices
        return m.toarray()


@testing.with_requires('scipy>=1.12')
class TestValueDtypeArithmetic:

    @pytest.mark.parametrize('dtype', _int_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_add(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        r = m + m
        assert r.dtype == dtype
        return r.toarray()

    @pytest.mark.parametrize('dtype', _int_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_sub(self, xp, sp, dtype):
        # uint wraps modularly; bool behaves like C bool arithmetic
        # (a - b != 0), both matching scipy.
        a = _make(xp, sp, dtype)
        b = sp.csr_matrix(
            xp.array([[1, 1, 0, 0],
                      [0, 0, 0, 2],
                      [0, 4, 0, 0]], dtype=dtype))
        return (a - b).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_mixed_dtype_add(self, xp, sp, dtype):
        # int + float32 promotes and uses the native cuSPARSE path.
        a = _make(xp, sp, dtype)
        b = _make(xp, sp, numpy.float32)
        return (a + b).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_scalar_mul(self, xp, sp, dtype):
        return (_make(xp, sp, dtype) * 2).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_truediv_scalar(self, xp, sp, dtype):
        return (_make(xp, sp, dtype) / 2).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_truediv_dense(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        div = xp.arange(1, 13, dtype=dtype).reshape(3, 4)
        r = m / div
        assert r.dtype == numpy.float64
        return r.toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_multiply(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        return m.multiply(m).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_multiply_dense(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        dn = xp.arange(12, dtype=dtype).reshape(3, 4)
        return m.multiply(dn).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_maximum_minimum(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        return m.maximum(m).toarray(), m.minimum(3).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_comparison(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        # 300 is out of int8 range: must compare by value, not wrap.
        return (m > 1).toarray(), (m != m).toarray(), (m > 300).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_power(self, xp, sp, dtype):
        # bool ** 2 promotes to int8 (numpy rule).
        return _make(xp, sp, dtype).power(2).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_neg_abs(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        return (-m).toarray(), abs(m).toarray()

    def test_bool_neg_raises(self):
        m = _make(cupy, cupyx.scipy.sparse, bool)
        with pytest.raises(NotImplementedError):
            -m


@testing.with_requires('scipy>=1.12')
class TestValueDtypeMatmul:

    @pytest.mark.parametrize('container',
                             ['csr_matrix', 'csc_matrix', 'csr_array'])
    @pytest.mark.parametrize('dtype', _int_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_spmv(self, xp, sp, dtype, container):
        m = _make(xp, sp, dtype, container)
        v = xp.array([1, 2, 0, 3], dtype=dtype)
        return m @ v

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_spmm(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        b = (xp.arange(8).reshape(4, 2) % 3).astype(dtype)
        return m @ b

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_spmv_mixed_float(self, xp, sp, dtype):
        # int sparse @ float dense promotes and stays on cuSPARSE.
        m = _make(xp, sp, dtype)
        v = xp.array([1, 2, 0, 3], dtype=numpy.float32)
        return m @ v

    @pytest.mark.parametrize('other_dtype',
                             [numpy.float16, numpy.int8, numpy.float32])
    def test_float16_matmul_dtype_promotion(self, other_dtype):
        # float16 matrix promotes with the dense operand:
        #   f16 @ f16/int8 -> f16 (native mixed-precision path)
        #   f16 @ f32      -> f32 (upcast path)
        d = cupy.array([[1, 0, 2, 0], [0, 3, 0, 1]], dtype=numpy.float16)
        m = cupyx.scipy.sparse.csr_matrix(d)
        v = cupy.array([1, 2, 0, 3], dtype=other_dtype)
        expected = numpy.promote_types(numpy.float16, other_dtype)
        r = m @ v
        assert r.dtype == expected
        cupy.testing.assert_allclose(
            r.astype(numpy.float32),
            (d.astype(numpy.float32) @ v.astype(numpy.float32)), rtol=1e-2)

    def test_int8_matmul_float16_dense(self):
        # int8 sparse @ f16 dense -> f16 (via the exact-upcast path).
        d = cupy.array([[1, 0, 2, 0], [0, 3, 0, 1]], dtype=numpy.int8)
        m = cupyx.scipy.sparse.csr_matrix(d)
        v = cupy.array([1, 2, 0, 3], dtype=numpy.float16)
        r = m @ v
        assert r.dtype == numpy.float16

    @pytest.mark.parametrize('dtype', _int_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_spgemm(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        r = m @ m.T
        assert r.dtype == dtype
        return r.toarray()

    def test_spgemm_int_exact_past_2_53(self):
        # Integer sparse @ sparse accumulates exactly (pure-CuPy),
        # including products/sums above float64's 2**53 mantissa limit.
        v = 1 << 30                       # v*v = 2**60 > 2**53
        a = cupyx.scipy.sparse.csr_matrix(
            cupy.array([[v, v], [0, v]], dtype=numpy.int64))
        c = a @ a.T
        ref = (numpy.array([[v, v], [0, v]], dtype=numpy.int64)
               @ numpy.array([[v, v], [0, v]], dtype=numpy.int64).T)
        cupy.testing.assert_array_equal(c.toarray(), cupy.asarray(ref))
        assert int(c.toarray()[0, 0]) == 2 * v * v

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_csr_matmul_csc(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        return (m @ m.T.tocsc()).toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    def test_spgemm_preserves_index_dtype(self, dtype):
        m = _make(cupy, cupyx.scipy.sparse, dtype)
        r = m @ m.T
        assert r.indices.dtype == m.indices.dtype

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_matmul_1d(self, xp, sp, dtype):
        arr = sp.csr_array(xp.array([1, 0, 2, 3], dtype=dtype))
        m = _make(xp, sp, dtype).T  # (4, 3)
        return arr @ m.tocsr().toarray()

    @pytest.mark.parametrize('power', [0, 1, 2, 3])
    @pytest.mark.parametrize('dtype', _int_dtypes_b + [bool])
    def test_matrix_power(self, power, dtype):
        # matrix_power builds on spgemm + eye_array, so it inherits full
        # value-dtype support (and int exactness).
        from cupyx.scipy.sparse.linalg import matrix_power
        d = numpy.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=dtype)
        m = cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
        r = matrix_power(m, power)
        assert r.dtype == numpy.dtype(dtype)
        if dtype == numpy.bool_:
            exp = numpy.linalg.matrix_power(d.astype(numpy.int64), power) != 0
        else:
            exp = numpy.linalg.matrix_power(d, power)  # wraps like scipy
        cupy.testing.assert_array_equal(r.toarray(), cupy.asarray(exp))

    def test_matrix_power_int64_exact(self):
        # int64 powering stays exact past float64's 2**53: the pure-CuPy
        # spgemm accumulates in int64, not float64.
        from cupyx.scipy.sparse.linalg import matrix_power
        k = 4 * 10 ** 15 + 1        # 3*k = 1.2e16 + 3, odd and > 2**53
        d = numpy.array([[1, k], [0, 1]], dtype=numpy.int64)
        m = cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
        r = matrix_power(m, 3)      # [[1, 3k], [0, 1]]
        assert int(r.toarray()[0, 1]) == 3 * k
        cupy.testing.assert_array_equal(
            r.toarray(), cupy.asarray(numpy.linalg.matrix_power(d, 3)))


@testing.with_requires('scipy>=1.12')
class TestValueDtypeReductions:

    @pytest.mark.parametrize('axis', [None, 0, 1])
    @pytest.mark.parametrize('dtype', _int_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_sum(self, xp, sp, dtype, axis):
        # scipy/numpy accumulate: bool/int -> int64, uint -> uint64.
        m = _make(xp, sp, dtype)
        return xp.asarray(m.sum(axis=axis))

    @pytest.mark.parametrize('container', ['csr_matrix', 'csc_matrix',
                                           'coo_matrix'])
    @pytest.mark.parametrize('axis', [None, 0, 1])
    def test_int64_sum_exact_past_2_53(self, axis, container):
        # sum() must accumulate in int64, not float64 (the matmul path):
        # values above 2**53 are not exactly representable in float64.
        import scipy.sparse
        big = 1 << 60
        d = numpy.array([[big + 7, big + 3, 0], [big + 5, 0, big + 1]],
                        dtype=numpy.int64)
        m = getattr(cupyx.scipy.sparse, container)(cupy.asarray(d))
        s = getattr(scipy.sparse, container)(d)
        got = cupy.asnumpy(cupy.asarray(m.sum(axis=axis))).ravel()
        ref = numpy.asarray(s.sum(axis=axis)).ravel()
        numpy.testing.assert_array_equal(got, ref)

    def test_uint64_sum_exact_past_2_53(self):
        big = 1 << 62
        d = numpy.array([[big + 7, big + 3, big + 5]], dtype=numpy.uint64)
        m = cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
        # Sum stays under 2**64 here, so it is exact (and > 2**53).
        assert int(m.sum()) == 3 * big + 15

    def test_dia_int64_sum_exact(self):
        # DIA ``data`` holds off-matrix padding; the exact path must
        # route through COO so padding is not counted.
        import scipy.sparse
        data = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=numpy.int64)
        offsets = numpy.array([0, -1])
        m = cupyx.scipy.sparse.dia_matrix(
            (cupy.asarray(data), cupy.asarray(offsets)), shape=(3, 3))
        s = scipy.sparse.dia_matrix((data, offsets), shape=(3, 3))
        assert int(m.sum()) == int(s.sum())

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        return xp.asarray(m.mean()), xp.asarray(m.mean(axis=1))

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_max_min(self, xp, sp, dtype):
        # max/min preserve the input dtype (scipy parity).
        m = _make(xp, sp, dtype)
        assert m.max().dtype == dtype
        return (xp.asarray(m.max()), xp.asarray(m.min()),
                m.max(axis=1).toarray(), m.min(axis=0).toarray())

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_argmax_argmin_axis(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        return (xp.asarray(m.argmax(axis=1)),
                xp.asarray(m.argmin(axis=0)))

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    def test_tuple_axis_full_reduction(self, dtype):
        # scipy accepts a 2-axis tuple as a full reduction; sum/mean and
        # max/min/argmax/argmin must all collapse it, not reject it.
        import scipy.sparse
        d = numpy.array([[1, 0, 3], [0, 2, 0], [4, 0, 5]], dtype=dtype)
        m = cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
        s = scipy.sparse.csr_matrix(d)
        assert int(m.sum(axis=(0, 1))) == int(s.sum(axis=(0, 1)))
        assert m.max(axis=(0, 1)) == s.max(axis=(0, 1))
        assert m.min(axis=(0, 1)) == s.min(axis=(0, 1))
        assert int(m.argmax(axis=(0, 1))) == int(s.argmax(axis=(0, 1)))
        assert int(m.argmin(axis=(0, 1))) == int(s.argmin(axis=(0, 1)))

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    def test_tuple_axis_length1(self, dtype):
        # A length-1 tuple ``(i,)`` means axis ``i``.
        import scipy.sparse
        d = numpy.array([[1, 0, 3], [0, 2, 0], [4, 0, 5]], dtype=dtype)
        m = cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
        s = scipy.sparse.csr_matrix(d)
        cupy.testing.assert_array_equal(
            m.max(axis=(1,)).toarray().ravel(),
            numpy.asarray(s.max(axis=(1,)).todense()).ravel())
        cupy.testing.assert_array_equal(
            cupy.asnumpy(m.argmin(axis=(0,))).ravel(),
            numpy.asarray(s.argmin(axis=(0,))).ravel())

    def test_int64_axis_reductions_exact_past_2_53(self):
        # Values above 2**53 are not exactly representable in float64,
        # so axis max/min and argmax must reduce in an int64 accumulator.
        import scipy.sparse
        big = (1 << 53) + 3
        d = numpy.array([[big, 0, big + 1], [0, big + 2, 0]],
                        dtype=numpy.int64)
        m = cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
        s = scipy.sparse.csr_matrix(d)
        cupy.testing.assert_array_equal(
            m.max(axis=1).toarray().ravel(),
            cupy.asarray(numpy.asarray(s.max(axis=1).todense()).ravel()))
        cupy.testing.assert_array_equal(
            cupy.asnumpy(m.argmax(axis=1)).ravel(),
            numpy.asarray(s.argmax(axis=1)).ravel())
        assert int(m.max()) == big + 2

    def test_uint64_axis_max_exact_past_2_53(self):
        big = (1 << 63) + 5
        d = numpy.array([[big, 0, big + 1], [0, big + 2, 0]],
                        dtype=numpy.uint64)
        m = cupyx.scipy.sparse.csr_matrix(cupy.asarray(d))
        cupy.testing.assert_array_equal(
            m.max(axis=1).toarray().ravel(),
            cupy.asarray(d.max(axis=1)))
        assert int(m.max()) == big + 2

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_getitem_and_slices(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        return (xp.asarray(m[0, 1]),
                m[0:2, :].toarray(),
                m[:, 1:3].toarray())

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_fancy_indexing(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        cols = xp.array([0, 2, 3])
        rows = xp.array([0, 2])
        return m[:, cols].toarray(), m[rows, :].toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_diagonal_setdiag(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        m.setdiag(9)
        return m.diagonal(), m.toarray()


@testing.with_requires('scipy>=1.12')
class TestValueDtypeExactInt64:
    """64-bit integer reductions must stay exact past float64's 2**53 mantissa
    limit -- a float64 accumulator (or comparison carrier) would round.  All
    magnitudes here exceed 2**53/2**63 yet keep sums in range, so any float64
    rounding diverges from scipy's exact integer result.
    """

    def _big(self, xp, dtype):
        # values > 2**53 (int64) / > 2**63 (uint64); sums stay in range
        base = 2 ** 63 if numpy.dtype(dtype).kind == 'u' else 2 ** 53
        return xp.array([[base + 1, 0, base + 5],
                         [0, base + 3, 0],
                         [base + 7, 0, base + 9]], dtype=dtype)

    @pytest.mark.parametrize('dtype', [numpy.int64, numpy.uint64])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_sum_exact(self, xp, sp, dtype):
        m = sp.csr_matrix(self._big(xp, dtype))
        # axis=None returns a scalar; wrap so the comparison sees the value
        return xp.array(m.sum()), m.sum(axis=0), m.sum(axis=1)

    @pytest.mark.parametrize('dtype', [numpy.int64, numpy.uint64])
    @pytest.mark.parametrize('container', ['csr_matrix', 'csc_matrix'])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_max_min_axis_exact(self, xp, sp, dtype, container):
        m = getattr(sp, container)(self._big(xp, dtype))
        return (m.max(axis=0).toarray(), m.max(axis=1).toarray(),
                m.min(axis=0).toarray(), m.min(axis=1).toarray())

    @pytest.mark.parametrize('dtype', [numpy.int64, numpy.uint64])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_argmax_axis_exact(self, xp, sp, dtype):
        m = sp.csr_matrix(self._big(xp, dtype))
        return m.argmax(axis=0), m.argmax(axis=1)


@testing.with_requires('scipy>=1.12')
class TestValueDtypeConstructFunctions:

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_stack_and_blocks(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        h = sp.hstack([m, m])
        v = sp.vstack([m, m])
        b = sp.block_diag([m, m])
        assert h.dtype == v.dtype == b.dtype == dtype
        return h.toarray(), v.toarray(), b.toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_kron(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        return (sp.kron(m, m).toarray(),
                sp.kronsum(m[:, :3], m[:, :3]).toarray())

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_eye_identity(self, xp, sp, dtype):
        e = sp.eye(4, dtype=dtype, format='csr')
        assert e.dtype == dtype
        return e.toarray(), sp.identity(3, dtype=dtype, format='coo').toarray()

    @pytest.mark.parametrize('dtype', _key_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_tril_triu_find(self, xp, sp, dtype):
        m = _make(xp, sp, dtype)
        i, j, v = sp.find(m)
        assert v.dtype == dtype
        return sp.tril(m).toarray(), sp.triu(m).toarray()

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.uint64])
    def test_random_int(self, dtype):
        m = cupyx.scipy.sparse.random(
            10, 10, density=0.5, dtype=dtype, random_state=7)
        assert m.dtype == dtype
        assert m.nnz == 50


class TestValueDtypeFloat16:
    """float16 is a CuPy extension: scipy.sparse rejects it, so these
    compare against dense cupy computation instead."""

    def _make(self, container='csr_matrix'):
        dense = cupy.array([[0, 1, 0, 2],
                            [3, 0, 0, 0],
                            [0, 4, 5, 0]], dtype=numpy.float16)
        return getattr(cupyx.scipy.sparse, container)(dense), dense

    @pytest.mark.parametrize('container', _containers_b)
    def test_construct_roundtrip(self, container):
        m, dense = self._make(container)
        assert m.dtype == numpy.float16
        cupy.testing.assert_array_equal(m.toarray(), dense)
        cupy.testing.assert_array_equal(m.tocsc().tocoo().tocsr().toarray(),
                                        dense)

    def test_arithmetic(self):
        m, dense = self._make()
        cupy.testing.assert_array_equal((m + m).toarray(), dense + dense)
        cupy.testing.assert_array_equal((m * 2).toarray(), dense * 2)
        cupy.testing.assert_array_equal(m.multiply(m).toarray(),
                                        dense * dense)
        # scipy rejects float16, so the oracle is the dense cupy ufunc:
        # a weak scalar keeps the width (float16 / 2 -> float16), unlike
        # scipy's "real division upcasts to float64" for float32/64.
        q = m / 2
        assert q.dtype == (dense / 2).dtype == numpy.float16
        cupy.testing.assert_array_equal(q.toarray(), dense / 2)

    def test_matmul(self):
        m, dense = self._make()
        v = cupy.array([1, 2, 0, 3], dtype=numpy.float16)
        r = m @ v
        assert r.dtype == numpy.float16
        cupy.testing.assert_allclose(r, dense @ v, rtol=1e-3)
        p = m @ m.T
        assert p.dtype == numpy.float16
        cupy.testing.assert_allclose(p.toarray(), dense @ dense.T, rtol=1e-3)

    def test_spgemm_heavy_accumulation_accuracy(self):
        # float16 sparse @ sparse must accumulate in float32, not
        # float16, or heavy per-element accumulation loses ~a decimal
        # digit.  Compare against the exact float64 reference.
        rng = numpy.random.RandomState(0)
        m, k, n = 40, 300, 40
        ad = rng.rand(m, k).astype(numpy.float16)
        bd = rng.rand(k, n).astype(numpy.float16)
        a = cupyx.scipy.sparse.csr_matrix(cupy.asarray(ad))
        b = cupyx.scipy.sparse.csr_matrix(cupy.asarray(bd))
        c = cupy.asnumpy((a @ b).toarray()).astype(numpy.float64)
        ref = ad.astype(numpy.float64) @ bd.astype(numpy.float64)
        rel = numpy.abs(c - ref) / (numpy.abs(ref) + 1e-9)
        # float32 accumulation keeps max rel error well under 1e-2; a
        # float16 accumulator would be ~1e-2 here.
        assert rel.max() < 2e-3, rel.max()

    def test_reductions(self):
        m, dense = self._make()
        assert m.sum().dtype == numpy.float16
        assert m.sum() == dense.sum()
        assert m.max() == dense.max()
        assert m.max().dtype == numpy.float16
        cupy.testing.assert_array_equal(
            m.argmax(axis=1), dense.argmax(axis=1)[:, None])

    def test_fancy_indexing(self):
        m, dense = self._make()
        cols = cupy.array([0, 2, 3])
        cupy.testing.assert_array_equal(m[:, cols].toarray(),
                                        dense[:, cols])

    def test_sum_duplicates(self):
        data = cupy.array([1.5, 2.5], dtype=numpy.float16)
        row = cupy.array([0, 0], 'i')
        col = cupy.array([0, 0], 'i')
        m = cupyx.scipy.sparse.coo_matrix((data, (row, col)), shape=(1, 1))
        m.sum_duplicates()
        assert m.dtype == numpy.float16
        assert m.toarray()[0, 0] == 4.0


# bfloat16 is provided by the optional ml_dtypes package (numpy has no
# native bfloat16); CuPy only supports it when ml_dtypes is installed AND
# numpy >= 2.1.2.  Skip on exactly the condition the sparse gate uses
# (``_sputils.bfloat16 is None`` captures both), while keeping the callable
# ml_dtypes type for constructing bfloat16 scalars/arrays in the tests.
try:
    import ml_dtypes as _ml_dtypes
    _bfloat16 = _ml_dtypes.bfloat16
except ImportError:
    _bfloat16 = None


@pytest.mark.skipif(cupyx.scipy.sparse._sputils.bfloat16 is None,
                    reason='requires ml_dtypes and numpy>=2.1.2')
class TestValueDtypeBfloat16:
    """bfloat16 is a CuPy/ml_dtypes extension (scipy.sparse rejects it, and
    numpy has no native bfloat16), so these compare against a float32 dense
    reference.  bfloat16 has ``kind == 'V'`` and only promotes with itself,
    float32/64 and complex -- mixing it with ints or float16 raises
    DTypePromotionError, exactly as dense numpy/CuPy do.
    """

    def _make(self, container='csr_matrix'):
        # integer-valued entries are exact in bfloat16, so elementwise ops
        # and small matmuls compare exactly against the float32 reference.
        ref = numpy.array([[0, 1, 0, 2],
                           [3, 0, 0, 0],
                           [0, 4, 5, 0]], dtype=numpy.float32)
        dense = cupy.asarray(ref.astype(_bfloat16))
        return getattr(cupyx.scipy.sparse, container)(dense), ref

    def _f32(self, x):
        if hasattr(x, 'toarray'):
            x = x.toarray()
        return cupy.asnumpy(x).astype(numpy.float32)

    def test_gate(self):
        from cupyx.scipy.sparse import _sputils
        bf = numpy.dtype(_bfloat16)
        assert _sputils.is_sparse_data_dtype(bf)
        assert _sputils.is_bfloat16(bf)
        assert not _sputils.is_bfloat16(numpy.float16)
        _sputils.check_data_dtype(bf)  # must not raise
        # bfloat16 preserved by upcast; promotes with float32 -> float32
        assert _sputils.upcast(bf) == bf
        assert _sputils.upcast(bf, numpy.float32) == numpy.float32

    @pytest.mark.parametrize('container', _containers_b)
    def test_construct_roundtrip(self, container):
        m, ref = self._make(container)
        assert m.dtype == numpy.dtype(_bfloat16)
        numpy.testing.assert_array_equal(self._f32(m), ref)
        numpy.testing.assert_array_equal(
            self._f32(m.tocsc().tocoo().tocsr()), ref)
        numpy.testing.assert_array_equal(self._f32(m.T), ref.T)

    def test_arithmetic(self):
        m, ref = self._make()
        assert (m + m).dtype == numpy.dtype(_bfloat16)
        numpy.testing.assert_array_equal(self._f32(m + m), ref + ref)
        numpy.testing.assert_array_equal(self._f32(m - m), ref - ref)
        numpy.testing.assert_array_equal(self._f32(m.multiply(m)), ref * ref)
        numpy.testing.assert_array_equal(
            self._f32(m * _bfloat16(2.0)), ref * 2)
        numpy.testing.assert_array_equal(self._f32(-m), -ref)
        # scipy rejects bfloat16, so the oracle is the dense ufunc -- and it
        # disagrees with numpy.result_type here: an integer weak scalar keeps
        # bfloat16 (bf / 2 -> bf) while a float weak scalar promotes to
        # float32 (bf / 2.0 -> float32), matching ``dense / scalar`` exactly.
        dense = cupy.asarray(ref.astype(_bfloat16))
        for scalar in (2, 2.0):
            q = m / scalar
            assert q.dtype == (dense / scalar).dtype, scalar
            numpy.testing.assert_array_equal(
                self._f32(q), self._f32(dense / scalar), err_msg=str(scalar))

    def test_comparison(self):
        # bfloat16 (kind 'V') must be treated as a float in _comparison:
        # otherwise the scalar keeps its int dtype and ``op(bf, int)`` raises
        # DTypePromotionError.  Every operator (esp. ==/!=/</<=) must match
        # the dense bfloat16 result, against both int and float scalars.
        m, ref = self._make()
        dense = cupy.asarray(ref.astype(_bfloat16))
        import operator
        for op in (operator.eq, operator.ne, operator.lt, operator.le,
                   operator.gt, operator.ge):
            for scalar in (0, 3, 2.0):
                numpy.testing.assert_array_equal(
                    self._f32(op(m, scalar)).astype(bool),
                    cupy.asnumpy(op(dense, scalar)).astype(bool),
                    err_msg=f'{op.__name__}({scalar})')

    def test_matmul(self):
        m, ref = self._make()
        v = cupy.asarray(numpy.array([1, 2, 0, 3],
                                     dtype=numpy.float32).astype(_bfloat16))
        r = m @ v
        assert r.dtype == numpy.dtype(_bfloat16)
        numpy.testing.assert_allclose(
            self._f32(r), ref @ numpy.array([1, 2, 0, 3], numpy.float32),
            rtol=1e-2)
        p = m @ m.T
        assert p.dtype == numpy.dtype(_bfloat16)
        numpy.testing.assert_allclose(self._f32(p), ref @ ref.T, rtol=1e-2)

    def test_spgemm_accuracy(self):
        # sparse @ sparse routes through the pure-CuPy fallback (bfloat16 is
        # not cuSPARSE-native below cc 8.0); it must widen to float32, not
        # accumulate in bfloat16.
        rng = numpy.random.RandomState(0)
        ad = rng.rand(30, 200).astype(_bfloat16)
        bd = rng.rand(200, 30).astype(_bfloat16)
        a = cupyx.scipy.sparse.csr_matrix(cupy.asarray(ad))
        b = cupyx.scipy.sparse.csr_matrix(cupy.asarray(bd))
        c = self._f32(a @ b).astype(numpy.float64)
        ref = ad.astype(numpy.float64) @ bd.astype(numpy.float64)
        rel = numpy.abs(c - ref) / (numpy.abs(ref) + 1e-9)
        # float32 accumulation keeps error near bfloat16's own resolution
        # (~8-bit mantissa); a bfloat16 accumulator would be far worse.
        assert rel.max() < 5e-2, rel.max()

    def test_reductions(self):
        m, ref = self._make()
        assert m.sum().dtype == numpy.dtype(_bfloat16)
        assert float(m.sum()) == ref.sum()
        numpy.testing.assert_array_equal(self._f32(m.sum(axis=0)),
                                         ref.sum(axis=0).reshape(1, -1))
        numpy.testing.assert_array_equal(self._f32(m.sum(axis=1)),
                                         ref.sum(axis=1).reshape(-1, 1))
        assert float(m.max()) == ref.max()
        assert m.max().dtype == numpy.dtype(_bfloat16)
        assert float(m.min()) == ref.min()
        # argmax/argmin stage through float32 (the bfloat16 arg kernel is
        # not instantiated); indices must still be exact.
        numpy.testing.assert_array_equal(
            cupy.asnumpy(m.argmax(axis=1)),
            ref.argmax(axis=1).reshape(-1, 1))
        numpy.testing.assert_array_equal(
            cupy.asnumpy(m.argmax(axis=0)),
            ref.argmax(axis=0).reshape(1, -1))

    def test_indexing(self):
        m, ref = self._make()
        assert float(m[1, 0]) == ref[1, 0]
        assert float(m[0, 0]) == ref[0, 0]  # stored/implicit zero
        cols = cupy.array([0, 2, 3])
        numpy.testing.assert_array_equal(self._f32(m[:, cols]),
                                         ref[:, [0, 2, 3]])
        numpy.testing.assert_array_equal(self._f32(m[1:3]), ref[1:3])
        numpy.testing.assert_array_equal(self._f32(m[[0, 2]]), ref[[0, 2]])

    def test_sum_duplicates(self):
        # Regression: the COO sum_duplicates scatter dispatches on
        # ``dtype.kind``; bfloat16's kind is 'V', so before the fix it
        # matched no branch and silently returned a zero-filled buffer.
        data = cupy.asarray(numpy.array([1.5, 2.5],
                                        dtype=numpy.float32).astype(_bfloat16))
        row = cupy.array([0, 0], 'i')
        col = cupy.array([0, 0], 'i')
        m = cupyx.scipy.sparse.coo_matrix((data, (row, col)), shape=(1, 1))
        m.sum_duplicates()
        assert m.dtype == numpy.dtype(_bfloat16)
        assert float(m.toarray()[0, 0]) == 4.0

    def test_mixed_dtype_promotes_like_dense(self):
        # bfloat16 has no scipy oracle, so ELEMENTWISE ops mixing it with an
        # int or float16 match dense CuPy (numpy.promote_types can't resolve
        # them, but the ufunc loops do): bf16+int32 -> float64, bf16+float16
        # -> float32.  Matmul and concatenation still raise, exactly as dense
        # does for ``bf16 @ int`` / ``concatenate([bf16, int])``.
        m, ref = self._make()
        mi = cupyx.scipy.sparse.csr_matrix(
            cupy.asarray((ref * 2).astype(numpy.int32)))
        r = m + mi
        assert r.dtype == numpy.float64
        numpy.testing.assert_allclose(self._f32(r), ref * 3, rtol=1e-2)
        assert m.multiply(mi).dtype == numpy.float64
        assert (m - mi).dtype == numpy.float64
        assert m.maximum(mi).dtype == numpy.float64   # binopt path
        assert m.minimum(mi).dtype == numpy.float64
        assert (m == mi).dtype == numpy.bool_
        mf16 = cupyx.scipy.sparse.csr_matrix(
            cupy.asarray(ref.astype(numpy.float16)))
        assert (m + mf16).dtype == numpy.float32
        with pytest.raises(numpy.exceptions.DTypePromotionError):
            (m @ mi.T).toarray()
        with pytest.raises(numpy.exceptions.DTypePromotionError):
            cupyx.scipy.sparse.hstack([m, mi])

    def test_random(self):
        m = cupyx.scipy.sparse.random(
            10, 10, density=0.5, dtype=_bfloat16, random_state=7)
        assert m.dtype == numpy.dtype(_bfloat16)
        assert m.nnz == 50

    def test_matrix_power(self):
        from cupyx.scipy.sparse.linalg import matrix_power
        d = cupy.asarray((numpy.eye(4) * 2).astype(_bfloat16))
        m = cupyx.scipy.sparse.csr_matrix(d)
        r = matrix_power(m, 3)
        assert r.dtype == numpy.dtype(_bfloat16)
        numpy.testing.assert_allclose(
            self._f32(r), numpy.eye(4, dtype=numpy.float32) * 8, rtol=1e-2)

    def test_stack_and_eye_preserve_dtype(self):
        m, ref = self._make()
        bf = numpy.dtype(_bfloat16)
        assert cupyx.scipy.sparse.eye(3, dtype=_bfloat16).dtype == bf
        assert cupyx.scipy.sparse.hstack([m, m]).dtype == bf
        assert cupyx.scipy.sparse.vstack([m, m]).dtype == bf
        numpy.testing.assert_array_equal(
            self._f32(cupyx.scipy.sparse.hstack([m, m])),
            numpy.hstack([ref, ref]))

    def test_1d_array(self):
        # 1-D sparse array preserves bfloat16 through construct/toarray/sum
        ref = numpy.array([0, 1.5, 0, 2.0], dtype=numpy.float32)
        v = cupyx.scipy.sparse.csr_array(cupy.asarray(ref.astype(_bfloat16)))
        assert v.dtype == numpy.dtype(_bfloat16)
        numpy.testing.assert_array_equal(self._f32(v), ref)
        assert float(v.sum()) == ref.sum()

    def test_str_does_not_crash(self):
        # scipy rejects bfloat16, so str() must fall back to repr (not raise)
        m, _ = self._make()
        s = str(m)
        assert 'bfloat16' in s and 'sparse' in s


@testing.with_requires('scipy>=1.12')
class TestValueDtypeStrRepr:
    """str()/print() must never crash.  For scipy-supported dtypes ``str``
    delegates to scipy; for CuPy-only float16 (which scipy.sparse rejects
    with ValueError) it falls back to ``repr`` -- a regression guard, since
    the fallback previously caught only RuntimeError/NotImplementedError.
    """

    @pytest.mark.parametrize('dtype', _key_dtypes_b + [bool, numpy.float64,
                                                       numpy.complex128])
    def test_str_scipy_dtypes(self, dtype):
        m = _make(cupy, cupyx.scipy.sparse, dtype)
        assert str(m)  # delegates to scipy, must not raise
        assert repr(m)

    def test_str_float16_falls_back_to_repr(self):
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.asarray(numpy.eye(3, dtype=numpy.float16)))
        s = str(m)  # scipy rejects float16 -> repr fallback (no ValueError)
        assert 'float16' in s and 'sparse' in s


@testing.with_requires('scipy>=1.12')
class TestValueDtypeComplexArgReduce:
    """argmax/argmin along an axis for complex reduces on the real part
    (matching scipy and how min/max cast complex->float64); the complex
    arg-reduction kernel is not instantiated, so this used to crash.
    """

    @pytest.mark.parametrize('dtype', [numpy.complex64, numpy.complex128])
    @pytest.mark.parametrize('op', ['argmax', 'argmin'])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_complex_argreduce_axis(self, xp, sp, dtype, op):
        dense = xp.array([[1 + 9j, 0, 3 + 1j],
                          [0, 2 + 0j, 0],
                          [4 + 2j, 0, 5 - 7j]], dtype=dtype)
        m = sp.csr_matrix(dense)
        return getattr(m, op)(axis=0), getattr(m, op)(axis=1)


@testing.with_requires('scipy>=1.12')
class TestValueDtypeComparisonsFull:
    """Every comparison operator, against int / out-of-range / fractional
    scalars, across formats -- must match scipy exactly (no int-scalar
    overflow or truncation).
    """

    @pytest.mark.parametrize('scalar', [2, 300, 2.5, -1])
    @pytest.mark.parametrize('op', ['_eq_', '_ne_', '_lt_', '_le_',
                                    '_gt_', '_ge_'])
    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int64, numpy.uint8])
    @pytest.mark.parametrize('container', ['csr_matrix', 'csc_matrix'])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_scalar_comparison(self, xp, sp, dtype, op, scalar, container):
        import operator
        m = _make(xp, sp, dtype, container)
        pyop = getattr(operator, op.strip('_'))
        return pyop(m, scalar).toarray()


@testing.with_requires('scipy>=1.12')
class TestValueDtypeRandomDtypes:
    """random() over every storable dtype: correct dtype, exact nnz, and
    dtype-appropriate values (ints span negatives, complex has imag).
    """

    @pytest.mark.parametrize('dtype', _int_dtypes_b + [bool, numpy.float32,
                                                       numpy.float64,
                                                       numpy.complex128])
    def test_random_dtype_and_values(self, dtype):
        m = cupyx.scipy.sparse.random(
            12, 12, density=0.5, dtype=dtype, random_state=3)
        assert m.dtype == numpy.dtype(dtype)
        assert m.nnz == 72
        data = cupy.asnumpy(m.data)
        if numpy.dtype(dtype).kind == 'c':
            assert numpy.any(data.imag != 0)
        elif numpy.dtype(dtype).kind == 'i':
            assert numpy.any(data < 0)

    def test_random_float16(self):
        m = cupyx.scipy.sparse.random(
            10, 10, density=0.3, dtype=numpy.float16, random_state=1)
        assert m.dtype == numpy.float16


@testing.with_requires('scipy>=1.12')
class TestValueDtypeGating:
    """Extended-precision dtypes the GPU cannot store are rejected; the
    identity/width gate must not be fooled by dtype-char aliasing.
    """

    @pytest.mark.parametrize('dtype', [numpy.longdouble, numpy.clongdouble])
    def test_reject_extended_precision(self, dtype):
        if numpy.dtype(dtype).itemsize <= 16 and dtype is numpy.clongdouble:
            pytest.skip('clongdouble not extended on this platform')
        d = cupy.zeros((2, 2), dtype=numpy.float64)
        with pytest.raises(ValueError, match='does not support dtype'):
            cupyx.scipy.sparse.csr_matrix(d, dtype=dtype)

    def test_reject_via_astype(self):
        m = cupyx.scipy.sparse.csr_matrix(cupy.asarray(numpy.eye(2)))
        with pytest.raises((ValueError, TypeError)):
            m.astype(numpy.longdouble)


@testing.with_requires('scipy>=1.12')
class TestValueDtypeRegressions:
    """Regression guards for value-dtype behaviours that are easy to break:
    exact bool/int counts, count_nonzero along an axis, explicit power dtype,
    and float16 matrix_power.
    """

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_bool_sum_is_exact_count(self, xp, sp):
        # bool sum must widen to int64 and count exactly, NOT collapse to a
        # bool "any-nonzero" flag.
        dense = xp.array([[True, False, True],
                          [True, True, False]], dtype=xp.bool_)
        m = sp.csr_matrix(dense)
        return xp.array(m.sum()), m.sum(axis=0), m.sum(axis=1)

    @pytest.mark.parametrize('dtype', [numpy.int16, numpy.int64, bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_count_nonzero_axis(self, xp, sp, dtype):
        # scipy's count_nonzero(axis) dtype is inconsistent (int64 for axis=0
        # but int32 for axis=1); mine is uniformly the platform int, so
        # compare the counts as int64.
        m = _make(xp, sp, dtype)
        return (m.count_nonzero(axis=0).astype(numpy.int64),
                m.count_nonzero(axis=1).astype(numpy.int64))

    @pytest.mark.parametrize('target', [numpy.int64, numpy.float64])
    @pytest.mark.parametrize('dtype', [numpy.int8, bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_power_explicit_dtype(self, xp, sp, dtype, target):
        m = _make(xp, sp, dtype)
        r = m.power(2, dtype=target)
        assert r.dtype == target
        return r.toarray()

    @pytest.mark.parametrize('n', [2, 3])
    def test_bool_power_0d_cupy_exponent(self, n):
        # ``isscalarlike`` accepts a 0-D cupy array, but the bool carrier is
        # derived with numpy (``np.ones(1, bool) ** n``), which rejects a
        # device operand -- power() must read it to host first.  A 0-D int64
        # array exponent matches numpy's ``np.int64`` carrier (int64), not
        # the python-int carrier (int8 at n == 2).
        dense = numpy.array([[True, False], [False, True]])
        m = cupyx.scipy.sparse.csr_array(cupy.asarray(dense))
        r = m.power(cupy.asarray(n))
        expect = (numpy.ones(1, numpy.bool_) ** numpy.int64(n)).dtype
        assert r.dtype == expect
        numpy.testing.assert_array_equal(
            cupy.asnumpy(r.toarray()), (dense ** n).astype(expect))

    def test_matrix_power_float16(self):
        from cupyx.scipy.sparse.linalg import matrix_power
        m = cupyx.scipy.sparse.csr_matrix(
            cupy.asarray(numpy.eye(4, dtype=numpy.float16) * 2))
        r = matrix_power(m, 3)
        assert r.dtype == numpy.float16
        cupy.testing.assert_allclose(
            r.toarray(), numpy.eye(4, dtype=numpy.float32) * 8, rtol=1e-2)

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int16, numpy.int32,
                                       numpy.int64, numpy.float16])
    def test_toarray_dedups_despite_stale_canonical_flag(self, dtype):
        # Duplicate coordinates must sum in toarray() even when
        # has_canonical_format is (wrongly) True.  The 8/16-bit direct-write
        # densify path used to trust the stale flag and skip the dedup,
        # diverging from the atomicAdd dtypes and scipy.
        data = cupy.asarray(numpy.array([5, 5, 3], dtype=dtype))
        indices = cupy.asarray(numpy.array([0, 0, 1], 'i'))
        indptr = cupy.asarray(numpy.array([0, 2, 3], 'i'))
        m = cupyx.scipy.sparse.csr_matrix((data, indices, indptr),
                                          shape=(2, 2))
        m.has_canonical_format = True  # stale: (0, 0) is duplicated
        numpy.testing.assert_array_equal(
            cupy.asnumpy(m.toarray()),
            numpy.array([[10, 0], [0, 3]], dtype=dtype))

    @pytest.mark.parametrize('dtype', _int_dtypes_b)
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_int_add_overflow(self, xp, sp, dtype):
        # A + B for integers must wrap modularly like numpy/scipy, on both
        # the cuSPARSE float64-upcast fast path (<=32-bit) and the pure-CuPy
        # fallback (64-bit).  A + A doubles every entry; near-max values
        # overflow the dtype and must wrap, not saturate.
        hi = int(numpy.iinfo(dtype).max)
        dense = xp.array([[hi, 0, hi // 2], [0, hi - 3, 0]], dtype=dtype)
        A = sp.csr_matrix(dense)
        return (A + A).toarray()

    @pytest.mark.parametrize('dtype', [numpy.int64, numpy.uint64, bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_sum_duplicates_nonsquare(self, xp, sp, dtype):
        # The composite-key ``sum_duplicates`` (row*ncols+col) must match
        # scipy for non-square 2-D shapes, with duplicate coordinates summed.
        row = xp.array([2, 0, 2, 0, 1], 'i')
        col = xp.array([4, 1, 4, 1, 0], 'i')
        data = xp.array([1, 2, 3, 4, 5], dtype=dtype)
        m = sp.coo_matrix((data, (row, col)), shape=(3, 6))
        m.sum_duplicates()
        assert m.dtype == dtype
        return m.toarray()

    @pytest.mark.parametrize('dtype', [numpy.float64, numpy.int64, bool])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_sum_duplicates_1d(self, xp, sp, dtype):
        # The composite key is 1-D-aware (via _shape_as_2d): a 1-D coo_array
        # uses the (1, N) backing, so ``row`` is all zeros and the key is
        # just ``col``.  Duplicate 1-D indices must coalesce like scipy.
        coords = xp.array([3, 1, 3, 1, 4], 'i')
        data = xp.array([1, 2, 3, 4, 5], dtype=dtype)
        m = sp.coo_array((data, (coords,)), shape=(6,))
        m.sum_duplicates()
        assert m.dtype == dtype
        return m.toarray()

    @pytest.mark.parametrize('container', ['coo_array', 'csr_array'])
    def test_1d_int64_sum_exact_past_float64(self, container):
        # A 1-D int64 array whose total is 2**53 + 1: the float64 axis=None
        # matmul path would round it to 2**53, but the exact-int path (sum
        # ``data`` directly) keeps it exact -- matching scipy/numpy.
        v = numpy.array([2 ** 53, 1], dtype=numpy.int64)
        m = getattr(cupyx.scipy.sparse, container)(
            (cupy.asarray(v), (cupy.asarray([0, 5]),)), shape=(6,))
        assert int(cupy.asnumpy(m.sum())) == 2 ** 53 + 1

    @pytest.mark.parametrize('dtype', [numpy.int8, numpy.int16, numpy.int32,
                                       numpy.uint8, numpy.uint32])
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_int_matmul_overflow(self, xp, sp, dtype):
        # sparse @ dense integer matmul whose result overflows the dtype must
        # wrap modularly like numpy, not saturate.  The exact int64-carrier
        # scatter accumulates in a 64-bit ring and wraps on the cast back.
        k = 40
        val = int(numpy.sqrt(int(numpy.iinfo(dtype).max)))  # k*val**2 >> max
        A = sp.csr_matrix(xp.full((1, k), val, dtype=dtype))
        B = xp.full((k, 1), val, dtype=dtype)
        return A @ B

    @pytest.mark.parametrize('container', ['csr_matrix', 'csc_matrix'])
    def test_int64_matmul_exact_past_float64(self, container):
        # int64 @ dense whose true result is 2**53 + 1: a float64 compute
        # rounds it to 2**53, but the exact int64-carrier scatter keeps it
        # exact (matches scipy/numpy).
        m = getattr(cupyx.scipy.sparse, container)(
            cupy.asarray(numpy.array([[2 ** 53, 1]], dtype=numpy.int64)))
        x = cupy.asarray(numpy.array([1, 1], dtype=numpy.int64))
        assert int(cupy.asnumpy(m @ x)[0]) == 2 ** 53 + 1

    def test_wide_operand_chunks_and_stays_exact(self, monkeypatch):
        # The exact int scatter materialises an (nnz, N) product, so a wide
        # dense operand is column-blocked to bound memory.  Force the chunk
        # loop (tiny free memory) and assert it still matches scipy exactly,
        # down to a 1-column block -- the blocking must not change the result.
        import scipy.sparse
        from cupyx import cusparse
        rng = numpy.random.default_rng(0)
        a = scipy.sparse.random(120, 120, density=0.1, format='csr',
                                dtype=numpy.float64)
        a.data = rng.integers(1, 20, a.nnz).astype(numpy.int64)
        a = a.astype(numpy.int64)
        x = rng.integers(1, 20, (120, 16)).astype(numpy.int64)
        ref = a @ x
        ag = cupyx.scipy.sparse.csr_matrix(a)
        xg = cupy.asarray(x)

        class _FakePool:
            def free_bytes(self):
                return 0
        monkeypatch.setattr(cusparse._cupy, 'get_default_memory_pool',
                            lambda: _FakePool())
        per_col = ag.nnz * 8
        for cols_per_block in (3, 1):  # force multi-col and 1-col blocks
            monkeypatch.setattr(
                cupy.cuda.runtime, 'memGetInfo',
                lambda cpb=cols_per_block: (4 * per_col * cpb, 1 << 34))
            got = cupy.asnumpy(cusparse._cupy_csr_dense_matmul(ag, xg))
            numpy.testing.assert_array_equal(got, ref)
