"""Tests for sparse array classes (csr_array, csc_array, coo_array, dia_array).
"""
from __future__ import annotations

import operator
import warnings

import numpy
import pytest
try:
    import scipy.sparse  # noqa: F401
except ImportError:
    pass

import cupy
from cupy import testing
from cupyx.scipy import sparse


SPARSE_FORMATS = ('csr', 'csc', 'coo', 'dia')


def _make_small_sparse(
        fmt, *, array=True, shape=(2, 3), dtype=numpy.float64,
        xp=cupy, spmod=sparse):
    """Return a small sparse array/matrix with values on the main diagonal."""
    kind = 'array' if array else 'matrix'
    if fmt == 'dia':
        width = min(shape)
        data = xp.ones((1, width), dtype=dtype)
        offsets = xp.array([0], dtype='i')
        cls = spmod.dia_array if array else spmod.dia_matrix
        return cls((data, offsets), shape=shape)
    cls = getattr(spmod, f'{fmt}_{kind}')
    dense = xp.zeros(shape, dtype=dtype)
    diag_n = min(shape)
    if diag_n:
        diag_idx = xp.arange(diag_n)
        dense[diag_idx, diag_idx] = xp.arange(1, diag_n + 1, dtype=dtype)
    return cls(dense)


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


def _make_csc(xp, sp, dtype, *, array=False):
    """3x3 diagonal CSC with values [1, 2, 3]."""
    data = xp.array([1, 2, 3], dtype)
    indices = xp.array([0, 1, 2], 'i')
    indptr = xp.array([0, 1, 2, 3], 'i')
    cls = sp.csc_array if array else sp.csc_matrix
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

    @pytest.mark.parametrize('fmt', SPARSE_FORMATS)
    def test_array_issparse(self, fmt):
        A = _make_small_sparse(fmt, array=True)
        assert sparse.issparse(A)

    @pytest.mark.parametrize('fmt', SPARSE_FORMATS)
    def test_array_not_isspmatrix(self, fmt):
        A = _make_small_sparse(fmt, array=True)
        assert not sparse.isspmatrix(A)

    @pytest.mark.parametrize('fmt', SPARSE_FORMATS)
    def test_array_isinstance_sparray(self, fmt):
        A = _make_small_sparse(fmt, array=True)
        assert isinstance(A, sparse.sparray)
        assert not isinstance(A, sparse.spmatrix)

    @pytest.mark.parametrize('fmt', SPARSE_FORMATS)
    def test_matrix_isinstance_spmatrix(self, fmt):
        M = _make_small_sparse(fmt, array=False)
        assert isinstance(M, sparse.spmatrix)
        assert not isinstance(M, sparse.sparray)

    @pytest.mark.parametrize('fmt', SPARSE_FORMATS)
    def test_matrix_issparse(self, fmt):
        M = _make_small_sparse(fmt, array=False)
        assert sparse.issparse(M)
        assert sparse.isspmatrix(M)

    def test_dense_not_sparse(self):
        assert not sparse.issparse(cupy.array([1, 2]))
        assert not sparse.isspmatrix(cupy.array([1, 2]))

    def test_coo_array_coords_property(self):
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype='i')
        col = cupy.array([2, 0, 1], dtype='i')
        A = sparse.coo_array((data, (row, col)), shape=(3, 3))
        assert isinstance(A.coords, tuple)
        assert len(A.coords) == 2
        assert A.coords[0] is A.row
        assert A.coords[1] is A.col

    def test_repr_array(self):
        A = sparse.csr_array(
            (cupy.array([1.0, 2.0]), cupy.array([0, 1], 'i'),
             cupy.array([0, 1, 2], 'i')), shape=(2, 2))
        r = repr(A)
        assert 'sparse array' in r
        assert 'sparse matrix' not in r
        assert 'Compressed Sparse Row' in r

    def test_repr_matrix(self):
        M = sparse.csr_matrix(
            (cupy.array([1.0, 2.0]), cupy.array([0, 1], 'i'),
             cupy.array([0, 1, 2], 'i')), shape=(2, 2))
        r = repr(M)
        assert 'sparse matrix' in r
        assert 'sparse array' not in r

    def test_mT_property(self):
        A = sparse.csr_array(
            (cupy.array([1.0, 2.0]), cupy.array([0, 1], 'i'),
             cupy.array([0, 1, 2], 'i')), shape=(2, 2))
        assert A.mT.shape == (2, 2)
        assert isinstance(A.mT, sparse.sparray)

    def test_block_array_returns_sparray(self):
        # 2x2 grid of 2x2 identity blocks tiles into a 4x4.
        A = sparse.csr_array(cupy.array([[1, 0], [0, 1]], dtype='d'))
        result = sparse.block_array([[A, A], [A, A]])
        assert isinstance(result, sparse.sparray)
        cupy.testing.assert_array_equal(
            result.toarray(),
            cupy.array([[1., 0., 1., 0.],
                        [0., 1., 0., 1.],
                        [1., 0., 1., 0.],
                        [0., 1., 0., 1.]]))

    @pytest.mark.parametrize('use_array', [True, False])
    def test_block_diag_type_and_values(self, use_array):
        cls = sparse.csr_array if use_array else sparse.csr_matrix
        expected_type = sparse.sparray if use_array else sparse.spmatrix
        A = cls(cupy.array([[1, 2]], dtype='d'))
        B = cls(cupy.array([[3]], dtype='d'))
        result = sparse.block_diag((A, B))
        assert isinstance(result, expected_type)
        cupy.testing.assert_array_equal(
            result.toarray(),
            cupy.array([[1., 2., 0.],
                        [0., 0., 3.]]))

    def test_safely_cast_index_arrays_csr(self):
        A = sparse.csr_array(cupy.array([[1, 0], [0, 2]], dtype='d'))
        ind, ptr = sparse.safely_cast_index_arrays(A, numpy.int32)
        assert ind.dtype == numpy.int32
        assert ptr.dtype == numpy.int32

    def test_safely_cast_index_arrays_coo(self):
        A = sparse.coo_array(
            (cupy.array([1.0]), (cupy.array([0], 'i'),
             cupy.array([0], 'i'))), shape=(2, 2))
        row, col = sparse.safely_cast_index_arrays(A, numpy.int32)
        assert row.dtype == numpy.int32
        assert col.dtype == numpy.int32

    def test_get_index_dtype_export(self):
        assert sparse.get_index_dtype(maxval=10) == numpy.int32
        assert (sparse.get_index_dtype(maxval=2**33)
                == numpy.int64)

    def test_nonzero_array(self):
        A = sparse.csr_array(
            cupy.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]))
        row, col = A.nonzero()
        assert row.shape == (3,)
        assert col.shape == (3,)
        cupy.testing.assert_array_equal(row, cupy.array([0, 0, 1]))
        cupy.testing.assert_array_equal(col, cupy.array([0, 1, 2]))

    def test_round_array(self):
        A = sparse.csr_array(
            cupy.array([[1.4, 2.6, 0.0], [0.0, 0.0, 3.5]]))
        R = round(A)
        assert isinstance(R, sparse.sparray)
        cupy.testing.assert_array_equal(
            R.toarray(),
            cupy.array([[1.0, 3.0, 0.0], [0.0, 0.0, 4.0]]))

    def test_matrix_transpose_function(self):
        dense = cupy.array([[1., 2.], [3., 4.]])
        A = sparse.csr_array(dense)
        T = sparse.matrix_transpose(A)
        # csr_array.T is csc_array (same data, different format).
        assert isinstance(T, sparse.sparray)
        cupy.testing.assert_array_equal(T.toarray(), dense.T)

    def test_swapaxes(self):
        dense = cupy.array([[1., 2., 3.], [4., 5., 6.]])
        A = sparse.csr_array(dense)
        T = sparse.swapaxes(A, 0, 1)
        assert isinstance(T, sparse.sparray)
        cupy.testing.assert_array_equal(T.toarray(), dense.T)
        T_neg = sparse.swapaxes(A, -2, -1)
        cupy.testing.assert_array_equal(T_neg.toarray(), dense.T)
        # Identity swap returns the same data.
        T_id = sparse.swapaxes(A, 0, 0)
        cupy.testing.assert_array_equal(T_id.toarray(), dense)

    def test_permute_dims(self):
        dense = cupy.array([[1., 2., 3.], [4., 5., 6.]])
        A = sparse.csr_array(dense)
        # Default reverses axes.
        P = sparse.permute_dims(A)
        cupy.testing.assert_array_equal(P.toarray(), dense.T)
        # Identity preserves data.
        P_id = sparse.permute_dims(A, (0, 1))
        cupy.testing.assert_array_equal(P_id.toarray(), dense)
        # Explicit reversal.
        P_rev = sparse.permute_dims(A, (1, 0))
        cupy.testing.assert_array_equal(P_rev.toarray(), dense.T)
        # Bad permutation raises.
        with pytest.raises(ValueError):
            sparse.permute_dims(A, (0, 0))

    @pytest.mark.parametrize('fmt', SPARSE_FORMATS)
    def test_array_maxprint_kwarg(self, fmt):
        if fmt == 'dia':
            data = cupy.array([[1., 2.]])
            offsets = cupy.array([0], dtype='i')
            A = sparse.dia_array((data, offsets), shape=(2, 2), maxprint=10)
            B = sparse.dia_array((data, offsets), shape=(2, 2))
        else:
            cls = getattr(sparse, f'{fmt}_array')
            A = cls(cupy.array([[1., 2.]]), maxprint=10)
            B = cls(cupy.array([[1., 2.]]))
        assert A.maxprint == 10
        assert B.maxprint == 50

    def test_negate_bool_array_raises(self):
        # Match scipy 1.17: NotImplementedError, not TypeError.
        B = sparse.csr_array(
            cupy.array([[True, False], [False, True]]))
        with pytest.raises(NotImplementedError, match='boolean'):
            -B

    def test_bool_sparse_mask_indexing_returns_dense(self):
        # Boolean sparse-array indexing returns a 1-D dense ndarray
        # (matches scipy 1.17).
        A = sparse.csr_array(
            cupy.array([[1., 2., 0.], [0., 0., 3.]]))
        mask = sparse.csr_array(
            cupy.array([[True, False, True], [True, False, True]]))
        res = A[mask]
        assert isinstance(res, cupy.ndarray)
        cupy.testing.assert_array_equal(
            res, cupy.array([1., 0., 0., 3.]))

    def test_copy_preserves_array_type(self):
        a = _make_csr(cupy, sparse, numpy.float64, array=True)
        assert isinstance(a.copy(), sparse.sparray)

    def test_csc_array_matmul_preserves_array_type(self):
        # SciPy gh-fix: csc_array @ csc_array (or csr_array) returns
        # csr_array, not csr_matrix.
        A = sparse.csc_array(cupy.array([[1., 2.], [3., 4.]]))
        B = sparse.csc_array(cupy.array([[1., 2.], [3., 4.]]))
        assert isinstance(A @ B, sparse.sparray)
        C = sparse.csr_array(cupy.array([[1., 2.], [3., 4.]]))
        assert isinstance(A @ C, sparse.sparray)

    def test_setitem_scalar_from_sparse_rhs(self):
        # Assigning a 1x1 sparse RHS to a scalar position works
        # (densifies the RHS first).  Used to crash with TypeError.
        A = sparse.csr_array(cupy.array([[1., 2.], [3., 4.]]))
        rhs = sparse.csr_array(cupy.array([[7.]]))
        A[0, 0] = rhs
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.array([[7., 2.], [3., 4.]]))

    def test_get_array_module_sparray(self):
        # cupy.get_array_module and cupyx.scipy.get_array_module recognize
        # sparray, not just spmatrix.
        import cupyx.scipy as cscipy
        A = sparse.csr_array(cupy.array([[1., 2.]]))
        M = sparse.csr_matrix(cupy.array([[1., 2.]]))
        assert cupy.get_array_module(A) is cupy
        assert cupy.get_array_module(M) is cupy
        assert cscipy.get_array_module(A).__name__ == 'cupyx.scipy'
        assert cscipy.get_array_module(M).__name__ == 'cupyx.scipy'

    def test_dia_tocsc_data_wider_than_matrix(self):
        # tocsc() must handle a DIA ``data`` buffer wider than the
        # matrix: the trailing columns fall outside ``shape`` and are
        # dropped.  A wider-than-num_cols buffer previously raised a
        # broadcast-shape ValueError here.
        data = cupy.array([[1., 2., 3., 4., 5., 6.]])
        offsets = cupy.array([0])
        m = sparse.dia_array((data, offsets), shape=(3, 4))
        cupy.testing.assert_array_equal(
            m.tocsc().toarray(),
            cupy.array([[1., 0., 0., 0.],
                        [0., 2., 0., 0.],
                        [0., 0., 3., 0.]]))

    def test_csc_setdiag(self):
        A = sparse.csc_matrix(cupy.zeros((3, 3)))
        A.setdiag(cupy.array([1., 2., 3.]))
        cupy.testing.assert_array_equal(
            A.toarray(),
            cupy.array([[1., 0., 0.],
                        [0., 2., 0.],
                        [0., 0., 3.]]))

    @pytest.mark.parametrize('fmt', ('csr', 'csc', 'coo'))
    def test_setdiag_python_types(self, fmt):
        # setdiag must accept Python scalars/lists, not just cupy
        # arrays: scipy's ``_spbase.setdiag`` coerces via
        # ``np.asarray`` first, and cupy mirrors that.  (It previously
        # called ``.astype``/``.ndim`` on the raw input, which raised
        # AttributeError for lists/scalars.)
        cls = getattr(sparse, f'{fmt}_array')
        expected = cupy.array([[10., 2., 3.],
                               [4., 20., 6.],
                               [7., 8., 9.]])
        for arg in ([10., 20.], (10., 20.), numpy.array([10., 20.])):
            A = cls(cupy.array([[1., 2., 3.],
                                [4., 5., 6.],
                                [7., 8., 9.]]))
            A.setdiag(arg)
            cupy.testing.assert_array_equal(A.toarray(), expected)
        # Scalar broadcasts to whole diagonal.
        A = cls(cupy.array([[1., 2., 3.],
                            [4., 5., 6.],
                            [7., 8., 9.]]))
        A.setdiag(99.0)
        cupy.testing.assert_array_equal(
            A.toarray(),
            cupy.array([[99., 2., 3.],
                        [4., 99., 6.],
                        [7., 8., 99.]]))

    @pytest.mark.parametrize('fmt', ('csr', 'csc', 'coo'))
    def test_inplace_scalar_preserves_identity(self, fmt):
        # ``A *= 2`` and ``A /= 2`` must mutate ``self.data`` in place
        # so the bound name still refers to the same object (matches
        # scipy's ``_data._data_matrix.__imul__``).  Without these
        # specials, ``A *= 2`` rebinds via ``A = A * 2`` and silently
        # allocates a new sparse matrix every time.
        cls = getattr(sparse, f'{fmt}_array')
        A = cls(cupy.array([[1., 2.], [3., 4.]]))
        old = A
        old_data = A.data
        A *= 2
        assert A is old
        assert A.data is old_data
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.array([[2., 4.], [6., 8.]]))
        A /= 2
        assert A is old
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.array([[1., 2.], [3., 4.]]))

    @testing.with_requires('scipy')
    @pytest.mark.parametrize('iop', [operator.imul, operator.itruediv],
                             ids=['imul', 'itruediv'])
    def test_inplace_bool_scalar_diverges_from_scipy(self, iop):
        # ``bool *= int`` / ``bool /= int`` violate numpy's same_kind
        # cast rule.  scipy raises ``UFuncTypeError``; CuPy intentionally
        # diverges, promoting ``self.data`` to a cuSPARSE-supported float
        # dtype in place while preserving ``self`` (but not ``self.data``)
        # identity.  Assert both branches directly.
        data = numpy.array([[True, False], [False, True]])

        s = scipy.sparse.csr_array(data)
        with pytest.raises(TypeError):
            iop(s, 2)

        A = sparse.csr_array(cupy.array(data))
        old, old_data = A, A.data
        iop(A, 2)
        assert A is old
        assert A.data is not old_data
        assert A.dtype == cupy.float64
        # Stored entries are the two ``True``s: ``True * 2 == 2.0``,
        # ``True / 2 == 0.5``.
        expected = float(iop(numpy.float64(1), 2))
        cupy.testing.assert_array_equal(A.data, cupy.full(2, expected))

    # Non-in-place ``/`` follows scipy's true-division dtype rules;
    # ``numpy_cupy_allclose`` checks both values and dtype against scipy.
    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_truediv_bool_promotes(self, xp, sp):
        a = sp.csr_array(xp.array([[True, False], [False, True]]))
        return a / 2

    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_truediv_float32_promotes(self, xp, sp):
        a = sp.csr_array(xp.array([[1., 2.]], dtype=xp.float32))
        return a / 2.0

    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_truediv_complex_preserved(self, xp, sp):
        a = sp.csr_array(xp.array([[1 + 2j, 3 + 4j]]))
        return a / 2

    def test_inplace_scalar_dia(self):
        # DIA only accepts the ``(data, offsets)`` tuple constructor
        # (C2 pre-existing limitation), so it gets its own test.
        data = cupy.array([[1., 2., 3.]])
        offsets = cupy.array([0])
        A = sparse.dia_array((data, offsets), shape=(3, 3))
        old = A
        A *= 2
        assert A is old
        cupy.testing.assert_array_equal(
            A.toarray(),
            cupy.array([[2., 0., 0.], [0., 4., 0.], [0., 0., 6.]]))

    def test_inplace_scalar_zero_preserves_structure(self):
        # ``A *= 0`` zeroes the data buffer in place but keeps the
        # stored structure (indices/indptr), matching scipy.
        A = sparse.csr_array(cupy.array([[1., 2.], [3., 4.]]))
        old = A
        old_indices = A.indices.copy()
        A *= 0
        assert A is old
        cupy.testing.assert_array_equal(A.toarray(), cupy.zeros((2, 2)))
        cupy.testing.assert_array_equal(A.indices, old_indices)

    def test_inplace_scalar_non_scalar_falls_back(self):
        # For a non-scalar operand ``__imul__`` returns NotImplemented,
        # so Python rebinds via ``A = A * other`` (element-wise
        # multiply): identity is NOT preserved, matching scipy.
        A = sparse.csr_array(cupy.array([[1., 0.], [0., 1.]]))
        B = sparse.csr_array(cupy.array([[2., 0.], [0., 3.]]))
        old = A
        A *= B
        assert A is not old
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.array([[2., 0.], [0., 3.]]))

    @pytest.mark.parametrize('fmt', ('csr', 'csc', 'coo'))
    def test_setdiag_does_not_mutate_input(self, fmt):
        # setdiag must not mutate the caller's ``values``.  Input is
        # coerced via ``cupy.asarray`` (no copy when the dtype already
        # matches), so the diagonal subtraction is out-of-place (``-``,
        # not ``-=``) to avoid writing through that shared view.
        cls = getattr(sparse, f'{fmt}_array')
        A = cls(cupy.array([[1., 2., 3.],
                            [4., 5., 6.],
                            [7., 8., 9.]]))
        v = cupy.array([10., 20., 30.])
        v_orig = v.copy()
        A.setdiag(v)
        cupy.testing.assert_array_equal(v, v_orig)

    def test_prune_csr(self):
        # ``prune()`` trims data/indices to ``indptr[-1]``.  The only
        # way to construct slack today is direct attribute mutation
        # (internal helpers always produce tight buffers); that path
        # is what ``prune()`` is for.
        A = sparse.csr_array(
            (cupy.array([1.0, 2.0, 3.0]),
             cupy.array([0, 1, 2], dtype='i'),
             cupy.array([0, 1, 2, 3], dtype='i')),
            shape=(3, 3))
        A.data = cupy.concatenate([A.data, cupy.array([99.0, 99.0])])
        A.indices = cupy.concatenate(
            [A.indices, cupy.array([99, 99], dtype='i')])
        assert A.data.shape == (5,)  # buffer has slack
        A.prune()
        assert A.data.shape == (3,)
        assert A.indices.shape == (3,)
        cupy.testing.assert_array_equal(A.data, cupy.array([1.0, 2.0, 3.0]))
        cupy.testing.assert_array_equal(A.indices, cupy.array([0, 1, 2]))

    def test_astype_copy_param(self):
        A = sparse.csr_array(cupy.array([[1., 2.]]))
        # No-op when dtype matches and copy=False
        B = A.astype(cupy.float64, copy=False)
        assert B is A
        # New object when copy=True
        C = A.astype(cupy.float64, copy=True)
        assert C is not A
        # Different dtype always returns new object
        D = A.astype(cupy.float32)
        assert D.dtype == cupy.float32
        assert D is not A

    def test_dia_todia_returns_self(self):
        # DIA todia() previously hit ``_csr_base.todia`` which is
        # ``raise NotImplementedError``.  ``_dia_base.todia`` overrides
        # to return ``self`` (or a copy), avoiding the round-trip.
        data = cupy.array([[1., 2., 3.]])
        offsets = cupy.array([0])
        m = sparse.dia_array((data, offsets), shape=(3, 3))
        assert m.todia() is m
        # copy=True returns a new object
        n = m.todia(copy=True)
        assert n is not m
        cupy.testing.assert_array_equal(n.toarray(), m.toarray())

    def test_block_diag_int_dense(self):
        # block_diag with integer-typed Python list/scalar/tuple input
        # used to fail because cuSPARSE rejects the int dtype; now we
        # promote to float64 first.
        res = sparse.block_diag([[[1, 2], [3, 4]], [[5]]])
        cupy.testing.assert_array_equal(
            res.toarray(),
            cupy.array([[1., 2., 0.],
                        [3., 4., 0.],
                        [0., 0., 5.]]))

    def test_block_diag_tuple_input(self):
        res = sparse.block_diag([(1, 2), [[3]]])
        cupy.testing.assert_array_equal(
            res.toarray(),
            cupy.array([[1., 2., 0.],
                        [0., 0., 3.]]))

    def test_block_diag_scalars(self):
        res = sparse.block_diag([1.0, 2.0, 3.0])
        cupy.testing.assert_array_equal(
            res.toarray(),
            cupy.array([[1., 0., 0.],
                        [0., 2., 0.],
                        [0., 0., 3.]]))

    def test_count_nonzero_csr_axis(self):
        # CSR/CSC count_nonzero(axis=...) uses the bincount fast path
        # (no tocoo round-trip).  Result matches per-row / per-col
        # counts of the dense form.
        A = sparse.csr_array(
            cupy.array([[1., 0., 2., 0.],
                        [0., 0., 0., 3.],
                        [4., 5., 0., 6.]]))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=0), cupy.array([2, 1, 1, 2]))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=1), cupy.array([2, 1, 3]))
        # Dedupes first: stored zero excluded.
        data = cupy.array([1.0, 2.0, 0.0, 3.0])
        ind = cupy.array([0, 1, 2, 0], dtype='i')
        ptr = cupy.array([0, 1, 3, 4], dtype='i')
        B = sparse.csr_array._from_parts(data, ind, ptr, (3, 3))
        assert B.count_nonzero() == 3  # explicit zero excluded

    def test_count_nonzero_coo_axis(self):
        A = sparse.coo_array(
            cupy.array([[1., 0., 2.],
                        [0., 3., 0.]]))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=0), cupy.array([1, 1, 1]))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=1), cupy.array([2, 1]))

    def test_count_nonzero_dia_axis_raises(self):
        # DIA axis-aware count_nonzero matches scipy: NotImplementedError.
        # Users can convert to CSR/CSC for per-axis counts.
        m = sparse.dia_array(
            (cupy.array([[1., 2., 3.]]), cupy.array([0])),
            shape=(3, 3))
        assert m.count_nonzero() == 3
        with pytest.raises(NotImplementedError):
            m.count_nonzero(axis=0)

    def test_count_nonzero_dia_data_wider_than_matrix(self):
        # When ``data`` is wider than the matrix, the pad columns lie
        # outside the matrix and must be excluded -- matching scipy.
        m = sparse.dia_array(
            (cupy.array([[1., 2., 3., 4., 5., 6.]]),
             cupy.array([0])),
            shape=(3, 4))
        # Only data[0:3] correspond to in-matrix positions (0,0), (1,1),
        # (2,2); the pad columns 3..5 are outside the (3, 4) matrix.
        assert m.count_nonzero() == 3

    def test_count_nonzero_dia_excludes_explicit_zero(self):
        # Explicit zero in the diagonal data is excluded.
        m = sparse.dia_array(
            (cupy.array([[0., 1., 2.]]), cupy.array([0])),
            shape=(3, 3))
        assert m.count_nonzero() == 2  # the 0 at (0, 0) doesn't count
        assert m.nnz == 3                # but it is stored

    @pytest.mark.parametrize('fmt', ('csr', 'csc', 'coo'))
    def test_count_nonzero_axis_empty(self, fmt):
        # count_nonzero(axis=) on an empty matrix returns scipy's
        # zero-filled axis vector.  Exercises the zero-size guard:
        # ``cupy.bincount`` rejects zero-size input even with
        # ``minlength``, so the empty case is special-cased.
        cls = getattr(sparse, f'{fmt}_array')
        A = cls((3, 5))
        assert A.count_nonzero() == 0
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=0), cupy.zeros(5, dtype=cupy.intp))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=1), cupy.zeros(3, dtype=cupy.intp))
        # Negative axes work the same.
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=-1), cupy.zeros(3, dtype=cupy.intp))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=-2), cupy.zeros(5, dtype=cupy.intp))

    @pytest.mark.parametrize('fmt', ('csr', 'csc', 'coo'))
    def test_count_nonzero_axis_all_explicit_zero(self, fmt):
        # When every stored value is an explicit zero, the
        # bincount-on-mask path also produces a zero-size input that
        # would otherwise crash.
        cls = getattr(sparse, f'{fmt}_array')
        A = cls(cupy.zeros((2, 3)) + 0.0)
        # Stored explicit zeros: build via _from_parts to keep them.
        if fmt == 'csr':
            A = sparse.csr_array._from_parts(
                cupy.zeros(3), cupy.array([0, 1, 2], 'i'),
                cupy.array([0, 2, 3], 'i'), (2, 3))
        elif fmt == 'csc':
            A = sparse.csc_array._from_parts(
                cupy.zeros(3), cupy.array([0, 1, 0], 'i'),
                cupy.array([0, 1, 2, 3], 'i'), (2, 3))
        else:
            A = sparse.coo_array._from_parts(
                cupy.zeros(3),
                cupy.array([0, 0, 1], 'i'),
                cupy.array([0, 1, 2], 'i'),
                (2, 3))
        assert A.count_nonzero() == 0
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=0), cupy.zeros(3, dtype=cupy.intp))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=1), cupy.zeros(2, dtype=cupy.intp))

    @testing.with_requires('scipy')
    @pytest.mark.parametrize('fmt', SPARSE_FORMATS)
    def test_type_system_matches_scipy(self, fmt):
        """CuPy and SciPy type predicates should agree."""
        sp_a = _make_small_sparse(fmt, xp=numpy, spmod=scipy.sparse)
        sp_m = _make_small_sparse(
            fmt, array=False, xp=numpy, spmod=scipy.sparse)
        cp_a = _make_small_sparse(fmt)
        cp_m = _make_small_sparse(fmt, array=False)

        assert sparse.issparse(cp_a) == scipy.sparse.issparse(sp_a)
        assert sparse.issparse(cp_m) == scipy.sparse.issparse(sp_m)
        assert sparse.isspmatrix(cp_a) == scipy.sparse.isspmatrix(sp_a)
        assert sparse.isspmatrix(cp_m) == scipy.sparse.isspmatrix(sp_m)


@pytest.mark.parametrize(
    'dtype', [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
@testing.with_requires('scipy')
class TestCsrArrayConstruction:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_from_data_indices_indptr(self, xp, sp, dtype):
        m = _make_csr(xp, sp, dtype, array=True)
        assert m.format == 'csr'
        assert isinstance(m, sp.sparray)
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_from_dense(self, xp, sp, dtype):
        dense = xp.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
        m = sp.csr_array(dense)
        assert isinstance(m, sp.sparray)
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_from_coo_tuple(self, xp, sp, dtype):
        data = xp.array([1, 2, 3], dtype)
        row = xp.array([0, 1, 2], 'i')
        col = xp.array([2, 0, 1], 'i')
        m = sp.csr_array((data, (row, col)), shape=(3, 3))
        assert isinstance(m, sp.sparray)
        return m

    def test_empty(self, dtype):
        m = sparse.csr_array((3, 4), dtype=dtype)
        assert m.shape == (3, 4)
        assert m.nnz == 0
        assert isinstance(m, sparse.sparray)

    def test_from_coo_tuple_preserves_int64_indices(self, dtype):
        # csr_array must preserve int64 indices.  The
        # ``(data, (row, col))`` tuple is routed through
        # ``self._coo_container`` (coo_array), not ``coo_matrix`` --
        # whose ``_get_index_dtype(check_contents=True)`` path would
        # silently downcast int64 row/col to int32.
        data = cupy.array([1.0, 2.0], dtype=dtype)
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, 1], dtype=cupy.int64)
        m = sparse.csr_array((data, (row, col)), shape=(3, 3))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        # csc_array path is symmetric.
        m = sparse.csc_array((data, (row, col)), shape=(3, 3))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        # Matrix variant should still downcast (matches scipy convention).
        m = sparse.csr_matrix((data, (row, col)), shape=(3, 3))
        assert m.indices.dtype == cupy.int32


@pytest.mark.parametrize(
    'dtype', [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
@testing.with_requires('scipy')
class TestCsrArrayStarIsElementwise:
    """Verify that * is element-wise for csr_array (matching scipy.sparse)."""

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_sparse(self, xp, sp, dtype):
        """array * array should be element-wise."""
        a = _make_csr_sq(xp, sp, dtype, array=True)
        b = _make_csr_sq(xp, sp, dtype, array=True)
        result = a * b
        assert isinstance(result, sp.sparray)
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_scalar(self, xp, sp, dtype):
        """array * scalar should be scalar multiplication."""
        a = _make_csr(xp, sp, dtype, array=True)
        result = a * dtype(2.0)
        assert isinstance(result, sp.sparray)
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rstar_scalar(self, xp, sp, dtype):
        """scalar * array should be scalar multiplication."""
        a = _make_csr(xp, sp, dtype, array=True)
        result = dtype(3.0) * a
        assert isinstance(result, sp.sparray)
        return result


@pytest.mark.parametrize(
    'dtype', [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
@testing.with_requires('scipy')
class TestCsrArrayMatmul:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_matmul_sparse(self, xp, sp, dtype):
        a = _make_csr(xp, sp, dtype, array=True)
        b = _make_for_matmul(xp, sp, dtype, array=True)
        result = a @ b
        assert isinstance(result, sp.sparray)
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_matmul_dense_vector(self, xp, sp, dtype):
        a = _make_csr(xp, sp, dtype, array=True)
        x = xp.arange(4).astype(dtype)
        return a @ x

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_matmul_dense_matrix(self, xp, sp, dtype):
        a = _make_csr(xp, sp, dtype, array=True)
        x = xp.arange(8).reshape(4, 2).astype(dtype)
        return a @ x

    def test_matmul_scalar_raises(self, dtype):
        a = _make_csr_sq(cupy, sparse, dtype, array=True)
        with pytest.raises(ValueError):
            a @ 5.0


@pytest.mark.parametrize(
    'dtype', [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
@testing.with_requires('scipy')
class TestCsrArrayPower:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_elementwise(self, xp, sp, dtype):
        """array ** n should be element-wise, not matrix power."""
        a = _make_csr_sq(xp, sp, dtype, array=True)
        result = a ** 2
        assert isinstance(result, sp.sparray)
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_matches_dense(self, xp, sp, dtype):
        """Verify ** gives same result as dense element-wise power."""
        a = _make_csr_sq(xp, sp, dtype, array=True)
        result_sparse = (a ** 2).toarray()
        result_dense = a.toarray() ** 2
        xp.testing.assert_allclose(result_sparse, result_dense)
        return result_sparse

    def test_power_zero_raises(self, dtype):
        """Array ** 0 raises NotImplementedError (would densify)."""
        a = _make_csr_sq(cupy, sparse, dtype, array=True)
        with pytest.raises(NotImplementedError):
            a ** 0


@testing.with_requires('scipy')
class TestPowerZeroDensifies:
    """Element-wise ``power(0)`` (and array ``** 0``) would densify --
    every implicit zero becomes ``0 ** 0 == 1`` -- so it raises
    NotImplementedError to match scipy rather than return a
    mathematically wrong sparse result.  A non-scalar exponent likewise
    raises (checked before any ``other == 0`` test, else the array
    comparison would raise "truth value ambiguous").  Matrix ``** 0`` is
    *matrix* power and is unaffected.

    ``accept_error`` requires both scipy and cupy to raise the same
    error, so these verify the divergence-free behavior against scipy.
    """

    @pytest.mark.parametrize('cls_name', ['csr_matrix', 'csr_array',
                                          'csc_matrix', 'csc_array',
                                          'coo_matrix', 'coo_array'])
    @testing.numpy_cupy_allclose(sp_name='sp',
                                 accept_error=NotImplementedError)
    def test_power_zero_method(self, xp, sp, cls_name):
        a = getattr(sp, cls_name)(xp.array([[1.0, 0, 2.0], [0, 3.0, 0]]))
        return a.power(0)

    @pytest.mark.parametrize('op', [
        lambda a, xp: a ** 0,
        lambda a, xp: a ** xp.array([2, 3, 4]),
    ], ids=['pow-zero', 'pow-nonscalar'])
    @testing.numpy_cupy_allclose(sp_name='sp',
                                 accept_error=NotImplementedError)
    def test_array_pow_raises(self, xp, sp, op):
        a = sp.csr_array(xp.array([[1.0, 0, 2.0]]))
        return op(a, xp)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_nonzero(self, xp, sp):
        a = sp.csr_array(xp.array([[2.0, 4.0]]))
        return a.power(2)


@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
@testing.with_requires('scipy')
class TestCsrMatrixStarIsMatmul:
    """Verify that * is still matmul for csr_matrix (unchanged)."""

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_matmul(self, xp, sp, dtype):
        """matrix * matrix should be matmul."""
        a = _make_csr(xp, sp, dtype)
        b = _make_for_matmul(xp, sp, dtype)
        return a * b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_scalar(self, xp, sp, dtype):
        """matrix * scalar should still work."""
        a = _make_csr(xp, sp, dtype)
        return a * dtype(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_pow_matrix_power(self, xp, sp, dtype):
        """matrix ** n should be matrix power."""
        a = _make_csr_sq(xp, sp, dtype)
        return a ** 2


class TestMultiplyMixedSparrayMatrix:
    """Cross-type element-wise multiply preserves the caller's type.

    ``multiply_by_csr`` swaps operands for performance when
    ``a.nnz > b.nnz``; before the ``out_cls`` parameter was added it
    would derive the result class from the post-swap first arg, which
    leaked the other operand's type into the result.
    """

    @pytest.fixture
    def dense_pair(self):
        hi_nnz = cupy.array([[1., 1., 1.],
                             [0., 1., 1.],
                             [0., 0., 1.]])
        lo_nnz = cupy.array([[1., 0., 0.],
                             [0., 1., 0.],
                             [0., 0., 1.]])
        return hi_nnz, lo_nnz

    def test_array_times_matrix_when_array_has_more_nnz(self, dense_pair):
        hi, lo = dense_pair
        A = sparse.csr_array(hi)  # 6 nnz
        M = sparse.csr_matrix(lo)  # 3 nnz -- swap fires
        result = A.multiply(M)
        assert isinstance(result, sparse.sparray)
        assert not isinstance(result, sparse.spmatrix)
        cupy.testing.assert_array_equal(result.toarray(), hi * lo)

    def test_matrix_times_array_when_matrix_has_more_nnz(self, dense_pair):
        hi, lo = dense_pair
        M = sparse.csr_matrix(hi)  # 6 nnz
        A = sparse.csr_array(lo)  # 3 nnz -- swap fires
        result = M.multiply(A)
        assert isinstance(result, sparse.spmatrix)
        assert not isinstance(result, sparse.sparray)
        cupy.testing.assert_array_equal(result.toarray(), hi * lo)

    def test_array_times_matrix_no_swap(self, dense_pair):
        hi, lo = dense_pair
        A = sparse.csr_array(lo)  # 3 nnz -- no swap
        M = sparse.csr_matrix(hi)  # 6 nnz
        result = A.multiply(M)
        assert isinstance(result, sparse.sparray)
        cupy.testing.assert_array_equal(result.toarray(), hi * lo)


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


@pytest.mark.parametrize(
    'dtype', [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
@testing.with_requires('scipy')
class TestCsrArrayArithmeticSciPyComparison:
    """SciPy-parity sweep over float and complex dtypes.

    Elementwise ``*``, ``@``, and ``**`` are covered across the same
    dtype set by ``TestCsrArrayStarIsElementwise`` /
    ``TestCsrArrayMatmul`` / ``TestCsrArrayPower``, so this class only
    covers the remaining ops (add/sub/neg/abs/transpose/conj).
    """

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add(self, xp, sp, dtype):
        a = _make_csr(xp, sp, dtype, array=True)
        b = _make_csr(xp, sp, dtype, array=True)
        result = a + b
        assert isinstance(result, sp.sparray)
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub(self, xp, sp, dtype):
        a = _make_csr(xp, sp, dtype, array=True)
        b = _make_csr(xp, sp, dtype, array=True)
        result = a - b
        assert isinstance(result, sp.sparray)
        return result.toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_neg(self, xp, sp, dtype):
        a = _make_csr(xp, sp, dtype, array=True)
        result = -a
        assert isinstance(result, sp.sparray)
        return result.toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_abs(self, xp, sp, dtype):
        a = _make_csr(xp, sp, dtype, array=True)
        result = abs(a)
        assert isinstance(result, sp.sparray)
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_transpose(self, xp, sp, dtype):
        a = _make_csr(xp, sp, dtype, array=True)
        result = a.T
        assert isinstance(result, sp.sparray)
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_conj(self, xp, sp, dtype):
        a = _make_csr(xp, sp, dtype, array=True)
        result = a.conj()
        assert isinstance(result, sp.sparray)
        return result


# Matrix-only APIs that don't exist on sparray.  In SciPy 1.14 most of
# these were deprecated; SciPy 1.17 un-deprecated everything except
# ``A`` and ``H`` (those still emit DeprecationWarning on CuPy until
# users have a release cycle to migrate).  The ``A`` / ``H``
# DeprecationWarnings are silenced here; warning behaviour itself is
# exercised by ``test_base.py::TestDeprecatedSpmatrixApi``.
@pytest.mark.filterwarnings(
    "ignore:`spmatrix\\.(A|H)`:DeprecationWarning"
)
class TestLegacyApiSurfaceVia:
    """Legacy array/matrix API surface checks using CSR as the
    representative sparse format.

    Intent: validate common array-vs-matrix API contract (attribute
    presence/absence and shape-setter behavior) without multiplying by
    all formats.  Some legacy methods (e.g. getrow/getcol) dispatch to
    format-specific hooks, so keeping this class CSR-focused avoids
    mixing format-implementation tests into API-surface checks.
    """

    @pytest.fixture
    def arr(self):
        return _make_small_sparse('csr', array=True, shape=(1, 2))

    @pytest.fixture
    def mat(self):
        return _make_small_sparse('csr', array=False, shape=(1, 2))

    @pytest.mark.parametrize(
        'attr', ['A', 'H', 'getrow', 'getcol', 'getH', 'asfptype',
                 'getformat', 'getmaxprint']
    )
    def test_array_has_no_matrix_only_api(self, arr, attr):
        with pytest.raises(AttributeError):
            getattr(arr, attr)

    def test_array_has_basic_surface(self, arr):
        assert arr.shape == (1, 2)
        assert arr.nnz == 1
        assert arr.format == 'csr'
        assert arr.ndim == 2

    def test_array_has_no_shape_setter(self, arr):
        with pytest.raises(AttributeError):
            arr.shape = (2, 1)

    def test_matrix_A_and_H_exist(self, mat):
        assert isinstance(mat.A, cupy.ndarray)
        assert sparse.issparse(mat.H)

    @pytest.mark.parametrize('name,args', [
        ('getH', ()),
        ('asfptype', ()),
        ('getrow', (0,)),
        ('getcol', (0,)),
    ])
    def test_matrix_legacy_methods_exist(self, mat, name, args):
        assert sparse.issparse(getattr(mat, name)(*args))

    def test_matrix_getformat_exists(self, mat):
        assert mat.getformat() == 'csr'

    def test_matrix_shape_setter_exists(self, mat):
        # shape setter exists on matrices (even if reshape is no-op here)
        mat.shape = (1, 2)


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

    def test_matrix_downcasts_int64_when_values_fit(self):
        # Matrix path keeps the legacy policy: ``check_contents=True``
        # downcasts int64 index buffers to int32 when every value fits
        # in int32 (the array path preserves int64 unconditionally;
        # see ``test_array_preserves_int64`` above).
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        M = sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        assert M.indices.dtype == cupy.int32
        assert M.indptr.dtype == cupy.int32

    def test_coo_array_preserves_int64(self):
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        A = sparse.coo_array((data, (row, col)), shape=(3, 3))
        assert A.row.dtype == cupy.int64
        assert A.col.dtype == cupy.int64


# Sparse array format conversions

@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
class TestSparseArrayConversions:

    @testing.with_requires('scipy')
    @pytest.mark.parametrize('src_fmt', SPARSE_FORMATS)
    @pytest.mark.parametrize('dst_fmt', SPARSE_FORMATS)
    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_conversion_values_match_scipy(
            self, xp, sp, src_fmt, dst_fmt, dtype):
        if dst_fmt == 'dia' and src_fmt != 'dia':
            # CuPy does not implement non-DIA -> DIA yet.
            pytest.skip('non-DIA -> DIA conversion is not implemented')
        A = _make_small_sparse(
            src_fmt, array=True, shape=(2, 2), dtype=dtype, xp=xp, spmod=sp)
        return getattr(A, f'to{dst_fmt}')().toarray()

    @testing.with_requires('scipy')
    @pytest.mark.parametrize('src_fmt', SPARSE_FORMATS)
    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_toarray(self, xp, sp, src_fmt, dtype):
        m = _make_small_sparse(
            src_fmt, array=True, shape=(2, 2), dtype=dtype, xp=xp, spmod=sp)
        return m.toarray()

    @pytest.mark.parametrize(
        'src_fmt,expected_fmt',
        [('csr', 'csc'),
         ('csc', 'csr'),
         ('coo', 'coo'),
         # DIA has no transpose override yet; routes via tocsr() → CSC.
         ('dia', 'csc')])
    def test_array_transpose_type(self, src_fmt, expected_fmt, dtype):
        A = _make_small_sparse(src_fmt, array=True, shape=(2, 2), dtype=dtype)
        AT = A.T
        assert isinstance(AT, sparse.sparray)
        assert AT.format == expected_fmt

    @pytest.mark.parametrize('src_fmt', SPARSE_FORMATS)
    @pytest.mark.parametrize('dst_fmt', SPARSE_FORMATS)
    def test_conversion_type_and_format(
            self, src_fmt, dst_fmt, dtype):
        A = _make_small_sparse(src_fmt, array=True, shape=(2, 2), dtype=dtype)
        op = getattr(A, f'to{dst_fmt}')
        if dst_fmt == 'dia' and src_fmt != 'dia':
            # Non-DIA -> DIA currently routes to CSR.todia(), which is not
            # implemented yet.
            with pytest.raises(NotImplementedError):
                op()
            return
        result = op()
        assert isinstance(result, sparse.sparray)
        assert result.format == dst_fmt
        if src_fmt == dst_fmt:
            assert result is A

    @pytest.mark.parametrize('src_fmt', SPARSE_FORMATS)
    @pytest.mark.parametrize('copy', [False, True])
    def test_same_format_conversion_copy_semantics(
            self, src_fmt, copy, dtype):
        A = _make_small_sparse(src_fmt, array=True, shape=(2, 2), dtype=dtype)
        result = getattr(A, f'to{src_fmt}')(copy=copy)
        assert isinstance(result, sparse.sparray)
        assert result.format == src_fmt
        if copy:
            assert result is not A
        else:
            assert result is A


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
        # Use explicit float dtype to sidestep the SciPy 1.17 FutureWarning
        # for integer-input ``diags_array`` calls (will become an error in
        # SciPy 1.19).  CuPy's ``diags_array`` already requires a float
        # storage dtype.
        return sp.diags_array([1, 2, 3], offsets=0, dtype=float).toarray()

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

    def _make_pair(self, use_array):
        cls = sparse.csr_array if use_array else sparse.csr_matrix
        d = cupy.array([[1, 0], [0, 2]], dtype='d')
        return cls(d), cls(d)

    @pytest.mark.parametrize('use_array', [True, False])
    @pytest.mark.parametrize('op_name', ['hstack', 'vstack', 'kron',
                                         'tril', 'triu'])
    def test_construct_type_propagation(self, use_array, op_name):
        pair = self._make_pair(use_array)
        expected_type = sparse.sparray if use_array else sparse.spmatrix
        op = getattr(sparse, op_name)
        if op_name in ('hstack', 'vstack'):
            result = op(list(pair))
        elif op_name == 'kron':
            result = op(*pair)
        else:
            result = op(pair[0])
        assert isinstance(result, expected_type)


# CSC array arithmetic

@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
@testing.with_requires('scipy')
class TestCscArrayArithmetic:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add(self, xp, sp, dtype):
        a = _make_csc(xp, sp, dtype, array=True)
        result = a + a
        assert isinstance(result, sp.sparray)
        return result

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_sub(self, xp, sp, dtype):
        a = _make_csc(xp, sp, dtype, array=True)
        result = a - a
        assert isinstance(result, sp.sparray)
        return result.toarray()

    def test_add_preserves_type(self, dtype):
        a = _make_csc(cupy, sparse, dtype, array=True)
        assert isinstance(a + a, sparse.sparray)


# Cross-format multiply

@testing.with_requires('scipy')
class TestCrossFormatMultiply:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_coo_star_coo(self, xp, sp):
        """COO * COO element-wise should work."""
        a = sp.coo_array(_make_csr_sq(xp, sp, numpy.float64, array=True))
        result = a * a
        assert isinstance(result, sp.sparray)
        return result.toarray()

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_csc_star_csc(self, xp, sp):
        """CSC * CSC element-wise should work."""
        a = _make_csc(xp, sp, numpy.float64, array=True)
        result = a * a
        assert isinstance(result, sp.sparray)
        return result.toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_csr_star_coo(self, xp, sp):
        """CSR * COO cross-format multiply should work."""
        a = _make_csr_sq(xp, sp, numpy.float64, array=True)
        b = sp.coo_array(a)
        result = a * b
        assert isinstance(result, sp.sparray)
        return result.toarray()


# Reduction 1D shaping

@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
@testing.with_requires('scipy')
class TestArrayReductions:

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_sum_axis0_ndim(self, xp, sp, dtype):
        m = _make_csr(xp, sp, dtype, array=True)
        result = m.sum(axis=0)
        return result.ndim

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_sum_axis1_ndim(self, xp, sp, dtype):
        m = _make_csr(xp, sp, dtype, array=True)
        result = m.sum(axis=1)
        return result.ndim

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_axis0_values(self, xp, sp, dtype):
        m = _make_csr(xp, sp, dtype, array=True)
        return m.sum(axis=0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_axis1_values(self, xp, sp, dtype):
        m = _make_csr(xp, sp, dtype, array=True)
        return m.sum(axis=1)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean_axis0(self, xp, sp, dtype):
        m = _make_csr(xp, sp, dtype, array=True)
        return m.mean(axis=0)

    def test_matrix_sum_stays_2d(self, dtype):
        """Matrix sum(axis=0) should still be 2D."""
        m = _make_csr(cupy, sparse, dtype, array=False)
        result = m.sum(axis=0)
        assert result.ndim == 2

    def test_min_max_axis_returns_2d_known_gap(self, dtype):
        # KNOWN GAP vs scipy: scipy sparse *arrays* return 1-D from min/max
        # over an axis (shape ``(M,)``), as sum()/argmin() already do here.
        # CuPy's COO is 2-D-only, so min/max(axis=) currently returns 2-D
        # (``(1, M)`` / ``(M, 1)``) for arrays too -- an inconsistency with
        # sum/argmin.  Pinned so the divergence is tracked; tighten to 1-D
        # if/when 1-D sparse arrays land.  See ``_data.py._min_or_max_axis``.
        a = sparse.csr_array(
            cupy.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=dtype))
        assert a.min(axis=0).shape == (1, 3)
        assert a.max(axis=1).shape == (2, 1)
        # Contrast: sum(axis=) IS 1-D for arrays (the consistent behavior).
        assert a.sum(axis=0).shape == (3,)


# DIA array

class TestDiaArrayBasic:

    def test_construction(self):
        A = _make_small_sparse('dia', array=True, shape=(3, 3))
        assert isinstance(A, sparse.sparray)
        assert A.format == 'dia'

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

    @pytest.mark.parametrize('use_array', [True, False])
    def test_spsolve_csr_input(self, use_array):
        from cupyx.scipy.sparse.linalg import spsolve
        n = 8
        A_dense = cupy.zeros((n, n), dtype=numpy.float64)
        A_dense[cupy.arange(n), cupy.arange(n)] = 4
        A_dense[cupy.arange(n - 1), cupy.arange(1, n)] = 1
        A_dense[cupy.arange(1, n), cupy.arange(n - 1)] = 1
        cls = sparse.csr_array if use_array else sparse.csr_matrix
        A = cls(A_dense)
        b = cupy.arange(1, n + 1, dtype=numpy.float64)
        x = spsolve(A, b)
        lhs = A @ x if use_array else A * x
        cupy.testing.assert_allclose(lhs, b, rtol=1e-10)


class TestNegativeShapeRejected:
    # Error message matches scipy: "'shape' elements cannot be negative".

    @pytest.mark.parametrize('cls_name', ['csr_matrix', 'csc_matrix',
                                          'coo_matrix', 'csr_array',
                                          'csc_array', 'coo_array',
                                          'dia_matrix', 'dia_array'])
    @pytest.mark.parametrize('shape', [(10, -5), (-5, 10), (-5, -5)])
    def test_negative_shape_raises(self, cls_name, shape):
        cls = getattr(sparse, cls_name)
        if cls_name.startswith('dia'):
            # DIA only accepts (data, offsets) + shape= kwarg.
            with pytest.raises(
                    ValueError,
                    match=r"'shape' elements cannot be negative"):
                cls((cupy.array([[1.0]]), cupy.array([0])), shape=shape)
        else:
            with pytest.raises(
                    ValueError,
                    match=r"'shape' elements cannot be negative"):
                cls(shape)

    @pytest.mark.parametrize('cls_name', ['csr_matrix', 'csc_matrix',
                                          'coo_matrix'])
    def test_negative_shape_kwarg_raises(self, cls_name):
        # Same check via the shape= keyword on a 3-tuple constructor.
        cls = getattr(sparse, cls_name)
        data = cupy.array([1.0])
        if cls_name == 'coo_matrix':
            row = cupy.array([0], dtype='i')
            col = cupy.array([0], dtype='i')
            with pytest.raises(
                    ValueError,
                    match=r"'shape' elements cannot be negative"):
                cls((data, (row, col)), shape=(10, -5))
        else:
            indices = cupy.array([0], dtype='i')
            indptr = cupy.array([0, 1, 1, 1], dtype='i')
            with pytest.raises(
                    ValueError,
                    match=r"'shape' elements cannot be negative"):
                cls((data, indices, indptr), shape=(10, -5))


@testing.with_requires('scipy')
class TestComparisonCrossFormat:
    """``_comparison`` and ``_maximum_minimum`` accept any sparse operand
    by routing through ``other.tocsr()`` (matches ``_add_sparse`` /
    ``multiply``), so ``csr.maximum(coo)``, ``csr == csc`` etc. don't
    bottom out in NotImplementedError.  Each stays sparse and matches
    scipy's value/dtype.
    """

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_maximum_csr_coo(self, xp, sp):
        a = sp.csr_matrix(xp.array([[1.0, 2.0], [3.0, 4.0]]))
        b = sp.coo_matrix(xp.array([[5.0, 1.0], [2.0, 8.0]]))
        c = a.maximum(b)
        assert sp.issparse(c)
        return c

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_minimum_csr_csc(self, xp, sp):
        a = sp.csr_matrix(xp.array([[5.0, 6.0], [7.0, 8.0]]))
        b = sp.csc_matrix(xp.array([[1.0, 2.0], [3.0, 4.0]]))
        c = a.minimum(b)
        assert sp.issparse(c)
        return c

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_eq_csr_coo(self, xp, sp):
        a = sp.csr_matrix(xp.array([[1.0, 2.0]]))
        b = sp.coo_matrix(xp.array([[1.0, 0.0]]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)
            c = a == b
        assert sp.issparse(c)
        return c

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_lt_csr_dia(self, xp, sp):
        a = sp.csr_matrix(xp.array([[1.0, 2.0], [3.0, 4.0]]))
        b = sp.dia_matrix(
            (xp.array([[5.0, 6.0]]), xp.array([0])), shape=(2, 2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)
            c = a < b
        assert sp.issparse(c)
        return c


class TestSetdiag2DRejected:

    def test_csr_setdiag_2d_raises(self):
        a = sparse.csr_matrix(cupy.zeros((3, 3)))
        with pytest.raises(ValueError, match='must be 0-d or 1-d'):
            a.setdiag(cupy.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_coo_setdiag_2d_raises(self):
        a = sparse.coo_matrix(cupy.zeros((3, 3)))
        with pytest.raises(ValueError, match='must be 0-d or 1-d'):
            a.setdiag(cupy.array([[1.0, 2.0], [3.0, 4.0]]))


class TestPublicCtorIndptrZeroCheck:

    def test_csr_indptr_must_start_at_zero(self):
        with pytest.raises(ValueError, match='start with 0'):
            sparse.csr_matrix(
                (cupy.array([1.0, 2.0]),
                 cupy.array([0, 1], dtype='i'),
                 cupy.array([5, 6, 7], dtype='i')),
                shape=(2, 2))

    def test_csc_indptr_must_start_at_zero(self):
        with pytest.raises(ValueError, match='start with 0'):
            sparse.csc_matrix(
                (cupy.array([1.0, 2.0]),
                 cupy.array([0, 1], dtype='i'),
                 cupy.array([5, 6, 7], dtype='i')),
                shape=(2, 2))


@testing.with_requires('scipy')
class TestSign:
    """``sign()`` delegates to ``cupy.sign`` with no sparse-specific
    special-casing, so it must match scipy elementwise.  scipy 1.16+
    (numpy 2.x) defines complex sign as ``z / abs(z)`` with
    ``sign(0+0j) == 0+0j``; ``cupy.sign`` follows the same rule (since
    gh-10034).  These compare against scipy to guard against
    reintroducing a divergent workaround in ``_data.py``.
    """

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sign_real(self, xp, sp):
        a = sp.csr_matrix(xp.array([[-3.0, 0, 2.0], [0, -1.0, 0]]))
        return a.sign()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sign_complex(self, xp, sp):
        a = sp.csr_matrix(
            xp.array([[1 + 1j, 0, -2 + 0j], [0, 0 + 3j, 0]]))
        return a.sign()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sign_stored_complex_zero(self, xp, sp):
        # An explicit stored ``0+0j`` must round-trip to ``0+0j``, not
        # ``nan+nanj`` -- the case ``cupy.sign`` only handled correctly
        # after gh-10034 (build via indptr so the zero stays stored).
        data = xp.array([0 + 0j, 1 + 2j, 0 + 0j])
        indices = xp.array([0, 1, 0], 'i')
        indptr = xp.array([0, 2, 3], 'i')
        a = sp.csr_matrix((data, indices, indptr), shape=(2, 2))
        return a.sign()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sign_array_preserves_type(self, xp, sp):
        a = sp.csr_array(xp.array([[-3.0, 0, 2.0]]))
        out = a.sign()
        assert isinstance(out, sp.sparray)
        return out
