"""Tests for sparse array classes (csr_array, csc_array, coo_array, dia_array).
"""
from __future__ import annotations

import numpy
import pytest
try:
    import scipy.sparse  # noqa: F401
except ImportError:
    pass

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

    def test_block_diag_arrays(self):
        A = sparse.csr_array(cupy.array([[1, 2]], dtype='d'))
        B = sparse.csr_array(cupy.array([[3]], dtype='d'))
        result = sparse.block_diag((A, B))
        assert isinstance(result, sparse.sparray)
        cupy.testing.assert_array_equal(
            result.toarray(),
            cupy.array([[1., 2., 0.],
                        [0., 0., 3.]]))

    def test_block_diag_matrices(self):
        A = sparse.csr_matrix(cupy.array([[1, 2]], dtype='d'))
        B = sparse.csr_matrix(cupy.array([[3]], dtype='d'))
        result = sparse.block_diag((A, B))
        assert isinstance(result, sparse.spmatrix)
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

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_array_maxprint_kwarg(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        A = cls(cupy.array([[1., 2.]]), maxprint=10)
        assert A.maxprint == 10
        # Default
        B = cls(cupy.array([[1., 2.]]))
        assert B.maxprint == 50

    def test_dia_array_maxprint_kwarg(self):
        data = cupy.array([[1., 2., 3.]])
        offsets = cupy.array([0])
        A = sparse.dia_array((data, offsets), shape=(3, 3), maxprint=10)
        assert A.maxprint == 10

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
        # Regression: DIA with data buffer wider than the matrix used to
        # raise a broadcast-shape ValueError in tocsc().
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

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_setdiag_python_types(self, fmt):
        # Regression: setdiag used to call ``values.astype(...)`` (or
        # ``values.ndim``) directly on the input, raising AttributeError
        # for Python lists / scalars.  scipy's ``_spbase.setdiag`` does
        # ``np.asarray(values)`` first; cupy now mirrors that.
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

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_inplace_scalar_preserves_identity(self, fmt):
        # ``A *= 2`` and ``A /= 2`` must mutate ``self.data`` in place
        # so the bound name still refers to the same object (matches
        # scipy's ``_data._data_matrix.__imul__``).  Without these
        # specials, ``A *= 2`` rebinds via ``A = A * 2`` and silently
        # allocates a new sparse matrix every time.
        cls = getattr(sparse, f'{fmt}_array')
        A = cls(cupy.array([[1., 2.], [3., 4.]]))
        old = A
        old_data_id = id(A.data)
        A *= 2
        assert A is old
        assert id(A.data) == old_data_id
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.array([[2., 4.], [6., 8.]]))
        A /= 2
        assert A is old
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.array([[1., 2.], [3., 4.]]))

    def test_inplace_scalar_promotes_dtype(self):
        # ``bool *= int`` violates numpy's same_kind cast rule and would
        # raise; CuPy intentionally diverges from scipy here (which
        # raises ``UFuncTypeError``) and reassigns ``self.data`` to a
        # cuSPARSE-supported dtype.  Object identity of ``self`` is
        # preserved; identity of ``self.data`` is not.
        A = sparse.csr_array(
            cupy.array([[True, False], [False, True]]))
        old = A
        old_data_id = id(A.data)
        A *= 2
        assert A is old
        assert A.dtype == cupy.float64
        assert id(A.data) != old_data_id
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.array([[2., 0.], [0., 2.]]))

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

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_setdiag_does_not_mutate_input(self, fmt):
        # Regression: CSR setdiag used ``x_data -= self.diagonal(k)``
        # in place.  Now that input is coerced via ``cupy.asarray``
        # (no copy when dtype matches), the in-place subtraction would
        # mutate the caller's array -- switched to out-of-place ``-``.
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

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_count_nonzero_axis_empty(self, fmt):
        # Regression: ``count_nonzero(axis=)`` on an empty sparse object
        # used to crash because ``cupy.bincount`` errors on zero-size
        # input even with ``minlength``.  scipy returns the zero-filled
        # axis vector.
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

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
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

    def test_from_coo_tuple_preserves_int64_indices(self):
        # Regression: csr_array((data, (row, col))) used to construct
        # an intermediate ``coo_matrix`` (not ``coo_array``), which ran
        # ``_get_index_dtype(check_contents=True)`` and silently
        # downcast int64 row/col arrays to int32.  Now uses
        # ``self._coo_container`` so the sparse-array dtype-preservation
        # promise is honored.
        data = cupy.array([1.0, 2.0], dtype=self.dtype)
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

    def test_power_zero_raises(self):
        """Array ** 0 raises NotImplementedError (would densify)."""
        a = _make_csr_sq(cupy, sparse, self.dtype, array=True)
        with pytest.raises(NotImplementedError):
            a ** 0


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


class TestCsrArrayTypePreservation:
    """Operations on csr_array should return csr_array (not csr_matrix)."""

    dtype = numpy.float64

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

    def test_no_getrow(self):
        with pytest.raises(AttributeError):
            self.arr.getrow(0)

    def test_no_getcol(self):
        with pytest.raises(AttributeError):
            self.arr.getcol(0)

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


# Matrix-only APIs that don't exist on sparray.  In SciPy 1.14 most of
# these were deprecated; SciPy 1.17 un-deprecated everything except
# ``A`` and ``H`` (those still emit DeprecationWarning on CuPy until
# users have a release cycle to migrate).  The ``A`` / ``H``
# DeprecationWarnings are silenced here; warning behaviour itself is
# exercised by ``test_base.py::TestDeprecatedSpmatrixApi``.
@pytest.mark.filterwarnings(
    "ignore:`spmatrix\\.(A|H)`:DeprecationWarning"
)
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

    def test_has_getrow(self):
        result = self.mat.getrow(0)
        assert sparse.issparse(result)

    def test_has_getcol(self):
        result = self.mat.getcol(0)
        assert sparse.issparse(result)

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
