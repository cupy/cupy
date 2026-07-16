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
import cupyx.scipy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import aslinearoperator, matrix_power, spsolve


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
        A = sparse.csr_array(cupy.array([[1., 2.]]))
        M = sparse.csr_matrix(cupy.array([[1., 2.]]))
        assert cupy.get_array_module(A) is cupy
        assert cupy.get_array_module(M) is cupy
        assert cupyx.scipy.get_array_module(A).__name__ == 'cupyx.scipy'
        assert cupyx.scipy.get_array_module(M).__name__ == 'cupyx.scipy'

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
        # zero-filled axis vector (exercises ``cupy.bincount`` on
        # zero-size input with ``minlength``).
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

    def test_min_max_axis_returns_1d_array(self, dtype):
        # Sparse *arrays* return a 1-D coo_array from min/max over an axis
        # (shape ``(M,)``), like sum()/argmin().  See
        # ``_data.py._min_or_max_axis``.
        a = sparse.csr_array(
            cupy.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=dtype))
        r0, r1 = a.min(axis=0), a.max(axis=1)
        assert r0.shape == (3,) and r0.ndim == 1
        assert r1.shape == (2,) and r1.ndim == 1
        assert isinstance(r0, sparse.coo_array) and r0.format == 'coo'
        # Consistent with sum(axis=), which is also 1-D for arrays.
        assert a.sum(axis=0).shape == (3,)

    def test_min_max_axis_matrix_stays_2d(self, dtype):
        # Matrices keep the legacy 2-D shape (1, M) / (M, 1).
        a = sparse.csr_matrix(
            cupy.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=dtype))
        assert a.min(axis=0).shape == (1, 3)
        assert a.max(axis=1).shape == (2, 1)
        assert isinstance(a.min(axis=0), sparse.coo_matrix)

    @pytest.mark.parametrize('fmt,method,axis', [
        ('csr', 'min', 0), ('csc', 'max', 1),
    ], ids=['min-axis0', 'max-axis1'])
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_minmax_axis_values(self, xp, sp, dtype, fmt, method, axis):
        m = getattr(sp, f'{fmt}_array')(xp.array(
            [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0],
             [4.0, 0.0, 0.0, 5.0]], dtype=dtype))
        return getattr(m, method)(axis=axis)


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

    def test_real_sign_unchanged(self):
        a = sparse.csr_matrix(cupy.array([[-2.0, 0, 3.0]]))
        b = a.sign()
        cupy.testing.assert_array_equal(b.data, cupy.array([-1.0, 1.0]))


# 1-D sparse arrays (coo_array / csr_array)

@testing.with_requires('scipy>=1.14')
class TestSparseArray1D:
    """1-D coo_array / csr_array parity with scipy.

    Covers construction, conversion, reductions, element-wise
    arithmetic, matmul, and indexing.  1-D is supported only for the
    *array* classes coo_array and csr_array (matching scipy); csc/dia and
    all matrix classes stay 2-D.
    """

    # -- construction --------------------------------------------------

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_construct_from_1d_dense(self, xp, sp):
        return sp.coo_array(xp.array([0., 1., 0., 2., 3.]))

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_construct_from_data_coords(self, xp, sp):
        data = xp.array([1., 2., 3.])
        coord = xp.array([1, 3, 4])
        return sp.coo_array((data, (coord,)), shape=(5,))

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_construct_empty_shape(self, xp, sp):
        return sp.coo_array((5,))

    def test_ndim_shape_coords(self):
        v = sparse.coo_array(cupy.array([0., 1., 0., 2.]))
        assert v.ndim == 1
        assert v.shape == (4,)
        assert isinstance(v, sparse.sparray)
        assert len(v.coords) == 1
        cupy.testing.assert_array_equal(v.coords[0], cupy.array([1, 3]))

    @pytest.mark.parametrize('fmt', ['csc', 'dia'])
    def test_non_1d_array_formats_reject_1d(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        with pytest.raises(ValueError):
            cls(cupy.array([0., 1., 2.]))

    def test_allow_nd_attribute(self):
        # Only coo/csr arrays advertise 1-D support.
        assert 1 in sparse.coo_array._allow_nd
        assert 1 in sparse.csr_array._allow_nd
        assert sparse.csc_array._allow_nd == (2,)
        assert sparse.coo_matrix._allow_nd == (2,)
        assert sparse.csr_matrix._allow_nd == (2,)

    @pytest.mark.parametrize('fmt', ['coo', 'csr', 'csc'])
    def test_matrix_classes_promote_1d_dense(self, fmt):
        # A 1-D dense input to a *matrix* class becomes 2-D (1, N).
        cls = getattr(sparse, f'{fmt}_matrix')
        m = cls(cupy.array([0., 1., 2.]))
        assert m.ndim == 2 and m.shape == (1, 3)

    def test_csr_array_from_1d_dense(self):
        v = sparse.csr_array(cupy.array([0., 1., 0., 2.]))
        assert isinstance(v, sparse.csr_array)
        assert v.ndim == 1 and v.shape == (4,)

    # -- reductions over an axis return 1-D (the primary goal) ---------

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_min_axis_returns_1d(self, xp, sp):
        a = sp.csr_array(xp.array(
            [[1., 0., 2., 0.], [0., 3., 0., 0.], [4., 0., 0., 5.]]))
        return a.min(axis=0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_max_axis_returns_1d(self, xp, sp):
        a = sp.csc_array(xp.array(
            [[1., 0., 2., 0.], [0., 3., 0., 0.], [4., 0., 0., 5.]]))
        return a.max(axis=1)

    # -- conversions ---------------------------------------------------

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsr_then_toarray(self, xp, sp):
        v = sp.coo_array(xp.array([0., 1., 0., 2., 3., 0.]))
        return v.tocsr()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_csr_1d_tocoo(self, xp, sp):
        v = sp.coo_array(xp.array([0., 1., 0., 2., 3., 0.])).tocsr()
        return v.tocoo()

    def test_tocsr_is_1d(self):
        v = sparse.coo_array(cupy.array([0., 1., 0., 2.])).tocsr()
        assert isinstance(v, sparse.csr_array)
        assert v.ndim == 1 and v.shape == (4,)

    @pytest.mark.parametrize('to', ['tocsc', 'todia'])
    def test_1d_to_2d_only_format_raises(self, to):
        v = sparse.coo_array(cupy.array([0., 1., 0., 2.]))
        with pytest.raises((ValueError, NotImplementedError)):
            getattr(v, to)()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_transpose_noop(self, xp, sp):
        v = sp.coo_array(xp.array([0., 1., 0., 2.]))
        return v.T

    # -- reshape 1-D <-> 2-D -------------------------------------------

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_reshape_1d_to_2d(self, xp, sp):
        v = sp.coo_array(xp.array([0., 1., 0., 2., 3., 0.]))
        return v.reshape(2, 3)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_reshape_2d_to_1d(self, xp, sp):
        a = sp.coo_array(xp.array([[0., 1., 0.], [2., 3., 0.]]))
        return a.reshape((6,))

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_reshape_2d_to_1d_fortran(self, xp, sp):
        a = sp.coo_array(xp.array([[0., 1., 0.], [2., 3., 0.]]))
        return a.reshape((6,), order='F')

    # -- scalar reductions on a 1-D array ------------------------------

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum(self, xp, sp):
        return sp.coo_array(xp.array([0., -1., 2., 3., -5.])).sum()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_axis0(self, xp, sp):
        return sp.coo_array(xp.array([0., -1., 2., 3., -5.])).sum(axis=0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean(self, xp, sp):
        return sp.coo_array(xp.array([0., -1., 2., 3., -5.])).mean()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_min_scalar(self, xp, sp):
        return sp.csr_array(xp.array([0., -1., 2., 3., -5.])).min()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_max_scalar(self, xp, sp):
        return sp.csr_array(xp.array([0., -1., 2., 3., -5.])).max()

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_argmax_scalar(self, xp, sp):
        return int(sp.csr_array(xp.array([0., -1., 2., 3., -5.])).argmax())

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_argmin_scalar(self, xp, sp):
        return int(sp.csr_array(xp.array([0., -1., 2., 3., -5.])).argmin())

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_count_nonzero(self, xp, sp):
        return sp.coo_array(xp.array([0., -1., 2., 0., -5.])).count_nonzero()

    # -- scalar / element-wise ops stay 1-D ----------------------------

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_scalar_multiply(self, xp, sp):
        # Value parity via toarray; cupy routes ``coo * scalar`` through
        # CSR (a pre-existing format choice), so compare densely.
        return (sp.coo_array(xp.array([0., 1., 0., 2.])) * 2).toarray()

    @pytest.mark.parametrize('op', [
        lambda a: -a,
        lambda a: abs(a),
        lambda a: a ** 2,
    ], ids=['negate', 'abs', 'power'])
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_unary_elementwise_stays_1d(self, xp, sp, op):
        return op(sp.coo_array(xp.array([0., -1., 0., 2.])))

    # -- 1-D construction / conversion / reduction edge cases ------------

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_coo_coords_as_2d_dense_array(self, xp, sp):
        # (data, (ndim, nnz)-array) coords form must still build 2-D
        # (regression: a dense-coords guard had rejected it).
        data = xp.array([1., 2., 3.])
        coords = xp.array([[0, 1, 0], [0, 1, 2]])
        return sp.coo_array((data, coords), shape=(2, 3))

    def test_coo_matrix_coords_as_2d_dense_array(self):
        m = sparse.coo_matrix(
            (cupy.array([1., 2., 3.]), cupy.array([[0, 0, 0], [1, 3, 4]])),
            shape=(1, 5))
        assert m.shape == (1, 5)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_csr_reshape_1d_to_2d(self, xp, sp):
        v = sp.csr_array(xp.array([0., 1., 0., 2., 3., 0.]))
        return v.reshape(2, 3)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_csr_reshape_2d_to_1d(self, xp, sp):
        a = sp.csr_array(xp.array([[0., 1., 0.], [2., 3., 0.]]))
        return a.reshape((6,))

    # scipy's count_nonzero gained the axis argument in 1.15.
    @testing.with_requires('scipy>=1.15')
    @testing.numpy_cupy_equal(sp_name='sp')
    def test_csr_count_nonzero_axis0(self, xp, sp):
        return sp.csr_array(xp.array([0., 1., 0., 2.])).count_nonzero(axis=0)

    def test_construct_csr_from_scipy_1d(self):
        s = scipy.sparse.coo_array(numpy.array([0., 1., 0., 2.]))
        v = sparse.csr_array(s)
        assert v.ndim == 1 and v.shape == (4,)
        cupy.testing.assert_array_equal(
            v.toarray(), cupy.array([0., 1., 0., 2.]))

    def test_construct_coo_from_scipy_1d(self):
        s = scipy.sparse.coo_array(numpy.array([0., 1., 0., 2.]))
        v = sparse.coo_array(s)
        assert v.ndim == 1 and v.shape == (4,)

    def test_nonzero_1d(self):
        # scipy returns int32 coords, cupy int64 -- compare values only.
        (idx,) = sparse.coo_array(
            cupy.array([0., 1., 0., 2., 3.])).nonzero()
        cupy.testing.assert_array_equal(idx, cupy.array([1, 3, 4]))

    def test_nonzero_1d_is_1tuple(self):
        nz = sparse.coo_array(cupy.array([0., 1., 0., 2.])).nonzero()
        assert isinstance(nz, tuple) and len(nz) == 1

    def test_csr_1d_3tuple_trims_to_nnz(self):
        # data/indices longer than indptr[-1] are trimmed (2-D parity).
        v = sparse.csr_array(
            (cupy.array([1., 2., 3., 4., 5.]), cupy.array([0, 1, 2, 3, 4]),
             cupy.array([0, 3])), shape=(5,))
        cupy.testing.assert_array_equal(
            v.toarray(), cupy.array([1., 2., 3., 0., 0.]))

    def test_csr_1d_3tuple_bad_indptr_start_raises(self):
        with pytest.raises(ValueError):
            sparse.csr_array(
                (cupy.array([1., 2., 3.]), cupy.array([0, 1, 2]),
                 cupy.array([1, 3])), shape=(5,))

    @pytest.mark.parametrize('op', ['maximum', 'minimum'])
    def test_maximum_minimum_sparse_1d(self, op):
        v = sparse.csr_array(cupy.array([0., 1., 0., 2.]))
        w = sparse.csr_array(cupy.array([1., 0., 0., 1.]))
        r = getattr(v, op)(w)
        assert r.ndim == 1 and r.shape == (4,)
        expected = getattr(cupy, op)(
            cupy.array([0., 1., 0., 2.]), cupy.array([1., 0., 0., 1.]))
        cupy.testing.assert_allclose(r.toarray(), expected)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_scalar_truediv_stays_1d(self, xp, sp):
        return (sp.coo_array(xp.array([0., 1., 0., 2.])) / 2).toarray()

    # -- 1-D structural ops, 0-D input, and 2-D-only op guards -----------

    @pytest.mark.parametrize('idx_dtype', [numpy.int32, numpy.int64])
    def test_csr_eliminate_zeros_1d(self, idx_dtype):
        v = sparse.csr_array(
            (cupy.array([1., 0., 2., 0., 3.]),
             cupy.arange(5, dtype=idx_dtype),
             cupy.array([0, 5], dtype=idx_dtype)), shape=(5,))
        v.eliminate_zeros()
        assert v.nnz == 3
        cupy.testing.assert_array_equal(
            v.toarray(), cupy.array([1., 0., 2., 0., 3.]))
        # A second call must not corrupt indptr.
        v.eliminate_zeros()
        cupy.testing.assert_array_equal(
            v.toarray(), cupy.array([1., 0., 2., 0., 3.]))

    def test_csr_prune_1d(self):
        v = sparse.coo_array(cupy.array([0., 1., 0., 2.])).tocsr()
        v.prune()
        assert v.nnz == 2

    def test_csr_sort_indices_1d(self):
        v = sparse.csr_array(
            (cupy.array([2., 1., 3.]), cupy.array([3, 1, 4]),
             cupy.array([0, 3])), shape=(5,))
        v.sort_indices()
        cupy.testing.assert_array_equal(v.indices, cupy.array([1, 3, 4]))
        cupy.testing.assert_array_equal(
            v.toarray(), cupy.array([0., 1., 0., 2., 3.]))

    def test_csr_sum_duplicates_1d(self):
        v = sparse.csr_array(
            (cupy.array([1., 2., 3.]), cupy.array([1, 1, 3]),
             cupy.array([0, 3])), shape=(5,))
        v.sum_duplicates()
        cupy.testing.assert_array_equal(
            v.toarray(), cupy.array([0., 3., 0., 3., 0.]))

    @pytest.mark.parametrize('fmt', ['coo', 'csr'])
    def test_coords_shape_mismatch_raises(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        # 2 coord arrays but a 1-D shape.
        with pytest.raises(ValueError):
            cls((cupy.array([1., 2., 3.]),
                 (cupy.array([0, 1, 2]), cupy.array([0, 1, 2]))), shape=(4,))
        # 1 coord array but a 2-D shape.
        with pytest.raises(ValueError):
            cls((cupy.array([1., 2.]), (cupy.array([0, 1]),)), shape=(1, 4))

    # scipy rejects 0-D array input with a format-dependent exception
    # type: TypeError for coo/csr, ValueError for csc ("CSC arrays don't
    # support 0D input").  Match the type per format.
    @pytest.mark.parametrize('fmt,exc', [
        ('coo', TypeError), ('csr', TypeError), ('csc', ValueError)])
    def test_zero_d_dense_rejected(self, fmt, exc):
        with pytest.raises(exc):
            getattr(sparse, f'{fmt}_array')(cupy.array(5.0))

    @pytest.mark.parametrize(
        'method', ['diagonal', 'trace', 'setdiag'])
    @pytest.mark.parametrize('fmt', ['coo', 'csr'])
    def test_diagonal_family_1d_raises(self, fmt, method):
        v = getattr(sparse, f'{fmt}_array')(cupy.array([0., 1., 2.]))
        with pytest.raises(ValueError):
            if method == 'setdiag':
                v.setdiag([1.])
            else:
                getattr(v, method)()

    # scipy accepts a length-1 tuple for the reduction axis since 1.16.
    @testing.with_requires('scipy>=1.16')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_tuple_axis(self, xp, sp):
        return sp.coo_array(xp.array([0., 1., 2., 3.])).sum(axis=(0,))

    # -- explicit shape vs. input dimensionality validation -------------
    # A ``shape`` whose ndim disagrees with a full sparse/dense input must
    # never silently reinterpret the data.  Before the fix,
    # ``coo_array(coo_2d, shape=(6,))`` built a corrupt 1-D array (non-zero
    # ``row``) whose ``toarray()`` was silently wrong, and
    # ``csr_array(csr_2d, shape=(6,))`` silently returned a 2-D array.

    @pytest.mark.parametrize('fmt', ['coo', 'csr'])
    def test_2d_input_1d_shape_rejected(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        dense = cupy.array([[1., 0., 2.], [0., 9., 0.]])
        with pytest.raises(ValueError):
            cls(dense, shape=(6,))                    # 2-D dense + 1-D shape
        with pytest.raises(ValueError):
            cls(cls(dense), shape=(6,))               # 2-D sparse (same fmt)
        with pytest.raises(ValueError):
            cls(sparse.coo_array(dense), shape=(6,))  # 2-D sparse (coo)

    def test_2d_coo_1d_shape_no_corruption(self):
        # The exact former corruption: a (2, 3) coo reinterpreted as (6,)
        # kept a non-zero ``row`` and densified to wrong values.
        a2 = sparse.coo_array(cupy.array([[1., 0., 2.], [0., 9., 0.]]))
        with pytest.raises(ValueError):
            sparse.coo_array(a2, shape=(6,))

    def test_1d_input_2d_shape_rejected_coo(self):
        # coo has no 1-D -> 2-D promotion; a 2-D shape on a 1-D input
        # (dense or sparse) is rejected, not silently reinterpreted.
        with pytest.raises(ValueError):
            sparse.coo_array(cupy.array([0., 1., 2.]), shape=(1, 3))
        with pytest.raises(ValueError):
            sparse.coo_array(
                sparse.coo_array(cupy.array([0., 1., 2.])), shape=(1, 3))

    def test_1d_sparse_incompatible_2d_shape_raises(self):
        # A 1-D csr routed to a 2-D shape it cannot fill (wrong indptr
        # length) must raise cleanly, never corrupt.
        v = sparse.coo_array(cupy.array([0., 1., 0., 2.])).tocsr()
        with pytest.raises(ValueError):
            sparse.csr_array(v, shape=(2, 2))

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_csr_dense_1d_promotes_to_2d(self, xp, sp):
        # Preserved: a 1-D dense with a (1, N) shape promotes to 2-D for
        # csr (matching scipy); the ndim guard must not block this.
        return sp.csr_array(
            xp.array([0., 1., 0., 2.]), shape=(1, 4)).toarray()

    def test_csr_1d_promotion_is_2d(self):
        r = sparse.csr_array(cupy.array([0., 1., 0., 2.]), shape=(1, 4))
        assert r.ndim == 2 and r.shape == (1, 4)
        v = sparse.csr_array(cupy.array([0., 1., 0., 2.]))
        r2 = sparse.csr_array(v, shape=(1, 4))
        assert r2.ndim == 2 and r2.shape == (1, 4)

    @pytest.mark.parametrize('fmt', ['coo', 'csr'])
    def test_matching_ndim_shape_still_accepted(self, fmt):
        # The guard rejects only ndim mismatches; a redundant matching
        # shape (1-D or 2-D) for a full input is still accepted.
        cls = getattr(sparse, f'{fmt}_array')
        a2 = cls(cupy.array([[1., 0.], [0., 2.]]))
        assert cls(a2, shape=(2, 2)).shape == (2, 2)
        v1 = cls(cupy.array([0., 1., 0., 2.]))
        assert cls(v1, shape=(4,)).shape == (4,)

    @pytest.mark.parametrize('fmt', ['coo', 'csr'])
    def test_1d_sparse_enlarging_shape_consistent(self, fmt):
        # A same-ndim (1-D) explicit shape that only enlarges is honored
        # consistently for coo and csr (csr must not silently drop it via
        # the same-format fast-adoption path), and the indices are
        # bounds-validated.
        cls = getattr(sparse, f'{fmt}_array')
        v = cls(cupy.array([0., 1., 0., 2.]))
        r = cls(v, shape=(6,))
        assert r.shape == (6,) and r.ndim == 1
        cupy.testing.assert_array_equal(
            r.toarray(), cupy.array([0., 1., 0., 2., 0., 0.]))

    def test_1d_sparse_shrinking_shape_out_of_bounds_raises(self):
        # Shrinking below the stored indices must raise (cupy is stricter
        # than scipy, which silently builds an out-of-bounds array).
        v = sparse.csr_array(cupy.array([0., 1., 0., 2.]))
        with pytest.raises(ValueError):
            sparse.csr_array(v, shape=(2,))

    # -- 1-D arithmetic / matmul / indexing ------------------------------

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_add(self, xp, sp):
        a = sp.coo_array(xp.array([0., 1., 0., 2., 3.]))
        b = sp.coo_array(xp.array([1., 0., 0., 1., 2.]))
        return a + b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_sub(self, xp, sp):
        a = sp.coo_array(xp.array([0., 1., 0., 2., 3.]))
        b = sp.coo_array(xp.array([1., 0., 0., 1., 2.]))
        return a - b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_multiply(self, xp, sp):
        a = sp.coo_array(xp.array([0., 1., 0., 2., 3.]))
        b = sp.coo_array(xp.array([1., 0., 0., 1., 2.]))
        return a.multiply(b)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_maximum(self, xp, sp):
        a = sp.csr_array(xp.array([0., 1., 0., 2., 3.]))
        b = sp.csr_array(xp.array([1., 0., 0., 1., 2.]))
        return a.maximum(b)

    def test_1d_divide_by_dense(self):
        a = sparse.csr_array(cupy.array([0., 1., 0., 2., 3.]))
        r = a / cupy.array([1., 2., 1., 2., 1.])
        assert r.ndim == 1 and r.shape == (5,)
        cupy.testing.assert_allclose(
            r.toarray(), cupy.array([0., 0.5, 0., 1., 3.]))

    def test_1d_comparison(self):
        a = sparse.csr_array(cupy.array([0., 1., 0., 2., 3.]))
        b = sparse.csr_array(cupy.array([1., 0., 0., 1., 2.]))
        r = a != b
        assert r.ndim == 1 and r.shape == (5,)
        cupy.testing.assert_array_equal(
            r.toarray(), cupy.array([1, 1, 0, 1, 1], dtype=bool))

    def test_1d_dot_scalar(self):
        a = sparse.coo_array(cupy.array([0., 1., 0., 2., 3.]))
        b = sparse.coo_array(cupy.array([1., 0., 2., 1., 0.]))
        r = a @ b
        assert cupy.ndim(r) == 0
        cupy.testing.assert_allclose(cupy.asarray(r), cupy.asarray(2.0))

    def test_1d_matmul_dense_vec(self):
        a = sparse.coo_array(cupy.array([0., 1., 0., 2., 3.]))
        r = a @ cupy.array([1., 2., 3., 4., 5.])
        cupy.testing.assert_allclose(cupy.asarray(r), cupy.asarray(25.0))

    # scipy's 2-D @ 1-D raised through 1.14; fixed in 1.15.
    @testing.with_requires('scipy>=1.15')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_2d_matmul_1d(self, xp, sp):
        M = sp.csr_array(xp.array([[1., 0, 2, 0], [0, 3, 0, 1]]))
        v = sp.coo_array(xp.array([1., 2., 3., 4.]))
        return M @ v

    # scipy 1.14 returned csr from 1-D @ 2-D; coo (as here) since 1.15.
    @testing.with_requires('scipy>=1.15')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_matmul_2d(self, xp, sp):
        M = sp.csr_array(xp.array([[1., 0, 2, 0], [0, 3, 0, 1]]))
        v = sp.coo_array(xp.array([1., 2.]))
        return v @ M

    def test_1d_getitem_int(self):
        v = sparse.csr_array(cupy.array([0., 1., 0., 2., 3.]))
        assert float(v[3]) == 2.0
        assert float(v[-1]) == 3.0

    def test_1d_getitem_slice(self):
        v = sparse.csr_array(cupy.array([0., 1., 0., 2., 3.]))
        r = v[1:4]
        assert r.ndim == 1 and r.shape == (3,)
        cupy.testing.assert_allclose(
            r.toarray(), cupy.array([1., 0., 2.]))

    def test_1d_getitem_fancy(self):
        v = sparse.csr_array(cupy.array([0., 1., 0., 2., 3.]))
        r = v[cupy.array([0, 3, 4])]
        assert r.ndim == 1 and r.shape == (3,)
        cupy.testing.assert_allclose(
            r.toarray(), cupy.array([0., 2., 3.]))

    def test_2d_array_int_index_is_1d(self):
        # scipy behavior: A[i], A[i,:], A[:,j] on a 2-D *array* -> 1-D.
        A = sparse.csr_array(cupy.array([[1., 0, 2], [0, 3, 0]]))
        assert A[1].shape == (3,) and A[1].ndim == 1
        assert A[0, :].shape == (3,)
        assert A[:, 2].shape == (2,)
        assert cupy.ndim(A[1, 2]) == 0
        cupy.testing.assert_allclose(
            A[0].toarray(), cupy.array([1., 0., 2.]))

    def test_2d_matrix_int_index_stays_2d(self):
        M = sparse.csr_matrix(cupy.array([[1., 0, 2], [0, 3, 0]]))
        assert M[1].shape == (1, 3) and M[1].ndim == 2

    def test_1d_setitem_int(self):
        v = sparse.csr_array(cupy.array([0., 1., 0., 2.]))
        v[2] = 9.0
        cupy.testing.assert_allclose(
            v.toarray(), cupy.array([0., 1., 9., 2.]))
        assert v.ndim == 1

    def test_1d_setitem_slice(self):
        v = sparse.csr_array(cupy.array([0., 1., 0., 2.]))
        v[0:2] = cupy.array([7., 8.])
        cupy.testing.assert_allclose(
            v.toarray(), cupy.array([7., 8., 0., 2.]))

    # -- 1-D iteration / stacking / random_array -------------------------

    def test_1d_iter_yields_elements(self):
        v = sparse.csr_array(cupy.array([1., 2., 0., 3.]))
        els = list(v)
        assert all(isinstance(e, cupy.ndarray) and e.ndim == 0 for e in els)
        cupy.testing.assert_array_equal(
            cupy.asarray([float(e) for e in els]),
            cupy.array([1., 2., 0., 3.]))

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_hstack_1d(self, xp, sp):
        a = sp.coo_array(xp.array([1., 2., 0., 3.]))
        b = sp.coo_array(xp.array([0., 5., 6., 0.]))
        return sp.hstack([a, b])

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_vstack_1d(self, xp, sp):
        a = sp.coo_array(xp.array([1., 2., 0., 3.]))
        b = sp.coo_array(xp.array([0., 5., 6., 0.]))
        return sp.vstack([a, b])

    def test_hstack_1d_returns_array(self):
        a = sparse.coo_array(cupy.array([1., 2., 0.]))
        r = sparse.hstack([a, a])
        assert isinstance(r, sparse.sparray) and r.shape == (1, 6)

    def test_random_array_1d(self):
        r = sparse.random_array((100,), density=0.1, rng=0)
        assert isinstance(r, sparse.sparray)
        assert r.ndim == 1 and r.shape == (100,) and r.format == 'coo'
        assert r.nnz == 10

    def test_random_array_1d_csr(self):
        r = sparse.random_array((50,), density=0.2, format='csr', rng=1)
        assert r.ndim == 1 and r.shape == (50,) and r.format == 'csr'

    # -- 1-D indexing format and CSC indexing ---------------------------

    def test_2d_csc_int_index_is_1d(self):
        # A 2-D csc_array reduces an integer-indexed axis to 1-D, as a
        # coo_array like scipy: the reduction must route through COO
        # because csc has no 1-D shape.
        A = sparse.csc_array(cupy.array([[1., 0, 2], [0, 3, 0]]))
        for r, expected in [
                (A[1], cupy.array([0., 3., 0.])),
                (A[0, :], cupy.array([1., 0., 2.])),
                (A[:, 2], cupy.array([2., 0.]))]:
            assert r.ndim == 1 and r.format == 'coo'
            cupy.testing.assert_allclose(r.toarray(), expected)
        assert cupy.ndim(A[1, 2]) == 0

    def test_1d_getitem_preserves_csr_format(self):
        # Slicing / fancy-indexing a 1-D csr_array keeps csr (scipy parity).
        v = sparse.csr_array(cupy.array([0., 1., 0., 2., 3.]))
        assert v[1:4].format == 'csr'
        assert v[cupy.array([0, 3, 4])].format == 'csr'
        assert v[cupy.array([True, False, True, False, True])].format == 'csr'

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_multiply_broadcast_2d_sparse(self, xp, sp):
        # 1-D * 2-D broadcasts to a genuinely 2-D result (scipy); the
        # (1, N)-backing result must not be squeezed back to 1-D.
        v = sp.csr_array(xp.array([1., 2., 3.]))
        M = sp.csr_array(xp.array([[1., 0, 1], [0, 2, 0]]))
        return v.multiply(M)

    def test_1d_multiply_broadcast_2d_dense(self):
        # Value/shape parity (cupy returns csr, scipy coo -- a 2-D
        # format choice -- so compare densified rather than via the
        # format-checking allclose decorator).
        v = sparse.csr_array(cupy.array([1., 2., 3.]))
        r = v.multiply(cupy.array([[1., 0, 1], [0, 2, 0]]))
        assert r.ndim == 2 and r.shape == (2, 3)
        cupy.testing.assert_allclose(
            r.toarray(), cupy.array([[1., 0., 3.], [0., 4., 0.]]))

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_maximum_broadcast_2d_dense(self, xp, sp):
        # A dense 2-D operand broadcasts the result to dense 2-D; it
        # must not be flattened back to 1-D.
        v = sp.csr_array(xp.array([1., 2., 3.]))
        return v.maximum(xp.array([[0., 3, 0], [2, 0, 2]]))

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_ne_broadcast_2d_dense(self, xp, sp):
        v = sp.csr_array(xp.array([1., 2., 3.]))
        return (v != xp.ones((2, 3))).astype(xp.int8)

    def test_1d_multiply_broadcast_2d_stays_2d(self):
        v = sparse.csr_array(cupy.array([1., 2., 3.]))
        r = v.multiply(cupy.ones((2, 3)))
        assert r.ndim == 2 and r.shape == (2, 3)

    # A 2-D operand with a *single* row is still 2-D: the result must stay
    # (1, N), decided by the operand's dimensionality, not the result shape
    # (regression: _squeeze_to_1d wrongly collapsed a real (1, N) to 1-D).
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_lt_dense_1row_stays_2d(self, xp, sp):
        return sp.csr_array(xp.array([1., 0., 2.])) < xp.array([[1., 0., 3.]])

    def test_1d_op_2d_one_row_shapes(self):
        v = sparse.csr_array(cupy.array([1., 0., 2.]))
        one_row = sparse.csr_array(cupy.array([[1., 0., 3.]]))
        # sparse and dense one-row 2-D operands both yield a 2-D result.
        assert v.multiply(one_row).shape == (1, 3)
        assert v.multiply(cupy.array([[1., 2., 3.]])).shape == (1, 3)
        assert (v < cupy.array([[1., 0., 3.]])).shape == (1, 3)
        assert (v / cupy.array([[1., 1., 1.]])).shape == (1, 3)

    @pytest.mark.parametrize('fmt', ['csr', 'csc'])
    def test_2d_one_row_plus_1d_stays_2d(self, fmt):
        # A genuinely 2-D (1, N) self plus a 1-D other is numpy-broadcast
        # to (1, N), not collapsed to 1-D (which for csc even crashed).
        A = getattr(sparse, f'{fmt}_array')(cupy.array([[1., 2., 3.]]))
        b = sparse.csr_array(cupy.array([1., 1., 1.]))
        assert (A + b).shape == (1, 3)
        assert (A - b).shape == (1, 3)
        cupy.testing.assert_array_equal(
            (A + b).toarray(), cupy.array([[2., 3., 4.]]))

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_1d_truediv_dense_2d_one_row(self, xp, sp):
        # 1-D / dense (1, N) returns a dense (1, N); the inner op returns
        # NotImplemented for a numpy RHS, so the operator protocol must fall
        # back rather than crash in _finalize_1d_op.
        v = sp.csr_array(xp.array([1., 0., 2.]))
        return v / xp.asarray([[1., 2., 3.]])

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_1d_mean_integer_dtype(self, xp, sp):
        # mean(dtype=<int>) must accumulate in float and cast once, not
        # truncate each 1/N-scaled term.  The first case gave 24 instead of
        # 25; the second is starker -- each 1/8-scaled term is < 1, so the
        # old per-element truncation collapsed the whole mean to 0.
        return (int(sp.coo_array(
                    xp.asarray([10., 20., 30., 40.])).mean(dtype=xp.int64)),
                int(sp.coo_array(xp.ones(8)).mean(dtype=xp.int64)))

    @pytest.mark.parametrize('op', [
        lambda a, b: a.multiply(b),
        lambda a, b: a < b,
        lambda a, b: a.maximum(b),
        lambda a, b: a + b,
    ], ids=['multiply', 'lt', 'maximum', 'add'])
    def test_1d_op_1d_stays_1d(self, op):
        # Both operands 1-D -> the result must stay 1-D.
        v = sparse.csr_array(cupy.array([1., 0., 2.]))
        r = op(v, sparse.csr_array(cupy.array([2., 1., 0.])))
        assert r.ndim == 1 and r.shape == (3,)

    def test_1d_add_2d_sparse_raises(self):
        # Adding a 1-D array to an incompatible 2-D sparse must raise
        # (scipy: inconsistent shapes), not silently drop rows.  1-D
        # arrays use int64 indices, exercising the int64 csrgeam guard.
        v = sparse.csr_array(cupy.array([1., 2., 3.]))
        M = sparse.csr_array(cupy.array([[1., 0, 1], [0, 2, 0]]))
        with pytest.raises(ValueError):
            v + M

    def test_int64_csrgeam_shape_mismatch_raises(self):
        # _cupy_csrgeam_int64 must validate shapes like csrgeam2/spgeam,
        # instead of silently dropping rows via COO concatenation.
        a = sparse.csr_array(cupy.array([[1., 2., 3.]]))
        b = sparse.csr_array(cupy.array([[1., 0, 1], [0, 2, 0]]))
        a.indices = a.indices.astype(cupy.int64)
        a.indptr = a.indptr.astype(cupy.int64)
        with pytest.raises(ValueError):
            a + b

    def test_1d_getitem_too_many_indices_raises(self):
        v = sparse.csr_array(cupy.array([0., 1., 0., 2.]))
        with pytest.raises(IndexError):
            v[0, 1]

    def test_1d_setitem_sparse_rhs(self):
        # Assigning a 1-D sparse RHS must not crash on x.shape[1].
        v = sparse.csr_array(cupy.array([0., 1., 0., 2.]))
        v[0:2] = sparse.csr_array(cupy.array([7., 8.]))
        assert v.ndim == 1
        cupy.testing.assert_allclose(
            v.toarray(), cupy.array([7., 8., 0., 2.]))

    # -- bool dtype indexing (supported data dtype) ----------------------

    def test_1d_bool_indexing(self):
        v = sparse.coo_array(
            cupy.array([True, False, True, False, True])).tocsr()
        assert bool(v[2]) and not bool(v[1])
        cupy.testing.assert_array_equal(
            v[1:4].toarray(), cupy.array([False, True, False]))
        cupy.testing.assert_array_equal(
            v[cupy.array([0, 3])].toarray(), cupy.array([True, False]))

    def test_2d_bool_indexing(self):
        # bool is a supported data dtype: scalar (via _get_intXint) and
        # fancy/minor-axis (via _fill_B) indexing must handle it.
        A = sparse.csr_array(
            cupy.array([[True, False, True], [False, True, False]]))
        assert bool(A[0, 2]) and not bool(A[0, 1])
        cupy.testing.assert_array_equal(
            A[1].toarray(), cupy.array([False, True, False]))
        cupy.testing.assert_array_equal(
            A[:, cupy.array([0, 2])].toarray(),
            cupy.array([[True, True], [False, False]]))


class Test1dIndexingEdges:
    """Edge semantics of 1-D indexing/assignment (numpy/scipy parity)."""

    def test_setitem_tuple_key_raises(self):
        # A tuple key on a 1-D array is too many indices; it must raise
        # (like numpy/scipy) rather than be misread as a fancy index.
        v = sparse.csr_array(cupy.array([1., 0., 2.]))
        with pytest.raises(IndexError, match='too many indices'):
            v[1, 2] = 9.
        cupy.testing.assert_array_equal(
            v.toarray(), cupy.array([1., 0., 2.]))

    def test_setitem_ellipsis(self):
        v = sparse.csr_array(cupy.array([1., 0., 2.]))
        with pytest.warns(sparse.SparseEfficiencyWarning):
            v[...] = 7.
        cupy.testing.assert_array_equal(
            v.toarray(), cupy.array([7., 7., 7.]))

    def test_getitem_none_adds_axis(self):
        # v[None] / v[None, :] -> (1, N); v[:, None] -> (N, 1).  The
        # promoted axis makes the result 2-D COO, matching scipy.
        v = sparse.csr_array(cupy.array([1., 0., 2.]))
        for key in (None, (None, slice(None))):
            r = v[key]
            assert r.shape == (1, 3) and r.format == 'coo'
            cupy.testing.assert_array_equal(
                r.toarray(), cupy.array([[1., 0., 2.]]))
        r = v[:, None]
        assert r.shape == (3, 1) and r.format == 'coo'
        cupy.testing.assert_array_equal(
            r.toarray(), cupy.array([[1.], [0.], [2.]]))

    def test_getitem_none_with_int(self):
        # An integer plus one new axis gives a length-1 sparse array of
        # the same format (scipy: csr stays csr).
        v = sparse.csr_array(cupy.array([1., 0., 2.]))
        for key in ((None, 2), (2, None)):
            r = v[key]
            assert r.shape == (1,) and r.format == 'csr'
            cupy.testing.assert_array_equal(r.toarray(), cupy.array([2.]))
        # A zero element round-trips as an empty (1,) array.
        assert v[None, 1].nnz == 0

    def test_getitem_none_beyond_2d_raises(self):
        v = sparse.csr_array(cupy.array([1., 0., 2.]))
        with pytest.raises(IndexError):
            v[None, None]


class TestCooArrayGetitem:
    """coo_array supports indexing (scipy >= 1.17); coo_matrix does not."""

    def test_coo_2d_getitem(self):
        dense = cupy.array([[1., 0., 2., 0.], [0., 3., 0., 0.],
                            [4., 0., 0., 5.]])
        A = sparse.coo_array(dense)
        r = A[1]
        assert r.shape == (4,) and r.format == 'coo'
        cupy.testing.assert_array_equal(r.toarray(), dense[1])
        assert float(A[2, 3]) == 5.
        r = A[:, 1]
        assert r.shape == (3,) and r.format == 'coo'
        cupy.testing.assert_array_equal(r.toarray(), dense[:, 1])
        r = A[0:2]
        assert r.shape == (2, 4) and r.format == 'coo'
        cupy.testing.assert_array_equal(r.toarray(), dense[0:2])
        r = A[cupy.array([0, 2])]
        assert r.shape == (2, 4) and r.format == 'coo'
        cupy.testing.assert_array_equal(
            r.toarray(), dense[cupy.array([0, 2])])

    def test_coo_1d_getitem(self):
        v = sparse.coo_array(cupy.array([1., 0., 2., 0., 3.]))
        assert float(v[4]) == 3.
        r = v[::2]
        assert r.shape == (3,) and r.format == 'coo'
        cupy.testing.assert_array_equal(
            r.toarray(), cupy.array([1., 2., 3.]))

    def test_coo_matrix_not_subscriptable(self):
        M = sparse.coo_matrix(cupy.eye(3))
        with pytest.raises(TypeError, match='not subscriptable'):
            M[0]


class TestBoolSum:
    """sum() on bool data (cuSPARSE has no bool arithmetic)."""

    @pytest.mark.parametrize('fmt', SPARSE_FORMATS)
    def test_bool_sum_matches_dense(self, fmt):
        dense = cupy.zeros((3, 4), dtype=bool)
        dense[cupy.arange(3), cupy.arange(3)] = True
        dense[0, 2] = True
        if fmt == 'dia':
            # DIA has no dense-input constructor form.
            A = sparse.dia_array(
                (cupy.ones((2, 3), dtype=bool),
                 cupy.array([0, 2], dtype='i')), shape=(3, 4))
            dense = A.astype(cupy.float64).toarray() != 0
        else:
            A = getattr(sparse, f'{fmt}_array')(dense)
        r = A.sum()
        assert r.dtype == cupy.int64
        assert int(r) == int(dense.sum())
        for axis in (0, 1):
            r = A.sum(axis=axis)
            assert r.dtype == cupy.int64
            cupy.testing.assert_array_equal(
                cupy.asarray(r).ravel(), dense.sum(axis=axis))

    def test_bool_sum_duplicates_axis(self):
        # A duplicate stored True coalesces (a bool coordinate is present
        # once), so the axis sum equals densify-then-sum -- i.e. numpy's
        # dense semantics.  This is deliberately NOT compared against
        # scipy: on a non-canonical bool COO scipy's axis sum double-counts
        # the duplicate ([0, 2, 0]) while its full sum coalesces (1), an
        # internal inconsistency cupy avoids.
        A = sparse.coo_array((cupy.array([True, True]),
                              (cupy.array([0, 0]), cupy.array([1, 1]))),
                             shape=(2, 3))
        dense = A.toarray()
        r = A.sum(axis=0)
        assert r.dtype == cupy.int64
        cupy.testing.assert_array_equal(r, dense.sum(axis=0))
        cupy.testing.assert_array_equal(r, cupy.array([0, 1, 0]))

    # scipy coalesces bool duplicates in the full sum only since 1.16
    # (1.14/1.15 counted them, returning 2); cupy matches the newer,
    # densify-consistent result, so compare against scipy >= 1.16.
    @testing.with_requires('scipy>=1.16')
    @testing.numpy_cupy_equal(sp_name='sp')
    def test_bool_sum_duplicates_scalar(self, xp, sp):
        A2 = sp.coo_array((xp.array([True, True]),
                           (xp.array([0, 0]), xp.array([1, 1]))), shape=(2, 3))
        v = sp.coo_array((xp.array([True, True]), (xp.array([1, 1]),)),
                         shape=(3,))
        return int(A2.sum()), int(v.sum())

    def test_matrix_bool_sum_stays_2d(self):
        M = sparse.csr_matrix(cupy.eye(3, dtype=bool))
        r = M.sum(axis=0)
        assert r.shape == (1, 3)
        cupy.testing.assert_array_equal(r, cupy.ones((1, 3), dtype=cupy.int64))

    @pytest.mark.parametrize('fmt', ['coo', 'csr', 'csc'])
    def test_bool_sum_does_not_mutate(self, fmt):
        # A reduction must not compact its operand: the bool coalescing is
        # done on a copy, so nnz/coordinates/flags are unchanged.
        A = sparse.coo_array(
            (cupy.array([True, True]),
             (cupy.array([0, 0]), cupy.array([1, 1]))), shape=(2, 3)
        ).asformat(fmt)
        n0 = A.nnz
        assert int(A.sum()) == 1
        assert A.nnz == n0

    def test_bool_sum_1d_does_not_mutate(self):
        v = sparse.coo_array(
            (cupy.array([True, True]), (cupy.array([1, 1]),)), shape=(3,))
        n0 = v.nnz
        assert int(v.sum()) == 1
        assert v.nnz == n0


class TestBoolMean:
    """mean() on bool coalesces duplicates so mean == sum/n (densify)."""

    def test_bool_mean_equals_sum_over_n(self):
        # A duplicate stored True coalesces (densify), so the mean divides
        # the coalesced sum by the element count -- unlike scipy, whose mean
        # counts duplicates while its sum coalesces (an inconsistency cupy
        # avoids).
        A = sparse.coo_array(
            (cupy.array([True, True]),
             (cupy.array([0, 0]), cupy.array([1, 1]))), shape=(2, 3))
        assert abs(float(A.mean()) - float(A.sum()) / 6) < 1e-12
        assert abs(float(A.mean()) - 1 / 6) < 1e-12

    @pytest.mark.parametrize('axis', [None, 0, 1])
    def test_bool_mean_matches_densify(self, axis):
        A = sparse.csr_array(cupy.eye(3, dtype=bool))
        dense = A.toarray().astype(cupy.float64)
        cupy.testing.assert_allclose(
            cupy.asarray(A.mean(axis=axis)), dense.mean(axis=axis))

    def test_bool_mean_does_not_mutate(self):
        A = sparse.coo_array(
            (cupy.array([True, True]),
             (cupy.array([0, 0]), cupy.array([1, 1]))), shape=(2, 3))
        n0 = A.nnz
        A.mean()
        assert A.nnz == n0


class TestBoolDenseConversion:
    def test_csc_bool_dense_round_trip(self):
        # cuSPARSE dense<->sparse conversion is float-only; the bool CSC
        # paths route through the pure-CuPy CSR fallback on the transpose.
        d = cupy.array([[True, False], [False, True], [True, True]])
        A = sparse.csc_array(d)
        assert A.format == 'csc' and A.dtype == bool
        cupy.testing.assert_array_equal(A.toarray(), d)
        f_out = A.toarray(order='F')
        assert f_out.flags.f_contiguous
        cupy.testing.assert_array_equal(f_out, d)

    def test_csr_bool_toarray_order_f(self):
        # order='F' affects the layout only; the values must match the
        # C-order result (csr2dense addresses the output by logical
        # position, letting the strides handle the layout).
        d = cupy.array([[True, False, True], [False, True, True]])
        A = sparse.csr_array(d)
        out = A.toarray(order='F')
        assert out.flags.f_contiguous
        cupy.testing.assert_array_equal(out, d)


class TestMeanEmpty:
    """mean() over a zero-length axis raises like scipy."""

    # scipy raises ZeroDivisionError on an empty mean; accept_error makes
    # the decorator require both libraries to raise the same type.
    @pytest.mark.parametrize('shape', [(0,), (0, 3)])
    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp',
                                 accept_error=ZeroDivisionError)
    def test_mean_empty_raises(self, xp, sp, shape):
        return sp.coo_array(xp.zeros(shape)).mean()

    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp',
                                 accept_error=ZeroDivisionError)
    def test_mean_empty_axis_raises(self, xp, sp):
        return sp.csr_array(xp.zeros((3, 0))).mean(axis=1)

    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean_values_2d(self, xp, sp):
        return sp.csr_array(
            xp.array([[1., 0., 2.], [0., 3., 0.]])).mean(axis=0)

    @testing.with_requires('scipy')
    @testing.numpy_cupy_equal(sp_name='sp')
    def test_mean_values_scalar(self, xp, sp):
        a = sp.csr_array(xp.array([[1., 0., 2.], [0., 3., 0.]]))
        v = sp.coo_array(xp.array([1., 0., 2.]))
        return float(a.mean()), float(v.mean())

    def test_1d_mean_out_shape_checked(self):
        # A wrong-shaped ``out`` must raise, not silently broadcast the
        # scalar mean; ``out=`` is forwarded to the reduction, which
        # validates it (like 1-D sum and numpy).
        v = sparse.coo_array(cupy.array([1., 2., 3.]))
        with pytest.raises(ValueError):
            v.mean(out=cupy.empty(5))
        buf = cupy.empty(())
        assert v.mean(out=buf) is buf
        cupy.testing.assert_allclose(buf, 2.0)

    def test_1d_sum_out_shape_checked(self):
        # Symmetric with mean: ``out=`` is forwarded to the reduction, so a
        # wrong-shaped ``out`` raises and a 0-D ``out`` is returned.
        v = sparse.coo_array(cupy.array([1., 2., 3.]))
        with pytest.raises(ValueError):
            v.sum(out=cupy.empty(5))
        buf = cupy.empty(())
        assert v.sum(out=buf) is buf
        cupy.testing.assert_allclose(buf, 6.0)


class TestTupleAxisValidation:
    """Tuple reduction axes: valid ones collapse, bogus ones raise."""

    @pytest.mark.parametrize('bad', [(None,), (0.0,), (0, 0), (0, 1, 2)])
    def test_bad_tuple_axis_raises(self, bad):
        A = sparse.csr_array(cupy.array([[1., 0., 2.], [0., 3., 0.]]))
        with pytest.raises((TypeError, ValueError)):
            A.sum(axis=bad)

    @testing.with_requires('scipy>=1.16')
    def test_none_tuple_axis_matches_scipy(self):
        # scipy rejects a non-integer axis element (given None) too.
        A = sparse.csr_array(cupy.array([[1., 0., 2.], [0., 3., 0.]]))
        An = scipy.sparse.csr_array(numpy.array([[1., 0., 2.], [0., 3., 0.]]))
        with pytest.raises(TypeError):
            A.sum(axis=(None,))
        with pytest.raises(TypeError):
            An.sum(axis=(None,))


class Test1dConstructorParity:
    """1-D constructor forms behave like their 2-D counterparts."""

    def test_1d_three_tuple_float_indices(self):
        # The 2-D 3-tuple path casts non-integer index arrays; the 1-D
        # form (which routes through it) must accept them identically.
        v = sparse.csr_array(
            (cupy.array([1., 2.]), cupy.array([0., 2.]),
             cupy.array([0, 2])), shape=(5,))
        assert v.shape == (5,)
        assert v.indices.dtype.kind == 'i'
        cupy.testing.assert_array_equal(
            v.toarray(), cupy.array([1., 0., 2., 0., 0.]))

    def test_1d_three_tuple_validation(self):
        with pytest.raises(ValueError, match='index pointer should start'):
            sparse.csr_array(
                (cupy.array([1.]), cupy.array([0]), cupy.array([1, 1])),
                shape=(3,))

    @testing.with_requires('scipy')
    def test_scalar_input_matches_scipy_exc_type(self):
        # cupy's 0-D-input exception type must match scipy's per format
        # (TypeError for coo/csr, ValueError for csc).
        for name in ('coo_array', 'csr_array', 'csc_array'):
            with pytest.raises(Exception) as cu:
                getattr(sparse, name)(cupy.array(5.))
            with pytest.raises(Exception) as sp:
                getattr(scipy.sparse, name)(numpy.array(5.))
            assert type(cu.value) is type(sp.value)

    def test_coo_bad_ndim_shape_tuple(self):
        # For a coo_array, an int-tuple shape of unsupported length reports
        # the dimensionality limit rather than a generic input error.
        with pytest.raises(ValueError, match='1-D and 2-D'):
            sparse.coo_array((2, 3, 4))
        # coo_matrix keeps scipy's generic TypeError('invalid input
        # format') for a 1-tuple shape (scipy's coo_matrix does the same).
        with pytest.raises(TypeError):
            sparse.coo_matrix((5,))


class TestReshapeAcrossNdim:
    def test_reshape_2d_to_1d(self):
        # scipy: reshaping a 2-D csr_array to 1-D yields a 1-D coo_array.
        A = sparse.csr_array(cupy.eye(4))
        r = A.reshape(16)
        assert r.shape == (16,) and r.format == 'coo'
        cupy.testing.assert_array_equal(r.toarray(), cupy.eye(4).ravel())

    def test_reshape_matrix_to_1d_raises(self):
        M = sparse.csr_matrix(cupy.eye(4))
        with pytest.raises(ValueError, match='shape must have length'):
            M.reshape(16)


class Test1dCountNonzeroAxis:
    @pytest.mark.parametrize('fmt', ('csr', 'coo'))
    def test_1d_count_nonzero_axis(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        v = cls((cupy.array([1., 0., 2.]), (cupy.array([0, 1, 3]),)),
                shape=(5,))
        # The stored explicit zero does not count.
        for axis in (None, 0, -1, (0,)):
            assert v.count_nonzero(axis=axis) == 2
        with pytest.raises(ValueError):
            v.count_nonzero(axis=1)


class TestMajorFancyFlagPreservation:
    def test_row_fancy_preserves_sort_flags(self):
        # Gathering whole rows preserves per-row order/uniqueness, so the
        # result inherits the flags without recomputing them on the GPU.
        A = sparse.csr_array(cupy.array([[1., 0., 2.], [0., 3., 0.]]))
        assert A.has_canonical_format  # dense conversion is canonical
        B = A[cupy.array([1, 0, 1]), :]
        assert getattr(B, '_has_canonical_format', None) is True
        cupy.testing.assert_array_equal(
            B.toarray(),
            cupy.array([[0., 3., 0.], [1., 0., 2.], [0., 3., 0.]]))


class TestMatrixPower:
    # Values compared against scipy's matrix_power; format is ignored
    # (cupy returns csr for power 0, scipy dia).
    @pytest.mark.parametrize('p', [0, 1, 2, 3, 5])
    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp', _check_sparse_format=False)
    def test_matrix_power_values(self, xp, sp, p):
        A = sp.csr_array(xp.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]))
        return sp.linalg.matrix_power(A, p)

    def test_matrix_power_zero_is_identity(self):
        A = sparse.csr_array(cupy.eye(3) * 2.)
        r = matrix_power(A, 0)
        assert isinstance(r, sparse.sparray)
        assert r.dtype == A.dtype
        cupy.testing.assert_allclose(r.toarray(), cupy.eye(3))

    def test_matrix_power_one_is_copy(self):
        A = sparse.csr_array(cupy.eye(2))
        r = matrix_power(A, 1)
        assert r is not A
        r.data[...] = 5.
        cupy.testing.assert_allclose(A.toarray(), cupy.eye(2))

    def test_matrix_power_matrix_input(self):
        M = sparse.csr_matrix(cupy.eye(3) * 2.)
        r = matrix_power(M, 2)
        assert isinstance(r, sparse.spmatrix)
        cupy.testing.assert_allclose(r.toarray(), cupy.eye(3) * 4.)

    def test_matrix_power_errors(self):
        A = sparse.csr_array(cupy.eye(3))
        with pytest.raises(ValueError, match='>= 0'):
            matrix_power(A, -1)
        with pytest.raises(ValueError, match='integer'):
            matrix_power(A, 2.0)
        with pytest.raises(TypeError, match='square'):
            matrix_power(sparse.csr_array(cupy.ones((2, 3))), 2)

    def test_matrix_pow_operator_keeps_matrix_kind(self):
        # ``matrix ** n`` (sharing matrix_power's recursion) must return a
        # matrix -- including ``** 0`` -- so ``*`` on the result stays
        # matmul rather than flipping to element-wise.
        M = sparse.csr_matrix(cupy.array([[0., 1.], [1., 0.]]))
        for p in (0, 1, 2, 3):
            r = M ** p
            assert isinstance(r, sparse.spmatrix)
        cupy.testing.assert_allclose((M ** 0).toarray(), cupy.eye(2))
        cupy.testing.assert_allclose((M ** 2).toarray(), cupy.eye(2))
        # ``*`` on the identity result is still matmul (2-D @ 2-D).
        assert ((M ** 0) * M).toarray().shape == (2, 2)

    def test_matrix_pow_errors_match_matrix_power(self):
        M = sparse.csr_matrix(cupy.eye(3))
        with pytest.raises(ValueError, match='>= 0'):
            M ** -1
        with pytest.raises(ValueError, match='integer'):
            M ** 2.5
        with pytest.raises(TypeError, match='square'):
            sparse.csr_matrix(cupy.ones((2, 3))) ** 2
