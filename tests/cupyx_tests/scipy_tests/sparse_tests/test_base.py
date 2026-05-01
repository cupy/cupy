from __future__ import annotations

import unittest
import warnings

import pytest
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import testing
from cupyx.scipy import sparse


@testing.with_requires('scipy>=1.14')
class TestSpmatrix(unittest.TestCase):

    def dummy_class(self, sp):
        if sp is sparse:
            class DummySparseGPU(sparse.spmatrix, sparse._spbase):

                def __init__(self, maxprint=50, shape=None, nnz=0):
                    super().__init__(maxprint=maxprint)
                    self._shape = shape
                    self._nnz = nnz

                def get_shape(self):
                    return self._shape

                def _getnnz(self, axis=None):
                    return self._nnz

            return DummySparseGPU
        else:
            class DummySparseCPU(scipy.sparse._base._spbase):

                def __init__(self, maxprint=50, shape=None, nnz=0):
                    super().__init__(
                        None, maxprint=maxprint)
                    self._shape = shape
                    self._nnz = nnz

                def _getnnz(self):
                    return self._nnz

            return DummySparseCPU

    def test_instantiation(self):
        # SciPy's _spbase rejects direct instantiation
        with pytest.raises((ValueError, TypeError)):
            scipy.sparse._base._spbase(None)
        # CuPy's spmatrix is a mixin — instantiation is technically
        # possible but useless.  We just check it doesn't crash.
        m = sparse.spmatrix()
        assert isinstance(m, sparse.spmatrix)

    def test_len(self):
        for sp in (scipy.sparse, sparse):
            s = self.dummy_class(sp)()
            with pytest.raises(TypeError):
                len(s)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_bool_true(self, xp, sp):
        s = self.dummy_class(sp)(shape=(1, 1), nnz=1)
        return bool(s)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_bool_false(self, xp, sp):
        s = self.dummy_class(sp)(shape=(1, 1), nnz=0)
        return bool(s)

    def test_bool_invalid(self):
        for sp in (scipy.sparse, sparse):
            s = self.dummy_class(sp)(shape=(2, 1))
            with pytest.raises(ValueError):
                bool(s)

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_asformat_none(self, xp, sp):
        s = self.dummy_class(sp)()
        assert s.asformat(None) is s

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_maxprint(self, xp, sp):
        s = self.dummy_class(sp)(maxprint=30)
        return s.maxprint


class TestIssparse:
    """Test issparse and isspmatrix with both arrays and matrices."""

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_issparse_matrix(self, fmt):
        cls = getattr(sparse, f'{fmt}_matrix')
        m = cls((2, 3))
        assert sparse.issparse(m)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_issparse_array(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        a = cls((2, 3))
        assert sparse.issparse(a)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_isspmatrix_matrix(self, fmt):
        cls = getattr(sparse, f'{fmt}_matrix')
        m = cls((2, 3))
        assert sparse.isspmatrix(m)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_isspmatrix_not_array(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        a = cls((2, 3))
        assert not sparse.isspmatrix(a)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_sparray_not_spmatrix(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        a = cls((2, 3))
        assert isinstance(a, sparse.sparray)
        assert not isinstance(a, sparse.spmatrix)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_spmatrix_not_sparray(self, fmt):
        cls = getattr(sparse, f'{fmt}_matrix')
        m = cls((2, 3))
        assert isinstance(m, sparse.spmatrix)
        assert not isinstance(m, sparse.sparray)


class TestDeprecatedSpmatrixApi:
    """Matrix-only APIs that don't exist on sparse arrays.

    SciPy 1.14 deprecated several matrix-only APIs that SciPy 1.17
    subsequently un-deprecated (kept on ``spmatrix`` only, no warning):
    ``asfptype``, ``get_shape``, ``getformat``, ``getmaxprint``,
    ``set_shape``, ``getrow``, ``getcol``.  CuPy follows the same
    policy.

    Two attributes are still removed on arrays *and* still warn on
    matrices: ``A`` (use ``.toarray()``) and ``H`` (use ``.T.conj()``).
    """

    @pytest.fixture
    def m(self):
        return sparse.csr_matrix(cupy.array([[1.0, 0.0, 0.0],
                                             [0.0, 2.0, 0.0],
                                             [0.0, 0.0, 3.0]]))

    @pytest.fixture
    def a(self):
        return sparse.csr_array(cupy.array([[1.0, 0.0, 0.0],
                                            [0.0, 2.0, 0.0],
                                            [0.0, 0.0, 3.0]]))

    def test_A_warns(self, m):
        with pytest.warns(DeprecationWarning, match=r"`spmatrix\.A`"):
            result = m.A
        testing.assert_array_equal(result, m.toarray())

    def test_H_warns(self, m):
        with pytest.warns(DeprecationWarning, match=r"`spmatrix\.H`"):
            result = m.H
        testing.assert_array_equal(
            result.toarray(), m.transpose().conj().toarray())

    @pytest.mark.parametrize('name,check', [
        ('asfptype', lambda r, m: r.dtype.kind == 'f'),
        ('get_shape', lambda r, m: r == m.shape),
        ('getformat', lambda r, m: r == m.format),
        ('getmaxprint', lambda r, m: r == m.maxprint),
        ('getnnz', lambda r, m: r == m.nnz),
    ])
    def test_plain_method_no_warn(self, m, name, check):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = getattr(m, name)()
        assert check(result, m)

    def test_set_shape_no_warn_and_in_place(self, m):
        old_shape = m.shape
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            m.set_shape((9, 1))
        assert m.shape == (9, 1)
        # Reset for cleanliness in case the fixture is reused.
        m.set_shape(old_shape)

    def test_shape_setter_does_not_warn(self, m):
        old_shape = m.shape
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            m.shape = (9, 1)
        assert m.shape == (9, 1)
        m.shape = old_shape

    def test_warning_is_attributed_to_caller(self, m):
        # ``stacklevel=2`` should make the warning point at the user's
        # frame (this test file) rather than ``_base.py``.
        with pytest.warns(DeprecationWarning) as record:
            m.A
        assert record[0].filename.endswith('test_base.py')

    @pytest.mark.parametrize(
        'attr', ['A', 'H', 'asfptype', 'get_shape', 'getformat',
                 'getmaxprint', 'getnnz', 'getrow', 'getcol', 'set_shape']
    )
    def test_array_does_not_have_deprecated_attr(self, a, attr):
        with pytest.raises(AttributeError):
            getattr(a, attr)


class TestSetShape:
    """``set_shape`` must mutate ``self`` in place.  Plain
    ``self.reshape(shape)`` returns a new matrix and discards the
    reshaped result; the in-place wrapper has to swap ``__dict__``.
    """

    def test_csr_set_shape_mutates(self):
        m = sparse.csr_matrix(cupy.array(
            [[1, 2, 0], [0, 3, 0], [0, 0, 4]], dtype=cupy.float64))
        m.set_shape((1, 9))
        assert m.shape == (1, 9)
        cupy.testing.assert_array_equal(
            m.toarray(),
            cupy.array([[1, 2, 0, 0, 3, 0, 0, 0, 4]], dtype=cupy.float64))

    def test_csc_set_shape_mutates(self):
        m = sparse.csc_matrix(cupy.eye(3, dtype=cupy.float64))
        m.set_shape((9, 1))
        assert m.shape == (9, 1)

    def test_set_shape_invalid_total(self):
        m = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        with pytest.raises(ValueError):
            m.set_shape((4, 4))
