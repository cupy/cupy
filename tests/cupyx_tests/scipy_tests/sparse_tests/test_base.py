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

                def getnnz(self):
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
    """SciPy 1.14 removed these matrix-only APIs from sparse arrays.
    CuPy mirrors the deprecation: ``spmatrix`` mixin emits
    ``DeprecationWarning``; arrays don't have the methods at all.

    The warnings live on the ``spmatrix`` mixin and aren't overridden by
    subclasses, so a single CSR fixture covers all matrix formats.
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

    def test_asfptype_warns(self, m):
        with pytest.warns(DeprecationWarning, match="asfptype"):
            result = m.asfptype()
        assert result.dtype.kind == 'f'

    def test_get_shape_warns(self, m):
        with pytest.warns(DeprecationWarning, match="get_shape"):
            result = m.get_shape()
        assert result == m.shape

    def test_getformat_warns(self, m):
        with pytest.warns(DeprecationWarning, match="getformat"):
            result = m.getformat()
        assert result == m.format

    def test_getmaxprint_warns(self, m):
        with pytest.warns(DeprecationWarning, match="getmaxprint"):
            result = m.getmaxprint()
        assert result == m.maxprint

    def test_set_shape_warns(self, m):
        with pytest.warns(DeprecationWarning, match="set_shape"):
            m.set_shape(m.shape)

    def test_shape_setter_does_not_warn(self, m):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            m.shape = m.shape

    def test_warning_is_attributed_to_caller(self, m):
        # ``stacklevel=2`` should make the warning point at the user's
        # frame (this test file) rather than ``_base.py``.
        with pytest.warns(DeprecationWarning) as record:
            m.A
        assert record[0].filename.endswith('test_base.py')

    @pytest.mark.parametrize(
        'attr', ['A', 'H', 'asfptype', 'getformat', 'getmaxprint', 'set_shape']
    )
    def test_array_does_not_have_deprecated_attr(self, a, attr):
        # `get_shape` is omitted: it currently leaks onto arrays via
        # _csr_base/_csc_base/_coo_base/_dia_base inheritance.  Tracked in
        # spai/FUTURE.md ("Format-base method leaks").
        with pytest.raises(AttributeError):
            getattr(a, attr)
