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
            class DummySparseGPU(sparse.spmatrix):

                def __init__(self, maxprint=50, shape=None, nnz=0):
                    super().__init__(maxprint)
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
        for sp in (scipy.sparse, sparse):
            with pytest.raises(ValueError):
                if sp is scipy.sparse:
                    sp._base._spbase(None)
                else:
                    # TODO(asi1024): Replace with sp._base._spbase
                    sp.spmatrix()

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


class TestSpmatrixOnlyApi:
    """Matrix-only APIs whose status mirrors scipy 1.17.

    SciPy 1.14 *removed* ``.A``/``.H`` from the array side.  CuPy keeps
    them on ``spmatrix`` for now but emits ``DeprecationWarning`` so user
    code has a migration breadcrumb.

    SciPy 1.14 deprecated and 1.17 *un-deprecated* ``asfptype``,
    ``getformat``, ``getmaxprint``, ``set_shape`` (et al.) — they remain
    plain matrix-side conveniences.  CuPy follows: no warning.
    """

    @pytest.fixture
    def m(self):
        return sparse.csr_matrix(cupy.array([[1.0, 0.0, 0.0],
                                             [0.0, 2.0, 0.0],
                                             [0.0, 0.0, 3.0]]))

    @pytest.mark.parametrize("attr", ["A", "H"])
    def test_attr_warns(self, m, attr):
        with pytest.warns(DeprecationWarning,
                          match=fr"`spmatrix\.{attr}`"):
            getattr(m, attr)

    @pytest.mark.parametrize("name", [
        "asfptype", "getformat", "getmaxprint",
        "getnnz", "get_shape", "set_shape",
    ])
    def test_plain_method_no_warn(self, m, name):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            if name == "set_shape":
                m.set_shape(m.shape)
            else:
                getattr(m, name)()

    def test_shape_setter_no_warn(self, m):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            m.shape = m.shape

    @pytest.mark.parametrize("via", ["set_shape", "shape="])
    def test_shape_mutates_in_place(self, via):
        # Regression for the silent no-op: ``reshape`` returned a new
        # matrix and the result was discarded.  Now the change must be
        # visible on ``self``.
        m = sparse.csr_matrix(cupy.array([[1.0, 0.0, 2.0],
                                          [0.0, 3.0, 0.0]]))
        if via == "set_shape":
            m.set_shape((3, 2))
        else:
            m.shape = (3, 2)
        assert m.shape == (3, 2)

    def test_warning_attributed_to_caller(self, m):
        # ``stacklevel=2`` should make the warning point at the user's
        # frame (this test file) rather than ``_base.py``.
        with pytest.warns(DeprecationWarning) as record:
            m.A
        assert record[0].filename.endswith("test_base.py")


class TestSetShape(unittest.TestCase):
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
