import re
import unittest
from unittest import mock

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
from cupyx.scipy.sparse import construct


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'format': ['csr', 'csc', 'coo'],
    'm': [3],
    'n': [None, 3, 2],
    'k': [0, 1],
}))
@testing.with_requires('scipy')
class TestEye(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_eye(self, xp, sp):
        x = sp.eye(
            self.m, n=self.n, k=self.k, dtype=self.dtype, format=self.format)
        assert isinstance(x, sp.spmatrix)
        assert x.format == self.format
        return x


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'format': ['csr', 'csc', 'coo'],
}))
@testing.with_requires('scipy')
class TestIdentity(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_eye(self, xp, sp):
        x = sp.identity(3, dtype=self.dtype, format=self.format)
        assert isinstance(x, sp.spmatrix)
        assert x.format == self.format
        return x


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@testing.with_requires('scipy')
class TestSpdiags(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_spdiags(self, xp, sp):
        data = xp.arange(12, dtype=self.dtype).reshape(3, 4)
        diags = xp.array([0, -1, 2], dtype='i')
        x = sp.spdiags(data, diags, 3, 4)
        return x


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64]
}))
class TestVstack(unittest.TestCase):

    def data(self):

        A = sparse.coo_matrix((cupy.asarray([1.0, 2.0, 3.0, 4.0]),
                               (cupy.asarray([0, 0, 1, 1]),
                                cupy.asarray([0, 1, 0, 1]))))
        B = sparse.coo_matrix((cupy.asarray([5.0, 6.0]),
                               (cupy.asarray([0, 0]),
                                cupy.asarray([0, 1]))))

        return A, B

    def expected(self):

        return cupy.asarray([[1, 2],
                             [3, 4],
                             [5, 6]], self.dtype)

    def test_basic_vstack(self):

        A, B = self.data()

        actual = construct.vstack([A, B]).todense()
        testing.assert_array_equal(actual, self.expected())

    def test_dtype(self):

        A, B = self.data()

        actual = construct.vstack([A, B], dtype=self.dtype)
        assert actual.dtype == self.dtype

    def test_csr(self):

        A, B = self.data()

        actual = construct.vstack([A.tocsr(), B.tocsr()]).todense()
        testing.assert_array_equal(actual, self.expected())

    def test_csr_with_dtype(self):

        A, B = self.data()

        actual = construct.vstack([A.tocsr(), B.tocsr()],
                                  dtype=self.dtype)
        assert actual.dtype == self.dtype
        assert actual.indices.dtype == cupy.int32
        assert actual.indptr.dtype == cupy.int32


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64]
}))
class TestHstack(unittest.TestCase):

    def data(self):

        A = sparse.coo_matrix((cupy.asarray([1.0, 2.0, 3.0, 4.0]),
                               (cupy.asarray([0, 0, 1, 1]),
                                cupy.asarray([0, 1, 0, 1]))))
        B = sparse.coo_matrix((cupy.asarray([5.0, 6.0]),
                               (cupy.asarray([0, 1]),
                                cupy.asarray([0, 0]))))

        return A, B

    def expected(self):

        return cupy.asarray([[1, 2, 5],
                             [3, 4, 6]])

    def test_basic_hstack(self):

        A, B = self.data()
        actual = construct.hstack([A, B], dtype=self.dtype).todense()
        testing.assert_array_equal(actual, self.expected())
        assert actual.dtype == self.dtype

    def test_csc(self):
        A, B = self.data()
        actual = construct.hstack([A.tocsc(), B.tocsc()],
                                  dtype=self.dtype).todense()
        testing.assert_array_equal(actual, self.expected())
        assert actual.dtype == self.dtype

    def test_csc_with_dtype(self):

        A, B = self.data()

        actual = construct.hstack([A.tocsc(), B.tocsc()],
                                  dtype=self.dtype)
        assert actual.indices.dtype == cupy.int32
        assert actual.indptr.dtype == cupy.int32


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64]
}))
class TestBmat(unittest.TestCase):

    def data(self):
        A = sparse.csr_matrix(cupy.asarray([[1, 2], [3, 4]],
                                           self.dtype)).tocoo()
        B = sparse.csr_matrix(cupy.asarray([[5], [6]],
                                           self.dtype)).tocoo()
        C = sparse.csr_matrix(cupy.asarray([[7]],
                                           self.dtype)).tocoo()
        D = sparse.coo_matrix((0, 0), dtype=self.dtype)

        return A, B, C, D

    def test_basic_inputs(self):

        A, B, C, D = self.data()

        expected = cupy.asarray([[1, 2, 5],
                                 [3, 4, 6],
                                 [0, 0, 7]], dtype=self.dtype)

        testing.assert_array_equal(
            construct.bmat([[A, B], [None, C]]).todense(), expected
        )

        expected = cupy.asarray([[1, 2, 0],
                                 [3, 4, 0],
                                 [0, 0, 7]])
        testing.assert_array_equal(
            construct.bmat([[A, None], [None, C]]).todense(), expected
        )

        expected = cupy.asarray([[0, 5],
                                 [0, 6],
                                 [7, 0]])

        testing.assert_array_equal(
            construct.bmat([[None, B], [C, None]]).todense(), expected
        )

    def test_empty(self):

        A, B, C, D = self.data()

        expected = cupy.empty((0, 0), dtype=self.dtype)
        testing.assert_array_equal(construct.bmat([[None, None]]).todense(),
                                   expected)
        testing.assert_array_equal(construct.bmat([[None, D], [D, None]])
                                   .todense(), expected)

    def test_edge_cases(self):
        """Catch-all for small edge cases"""

        A, B, C, D = self.data()

        expected = cupy.asarray([[7]], dtype=self.dtype)
        testing.assert_array_equal(construct.bmat([[None, D], [C, None]])
                                   .todense(), expected)

    def test_failure_cases(self):

        A, B, C, D = self.data()

        match = r'.*Got blocks\[{}\]\.shape\[{}\] == 1, expected 2'

        # test failure cases
        message1 = re.compile(match.format('1,0', '1'))
        with pytest.raises(ValueError, match=message1):
            construct.bmat([[A], [B]], dtype=self.dtype)

        message2 = re.compile(match.format('0,1', '0'))
        with pytest.raises(ValueError, match=message2):
            construct.bmat([[A, C]], dtype=self.dtype)


@testing.parameterize(*testing.product({
    'random_method': ['random', 'rand'],
    'dtype': [numpy.float32, numpy.float64],
    'format': ['csr', 'csc', 'coo'],
}))
class TestRandom(unittest.TestCase):

    def test_random(self):
        x = getattr(sparse, self.random_method)(
            3, 4, density=0.1,
            format=self.format, dtype=self.dtype)
        assert x.shape == (3, 4)
        assert x.dtype == self.dtype
        assert x.format == self.format

    def test_random_with_seed(self):
        x = getattr(sparse, self.random_method)(
            3, 4, density=0.1,
            format=self.format, dtype=self.dtype,
            random_state=1)
        assert x.shape == (3, 4)
        assert x.dtype == self.dtype
        assert x.format == self.format

        y = getattr(sparse, self.random_method)(
            3, 4, density=0.1,
            format=self.format, dtype=self.dtype,
            random_state=1)

        testing.assert_array_equal(x.toarray(), y.toarray())

    def test_random_with_state(self):
        state1 = cupy.random.RandomState(1)
        x = getattr(sparse, self.random_method)(
            3, 4, density=0.1,
            format=self.format, dtype=self.dtype,
            random_state=state1)
        assert x.shape == (3, 4)
        assert x.dtype == self.dtype
        assert x.format == self.format

        state2 = cupy.random.RandomState(1)
        y = getattr(sparse, self.random_method)(
            3, 4, density=0.1,
            format=self.format, dtype=self.dtype,
            random_state=state2)

        testing.assert_array_equal(x.toarray(), y.toarray())

    def test_random_with_data_rvs(self):
        if self.random_method == 'rand':
            pytest.skip('cupyx.scipy.sparse.rand does not support data_rvs')
        data_rvs = mock.MagicMock(side_effect=cupy.zeros)
        x = getattr(sparse, self.random_method)(
            3, 4, density=0.1, data_rvs=data_rvs,
            format=self.format, dtype=self.dtype)
        assert x.shape == (3, 4)
        assert x.dtype == self.dtype
        assert x.format == self.format

        assert data_rvs.call_count == 1
        # Note that its value is generated randomly
        assert isinstance(data_rvs.call_args[0][0], int)


@testing.with_requires('scipy')
class TestRandomInvalidArgument(unittest.TestCase):

    def test_too_small_density(self):
        for sp in (scipy.sparse, sparse):
            with pytest.raises(ValueError):
                sp.random(3, 4, density=-0.1)

    def test_too_large_density(self):
        for sp in (scipy.sparse, sparse):
            with pytest.raises(ValueError):
                sp.random(3, 4, density=1.1)

    def test_invalid_dtype(self):
        # Note: SciPy 1.12+ accepts integer.
        with self.assertRaises(NotImplementedError):
            sparse.random(3, 4, dtype='i')


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'format': ['dia', 'csr', 'csc', 'coo'],
}))
@testing.with_requires('scipy')
class TestDiags(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_diags_scalar_offset(self, xp, sp):
        x = sp.diags(
            xp.arange(16), offsets=0, dtype=self.dtype, format=self.format)
        assert isinstance(x, sp.spmatrix)
        assert x.format == self.format
        return x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_diags_single_element_lists(self, xp, sp):
        x = sp.diags(
            [xp.arange(16)], offsets=[0], dtype=self.dtype, format=self.format)
        assert isinstance(x, sp.spmatrix)
        assert x.format == self.format
        return x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_diags_multiple(self, xp, sp):
        x = sp.diags(
            [xp.arange(15), xp.arange(16), xp.arange(15), xp.arange(13)],
            offsets=[-1, 0, 1, 3],
            dtype=self.dtype, format=self.format)
        assert isinstance(x, sp.spmatrix)
        assert x.format == self.format
        return x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_diags_offsets_as_array(self, xp, sp):
        x = sp.diags(
            [xp.arange(15), xp.arange(16), xp.arange(15), xp.arange(13)],
            offsets=xp.array([-1, 0, 1, 3]),
            dtype=self.dtype, format=self.format)
        assert isinstance(x, sp.spmatrix)
        assert x.format == self.format
        return x

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_diags_non_square(self, xp, sp):
        x = sp.diags(
            [xp.arange(5), xp.arange(3)],
            offsets=[0, -2], shape=(5, 10),
            dtype=self.dtype, format=self.format)
        assert isinstance(x, sp.spmatrix)
        assert x.format == self.format
        return x


# borrowed from scipy:
arrs = [[[0]],
        [[-1]],
        [[4]],
        [[10]],
        [[0], [0]],
        [[0, 0]],
        [[1, 2], [3, 4]],
        [[0, 2], [5, 0]],
        [[0, 2, -6], [8, 0, 14]],
        [[5, 4], [0, 0], [6, 0]],
        [[5, 4, 4], [1, 0, 0], [6, 0, 8]],
        [[0, 1, 0, 2, 0, 5, 8]],
        [[0.5, 0.125, 0, 3.25], [0, 2.5, 0, 0]], ]


@testing.parameterize(*testing.product({
    'dtype': (numpy.float32, numpy.float64, numpy.complex64, numpy.complex128),
    'format': ('csr', 'csc', 'coo'),
    'arrA': arrs,
    'arrB': arrs,
}))
@testing.with_requires('scipy>=1.6')
class TestKron(unittest.TestCase):

    def _make_sp_mat(self, xp, sp, arr, dtype):
        a = xp.array(arr, dtype=dtype)
        a = sp.csr_matrix(a)
        return a

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_kron(self, xp, sp):
        a = self._make_sp_mat(xp, sp, self.arrA, self.dtype)
        b = self._make_sp_mat(xp, sp, self.arrB, self.dtype)
        kron = sp.kron(a, b, format=self.format)
        assert kron.shape == (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])
        assert kron.nnz == a.nnz * b.nnz
        return kron

    # TODO(leofang): check oversize inputs as in scipy/scipy#11879 after
    # #3513 is fixed
