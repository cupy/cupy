from cupyx.scipy.sparse.linalg import interface
from functools import partial
import cupy as cp
import unittest

import numpy
import pytest
try:
    import scipy.sparse
    import scipy.sparse.linalg
    import scipy.stats
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import testing
from cupy.testing import condition
from cupyx.scipy import sparse
import cupyx.scipy.sparse.linalg  # NOQA


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestLsqr(unittest.TestCase):

    def setUp(self):
        rvs = scipy.stats.randint(0, 15).rvs
        self.A = scipy.sparse.random(50, 50, density=0.2, data_rvs=rvs)
        self.b = numpy.random.randint(15, size=50)

    def test_size(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            A = sp.csr_matrix(self.A, dtype=self.dtype)
            b = xp.array(numpy.append(self.b, [1]), dtype=self.dtype)
            with pytest.raises(ValueError):
                sp.linalg.lsqr(A, b)

    def test_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            A = sp.csr_matrix(self.A, dtype=self.dtype)
            b = xp.array(numpy.tile(self.b, (2, 1)), dtype=self.dtype)
            with pytest.raises(ValueError):
                sp.linalg.lsqr(A, b)

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_csrmatrix(self, xp, sp):
        A = sp.csr_matrix(self.A, dtype=self.dtype)
        b = xp.array(self.b, dtype=self.dtype)
        x = sp.linalg.lsqr(A, b)
        return x[0]

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_ndarray(self, xp, sp):
        A = xp.array(self.A.A, dtype=self.dtype)
        b = xp.array(self.b, dtype=self.dtype)
        x = sp.linalg.lsqr(A, b)
        return x[0]


@testing.parameterize(*testing.product({
    'ord': [None, -numpy.Inf, -2, -1, 0, 1, 2, 3, numpy.Inf, 'fro'],
    'dtype': [
        numpy.float32,
        numpy.float64,
        numpy.complex64,
        numpy.complex128
    ],
    'axis': [None, (0, 1), (1, -2)],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
@testing.gpu
class TestMatrixNorm(unittest.TestCase):

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, sp_name='sp',
                                 accept_error=(ValueError,
                                               NotImplementedError))
    def test_matrix_norm(self, xp, sp):
        a = xp.arange(9, dtype=self.dtype) - 4
        b = a.reshape((3, 3))
        b = sp.csr_matrix(b, dtype=self.dtype)
        return sp.linalg.norm(b, ord=self.ord, axis=self.axis)


@testing.parameterize(*testing.product({
    'ord': [None, -numpy.Inf, -2, -1, 0, 1, 2, numpy.Inf, 'fro'],
    'dtype': [
        numpy.float32,
        numpy.float64,
        numpy.complex64,
        numpy.complex128
    ],
    'transpose': [True, False],
    'axis': [0, (1,), (-2,), -1],
})
)
@unittest.skipUnless(scipy_available, 'requires scipy')
@testing.gpu
class TestVectorNorm(unittest.TestCase):
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, sp_name='sp',
                                 accept_error=(ValueError,))
    def test_vector_norm(self, xp, sp):
        a = xp.arange(9, dtype=self.dtype) - 4
        b = a.reshape((3, 3))
        b = sp.csr_matrix(b, dtype=self.dtype)
        if self.transpose:
            b = b.T
        return sp.linalg.norm(b, ord=self.ord, axis=self.axis)

# TODO : TestVsNumpyNorm


@testing.parameterize(*testing.product({
    'which': ['LM', 'LA'],
    'k': [3, 6, 12],
    'return_eigenvectors': [True, False],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestEigsh(unittest.TestCase):
    n = 30
    density = 0.33
    _tol = {'f': 1e-5, 'd': 1e-12}

    def _make_matrix(self, dtype, xp):
        shape = (self.n, self.n)
        a = testing.shaped_random(shape, xp, dtype=dtype)
        mask = testing.shaped_random(shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        a = a * a.conj().T
        return a

    def _test_eigsh(self, a, xp, sp):
        ret = sp.linalg.eigsh(a, k=self.k, which=self.which,
                              return_eigenvectors=self.return_eigenvectors)
        if self.return_eigenvectors:
            w, x = ret
            # Check the residuals to see if eigenvectors are correct.
            ax_xw = a @ x - xp.multiply(x, w.reshape(1, self.k))
            res = xp.linalg.norm(ax_xw) / xp.linalg.norm(w)
            tol = self._tol[numpy.dtype(a.dtype).char.lower()]
            assert(res < tol)
        else:
            w = ret
        return xp.sort(w)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_sparse(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        a = sp.csr_matrix(a)
        return self._test_eigsh(a, xp, sp)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        return self._test_eigsh(a, xp, sp)

    def test_invalid(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a = xp.diag(xp.ones((self.n, ), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.eigsh(xp.ones((2, 1), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.eigsh(a, k=0)
        xp, sp = cupy, sparse
        a = xp.diag(xp.ones((self.n, ), dtype='f'))
        with pytest.raises(ValueError):
            sp.linalg.eigsh(xp.ones((1,), dtype='f'))
        with pytest.raises(TypeError):
            sp.linalg.eigsh(xp.ones((2, 2), dtype='i'))
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=self.n)
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=self.k, which='SM')
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=self.k, which='SA')


@testing.parameterize(*testing.product({
    'shape': [(30, 29), (29, 29), (29, 30)],
    'k': [3, 6, 12],
    'return_vectors': [True, False],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestSvds(unittest.TestCase):
    density = 0.33

    def _make_matrix(self, dtype, xp):
        a = testing.shaped_random(self.shape, xp, dtype=dtype)
        mask = testing.shaped_random(self.shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        return a

    def _test_svds(self, a, xp, sp):
        ret = sp.linalg.svds(a, k=self.k,
                             return_singular_vectors=self.return_vectors)
        if self.return_vectors:
            u, s, vt = ret
            # Check the results with u @ s @ vt, as singular vectors don't
            # necessarily match.
            return u @ xp.diag(s) @ vt
        else:
            return xp.sort(ret)

    @testing.for_dtypes('fF')
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-4, sp_name='sp')
    def test_sparse_f(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        a = sp.csr_matrix(a)
        return self._test_svds(a, xp, sp)

    @testing.for_dtypes('fF')
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-4, sp_name='sp')
    def test_dense_f(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        return self._test_svds(a, xp, sp)

    @testing.for_dtypes('dD')
    @testing.numpy_cupy_allclose(rtol=1e-12, atol=1e-12, sp_name='sp')
    def test_sparse_d(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        a = sp.csr_matrix(a)
        return self._test_svds(a, xp, sp)

    @testing.for_dtypes('dD')
    @testing.numpy_cupy_allclose(rtol=1e-12, atol=1e-12, sp_name='sp')
    def test_dense_d(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        return self._test_svds(a, xp, sp)

    def test_invalid(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a = xp.diag(xp.ones(self.shape, dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.svds(a, k=0)
        xp, sp = cupy, sparse
        a = xp.diag(xp.ones(self.shape, dtype='f'))
        with pytest.raises(ValueError):
            sp.linalg.svds(xp.ones((1,), dtype='f'))
        with pytest.raises(TypeError):
            sp.linalg.svds(xp.ones((2, 2), dtype='i'))
        with pytest.raises(ValueError):
            sp.linalg.svds(a, k=min(self.shape))
        with pytest.raises(ValueError):
            sp.linalg.svds(a, k=self.k, which='SM')


@testing.parameterize(*testing.product({
    'x0': [None, 'ones'],
    'M': [None, 'jacobi'],
    'atol': [None, 'select-by-dtype'],
    'b_ndim': [1, 2],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
@testing.gpu
class TestCg(unittest.TestCase):
    n = 30
    density = 0.33
    _atol = {'f': 1e-5, 'd': 1e-12}

    def _make_matrix(self, dtype, xp):
        dtype = numpy.dtype(dtype)
        shape = (self.n, 10)
        a = testing.shaped_random(shape, xp, dtype=dtype.char.lower(), scale=1)
        if dtype.char in 'FD':
            a = a + 1j * testing.shaped_random(
                shape, xp, dtype=dtype.char.lower(), scale=1)
        mask = testing.shaped_random(shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        a = a @ a.conj().T
        a = a + xp.diag(xp.ones((self.n,), dtype=dtype.char.lower()))
        M = None
        if self.M == 'jacobi':
            M = xp.diag(1.0 / xp.diag(a))
        return a, M

    def _make_normalized_vector(self, dtype, xp):
        b = testing.shaped_random((self.n,), xp, dtype=dtype)
        return b / xp.linalg.norm(b)

    def _test_cg(self, dtype, xp, sp, a, M):
        dtype = numpy.dtype(dtype)
        b = self._make_normalized_vector(dtype, xp)
        if self.b_ndim == 2:
            b = b.reshape(self.n, 1)
        x0 = None
        if self.x0 == 'ones':
            x0 = xp.ones((self.n,), dtype=dtype)
        atol = None
        if self.atol == 'select-by-dtype':
            atol = self._atol[dtype.char.lower()]
        if atol is None and xp == numpy:
            # Note: If atol is None or not specified, Scipy (at least 1.5.3)
            # raises DeprecationWarning
            with pytest.deprecated_call():
                return sp.linalg.cg(a, b, x0=x0, M=M, atol=atol)
        else:
            return sp.linalg.cg(a, b, x0=x0, M=M, atol=atol)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        return self._test_cg(dtype, xp, sp, a, M)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_csr(self, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        a = sp.csr_matrix(a)
        if M is not None:
            M = sp.csr_matrix(M)
        return self._test_cg(dtype, xp, sp, a, M)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_csc(self, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        a = sp.csc_matrix(a)
        if M is not None:
            M = sp.csc_matrix(M)
        return self._test_cg(dtype, xp, sp, a, M)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_coo(self, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a)
        if M is not None:
            M = sp.coo_matrix(M)
        return self._test_cg(dtype, xp, sp, a, M)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_empty(self, dtype, xp, sp):
        if self.x0 is not None or self.M is not None or self.atol is not None:
            raise unittest.SkipTest
        a = xp.empty((0, 0), dtype=dtype)
        b = xp.empty((0,), dtype=dtype)
        if self.atol is None and xp == numpy:
            # Note: If atol is None or not specified, Scipy (at least 1.5.3)
            # raises DeprecationWarning
            with pytest.deprecated_call():
                return sp.linalg.cg(a, b)
        else:
            return sp.linalg.cg(a, b)

    @testing.for_dtypes('fdFD')
    def test_callback(self, dtype):
        if self.x0 is not None or self.M is not None or self.atol is not None:
            raise unittest.SkipTest
        xp, sp = cupy, sparse
        a, M = self._make_matrix(dtype, xp)
        b = self._make_normalized_vector(dtype, xp)
        is_called = False

        def callback(x):
            print(xp.linalg.norm(b - a @ x))
            nonlocal is_called
            is_called = True
        sp.linalg.cg(a, b, callback=callback)
        assert is_called

    def test_invalid(self):
        if self.x0 is not None or self.M is not None or self.atol is not None:
            raise unittest.SkipTest
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a, M = self._make_matrix('f', xp)
            b = self._make_normalized_vector('f', xp)
            ng_a = xp.ones((self.n, ), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(ng_a, b, atol=self.atol)
            ng_a = xp.ones((self.n, self.n + 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(ng_a, b, atol=self.atol)
            ng_a = xp.ones((self.n, self.n, 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(ng_a, b, atol=self.atol)
            ng_b = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(a, ng_b, atol=self.atol)
            ng_b = xp.ones((self.n, 2), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(a, ng_b, atol=self.atol)
            ng_x0 = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(a, b, x0=ng_x0, atol=self.atol)
            ng_M = xp.diag(xp.ones((self.n + 1,), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.cg(a, b, M=ng_M, atol=self.atol)
        xp, sp = cupy, sparse
        b = self._make_normalized_vector('f', xp)
        ng_a = xp.ones((self.n, self.n), dtype='i')
        with pytest.raises(TypeError):
            sp.linalg.cg(ng_a, b, atol=self.atol)


@testing.parameterize(*testing.product({
    'n': [10, 30, 100, 300, 1000],
    'dtype': [cp.float32, cp.float64, cp.complex_]
}))
@testing.gpu
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestLinearOperator(unittest.TestCase):
    density = 0.33
    _tol = {'f': 1e-5, 'd': 1e-12}

    # class that defines parametrized custom cases
    # adapted from scipy's analogous tests
    def _define_cases(self, original, dtype):
        cases = []
        cases.append((cp.array(original, dtype=dtype), original))
        cases.append((cupyx.scipy.sparse.csr_matrix(
            original, dtype=dtype), original))

        # Test default implementations of _adjoint and _rmatvec, which
        # refer to each other.
        def mv(x, dtype):
            y = original.dot(x)
            if len(x.shape) == 2:
                y = y.reshape(-1, 1)
            return y

        def rmv(x, dtype):
            return original.T.conj().dot(x)

        class BaseMatlike(interface.LinearOperator):
            args = ()

            def __init__(self, dtype):
                self.dtype = cp.dtype(dtype)
                self.shape = original.shape

            def _matvec(self, x):
                return mv(x, self.dtype)

        class HasRmatvec(BaseMatlike):
            args = ()

            def _rmatvec(self, x):
                return rmv(x, self.dtype)

        class HasAdjoint(BaseMatlike):
            args = ()

            def _adjoint(self):
                shape = self.shape[1], self.shape[0]
                matvec = partial(rmv, dtype=self.dtype)
                rmatvec = partial(mv, dtype=self.dtype)
                return interface.LinearOperator(matvec=matvec,
                                                rmatvec=rmatvec,
                                                dtype=self.dtype,
                                                shape=shape)

        class HasRmatmat(HasRmatvec):
            def _matmat(self, x):
                return original.dot(x)

            def _rmatmat(self, x):
                return original.T.conj().dot(x)

        cases.append((HasRmatvec(dtype), original))
        cases.append((HasAdjoint(dtype), original))
        cases.append((HasRmatmat(dtype), original))
        return cases

    def _make_cases(self):
        self.cases = []
        if(self.dtype != cp.complex_):
            original = cp.array([[1., 2., 3.], [4., 5., 6.]])
            self.cases += self._define_cases(original, self.dtype)
            self.cases += [(interface.aslinearoperator(M).T, A.T)
                        for M, A in self._define_cases(original.T, self.dtype)]
            self.cases += [(interface.aslinearoperator(M).H, A.T.conj())
                        for M, A in self._define_cases(original.T, self.dtype)]

        else:
            original = cp.array([[1, 2j, 3j], [4j, 5j, 6]])
            self.cases += self._define_cases(original, self.dtype)
            self.cases += [(interface.aslinearoperator(M).T, A.T)
                        for M, A in self._define_cases(original.T, self.dtype)]
            self.cases += [(interface.aslinearoperator(M).H, A.T.conj())
                        for M, A in self._define_cases(original.T, self.dtype)]

    def test_basic(self):
        self._make_cases()
        for M, A_array in self.cases:
            A = interface.aslinearoperator(M)
            M, N = A.shape

            xs = [cp.array([1, 2, 3]),
                  cp.array([[1], [2], [3]])]
            ys = [cp.array([1, 2]), cp.array([[1], [2]])]

            if A.dtype == cp.complex_:
                xs += [cp.array([1, 2j, 3j]),
                       cp.array([[1], [2j], [3j]])]
                ys += [cp.array([1, 2j]), cp.array([[1], [2j]])]

            x2 = cp.array([[1, 4], [2, 5], [3, 6]])

            for x in xs:
                cp.testing.assert_array_equal(A.matvec(x), A_array.dot(x))
                cp.testing.assert_array_equal(A * x, A_array.dot(x))

            cp.testing.assert_array_equal(A.matmat(x2), A_array.dot(x2))
            cp.testing.assert_array_equal(A * x2, A_array.dot(x2))

            for y in ys:
                cp.testing.assert_array_equal(
                    A.rmatvec(y), A_array.T.conj().dot(y))
                cp.testing.assert_array_equal(A.T.matvec(y), A_array.T.dot(y))
                cp.testing.assert_array_equal(
                    A.H.matvec(y), A_array.T.conj().dot(y))

            for y in ys:
                if y.ndim < 2:
                    continue
                cp.testing.assert_array_equal(
                    A.rmatmat(y), A_array.T.conj().dot(y))
                cp.testing.assert_array_equal(A.T.matmat(y), A_array.T.dot(y))
                cp.testing.assert_array_equal(
                    A.H.matmat(y), A_array.T.conj().dot(y))

            if hasattr(M, 'dtype'):
                cp.testing.assert_array_equal(A.dtype, M.dtype)

            assert(hasattr(A, 'args'))

    def test_dot(self):
        self._make_cases()
        for M, A_array in self.cases:
            A = interface.aslinearoperator(M)
            M, N = A.shape

            x0 = cp.array([1, 2, 3])
            x1 = cp.array([[1], [2], [3]])
            x2 = cp.array([[1, 4], [2, 5], [3, 6]])

            cp.testing.assert_array_equal(A.dot(x0), A_array.dot(x0))
            cp.testing.assert_array_equal(A.dot(x1), A_array.dot(x1))
            cp.testing.assert_array_equal(A.dot(x2), A_array.dot(x2))

    # generate random matrix
    def _make_matrix(self, dtype, xp):
        shape = (self.n, self.n)
        a = testing.shaped_random(shape, xp, dtype=dtype)
        mask = testing.shaped_random(shape, xp, dtype=dtype, scale=1)
        a[mask > self.density] = 0
        return a

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_sparse(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        a = sp.csr_matrix(a)
        A = sp.linalg.aslinearoperator(a)
        return A(xp.eye(self.n, dtype=dtype))

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        A = sp.linalg.aslinearoperator(a)
        return A(xp.eye(self.n, dtype=dtype))
