import contextlib
import re
import cupy
import io
import unittest
import warnings


import numpy
import pytest
try:
    import scipy.sparse
    import scipy.sparse.linalg
    import scipy.stats
    scipy_available = True
except ImportError:
    scipy_available = False

from cupy import testing
from cupy.testing import _condition
from cupyx.scipy import sparse
import cupyx.scipy.sparse.linalg  # NOQA


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
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

    @_condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_csrmatrix(self, xp, sp):
        A = sp.csr_matrix(self.A, dtype=self.dtype)
        b = xp.array(self.b, dtype=self.dtype)
        x = sp.linalg.lsqr(A, b)
        return x[0]

    @_condition.retry(10)
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
@testing.with_requires('scipy')
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
@testing.with_requires('scipy')
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
    'use_linear_operator': [True, False],
}))
@testing.with_requires('scipy')
class TestEigsh:
    n = 30
    density = 0.33
    tol = {numpy.float32: 1e-5, numpy.complex64: 1e-5, 'default': 1e-12}
    res_tol = {'f': 1e-5, 'd': 1e-12}

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
            tol = self.res_tol[numpy.dtype(a.dtype).char.lower()]
            assert(res < tol)
        else:
            w = ret
        return xp.sort(w)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=tol, atol=tol, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_eigsh(a, xp, sp)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=tol, atol=tol, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_eigsh(a, xp, sp)

    def test_invalid(self):
        if self.use_linear_operator is True:
            raise unittest.SkipTest
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
    'use_linear_operator': [True, False],
}))
@testing.with_requires('scipy')
class TestSvds:
    density = 0.33
    tol = {numpy.float32: 1e-4, numpy.complex64: 1e-4, 'default': 1e-12}

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

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=tol, atol=tol, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_svds(a, xp, sp)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=tol, atol=tol, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_svds(a, xp, sp)

    def test_invalid(self):
        if self.use_linear_operator is True:
            raise unittest.SkipTest
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
    'use_linear_operator': [False, True],
}))
@testing.with_requires('scipy')
@testing.gpu
class TestCg:
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
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
            if M is not None:
                M = sp.linalg.aslinearoperator(M)
        return self._test_cg(dtype, xp, sp, a, M)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        if M is not None:
            M = sp.coo_matrix(M).asformat(format)
            if self.use_linear_operator:
                M = sp.linalg.aslinearoperator(M)
        return self._test_cg(dtype, xp, sp, a, M)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_empty(self, dtype, xp, sp):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
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
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
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
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
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
    'x0': [None, 'ones'],
    'M': [None, 'jacobi'],
    'atol': [None, 'select-by-dtype'],
    'b_ndim': [1, 2],
    'restart': [None, 10],
    'use_linear_operator': [False, True],
}))
@testing.with_requires('scipy>=1.4')
@testing.gpu
class TestGmres:
    n = 30
    density = 0.2
    _atol = {'f': 1e-5, 'd': 1e-12}

    # TODO(kataoka): Fix the `lstsq` call in CuPy's `gmres`
    @pytest.fixture(autouse=True)
    def ignore_futurewarning(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', '`rcond` parameter will change', FutureWarning,
            )
            yield

    def _make_matrix(self, dtype, xp):
        dtype = numpy.dtype(dtype)
        shape = (self.n, self.n)
        a = testing.shaped_random(shape, xp, dtype=dtype, scale=1)
        mask = testing.shaped_random(shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        diag = xp.diag(testing.shaped_random(
            (self.n,), xp, dtype=dtype.char.lower(), scale=1) + 1)
        a[diag > 0] = 0
        a = a + diag
        M = None
        if self.M == 'jacobi':
            M = xp.diag(1.0 / xp.diag(a))
        return a, M

    def _make_normalized_vector(self, dtype, xp):
        b = testing.shaped_random((self.n,), xp, dtype=dtype, scale=1)
        return b / xp.linalg.norm(b)

    def _test_gmres(self, dtype, xp, sp, a, M):
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
                return sp.linalg.gmres(
                    a, b, x0=x0, restart=self.restart, M=M, atol=atol)
        else:
            return sp.linalg.gmres(
                a, b, x0=x0, restart=self.restart, M=M, atol=atol)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
            if M is not None:
                M = sp.linalg.aslinearoperator(M)
        return self._test_gmres(dtype, xp, sp, a, M)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        if M is not None:
            M = sp.coo_matrix(M).asformat(format)
            if self.use_linear_operator:
                M = sp.linalg.aslinearoperator(M)
        return self._test_gmres(dtype, xp, sp, a, M)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_empty(self, dtype, xp, sp):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.restart is None and self.use_linear_operator is False):
            raise unittest.SkipTest
        a = xp.empty((0, 0), dtype=dtype)
        b = xp.empty((0,), dtype=dtype)
        if self.atol is None and xp == numpy:
            # Note: If atol is None or not specified, Scipy (at least 1.5.3)
            # raises DeprecationWarning
            with pytest.deprecated_call():
                return sp.linalg.gmres(a, b)
        else:
            return sp.linalg.gmres(a, b)

    @testing.for_dtypes('fdFD')
    def test_callback(self, dtype):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.restart is None and self.use_linear_operator is False):
            raise unittest.SkipTest
        xp, sp = cupy, sparse
        a, M = self._make_matrix(dtype, xp)
        b = self._make_normalized_vector(dtype, xp)
        is_called = False

        def callback1(x):
            print(xp.linalg.norm(b - a @ x))
            nonlocal is_called
            is_called = True
        sp.linalg.gmres(a, b, callback=callback1, callback_type='x')
        assert is_called
        is_called = False

        def callback2(pr_norm):
            print(pr_norm)
            nonlocal is_called
            is_called = True
        sp.linalg.gmres(a, b, callback=callback2, callback_type='pr_norm')
        assert is_called

    def test_invalid(self):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.restart is None and self.use_linear_operator is False):
            raise unittest.SkipTest
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a, M = self._make_matrix('f', xp)
            b = self._make_normalized_vector('f', xp)
            ng_a = xp.ones((self.n, ), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(ng_a, b)
            ng_a = xp.ones((self.n, self.n + 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(ng_a, b)
            ng_a = xp.ones((self.n, self.n, 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(ng_a, b)
            ng_b = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, ng_b)
            ng_b = xp.ones((self.n, 2), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, ng_b)
            ng_x0 = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, b, x0=ng_x0)
            ng_M = xp.diag(xp.ones((self.n + 1,), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, b, M=ng_M)
            ng_callback_type = '?'
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, b, callback_type=ng_callback_type)
        xp, sp = cupy, sparse
        b = self._make_normalized_vector('f', xp)
        ng_a = xp.ones((self.n, self.n), dtype='i')
        with pytest.raises(TypeError):
            sp.linalg.gmres(ng_a, b)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'outer_modification': [
        'normal', 'transpose', 'hermitian'],
    'inner_modification': [
        'normal', 'sparse', 'linear_operator', 'class_matvec', 'class_matmat'],
    'M': [1, 6],
    'N': [1, 7],
}))
@testing.gpu
@testing.with_requires('scipy>=1.4')
class TestLinearOperator(unittest.TestCase):

    # modified from scipy
    # class that defines parametrized custom cases
    # adapted from scipy's analogous tests
    def _inner_cases(self, xp, sp, A):
        # creating base-matrix-like class with default
        # matrix-vector and adjoint-matrix-vector impl

        def mv(x):
            return A.dot(x)

        def rmv(x):
            return A.T.conj().dot(x)

        # defining the user-defined classes
        class BaseMatlike(sp.linalg.LinearOperator):

            def __init__(self):
                self.dtype = A.dtype
                self.shape = A.shape

            def _adjoint(self):
                shape = self.shape[1], self.shape[0]
                return sp.linalg.LinearOperator(
                    matvec=rmv, rmatvec=mv, dtype=self.dtype, shape=shape)

        class HasMatvec(BaseMatlike):

            def _matvec(self, x):
                return mv(x)

        class HasMatmat(BaseMatlike):

            def _matmat(self, x):
                return mv(x)

        if self.inner_modification == 'normal':
            return sp.linalg.aslinearoperator(A)
        if self.inner_modification == 'sparse':
            # TODO(asi1024): Fix to return contiguous matrix.
            return sp.linalg.aslinearoperator(sp.csr_matrix(A))
        if self.inner_modification == 'linear_operator':
            return sp.linalg.LinearOperator(
                matvec=mv, rmatvec=rmv, dtype=A.dtype, shape=A.shape)
        if self.inner_modification == 'class_matvec':
            return HasMatvec()
        if self.inner_modification == 'class_matmat':
            return HasMatmat()
        assert False

    def _generate_linear_operator(self, xp, sp):
        A = testing.shaped_random((self.M, self.N), xp, self.dtype)

        if self.outer_modification == 'normal':
            return self._inner_cases(xp, sp, A)
        if self.outer_modification == 'transpose':
            # From SciPy 1.4 (scipy/scipy#9064)
            return self._inner_cases(xp, sp, A.T).T
        if self.outer_modification == 'hermitian':
            return self._inner_cases(xp, sp, A.T.conj()).H
        assert False

    @testing.numpy_cupy_allclose(sp_name='sp', rtol=1e-6)
    def test_matvec(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x_1dim = testing.shaped_random((self.N,), xp, self.dtype)
        x_2dim = testing.shaped_random((self.N, 1), xp, self.dtype)
        return linop.matvec(x_1dim), linop.matvec(x_2dim)

    @testing.numpy_cupy_allclose(
        sp_name='sp', rtol=1e-6, contiguous_check=False)
    def test_matmat(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x = testing.shaped_random((self.N, 8), xp, self.dtype)
        return linop.matmat(x)

    @testing.numpy_cupy_allclose(sp_name='sp', rtol=1e-6)
    def test_rmatvec(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x_1dim = testing.shaped_random((self.M,), xp, self.dtype)
        x_2dim = testing.shaped_random((self.M, 1), xp, self.dtype)
        return linop.rmatvec(x_1dim), linop.rmatvec(x_2dim)

    @testing.numpy_cupy_allclose(
        sp_name='sp', rtol=1e-6, contiguous_check=False)
    def test_rmatmat(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x = testing.shaped_random((self.M, 8), xp, self.dtype)
        return linop.rmatmat(x)

    @testing.numpy_cupy_allclose(
        sp_name='sp', rtol=1e-6, contiguous_check=False)
    def test_dot(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x0 = testing.shaped_random((self.N,), xp, self.dtype)
        x1 = testing.shaped_random((self.N, 1), xp, self.dtype)
        x2 = testing.shaped_random((self.N, 8), xp, self.dtype)
        return linop.dot(x0), linop.dot(x1), linop.dot(x2)

    @testing.numpy_cupy_allclose(
        sp_name='sp', rtol=1e-6, contiguous_check=False)
    def test_mul(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x0 = testing.shaped_random((self.N,), xp, self.dtype)
        x1 = testing.shaped_random((self.N, 1), xp, self.dtype)
        x2 = testing.shaped_random((self.N, 8), xp, self.dtype)
        return linop * x0, linop * x1, linop * x2


@testing.parameterize(*testing.product({
    'lower': [True, False],
    'unit_diagonal': [True, False],
    'nrhs': [None, 1, 4],
    'order': ['C', 'F']
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestSpsolveTriangular:

    n = 10
    density = 0.5

    def _make_matrix(self, dtype, xp):
        a_shape = (self.n, self.n)
        a = testing.shaped_random(a_shape, xp, dtype=dtype, scale=1)
        mask = testing.shaped_random(a_shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        diag = xp.diag(xp.ones((self.n,), dtype=dtype))
        a = a + diag
        if self.lower:
            a = xp.tril(a)
        else:
            a = xp.triu(a)
        b_shape = (self.n,) if self.nrhs is None else (self.n, self.nrhs)
        b = testing.shaped_random(b_shape, xp, dtype=dtype, order=self.order)
        return a, b

    def _test_spsolve_triangular(self, sp, a, b):
        return sp.linalg.spsolve_triangular(a, b, lower=self.lower,
                                            unit_diagonal=self.unit_diagonal)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        a, b = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        return self._test_spsolve_triangular(sp, a, b)

    def test_invalid_cases(self):
        dtype = 'float64'
        if not (self.lower and self.unit_diagonal and self.nrhs == 4 and
                self.order == 'C'):
            raise unittest.SkipTest

        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a, b = self._make_matrix(dtype, xp)
            a = sp.csr_matrix(a)

            # a is not a square matrix
            ng_a = sp.csr_matrix(xp.ones((self.n + 1, self.n), dtype=dtype))
            with pytest.raises(ValueError):
                self._test_spsolve_triangular(sp, ng_a, b)
            # b is not a 1D/2D matrix
            ng_b = xp.ones((1, self.n, self.nrhs), dtype=dtype)
            with pytest.raises(ValueError):
                self._test_spsolve_triangular(sp, a, ng_b)
            # mismatched shape
            ng_b = xp.ones((self.n + 1, self.nrhs), dtype=dtype)
            with pytest.raises(ValueError):
                self._test_spsolve_triangular(sp, a, ng_b)

        xp, sp = cupy, sparse
        a, b = self._make_matrix(dtype, xp)
        a = sp.csr_matrix(a)

        # unsupported dtype
        ng_a = sp.csr_matrix(xp.ones((self.n, self.n), dtype='bool'))
        with pytest.raises(TypeError):
            self._test_spsolve_triangular(sp, ng_a, b)
        # a is not spmatrix
        ng_a = xp.ones((self.n, self.n), dtype=dtype)
        with pytest.raises(TypeError):
            self._test_spsolve_triangular(sp, ng_a, b)
        # b is not cupy ndarray
        ng_b = numpy.ones((self.n, self.nrhs), dtype=dtype)
        with pytest.raises(TypeError):
            self._test_spsolve_triangular(sp, a, ng_b)


@testing.parameterize(*testing.product({
    'tol': [0, 1e-5],
    'reorder': [0, 1, 2, 3],
}))
@testing.with_requires('scipy')
class TestCsrlsvqr(unittest.TestCase):

    n = 8
    density = 0.75
    _test_tol = {'f': 1e-5, 'd': 1e-12}

    def _setup(self, dtype):
        dtype = numpy.dtype(dtype)
        a_shape = (self.n, self.n)
        a = testing.shaped_random(
            a_shape, numpy, dtype=dtype, scale=2 / self.n)
        a_mask = testing.shaped_random(a_shape, numpy, dtype='f', scale=1)
        a[a_mask > self.density] = 0
        a_diag = numpy.diag(numpy.ones((self.n,), dtype=dtype))
        a = a + a_diag
        b = testing.shaped_random((self.n,), numpy, dtype=dtype)
        test_tol = self._test_tol[dtype.char.lower()]
        return a, b, test_tol

    @testing.for_dtypes('fdFD')
    def test_csrlsvqr(self, dtype):
        if not cupy.cusolver.check_availability('csrlsvqr'):
            unittest.SkipTest('csrlsvqr is not available')
        a, b, test_tol = self._setup(dtype)
        ref_x = numpy.linalg.solve(a, b)
        cp_a = cupy.array(a)
        sp_a = cupyx.scipy.sparse.csr_matrix(cp_a)
        cp_b = cupy.array(b)
        x = cupy.cusolver.csrlsvqr(sp_a, cp_b, tol=self.tol,
                                   reorder=self.reorder)
        cupy.testing.assert_allclose(x, ref_x, rtol=test_tol,
                                     atol=test_tol)


def _eigen_vec_transform(block_vec, xp):
    """Helper to swap sign of each eigen vector based on the first
    non-zero element. ie, to standardize the first non-zero element
    of eigen vector as positive. This helps in comparing equivalence
    of eigen vectors"""
    direction = testing.shaped_random((block_vec.shape[0], 1),
                                      xp=xp, seed=123)
    direction = xp.where(block_vec.T.dot(direction) >= 0, 1, -1).T
    # shape of mask: (block_vec.shape[0], 1)
    # each eigenvector is multiplied by a 1 or -1 (scalar)
    # this is done by broadcasting mask
    return block_vec * direction


@testing.with_requires('scipy>=1.4')
@testing.gpu
# tests adapted from scipy's tests of lobpcg
class TestLOBPCG(unittest.TestCase):

    def _generate_input_for_elastic_rod(self, n, xp):
        """Build the matrices for the generalized eigenvalue problem of the
        fixed-free elastic rod vibration model.
        """
        L = 1.0
        le = L / n
        rho = 7.85e3
        S = 1.e-4
        E = 2.1e11
        mass = rho * S * le / 6.
        k = E * S / le
        A = k * (xp.diag(xp.r_[2. * xp.ones(n - 1), 1]) -
                 xp.diag(xp.ones(n - 1), 1) - xp.diag(xp.ones(n - 1), -1))
        B = mass * (xp.diag(xp.r_[4. * xp.ones(n - 1), 2]) +
                    xp.diag(xp.ones(n - 1), 1) + xp.diag(xp.ones(n - 1), -1))
        return A, B

    def _generate_input_for_mikota_pair(self, n, xp):
        """Build a pair of full diagonal matrices for the generalized eigenvalue
        problem. The Mikota pair acts as a nice test since the eigenvalues are
        the squares of the integers n, n=1,2,...
        """
        x = xp.arange(1, n + 1)
        B = xp.diag(1. / x)
        y = xp.arange(n - 1, 0, -1)
        z = xp.arange(2 * n - 1, 0, -2)
        A = xp.diag(z) - xp.diag(y, -1) - xp.diag(y, 1)
        return A, B

    def _generate_random_initial_ortho_eigvec(self, m, n, xp=numpy, seed=0):
        """helper to generate orthogonal, random initial approximation for
        eigen vectors.
        """
        V = testing.shaped_random((m, n), xp=numpy, seed=seed)
        # TODO : use cupy's native linalg.orth() once implemented
        X = scipy.linalg.orth(V)
        return xp.asarray(X)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_small_generate_input_for_elastic_rod(self, xp, sp):
        A, B = self._generate_input_for_elastic_rod(10, xp)
        n = A.shape[0]
        X = self._generate_random_initial_ortho_eigvec(n, 10, xp)
        eigvals, eigvecs = sp.linalg.lobpcg(A,
                                            X, B=B,
                                            tol=1e-5, maxiter=30,
                                            largest=False)
        return eigvals, _eigen_vec_transform(eigvecs, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_small_generate_input_for_mikota_pair(self, xp, sp):
        A, B = self._generate_input_for_mikota_pair(10, xp)
        n = A.shape[0]
        X = self._generate_random_initial_ortho_eigvec(n, 10, xp)
        eigvals, eigvecs = sp.linalg.lobpcg(A,
                                            X, B=B,
                                            tol=1e-5, maxiter=30,
                                            largest=False)
        return eigvals, _eigen_vec_transform(eigvecs, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_generate_input_for_elastic_rod(self, xp, sp):
        A, B = self._generate_input_for_elastic_rod(100, xp)
        n = A.shape[0]
        X = self._generate_random_initial_ortho_eigvec(n, 20, xp)
        eigvals, eigvecs = sp.linalg.lobpcg(A,
                                            X, B=B,
                                            tol=1e-5, maxiter=30,
                                            largest=False)
        return eigvals, _eigen_vec_transform(eigvecs, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_generate_input_for_mikota_pair(self, xp, sp):
        A, B = self._generate_input_for_mikota_pair(100, xp)
        n = A.shape[0]
        X = self._generate_random_initial_ortho_eigvec(n, 20, xp)
        eigvals, eigvecs = sp.linalg.lobpcg(A,
                                            X, B=B,
                                            tol=1e-5, maxiter=30,
                                            largest=False)
        return eigvals, _eigen_vec_transform(eigvecs, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_regression(self, xp, sp):
        """Check the eigenvalue of the identity matrix is one.
        """
        n = 10
        X = xp.ones((n, 1))
        A = xp.identity(n)
        w, v = sp.linalg.lobpcg(A, X)
        return w, _eigen_vec_transform(v, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_diagonal(self, xp, sp):
        """Check for diagonal matrices.
        """
        # The system of interest is of size n x n.
        n = 100
        # We care about only m eigenpairs.
        m = 4
        # Define the generalized eigenvalue problem Av = cBv
        # where (c, v) is a generalized eigenpair,
        # We choose A to be the diagonal matrix whose entries are 1..n
        # and where B is chosen to be the identity matrix.
        vals = xp.arange(1, n + 1, dtype=float)
        A = sp.diags([vals], [0], (n, n))
        B = sp.eye(n)
        # Let the preconditioner M be the inverse of A.
        M = sp.diags([1. / vals], [0], (n, n))
        # Pick random initial vectors.
        X = testing.shaped_random((n, m), xp=xp, seed=1234)
        # Require that the returned eigenvectors be in the orthogonal
        # complement of the first few standard basis vectors (Y)
        m_excluded = 3
        Y = xp.eye(n, m_excluded)
        eigvals, vecs = sp.linalg.lobpcg(A, X, B, M=M, Y=Y, tol=1e-4,
                                         maxiter=40, largest=False)
        return eigvals, _eigen_vec_transform(vecs, xp)

    def _generate_A_for_fiedler(self, n, p, xp):
        """Check for fiedler vector computation"""
        # fiedler vector computation based on scipy's tests
        # https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/sparse/linalg/eigen/lobpcg/tests/test_lobpcg.py#L140
        col = numpy.zeros(n)
        col[1] = 1
        A = scipy.linalg.toeplitz(col)
        D = numpy.diag(A.sum(axis=1))
        return xp.asarray(D - A)

    def _generate_small_X_for_fiedler(self, n, p, xp):
        tmp = xp.pi * xp.arange(n) / n
        analytic_V = xp.cos(xp.outer(xp.arange(n) + 1 / 2, tmp))
        return analytic_V[:, :p]

    def _generate_large_X_for_fiedler(self, n, p, xp):
        tmp = xp.pi * xp.arange(n) / n
        analytic_V = xp.cos(xp.outer(xp.arange(n) + 1 / 2, tmp))
        return analytic_V[:, -p:]

    def _generate_approximate_X_for_fiedler(self, n, p, xp):
        fiedler_guess = xp.concatenate((xp.ones(n // 2),
                                        -xp.ones(n - n // 2)))
        return xp.vstack((xp.ones(n), fiedler_guess)).T

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_small_8(self, xp, sp):
        """Check the dense workaround path for small matrices
           for small fiedler eigen values and vectors
        """
        # This triggers the dense path because 8 < 2*5.
        A = self._generate_A_for_fiedler(8, 2, xp)
        X = self._generate_small_X_for_fiedler(8, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_large_8(self, xp, sp):
        """Check the dense workaround path for small matrices
           for large fiedler eigen values and vectors
        """
        # This triggers the dense path because 8 < 2*5.
        A = self._generate_A_for_fiedler(8, 2, xp)
        X = self._generate_large_X_for_fiedler(8, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_approximate_8(self, xp, sp):
        """Check the dense workaround path for small matrices
           for approximately-formed fiedler eigen values and vectors
        """
        # This triggers the dense path because 8 < 2*5.
        A = self._generate_A_for_fiedler(8, 2, xp)
        X = self._generate_approximate_X_for_fiedler(8, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_small_12(self, xp, sp):
        """Check the dense workaround path is avoided for non-small
           fiedler matrices and small eigen values and vectors
        """
        A = self._generate_A_for_fiedler(12, 2, xp)
        X = self._generate_small_X_for_fiedler(12, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_large_12(self, xp, sp):
        """Check the dense workaround path is avoided for non-small
           fiedler matrices and large eigen values and vectors
        """
        A = self._generate_A_for_fiedler(12, 2, xp)
        X = self._generate_large_X_for_fiedler(12, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_approximate_12(self, xp, sp):
        """Check the dense workaround path is avoided for non-small,
           approximately generated fiedler matrices
        """
        A = self._generate_A_for_fiedler(12, 2, xp)
        X = self._generate_approximate_X_for_fiedler(12, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    def _verbosity_helper(self, xp, sp):
        """Helper to capture the verbose output from stdout
        """
        A, B = self._generate_input_for_elastic_rod(100, xp)
        n = A.shape[0]
        m = 20
        X = self._generate_random_initial_ortho_eigvec(n, m, xp)
        saved_stdout = io.StringIO()
        with contextlib.redirect_stdout(saved_stdout):
            _, _ = sp.linalg.lobpcg(A, X, B=B, tol=1e-5,
                                    maxiter=30, largest=False,
                                    verbosityLevel=9)
        output = saved_stdout.getvalue().strip()
        return output

    def test_verbosity(self):
        """Check that nonzero verbosity level code runs
           and is identical to scipy's output format.
        """
        stdout_cupy = self._verbosity_helper(cupy, cupyx.scipy.sparse)
        stdout_numpy = self._verbosity_helper(numpy, scipy.sparse)
        # getting rid of the numbers and whitespaces, we care only about
        # format of printed output.
        # also, due to the fact that there are unpredictable (but minor)
        # differences in decimal digits between scipy and cupy verbose output
        stdout_cupy = re.sub(r'[-+]?\d+\.?\d*[ ]*', '{number}', stdout_cupy)
        stdout_numpy = re.sub(r'[-+]?\d+\.?\d*[ ]*', '{number}', stdout_numpy)
        assert stdout_numpy == stdout_cupy, '''numpy: %s
                                               cupy: %s''' % (stdout_numpy,
                                                              stdout_cupy)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-3, sp_name='sp',
                                 contiguous_check=False)
    def test_random_initial_float32(self, xp, sp):
        """Check lobpcg in float32 for specific initial.
        """
        n = 50
        m = 4
        vals = -xp.arange(1, n + 1)
        A = sp.diags([vals], [0], (n, n))
        A = A.astype(xp.float32)
        X = testing.shaped_random((n, m), xp=xp, seed=3)
        eigvals, vecs = sp.linalg.lobpcg(A, X, tol=1e-3, maxiter=50,
                                         verbosityLevel=1)
        return eigvals, _eigen_vec_transform(vecs, xp)

    def test_maxit_None(self):
        """Check lobpcg if maxit=None runs 20 iterations (the default)
        by checking the size of the iteration history output, which should
        be the number of iterations plus 2 (initial and final values).
        """
        n = 50
        m = 4
        vals = -cupy.arange(1, n + 1)
        A = sparse.diags([vals], [0], (n, n))
        A = A.astype(cupy.float32)
        X = testing.shaped_random((n, m), xp=cupy, seed=1566950023)
        _, _, l_h = sparse.linalg.lobpcg(A, X, tol=1e-8, maxiter=None,
                                         retLambdaHistory=True)
        assert len(l_h) == 22


@testing.with_requires('scipy>=1.4')
@testing.gpu
@testing.parameterize(*testing.product({
    'A_sparsity': [True, False],
    'B_sparsity': [True, False],
    'A_dtype': [cupy.float32, cupy.float64],
    'preconditioner_sparsity': [True, False],
    'preconditioner_dtype': [None, cupy.float32, cupy.float64],
    'X_dtype': [cupy.float32, cupy.float64],
    'Y_dtype': [cupy.float32, cupy.float64],
    'sparse_format': ['coo', 'csr', 'csc']
}))
# test class for testing against diagonal matrices overall various data types
class TestLOBPCGForDiagInput(unittest.TestCase):

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    @pytest.mark.slow
    def test_diagonal_data_types(self, xp, sp):
        """Check lobpcg for diagonal matrices for all matrix types.
        """
        n = 40
        m = 4
        # Define the generalized eigenvalue problem Av = cBv
        # where (c, v) is a generalized eigenpair,
        # and where we choose A  and B to be diagonal.
        vals = xp.arange(1, n + 1)
        # A and B matrices based on parametrization
        A = sp.diags([vals * vals], [0], (n, n), format=self.sparse_format)
        A = A.astype(xp.dtype(self.A_dtype))
        A = A if self.A_sparsity is True else A.toarray()

        B = sp.diags([vals], [0], (n, n), format=self.sparse_format)
        B = B if self.B_sparsity is True else B.toarray()

        M_LO = None
        if self.preconditioner_dtype is not None:
            M = sp.diags([1. / vals], [0], (n, n), format=self.sparse_format)
            M = M if self.preconditioner_sparsity else M.toarray()

            def fun(x):
                return M @ x
            # Define Preconditioner function as Linear Operator
            M_LO = sp.linalg.LinearOperator(matvec=fun,
                                            matmat=fun,
                                            shape=(n, n),
                                            dtype=xp.dtype(self.preconditioner_dtype))  # NOQA

        # Cannot be sparse array
        X = testing.shaped_random((n, m), xp=xp, dtype=xp.dtype(self.X_dtype),
                                  seed=1234)

        # Require tht returned eigenvectors be in the orthogonal
        # complement of the first few standard basis vectors
        # (Cannot be sparse array)
        m_excluded = 3
        Y = xp.eye(n, m_excluded, dtype=xp.dtype(self.Y_dtype))
        # core call to lobpcg solver
        eigvals, eigvecs = sp.linalg.lobpcg(A, X, B=B, M=M_LO, Y=Y,
                                            tol=1e-4, maxiter=100,
                                            largest=False)
        return eigvals, _eigen_vec_transform(eigvecs, xp)


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc', 'coo'],
    'nrhs': [None, 1, 4],
    'order': ['C', 'F']
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
@testing.gpu
class TestSplu(unittest.TestCase):

    n = 10
    density = 0.5

    def _make_matrix(self, dtype, xp, sp, density=None):
        if density is None:
            density = self.density
        a_shape = (self.n, self.n)
        a = testing.shaped_random(a_shape, xp, dtype=dtype, scale=2 / self.n)
        mask = testing.shaped_random(a_shape, xp, dtype='f', scale=1)
        a[mask > density] = 0
        diag = xp.diag(xp.ones((self.n,), dtype=dtype))
        a = a + diag
        if self.format == 'csr':
            a = sp.csr_matrix(a)
        elif self.format == 'csc':
            a = sp.csc_matrix(a)
        elif self.format == 'coo':
            a = sp.coo_matrix(a)
        b_shape = (self.n,) if self.nrhs is None else (self.n, self.nrhs)
        b = testing.shaped_random(b_shape, xp, dtype=dtype, order=self.order)
        return a, b

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_splu(self, dtype, xp, sp):
        a, b = self._make_matrix(dtype, xp, sp)
        return sp.linalg.splu(a).solve(b)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_factorized(self, dtype, xp, sp):
        a, b = self._make_matrix(dtype, xp, sp)
        return sp.linalg.factorized(a)(b)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_spilu(self, dtype, xp, sp):
        a, b = self._make_matrix(dtype, xp, sp)
        return sp.linalg.spilu(a).solve(b)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_spilu_0(self, dtype, xp, sp):
        # Note: We don't know how to compute ILU(0) with
        # scipy.sprase.linalg.spilu, so in this test we use a matrix where the
        # format is a sparse matrix but is actually a dense matrix.
        a, b = self._make_matrix(dtype, xp, sp, density=1.0)
        if xp == cupy:
            # Set fill_factor=1 to computes ILU(0) using cuSparse
            ainv = sp.linalg.spilu(a, fill_factor=1)
        else:
            ainv = sp.linalg.spilu(a)
        return ainv.solve(b)
