import cupy
import itertools
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

from cupy import testing
from cupy.testing import condition
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
}))
@testing.with_requires('scipy')
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
@testing.with_requires('scipy')
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


@testing.with_requires('scipy>=1.4')
@testing.gpu
# tests adapted from scipy's tests of lobpcg
class TestLOBPCG:

    def _elasticRod(self, n, xp):
        """Build the matrices for the generalized eigenvalue problem of the
        fixed-free elastic rod vibration model.
        """
        L = 1.0
        le = L/n
        rho = 7.85e3
        S = 1.e-4
        E = 2.1e11
        mass = rho*S*le/6.
        k = E*S/le
        A = k*(xp.diag(xp.r_[2.*xp.ones(n-1), 1])-xp.diag(xp.ones(n-1), 1) -
               xp.diag(xp.ones(n-1), -1))
        B = mass*(xp.diag(xp.r_[4.*xp.ones(n-1), 2])+xp.diag(xp.ones(n-1), 1)
                  + xp.diag(xp.ones(n-1), -1))
        return A, B

    def _mikotaPair(self, n, xp):
        """Build a pair of full diagonal matrices for the generalized eigenvalue
        problem. The Mikota pair acts as a nice test since the eigenvalues are
        the squares of the integers n, n=1,2,...
        """
        x = xp.arange(1, n+1)
        B = xp.diag(1./x)
        y = xp.arange(n-1, 0, -1)
        z = xp.arange(2*n-1, 0, -2)
        A = xp.diag(z)-xp.diag(y, -1)-xp.diag(y, 1)
        return A, B

    def _compare_solutions(self, A, B, m, xp, sp):
        """Check eig vs. lobpcg consistency.
        """
        n = A.shape[0]
        numpy.random.seed(0)  # seeding is different in numpy and cupy!
        V = numpy.random.rand(n, m)
        X = scipy.linalg.orth(V)
        eigvals, _ = sp.linalg.lobpcg(A, xp.asarray(X), B=B, tol=1e-5,
                                      maxiter=30, largest=False)
        eigvals.sort()
        # converting to numpy below as there is no cupy general eigen value
        # in cupy at the moment
        w, _ = scipy.linalg.eig(cupy.asnumpy(A), b=cupy.asnumpy(B))
        w.sort()
        cupy.testing.assert_array_almost_equal(
            w[:int(m/2)], cupy.asnumpy(eigvals[:int(m/2)]), decimal=2)
        return eigvals

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_Small(self, xp, sp):
        eval_list = []
        A, B = self._elasticRod(10, xp)
        eval_list.append(self._compare_solutions(A, B, 10, xp, sp))
        A, B = self._mikotaPair(10, xp)
        eval_list.append(self._compare_solutions(A, B, 10, xp, sp))
        return eval_list

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_ElasticRod(self, xp, sp):
        A, B = self._elasticRod(100, xp)
        return self._compare_solutions(A, B, 20, xp, sp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_MikotaPair(self, xp, sp):
        A, B = self._mikotaPair(100, xp)
        return self._compare_solutions(A, B, 20, xp, sp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_regression(self, xp, sp):
        """Check the eigenvalue of the identity matrix is one.
        """
        n = 10
        X = xp.ones((n, 1))
        A = xp.identity(n)
        w, _ = sp.linalg.lobpcg(A, X)
        cupy.testing.assert_allclose(w, xp.array([1]))
        return w

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_diagonal(self, xp, sp):
        """Check for diagonal matrices.
        """
        numpy.random.seed(1234)
        # The system of interest is of size n x n.
        n = 100
        # We care about only m eigenpairs.
        m = 4
        # Define the generalized eigenvalue problem Av = cBv
        # where (c, v) is a generalized eigenpair,
        # We choose A to be the diagonal matrix whose entries are 1..n
        # and where B is chosen to be the identity matrix.
        vals = xp.arange(1, n+1, dtype=float)
        A = sp.diags([vals], [0], (n, n))
        B = sp.eye(n)
        # Let the preconditioner M be the inverse of A.
        M = sp.diags([1./vals], [0], (n, n))
        # Pick random initial vectors.
        X = xp.asarray(numpy.random.rand(n, m))
        # Require that the returned eigenvectors be in the orthogonal
        # complement of the first few standard basis vectors (Y)
        m_excluded = 3
        Y = xp.eye(n, m_excluded)

        eigvals, vecs = sp.linalg.lobpcg(A, X, B, M=M, Y=Y, tol=1e-4,
                                         maxiter=40, largest=False)

        cupy.testing.assert_allclose(eigvals, xp.arange(1+m_excluded,
                                                        1+m_excluded+m))
        self._check_eigen(A, eigvals, vecs, xp, sp, rtol=1e-3, atol=1e-3)
        return eigvals

    def _check_eigen(self, M, w, V, xp, sp, rtol=1e-8, atol=1e-14):
        """Check if the eigenvalue residual is small.
        """
        mult_wV = xp.multiply(w, V)
        dot_MV = M.dot(V)
        cupy.testing.assert_allclose(mult_wV, dot_MV, rtol=rtol, atol=atol)

    def _check_fiedler(self, n, p, xp, sp):
        """Check the Fiedler vector computation.
        """
        eval_list = []
        numpy.random.seed(1234)
        col = numpy.zeros(n)
        col[1] = 1
        A = scipy.linalg.toeplitz(col)
        D = numpy.diag(A.sum(axis=1))
        L = D - A
        # Compute the full eigendecomposition using tricks, e.g.
        # http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
        tmp = numpy.pi * numpy.arange(n) / n
        analytic_w = 2 * (1 - numpy.cos(tmp))
        analytic_V = numpy.cos(numpy.outer(numpy.arange(n) + 1/2, tmp))
        self._check_eigen(L, analytic_w, analytic_V, numpy, scipy.sparse)
        # Compute the full eigendecomposition using eigh.
        eigh_w, eigh_V = scipy.linalg.eigh(L)
        self._check_eigen(L, eigh_w, eigh_V, numpy, scipy.sparse)
        # Check that the first eigenvalue is near zero and that the rest agree.
        cupy.testing.assert_array_less(numpy.abs([eigh_w[0], analytic_w[0]]),
                                       1e-14)
        cupy.testing.assert_allclose(eigh_w[1:], analytic_w[1:])

        # Check small lobpcg eigenvalues.
        X = analytic_V[:, :p]
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(xp.asarray(L), xp.asarray(X),
                                              largest=False)
        eval_list.append(lobpcg_w)
        cupy.testing.assert_array_equal(lobpcg_w.shape, (p,))
        cupy.testing.assert_array_equal(lobpcg_V.shape, (n, p))
        self._check_eigen(xp.asarray(L), lobpcg_w, lobpcg_V, xp, sp)
        cupy.testing.assert_array_less(xp.abs(xp.min(lobpcg_w)),
                                       xp.array(1e-14))
        cupy.testing.assert_allclose(xp.sort(lobpcg_w)[1:],
                                     xp.asarray(analytic_w[1:p]))

        # Check large lobpcg eigenvalues.
        X = analytic_V[:, -p:]
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(xp.asarray(L), xp.asarray(X),
                                              largest=True)
        eval_list.append(lobpcg_w)
        cupy.testing.assert_array_equal(lobpcg_w.shape, (p,))
        cupy.testing.assert_array_equal(lobpcg_V.shape, (n, p))
        self._check_eigen(xp.asarray(L), lobpcg_w, lobpcg_V, xp, sp)
        cupy.testing.assert_allclose(xp.sort(lobpcg_w),
                                     xp.asarray(analytic_w[-p:]))

        fiedler_guess = numpy.concatenate((numpy.ones(n//2),
                                           -numpy.ones(n-n//2)))
        X = numpy.vstack((numpy.ones(n), fiedler_guess)).T
        lobpcg_w, _ = sp.linalg.lobpcg(xp.asarray(L), xp.asarray(X),
                                       largest=False)
        # Mathematically, the smaller eigenvalue should be zero
        # and the larger should be the algebraic connectivity.
        lobpcg_w = xp.sort(lobpcg_w)
        eval_list.append(lobpcg_w)
        cupy.testing.assert_allclose(lobpcg_w, xp.asarray(analytic_w[:2]),
                                     atol=1e-14)
        return eval_list

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_fiedler_small_8(self, xp, sp):
        """Check the dense workaround path for small matrices.
        """
        # This triggers the dense path because 8 < 2*5.
        return self._check_fiedler(8, 2, xp, sp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_fiedler_large_12(self, xp, sp):
        """Check the dense workaround path avoided for non-small matrices.
        """
        # This does not trigger the dense path, because 2*5 <= 12.
        return self._check_fiedler(12, 2, xp, sp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_eigs_consistency(self, xp, sp):
        """Check eigs vs. lobpcg consistency.
        """
        # The n=5 case tests the alternative small matrix code path
        param_arr = [(20, 1e-3), (5, 1e-8)]
        eval_list = []
        for n, _atol in param_arr:
            vals = xp.arange(1, n+1, dtype=xp.float64)
            A = sp.spdiags(vals, 0, n, n)
            numpy.random.seed(345678)
            X = xp.asarray(numpy.random.rand(n, 2))
            lvals, lvecs = sp.linalg.lobpcg(A, X, largest=True, maxiter=100)
            eval_list.append(lvals)
            try:  # cupyx.sparse.linalg.eigsh might not converge!
                vals, _ = sp.linalg.eigsh(A, k=2)
            except Exception as e:
                print("Exception: {} \n".format(e))
                continue
            self._check_eigen(A, lvals, lvecs, xp, sp, atol=_atol, rtol=0)
            cupy.testing.assert_allclose(xp.sort(vals), xp.sort(lvals),
                                         atol=1e-14)
        return eval_list

    def test_verbosity(self):
        """Check that nonzero verbosity level code runs.
        """
        A, B = self._elasticRod(100, cupy)
        n = A.shape[0]
        m = 20
        cupy.random.seed(0)
        V = numpy.random.rand(n, m)
        X = scipy.linalg.orth(V)
        _, _ = sparse.linalg.lobpcg(A, cupy.asarray(X), B=B, tol=1e-5,
                                    maxiter=30, largest=False,
                                    verbosityLevel=9)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_random_initial_float32(self, xp, sp):
        """Check lobpcg in float32 for specific initial.
        """
        numpy.random.seed(3)
        n = 50
        m = 4
        vals = -xp.arange(1, n + 1)
        A = sp.diags([vals], [0], (n, n))
        A = A.astype(xp.float32)
        X = xp.asarray(numpy.random.rand(n, m))
        X = X.astype(xp.float32)
        eigvals, _ = sp.linalg.lobpcg(A, X, tol=1e-3, maxiter=50,
                                      verbosityLevel=1)
        cupy.testing.assert_allclose(eigvals, -xp.arange(1, 1 + m), atol=1e-2)
        return eigvals

    def test_maxit_None(self):
        """Check lobpcg if maxit=None runs 20 iterations (the default)
        by checking the size of the iteration history output, which should
        be the number of iterations plus 2 (initial and final values).
        """
        cupy.random.seed(1566950023)
        n = 50
        m = 4
        vals = -cupy.arange(1, n + 1)
        A = sparse.diags([vals], [0], (n, n))
        A = A.astype(cupy.float32)
        X = cupy.random.randn(n, m)
        X = X.astype(cupy.float32)
        _, _, l_h = sparse.linalg.lobpcg(A, X, tol=1e-8, maxiter=20,
                                         retLambdaHistory=True)
        cupy.testing.assert_allclose(cupy.array(len(l_h)), cupy.array(20+2))

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    @pytest.mark.slow
    def test_diagonal_data_types(self, xp, sp):
        """Check lobpcg for diagonal matrices for all matrix types.
        """
        numpy.random.seed(1234)
        n = 40
        m = 4
        # Define the generalized eigenvalue problem Av = cBv
        # where (c, v) is a generalized eigenpair,
        # and where we choose A  and B to be diagonal.
        vals = xp.arange(1, n + 1)
        eval_list = []

        list_sparse_format = ['coo', 'csc', 'csr']
        sparse_formats = len(list_sparse_format)
        for s_f_i, s_f in enumerate(list_sparse_format):

            As64 = sp.diags([vals * vals], [0], (n, n), format=s_f)
            As32 = As64.astype(xp.float32)
            Af64 = As64.toarray()
            Af32 = Af64.astype(xp.float32)
            listA = [Af64, As64, Af32, As32]

            Bs64 = sp.diags([vals], [0], (n, n), format=s_f)
            Bf64 = Bs64.toarray()
            listB = [Bf64, Bs64]

            # Define the preconditioner function as LinearOperator.
            Ms64 = sp.diags([1./vals], [0], (n, n), format=s_f)

            def Ms64precond(x):
                return Ms64 @ x
            Ms64precondLO = sp.linalg.LinearOperator(matvec=Ms64precond,
                                                     matmat=Ms64precond,
                                                     shape=(n, n), dtype=float)
            Mf64 = Ms64.toarray()

            def Mf64precond(x):
                return Mf64 @ x
            Mf64precondLO = sp.linalg.LinearOperator(matvec=Mf64precond,
                                                     matmat=Mf64precond,
                                                     shape=(n, n), dtype=float)
            Ms32 = Ms64.astype(xp.float32)

            def Ms32precond(x):
                return Ms32 @ x
            Ms32precondLO = sp.linalg.LinearOperator(matvec=Ms32precond,
                                                     matmat=Ms32precond,
                                                     shape=(n, n),
                                                     dtype=xp.float32)
            Mf32 = Ms32.toarray()

            def Mf32precond(x):
                return Mf32 @ x
            Mf32precondLO = sp.linalg.LinearOperator(matvec=Mf32precond,
                                                     matmat=Mf32precond,
                                                     shape=(n, n),
                                                     dtype=xp.float32)
            listM = [None, Ms64precondLO, Mf64precondLO,
                     Ms32precondLO, Mf32precondLO]

            # Setup matrix of the initial approximation to the eigenvectors
            # (cannot be sparse array).
            Xf64 = xp.asarray(numpy.random.rand(n, m))
            Xf32 = Xf64.astype(xp.float32)
            listX = [Xf64, Xf32]

            # Require that returned eigenvectors be in the orthogonal
            # complement
            # of the first few standard basis vectors (cannot be sparse array).
            m_excluded = 3
            Yf64 = xp.eye(n, m_excluded, dtype=float)
            Yf32 = xp.eye(n, m_excluded, dtype=xp.float32)
            listY = [Yf64, Yf32]

            tests = list(itertools.product(listA, listB, listM, listX, listY))
            # to test here, instead of checking product of all input, output
            # types test each configuration for the first sparse format, and
            #  then for one additional sparse format. this takes 2/7=30% as
            # long as testing all configurations for all sparse formats.
            if s_f_i > 0:
                tests = tests[s_f_i - 1::sparse_formats-1]

            for A, B, M, X, Y in tests:
                eigvals, _ = sp.linalg.lobpcg(A, X, B=B, M=M, Y=Y, tol=1e-4,
                                              maxiter=100, largest=False)
                eval_list.append(eigvals)
                cupy.testing.assert_allclose(eigvals,
                                             xp.arange(1 + m_excluded,
                                                       1 + m_excluded + m))
        return eval_list
