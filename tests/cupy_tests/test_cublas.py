import numpy
import pytest

import cupy
from cupy import cublas
from cupy import testing
from cupy.testing import _attr


@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n': [10, 33, 100],
    'bs': [None, 1, 10],
    'nrhs': [None, 1, 10],
}))
@_attr.gpu
class TestBatchedGesv:
    _tol = {'f': 5e-5, 'd': 1e-12}

    def _make_random_matrices(self, shape, xp):
        a = testing.shaped_random(shape, xp, dtype=self.r_dtype, scale=1)
        if self.dtype.char in 'FD':
            a = a + 1j * testing.shaped_random(shape, xp, dtype=self.r_dtype,
                                               scale=1)
        return a

    def _make_well_conditioned_matrices(self, shape):
        a = self._make_random_matrices(shape, numpy)
        u, s, vh = numpy.linalg.svd(a)
        s = testing.shaped_random(s.shape, numpy, dtype=self.r_dtype,
                                  scale=1) + 1
        a = numpy.einsum('...ik,...k,...kj->...ij', u, s, vh)
        return cupy.array(a)

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.dtype = numpy.dtype(self.dtype)
        if self.dtype.char in 'fF':
            self.r_dtype = numpy.float32
        else:
            self.r_dtype = numpy.float64
        n = self.n
        bs = 1 if self.bs is None else self.bs
        nrhs = 1 if self.nrhs is None else self.nrhs
        a = self._make_well_conditioned_matrices((bs, n, n))
        x = self._make_random_matrices((bs, n, nrhs), cupy)
        b = cupy.matmul(a, x)
        a_shape = (n, n) if self.bs is None else (bs, n, n)
        b_shape = [n]
        if self.bs is not None:
            b_shape.insert(0, bs)
        if self.nrhs is not None:
            b_shape.append(nrhs)
        self.a = a.reshape(a_shape)
        self.b = b.reshape(b_shape)
        self.x_ref = x.reshape(b_shape)
        if self.r_dtype == numpy.float32:
            self.tol = self._tol['f']
        elif self.r_dtype == numpy.float64:
            self.tol = self._tol['d']

    def test_batched_gesv(self):
        x = cublas.batched_gesv(self.a, self.b)
        cupy.testing.assert_allclose(x, self.x_ref,
                                     rtol=self.tol, atol=self.tol)


@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n': [10, 100],
    'mode': [None, numpy, cupy],
}))
@_attr.gpu
class TestLevel1Functions:
    _tol = {'f': 1e-5, 'd': 1e-12}

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.dtype = numpy.dtype(self.dtype)
        self.tol = self._tol[self.dtype.char.lower()]

    def _make_random_vector(self):
        return testing.shaped_random((self.n,), cupy, dtype=self.dtype)

    def _make_out(self, dtype):
        out = None
        if self.mode is not None:
            out = self.mode.empty([], dtype=dtype)
        return out

    def _check_pointer(self, a, b):
        if a is not None and b is not None:
            assert self._get_pointer(a) == self._get_pointer(b)

    def _get_pointer(self, a):
        if isinstance(a, cupy.ndarray):
            return a.data.ptr
        else:
            return a.ctypes.data

    def test_iamax(self):
        x = self._make_random_vector()
        ref = cupy.argmax(cupy.absolute(x.real) + cupy.absolute(x.imag))
        out = self._make_out('i')
        res = cublas.iamax(x, out=out)
        self._check_pointer(res, out)
        # Note: iamax returns 1-based index
        cupy.testing.assert_array_equal(res - 1, ref)

    def test_iamin(self):
        x = self._make_random_vector()
        ref = cupy.argmin(cupy.absolute(x.real) + cupy.absolute(x.imag))
        out = self._make_out('i')
        res = cublas.iamin(x, out=out)
        self._check_pointer(res, out)
        # Note: iamin returns 1-based index
        cupy.testing.assert_array_equal(res - 1, ref)

    def test_asum(self):
        x = self._make_random_vector()
        ref = cupy.sum(cupy.absolute(x.real) + cupy.absolute(x.imag))
        out = self._make_out(self.dtype.char.lower())
        res = cublas.asum(x, out=out)
        self._check_pointer(res, out)
        cupy.testing.assert_allclose(res, ref, rtol=self.tol, atol=self.tol)

    def test_axpy(self):
        x = self._make_random_vector()
        y = self._make_random_vector()
        a = 1.1
        if self.dtype.char in 'FD':
            a = a - 1j * 0.9
        ref = a * x + y
        if self.mode is not None:
            a = self.mode.array(a, dtype=self.dtype)
        cublas.axpy(a, x, y)
        cupy.testing.assert_allclose(y, ref, rtol=self.tol, atol=self.tol)

    def test_dot(self):
        x = self._make_random_vector()
        y = self._make_random_vector()
        ref = x.dot(y)
        out = self._make_out(self.dtype)
        if self.dtype.char in 'FD':
            with pytest.raises(TypeError):
                res = cublas.dot(x, y, out=out)
            return
        res = cublas.dot(x, y, out=out)
        self._check_pointer(res, out)
        cupy.testing.assert_allclose(res, ref, rtol=self.tol, atol=self.tol)

    def test_dotu(self):
        x = self._make_random_vector()
        y = self._make_random_vector()
        ref = x.dot(y)
        out = self._make_out(self.dtype)
        res = cublas.dotu(x, y, out=out)
        self._check_pointer(res, out)
        cupy.testing.assert_allclose(res, ref, rtol=self.tol, atol=self.tol)

    def test_dotc(self):
        x = self._make_random_vector()
        y = self._make_random_vector()
        ref = x.conj().dot(y)
        out = self._make_out(self.dtype)
        res = cublas.dotc(x, y, out=out)
        self._check_pointer(res, out)
        cupy.testing.assert_allclose(res, ref, rtol=self.tol, atol=self.tol)

    def test_nrm2(self):
        x = self._make_random_vector()
        ref = cupy.linalg.norm(x)
        out = self._make_out(self.dtype.char.lower())
        res = cublas.nrm2(x, out=out)
        self._check_pointer(res, out)
        cupy.testing.assert_allclose(res, ref, rtol=self.tol, atol=self.tol)

    def test_scal(self):
        x = self._make_random_vector()
        a = 1.1
        if self.dtype.char in 'FD':
            a = a - 1j * 0.9
        ref = a * x
        if self.mode is not None:
            a = self.mode.array(a, dtype=self.dtype)
        cublas.scal(a, x)
        cupy.testing.assert_allclose(x, ref, rtol=self.tol, atol=self.tol)


@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'shape': [(10, 9), (9, 10)],
    'trans': ['N', 'T', 'H'],
    'order': ['C', 'F'],
    'mode': [None, numpy, cupy],
}))
@_attr.gpu
class TestGemv:
    _tol = {'f': 1e-5, 'd': 1e-12}

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.dtype = numpy.dtype(self.dtype)
        self.tol = self._tol[self.dtype.char.lower()]

    def test_gemv(self):
        a = testing.shaped_random(self.shape, cupy, dtype=self.dtype,
                                  order=self.order)
        if self.trans == 'N':
            ylen, xlen = self.shape
        else:
            xlen, ylen = self.shape
        x = testing.shaped_random((xlen,), cupy, dtype=self.dtype)
        y = testing.shaped_random((ylen,), cupy, dtype=self.dtype)
        alpha = 0.9
        beta = 0.8
        if self.dtype.char in 'FD':
            alpha = alpha - 1j * 0.7
            beta = beta - 1j * 0.6
        if self.trans == 'N':
            ref = alpha * a.dot(x) + beta * y
        elif self.trans == 'T':
            ref = alpha * a.T.dot(x) + beta * y
        elif self.trans == 'H':
            ref = alpha * a.T.conj().dot(x) + beta * y
        if self.mode is not None:
            alpha = self.mode.array(alpha)
            beta = self.mode.array(beta)
        cupy.cublas.gemv(self.trans, alpha, a, x, beta, y)
        cupy.testing.assert_allclose(y, ref, rtol=self.tol, atol=self.tol)


@testing.parameterize(*testing.product({
    'rank': [5, 9],
    'band': [0, 1, 3],
    'lower': [0, 1],
    'order': ['C', 'F'],
    'mode': [None, numpy, cupy],
}))
@_attr.gpu
class TestSbmv:
    _tol = {'f': 1e-5, 'd': 1e-12}

    def _gen2band(self, A, ku=0, kl=0, order='C'):
        assert A.ndim == 2
        n, m = A.shape
        ldm, lda = n, 1 + ku + kl
        B = numpy.zeros((lda, ldm), dtype=A.dtype, order=order)
        for j in range(n):
            k = ku - j
            for i in range(max(0, j-ku), min(m, j + kl + 1)):
                B[(k + i), j] = A[i, j]
        return B

    @testing.for_dtypes('fd')
    def test_sbmv(self, dtype):
        dtype = numpy.dtype(dtype)
        alpha, beta = 3.0, 2.0
        n, k = self.rank, self.band
        a = numpy.eye(n, n, 0, dtype, self.order)
        a *= numpy.random.randint(20)
        for i in range(1, k+1):
            band = numpy.random.randint(20, size=n-i)
            a += numpy.diag(band, k=+i)
            a += numpy.diag(band, k=-i)
        x = numpy.random.randint(20, size=n).astype(a.dtype)
        y = numpy.random.randint(20, size=n).astype(a.dtype)
        ku, kl = k, 0
        if self.lower == 1:
            ku, kl = kl, ku
        b = self._gen2band(a, ku, kl)
        a, b = cupy.asarray(a), cupy.asarray(b)
        x, y = cupy.asarray(x), cupy.asarray(y)
        ref = alpha * a.dot(x) + beta * y
        if self.mode is not None:
            alpha = self.mode.array(alpha)
            beta = self.mode.array(beta)
        y_ret = cupy.cublas.sbmv(k, alpha, b, x, beta, y, lower=self.lower)
        tol = self._tol[dtype.char.lower()]
        cupy.testing.assert_allclose(y, ref, rtol=tol, atol=tol)
        cupy.testing.assert_allclose(y_ret, ref, rtol=tol, atol=tol)


@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'shape': [(10, 9), (9, 10)],
    'order': ['C', 'F'],
    'mode': [None, numpy, cupy],
}))
@_attr.gpu
class TestGer:
    _tol = {'f': 1e-5, 'd': 1e-12}

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.dtype = numpy.dtype(self.dtype)
        self.tol = self._tol[self.dtype.char.lower()]
        self.a = testing.shaped_random(self.shape, cupy, dtype=self.dtype,
                                       order=self.order)
        self.x = testing.shaped_random((self.shape[0],), cupy,
                                       dtype=self.dtype)
        self.y = testing.shaped_random((self.shape[1],), cupy,
                                       dtype=self.dtype)
        self.alpha = 1.1
        if self.dtype.char in 'FD':
            self.alpha = self.alpha - 1j * 0.9

    def test_ger(self):
        if self.dtype.char in 'FD':
            with pytest.raises(TypeError):
                cublas.ger(self.alpha, self.x, self.y, self.a)
            return
        ref = self.alpha * cupy.outer(self.x, self.y) + self.a
        if self.mode is not None:
            self.alpha = self.mode.array(self.alpha)
        cublas.ger(self.alpha, self.x, self.y, self.a)
        cupy.testing.assert_allclose(self.a, ref, rtol=self.tol, atol=self.tol)

    def test_geru(self):
        ref = self.alpha * cupy.outer(self.x, self.y) + self.a
        if self.mode is not None:
            self.alpha = self.mode.array(self.alpha)
        cublas.geru(self.alpha, self.x, self.y, self.a)
        cupy.testing.assert_allclose(self.a, ref, rtol=self.tol, atol=self.tol)

    def test_gerc(self):
        ref = self.alpha * cupy.outer(self.x, self.y.conj()) + self.a
        if self.mode is not None:
            self.alpha = self.mode.array(self.alpha)
        cublas.gerc(self.alpha, self.x, self.y, self.a)
        cupy.testing.assert_allclose(self.a, ref, rtol=self.tol, atol=self.tol)


@testing.parameterize(*testing.product({
    'nk': [(5, 9), (9, 5)],
    'transa': ['N', 'T'],
    'ordera': ['F', 'C'],
    'orderc': ['F', 'C'],
    'lower': [0, 1],
    'mode': [None, numpy, cupy]
}))
@_attr.gpu
class TestSyrk:
    _tol = {'f': 1e-5, 'd': 1e-12}

    def _make_matrix(self, m, n, trans, order, dtype):
        if trans == 'N':
            shape = (m, n)
        else:
            shape = (n, m)
        return testing.shaped_random(shape, cupy, dtype=dtype, order=order,
                                     scale=1.0)

    def _trans_matrix(self, a, trans):
        if trans == 'N':
            return a
        return a.T

    @testing.for_dtypes('fdFD')
    def test_syrk(self, dtype):
        dtype = numpy.dtype(dtype)
        tol = self._tol[dtype.char.lower()]
        alpha, beta = 3.0, 2.0
        if dtype.char in 'FD':
            alpha = alpha - 1j * 2.0
            beta = beta + 1j * 5.0
        n, k = self.nk
        a = self._make_matrix(n, k, self.transa, self.ordera, dtype)
        aa = self._trans_matrix(a, self.transa)
        ref = alpha * aa.dot(aa.T)  # beta is used as a placeholder only
        c = cublas.syrk(self.transa, a, alpha=alpha, beta=beta,
                        lower=self.lower)
        rr, cc = cupy.asnumpy(ref), cupy.asnumpy(c)
        if self.lower:
            rr[numpy.triu_indices_from(rr,  1)] = 0
        else:
            rr[numpy.tril_indices_from(rr, -1)] = 0
        rru = rr[numpy.triu_indices_from(rr)]
        ccu = cc[numpy.triu_indices_from(cc)]
        rrl = rr[numpy.tril_indices_from(rr)]
        ccl = cc[numpy.tril_indices_from(cc)]
        cupy.testing.assert_allclose(ccu, rru, rtol=tol, atol=tol)
        cupy.testing.assert_allclose(ccl, rrl, rtol=tol, atol=tol)

    @testing.for_dtypes('fdFD')
    def test_syrk_out(self, dtype):
        dtype = numpy.dtype(dtype)
        tol = self._tol[dtype.char.lower()]
        alpha, beta = 2.3, 1.7
        if dtype.char in 'FD':
            alpha = alpha - 1j * 0.7
            beta = beta + 1j * 2.3
        n, k = self.nk
        a = self._make_matrix(n, k, self.transa, self.ordera, dtype)
        aa = self._trans_matrix(a, self.transa)
        m = aa.shape[0]
        c = self._make_matrix(m, m, 'N', self.orderc, dtype)
        c0 = cupy.array(c)
        ref = alpha * aa.dot(aa.T) + beta * c
        cublas.syrk(self.transa, a, out=c, alpha=alpha, beta=beta,
                    lower=self.lower)
        rr, c0, cc = cupy.asnumpy(ref), cupy.asnumpy(c0), cupy.asnumpy(c)
        if self.lower:
            trii = numpy.triu_indices_from(rr,  1)
        else:
            trii = numpy.tril_indices_from(rr, -1)
        rr[trii] = c0[trii]
        rru = rr[numpy.triu_indices_from(rr)]
        ccu = cc[numpy.triu_indices_from(cc)]
        rrl = rr[numpy.tril_indices_from(rr)]
        ccl = cc[numpy.tril_indices_from(cc)]
        cupy.testing.assert_allclose(ccu, rru, rtol=tol, atol=tol)
        cupy.testing.assert_allclose(ccl, rrl, rtol=tol, atol=tol)


@testing.parameterize(*testing.product({
    'mnk': [(8, 9, 10), (10, 9, 8)],
    'transa': ['N', 'T', 'H'],
    'transb': ['N', 'T', 'H'],
    'ordera': ['C', 'F'],
    'orderb': ['C', 'F'],
    'orderc': ['C', 'F'],
    'mode': [None, numpy, cupy],
}))
@_attr.gpu
class TestGemmAndGeam:
    _tol = {'f': 1e-5, 'd': 1e-12}

    def _make_matrix(self, m, n, trans, order, dtype):
        if trans == 'N':
            shape = (m, n)
        else:
            shape = (n, m)
        return testing.shaped_random(shape, cupy, dtype=dtype, order=order,
                                     scale=1.0)

    def _trans_matrix(self, a, trans):
        if trans == 'T':
            a = a.T
        elif trans == 'H':
            a = a.T.conj()
        return a

    @testing.for_dtypes('fdFD')
    def test_gemm(self, dtype):
        if not (self.mode is None and self.orderc == 'C'):
            pytest.skip()
        dtype = numpy.dtype(dtype)
        tol = self._tol[dtype.char.lower()]
        m, n, k = self.mnk
        a = self._make_matrix(m, k, self.transa, self.ordera, dtype)
        b = self._make_matrix(k, n, self.transb, self.orderb, dtype)
        aa = self._trans_matrix(a, self.transa)
        bb = self._trans_matrix(b, self.transb)
        ref = aa.dot(bb)
        c = cublas.gemm(self.transa, self.transb, a, b)
        cupy.testing.assert_allclose(c, ref, rtol=tol, atol=tol)

    @testing.for_dtypes('fdFD')
    def test_gemm_out(self, dtype):
        dtype = numpy.dtype(dtype)
        tol = self._tol[dtype.char.lower()]
        m, n, k = self.mnk
        a = self._make_matrix(m, k, self.transa, self.ordera, dtype)
        b = self._make_matrix(k, n, self.transb, self.orderb, dtype)
        c = self._make_matrix(m, n, 'N', self.orderc, dtype)
        alpha = 0.9
        beta = 0.8
        if dtype.char in 'FD':
            alpha = alpha - 1j * 0.7
            beta = beta - 1j * 0.6
        aa = self._trans_matrix(a, self.transa)
        bb = self._trans_matrix(b, self.transb)
        ref = alpha * aa.dot(bb) + beta * c
        if self.mode is not None:
            alpha = self.mode.array(alpha)
            beta = self.mode.array(beta)
        cublas.gemm(self.transa, self.transb, a, b, out=c,
                    alpha=alpha, beta=beta)
        cupy.testing.assert_allclose(c, ref, rtol=tol, atol=tol)

    @testing.for_dtypes('fdFD')
    def test_geam(self, dtype):
        if self.orderc != 'F':
            pytest.skip()
        dtype = numpy.dtype(dtype)
        tol = self._tol[dtype.char.lower()]
        m, n, _ = self.mnk
        a = self._make_matrix(m, n, self.transa, self.ordera, dtype)
        b = self._make_matrix(m, n, self.transb, self.orderb, dtype)
        alpha = 0.9
        beta = 0.8
        if dtype.char in 'FD':
            alpha = alpha - 1j * 0.7
            beta = beta - 1j * 0.6
        aa = self._trans_matrix(a, self.transa)
        bb = self._trans_matrix(b, self.transb)
        ref = alpha * aa + beta * bb
        if self.mode is not None:
            alpha = self.mode.array(alpha)
            beta = self.mode.array(beta)
        c = cublas.geam(self.transa, self.transb, alpha, a, beta, b)
        cupy.testing.assert_allclose(c, ref, rtol=tol, atol=tol)

    @testing.for_dtypes('fdFD')
    def test_geam_out(self, dtype):
        dtype = numpy.dtype(dtype)
        tol = self._tol[dtype.char.lower()]
        m, n, _ = self.mnk
        a = self._make_matrix(m, n, self.transa, self.ordera, dtype)
        b = self._make_matrix(m, n, self.transb, self.orderb, dtype)
        c = self._make_matrix(m, n, 'N', self.orderc, dtype)
        alpha = 0.9
        beta = 0.8
        if dtype.char in 'FD':
            alpha = alpha - 1j * 0.7
            beta = beta - 1j * 0.6
        aa = self._trans_matrix(a, self.transa)
        bb = self._trans_matrix(b, self.transb)
        ref = alpha * aa + beta * bb
        if self.mode is not None:
            alpha = self.mode.array(alpha)
            beta = self.mode.array(beta)
        cublas.geam(self.transa, self.transb, alpha, a, beta, b, out=c)
        cupy.testing.assert_allclose(c, ref, rtol=tol, atol=tol)


@testing.parameterize(*testing.product({
    'shape': [(9, 10), (10, 9)],
    'side': ['L', 'R'],
    'ordera': ['C', 'F'],
    'orderc': ['C', 'F'],
}))
@_attr.gpu
class TestDgmm:
    _tol = {'f': 1e-5, 'd': 1e-12}

    def _setup(self, dtype, xdim=1):
        self.dtype = numpy.dtype(dtype)
        self.tol = self._tol[self.dtype.char.lower()]
        self.a = testing.shaped_random(self.shape, cupy, dtype=dtype,
                                       order=self.ordera, scale=1.0)
        if self.side == 'L':
            xlen = self.shape[0]
        elif self.side == 'R':
            xlen = self.shape[1]
        if xdim == 0:
            self.x = cupy.array(1.1, dtype=dtype)
        elif xdim == 1:
            self.x = testing.shaped_random(
                (xlen,), cupy, dtype=dtype, scale=1.0)
        elif xdim == 2:
            self.x = testing.shaped_random(
                (xlen, xlen), cupy, dtype=dtype, scale=1.0)

    @testing.for_dtypes('fdFD')
    def test_dgmm(self, dtype):
        if self.orderc != 'F':
            pytest.skip()
        self._setup(dtype)
        if self.side == 'L':
            ref = cupy.diag(self.x) @ self.a
        elif self.side == 'R':
            ref = self.a @ cupy.diag(self.x)
        c = cublas.dgmm(self.side, self.a, self.x)
        cupy.testing.assert_allclose(c, ref, rtol=self.tol, atol=self.tol)

    @testing.for_dtypes('fdFD')
    def test_dgmm_out(self, dtype):
        self._setup(dtype)
        if self.side == 'L':
            ref = cupy.diag(self.x) @ self.a
        elif self.side == 'R':
            ref = self.a @ cupy.diag(self.x)
        c = cupy.empty(self.shape, order=self.orderc, dtype=dtype)
        cublas.dgmm(self.side, self.a, self.x, out=c)
        cupy.testing.assert_allclose(c, ref, rtol=self.tol, atol=self.tol)

    @testing.for_dtypes('fdFD')
    def test_dgmm_inplace(self, dtype):
        if self.orderc != 'F':
            pytest.skip()
        self._setup(dtype)
        if self.side == 'L':
            ref = cupy.diag(self.x) @ self.a
        elif self.side == 'R':
            ref = self.a @ cupy.diag(self.x)
        cublas.dgmm(self.side, self.a, self.x, out=self.a)
        cupy.testing.assert_allclose(self.a, ref, rtol=self.tol, atol=self.tol)

    _dgmm_incx_minus_one_hip_skip_condition = [
        ('C', 'F', (9, 10), 'R'),
        ('C', 'F', (10, 9), 'R'),
        ('F', 'F', (9, 10), 'L'),
        ('F', 'F', (10, 9), 'L'),
    ]

    def _check_dgmm_incx_minus_one_hip_skip_condition(self):
        return (self.ordera, self.orderc, self.shape, self.side) in \
            self._dgmm_incx_minus_one_hip_skip_condition

    @testing.for_dtypes('fdFD')
    def test_dgmm_incx_minus_one(self, dtype):
        if self.orderc != 'F':
            pytest.skip()
        if cupy.cuda.runtime.is_hip:
            if self._check_dgmm_incx_minus_one_hip_skip_condition():
                pytest.xfail('HIP dgmm may have a bug')
        self._setup(dtype)
        if self.side == 'L':
            ref = cupy.diag(self.x[::-1]) @ self.a
        elif self.side == 'R':
            ref = self.a @ cupy.diag(self.x[::-1])
        c = cublas.dgmm(self.side, self.a, self.x, incx=-1)
        cupy.testing.assert_allclose(c, ref, rtol=self.tol, atol=self.tol)

    @testing.for_dtypes('fdFD')
    def test_dgmm_x_scalar(self, dtype):
        if self.orderc != 'F':
            pytest.skip()
        self._setup(dtype, xdim=0)
        ref = self.x * self.a
        c = cublas.dgmm(self.side, self.a, self.x, incx=0)
        cupy.testing.assert_allclose(c, ref, rtol=self.tol, atol=self.tol)

    @testing.for_dtypes('fdFD')
    def test_dgmm_x_matrix(self, dtype):
        if self.orderc != 'F':
            pytest.skip()
        self._setup(dtype, xdim=2)
        if self.side == 'L':
            ref = cupy.diag(cupy.diag(self.x)) @ self.a
            incx = self.shape[0] + 1
        elif self.side == 'R':
            ref = self.a @ cupy.diag(cupy.diag(self.x))
            incx = self.shape[1] + 1
        c = cublas.dgmm(self.side, self.a, self.x, incx=incx)
        cupy.testing.assert_allclose(c, ref, rtol=self.tol, atol=self.tol)
