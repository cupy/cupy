from __future__ import annotations

import unittest

import numpy
import pytest
try:
    import scipy.linalg
except ImportError:
    pass

import cupy
from cupy import testing
from cupy.cuda import driver
from cupy.cuda import runtime
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import _lobpcg


def _random_array(shape, dtype, seed):
    rng = numpy.random.default_rng(seed)
    a = rng.standard_normal(shape)
    if numpy.dtype(dtype).kind == 'c':
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'generalized': [False, True],
    'precomputed': [False, True],
}))
class TestLOBPCGOrthonormalize(unittest.TestCase):

    def test_orthonormalize(self):
        n, k = 12, 3
        x = cupy.asarray(_random_array((n, k), self.dtype, 119))
        z = cupy.asarray(_random_array((n, n), self.dtype, 120))
        b = z.conj().T @ z / n + cupy.eye(n, dtype=self.dtype)
        operator = (lambda v: b @ v) if self.generalized else None
        bx = b @ x if self.generalized else x
        q, bq, inv_r, normalization = _lobpcg._b_orthonormalize(
            operator, x, bx if self.precomputed else None, retInvR=True)

        tol = 50 * numpy.finfo(self.dtype).eps
        expected_bq = b @ q if self.generalized else q
        testing.assert_allclose(
            q.conj().T @ expected_bq, cupy.eye(k), rtol=tol, atol=tol)
        testing.assert_allclose(
            (x / normalization) @ inv_r, q, rtol=tol, atol=tol)
        if self.generalized:
            testing.assert_allclose(bq, expected_bq, rtol=tol, atol=tol)
        else:
            assert bq is None


@testing.with_requires('scipy')
@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'generalized': [False, True],
}))
@pytest.mark.skipif(runtime.is_hip and driver.get_build_version() < 402,
                    reason='syevj not available')
class TestLOBPCGDenseEigh(unittest.TestCase):

    def test_eigh(self):
        n = 12
        z = _random_array((n, n), self.dtype, 119)
        a = z + z.conj().T
        z = _random_array((n, n), self.dtype, 120)
        b = z.conj().T @ z / n + numpy.eye(n, dtype=self.dtype)
        a_gpu = cupy.asarray(a)
        b_gpu = cupy.asarray(b) if self.generalized else None
        w, v = _lobpcg._eigh(a_gpu, b_gpu)

        expected = scipy.linalg.eigh(
            a, b if self.generalized else None, eigvals_only=True)
        tol = 100 * numpy.finfo(self.dtype).eps
        testing.assert_allclose(w, expected, rtol=tol, atol=tol)
        bv = b_gpu @ v if self.generalized else v
        testing.assert_allclose(a_gpu @ v, bv * w, rtol=tol, atol=tol)
        testing.assert_allclose(
            v.conj().T @ bv, cupy.eye(n), rtol=tol, atol=tol)


@testing.with_requires('scipy')
@testing.parameterize(*testing.product({
    'dtype': [numpy.complex64, numpy.complex128],
    'n': [12, 48],
    'generalized': [False, True],
    'largest': [False, True],
}))
@pytest.mark.skipif(runtime.is_hip and driver.get_build_version() < 402,
                    reason='syevj not available')
class TestLOBPCGComplex(unittest.TestCase):

    def test_eigenpairs(self):
        n, k = self.n, 3
        # Both matrices have complex off-diagonals; B is positive definite.
        a = numpy.diag(numpy.linspace(1, 4, n)).astype(self.dtype)
        off_diagonal = numpy.full(n - 1, 0.05 + 0.1j, dtype=self.dtype)
        a += numpy.diag(off_diagonal, 1)
        a += numpy.diag(off_diagonal.conj(), -1)
        b = 2 * numpy.eye(n, dtype=self.dtype)
        off_diagonal = numpy.full(n - 1, 0.05 - 0.03j, dtype=self.dtype)
        b += numpy.diag(off_diagonal, 1)
        b += numpy.diag(off_diagonal.conj(), -1)
        a_gpu = sparse.csr_matrix(cupy.asarray(a))
        b_gpu = sparse.csr_matrix(cupy.asarray(
            b)) if self.generalized else None
        x = cupy.asarray(_random_array((n, k), self.dtype, 121))
        tol = 2e-4 if self.dtype == numpy.complex64 else 1e-8

        # n=12 exercises the dense fallback; n=48 uses the iterative solver.
        w, v = sparse.linalg.lobpcg(
            a_gpu, x, B=b_gpu, largest=self.largest, tol=tol, maxiter=100)

        expected = scipy.linalg.eigh(
            a, b if self.generalized else None, eigvals_only=True)
        expected = expected[-k:][::-1] if self.largest else expected[:k]
        eigenvalue_tol = 100 * numpy.finfo(self.dtype).eps
        testing.assert_allclose(
            w, expected, rtol=eigenvalue_tol, atol=eigenvalue_tol)
        bv = b_gpu @ v if self.generalized else v
        residuals = cupy.linalg.norm(a_gpu @ v - bv * w, axis=0)
        testing.assert_array_less(residuals, 5 * tol)
        testing.assert_allclose(
            v.conj().T @ bv, cupy.eye(k), rtol=eigenvalue_tol,
            atol=eigenvalue_tol)
