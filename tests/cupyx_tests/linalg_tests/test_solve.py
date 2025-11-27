from __future__ import annotations

import unittest

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupyx import cusolver
from cupy import testing
import cupyx


@testing.parameterize(*testing.product({
    'size': [5, 9, 17, 33],
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@pytest.mark.xfail(runtime.is_hip,
                   reason='rocSOLVER does not implement potrs yet.')
class TestInvh(unittest.TestCase):

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_invh(self, xp):
        a = self._create_symmetric_matrix(xp, self.size, self.dtype)
        if xp == cupy:
            return cupyx.linalg.invh(a)
        else:
            return numpy.linalg.inv(a)

    def _create_symmetric_matrix(self, xp, n, dtype):
        if dtype == numpy.complex128:
            f_dtype = numpy.float64
        elif dtype == numpy.complex64:
            f_dtype = numpy.float32
        else:
            f_dtype = dtype
        a = testing.shaped_random((n, n), xp, f_dtype, scale=1)
        a = a + a.T + xp.eye(n, dtype=f_dtype) * n
        if dtype in (numpy.complex64, numpy.complex128):
            b = testing.shaped_random((n, n), xp, f_dtype, scale=1)
            b = b - b.T
            a = a + 1j * b
        return a


@testing.parameterize(*testing.product({
    'size': [8],
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
class TestErrorInvh(unittest.TestCase):

    @pytest.mark.skipif(
        cupy.cuda.runtime.runtimeGetVersion() == 12000,
        reason='This fails with CUDA 12.0.0 but pass in CUDA 12.0.1. (#7309)')
    def test_invh(self):
        a = self._create_symmetric_matrix(self.size, self.dtype)
        with cupyx.errstate(linalg='raise'):
            with self.assertRaises(numpy.linalg.LinAlgError):
                cupyx.linalg.invh(a)

    def _create_symmetric_matrix(self, n, dtype):
        a = testing.shaped_random((n, n), cupy, dtype, scale=1)
        a = a + a.T - cupy.eye(n, dtype=dtype)
        return a


# TODO: cusolver does not support nrhs > 1 for potrsBatched
@testing.parameterize(*testing.product({
    'shape': [(2, 3, 3)],
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
class TestXFailBatchedInvh(unittest.TestCase):

    def test_invh(self):
        if not cusolver.check_availability('potrsBatched'):
            pytest.skip('potrsBatched is not available')
        a = self._create_symmetric_matrix(self.shape, self.dtype)
        with cupyx.errstate(linalg='raise'):
            with self.assertRaises(numpy.linalg.LinAlgError):
                cupyx.linalg.invh(a)

    def _create_symmetric_matrix(self, shape, dtype):
        a = testing.shaped_random(shape, cupy, dtype, scale=1)
        a = a @ a.transpose(0, 2, 1)
        return a


@testing.parameterize(*testing.product({
    'shape': [(3, 4, 2, 2), (5, 3, 3), (7, 7)],
    'nrhs': [None, 1, 8],
    'lower': [True, False],
    'order': ['C', 'F'],
    'overwrite_b': [True, False]
}))
class TestChosolve(unittest.TestCase):

    @staticmethod
    def _solve(a, b):
        if (
            numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0"
            or a.shape[:-1] != b.shape
        ):
            return numpy.linalg.solve(a, b)
        b = b[..., numpy.newaxis]
        return numpy.linalg.solve(a, b)[..., 0]

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(atol=1e-5, contiguous_check=False)
    def test_chosolve(self, xp, dtype):

        if (len(self.shape) > 2 and
                not cusolver.check_availability('potrsBatched')):
            pytest.skip('potrsBatched is not available')

        a = self._create_posdef_matrix(xp, self.shape, dtype)
        b_shape = list(self.shape[:-1])
        if self.nrhs is not None:
            b_shape.append(self.nrhs)
        b = testing.shaped_random(b_shape, xp, dtype=dtype)

        if xp == cupy:
            L = xp.linalg.cholesky(a)
            if self.lower:
                indexes = numpy.triu_indices(L.shape[-1], 1)
            else:
                L = xp.moveaxis(L, -1, -2).conj()
                indexes = numpy.tril_indices(L.shape[-1], -1)
            L[..., indexes[0], indexes[1]] = numpy.nan
            L = cupy.asarray(L, order=self.order)
            return cupyx.linalg.cho_solve(
                (L, self.lower), b, overwrite_b=self.overwrite_b)
        else:
            return self._solve(a, b)

    @testing.for_dtypes('fdFD')
    def test_chosolve_does_not_overwrite_b(self, dtype):

        if (len(self.shape) > 2 and
                not cusolver.check_availability('potrsBatched')):
            pytest.skip('potrsBatched is not available')

        a = self._create_posdef_matrix(cupy, self.shape, dtype)
        b_shape = list(self.shape[:-1])
        if self.nrhs is not None:
            b_shape.append(self.nrhs)
        b = testing.shaped_random(b_shape, cupy, dtype=dtype)

        L = cupy.linalg.cholesky(a)
        if self.lower:
            indexes = numpy.triu_indices(L.shape[-1], 1)
        else:
            L = cupy.moveaxis(L, -1, -2).conj()
            indexes = numpy.tril_indices(L.shape[-1], -1)
        L[..., indexes[0], indexes[1]] = numpy.nan
        L = cupy.asarray(L, order=self.order)
        b_copy = cupy.asarray(b, copy=True)
        cupyx.linalg.cho_solve((L, self.lower), b, overwrite_b=False)
        assert cupy.all(b == b_copy).get()

    def _create_posdef_matrix(self, xp, shape, dtype):
        n = shape[-1]
        a = testing.shaped_random(shape, xp, dtype, scale=1)
        a = a @ a.swapaxes(-2, -1).conjugate()
        a = a + n * xp.eye(n)
        return a
