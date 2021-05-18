import unittest

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import cusolver
from cupy import testing
import cupyx


@testing.parameterize(*testing.product({
    'size': [5, 9, 17, 33],
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@testing.gpu
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
@testing.gpu
class TestErrorInvh(unittest.TestCase):

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
@testing.gpu
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
