import unittest

import numpy

from cupy.cuda import cusolver_enabled
from cupy import testing


@unittest.skipUnless(
    cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestCholeskyDecomposition(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_L_float(self, array, xp, dtype):
        a = xp.asarray(array, dtype=dtype)
        return xp.linalg.cholesky(a)

    def test_float_dtypes(self):
        # A normal positive definite matrix
        A = numpy.random.randn(5, 5)
        A = numpy.dot(A, A.transpose())
        self.check_L_float(A)
        # np.linalg.cholesky only uses a lower triangle of an array
        self.check_L_float(numpy.array([[1, 2], [1, 9]]))

    @testing.for_dtypes([numpy.int32, numpy.int64, numpy.uint32, numpy.uint64])
    @testing.numpy_cupy_allclose(atol=1e-4)
    def check_L_int(self, array, xp, dtype):
        a = xp.asarray(array, dtype=dtype)
        return xp.linalg.cholesky(a)

    def test_int_dtypes(self):
        A = numpy.random.randint(0, 100, size=(5, 5))
        A = numpy.dot(A, A.transpose())
        self.check_L_int(A)
        self.check_L_int(numpy.array([[1, 2], [1, 9]]))
