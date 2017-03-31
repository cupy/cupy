import unittest

import numpy

from cupy import cuda
from cupy import testing


@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestCholeskyDecomposition(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_dtypes([
        numpy.int32, numpy.int64, numpy.uint32, numpy.uint64,
        numpy.float32, numpy.float64])
    @testing.numpy_cupy_allclose(atol=1e-3)
    def check_L(self, array, xp, dtype):
        a = xp.asarray(array, dtype=dtype)
        return xp.linalg.cholesky(a)

    def test_decomposition(self):
        # A normal positive definite matrix
        A = numpy.random.randint(0, 100, size=(5, 5))
        A = numpy.dot(A, A.transpose())
        self.check_L(A)
        # np.linalg.cholesky only uses a lower triangle of an array
        self.check_L(numpy.array([[1, 2], [1, 9]]))
