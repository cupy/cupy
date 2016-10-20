import operator
import sys
import unittest

import numpy

import cupy

from cupy import testing


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            ((3, 2), (2, 4)),
            ((2,), (2, 4)),
            ((3, 2), (2,)),
            ((2,), (2,)),
            ((5, 3, 2), (5, 2, 4)),
            ((5, 3, 2), (2, 4)),
            ((3, 2), (5, 2, 4)),
            ((5, 3, 2), (1, 2, 4)),
            ((1, 3, 2), (5, 2, 4)),
            ((5, 3, 2), (2,)),
            ((2,), (5, 2, 4)),
            ((6, 5, 3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (6, 1, 2, 4)),
            ((6, 5, 3, 2), (1, 5, 2, 4)),
            ((6, 5, 3, 2), (1, 1, 2, 4)),
            ((6, 1, 3, 2), (6, 5, 2, 4)),
            ((1, 5, 3, 2), (6, 5, 2, 4)),
            ((1, 1, 3, 2), (6, 5, 2, 4)),
            ((3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (2, 4)),
            ((2,), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (2,)),
        ],
    }))
@testing.gpu
class TestMatmul(unittest.TestCase):

    # _multiprocess_can_split_ = True

    def setUp(self):
        self.x1 = numpy.random.randn(*self.shape_pair[0])
        self.x2 = numpy.random.randn(*self.shape_pair[1])

    @unittest.skipUnless(sys.version_info >= (3, 5),
                         'Only for Python3.5 or higher')
    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_almost_equal(5)  # required for uint8
    def test_operator_matmul(
            self, xp, dtype1=numpy.float32, dtype2=numpy.float32):
        if not numpy.result_type(dtype1, dtype2) == numpy.float32:
            return xp.array([])
        x1 = xp.array(self.x1, dtype=dtype1)
        x2 = xp.array(self.x2, dtype=dtype2)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_almost_equal(5)  # required for uint8
    def test_cupy_matmul(
            self, xp, dtype1=numpy.float32, dtype2=numpy.float32):
        if not numpy.result_type(dtype1, dtype2) == numpy.float32:
            return xp.array([])
        x1 = xp.array(self.x1, dtype=dtype1)
        x2 = xp.array(self.x2, dtype=dtype2)
        if xp == numpy:
            return numpy.matmul(x1, x2)
        else:
            return cupy.matmul(x1, x2)
