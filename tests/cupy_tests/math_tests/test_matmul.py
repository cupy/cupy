import operator
import sys
import unittest

import numpy

from cupy import testing


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            ((3, 2), (2, 4)),
            ((3, 0), (0, 4)),
            ((0, 2), (2, 4)),
            ((3, 2), (2, 0)),
            ((2,), (2, 4)),
            ((0,), (0, 4)),
            ((3, 2), (2,)),
            ((3, 0), (0,)),
            ((2,), (2,)),
            ((0,), (0,)),
            ((5, 3, 2), (5, 2, 4)),
            ((0, 3, 2), (0, 2, 4)),
            ((5, 3, 2), (2, 4)),
            ((0, 3, 2), (2, 4)),
            ((3, 2), (5, 2, 4)),
            ((3, 2), (0, 2, 4)),
            ((5, 3, 2), (1, 2, 4)),
            ((0, 3, 2), (1, 2, 4)),
            ((1, 3, 2), (5, 2, 4)),
            ((1, 3, 2), (0, 2, 4)),
            ((5, 3, 2), (2,)),
            ((5, 3, 0), (0,)),
            ((2,), (5, 2, 4)),
            ((0,), (5, 0, 4)),
        ],
    }))
@testing.gpu
class TestMatmul(unittest.TestCase):

    @unittest.skipUnless(sys.version_info >= (3, 5),
                         'Only for Python3.5 or higher')
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return operator.matmul(x1, x2)

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'shape_pair': [
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
class TestMatmulLarge(unittest.TestCase):

    # Avoid overflow
    skip_dtypes = {
        (numpy.int8, numpy.uint8),
        (numpy.int8, numpy.int16),
        (numpy.int8, numpy.float16),
        (numpy.uint8, numpy.uint8),
        (numpy.uint8, numpy.int16),
        (numpy.uint8, numpy.uint16),
        (numpy.int16, numpy.int16),
        (numpy.uint16, numpy.uint16),
    }

    @unittest.skipUnless(sys.version_info >= (3, 5),
                         'Only for Python3.5 or higher')
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1, dtype2):
        if ((dtype1, dtype2) in self.skip_dtypes or
                (dtype2, dtype1) in self.skip_dtypes):
            return xp.array([])
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return operator.matmul(x1, x2)

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1, dtype2):
        if ((dtype1, dtype2) in self.skip_dtypes or
                (dtype2, dtype1) in self.skip_dtypes):
            return xp.array([])
        shape1, shape2 = self.shape_pair
        x1 = testing.shaped_arange(shape1, xp, dtype1)
        x2 = testing.shaped_arange(shape2, xp, dtype2)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            ((5, 3, 1), (3, 1, 4)),
            ((3, 2, 3), (3, 2, 4)),
            ((3, 2), ()),
            ((), (3, 2)),
            ((), ()),
            ((3, 2), (1,)),
            ((0, 2), (3, 0)),
            ((0, 1, 1), (2, 1, 1)),
        ],
    }))
@testing.gpu
class TestMatmulInvalidShape(unittest.TestCase):

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_invalid_shape(self, xp):
        shape1, shape2 = self.shape_pair
        x1 = testing.shaped_arange(shape1, xp, numpy.float32)
        x2 = testing.shaped_arange(shape2, xp, numpy.float32)
        xp.matmul(x1, x2)
