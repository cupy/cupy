import unittest

import numpy

import chainer.functions as F
from chainer import testing


# sqrt

def make_data(shape, dtype):
    x = numpy.random.uniform(0.1, 5, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    return x, gy


@testing.unary_math_function_unittest(F.Sqrt(), make_data=make_data)
class TestSqrt(unittest.TestCase):
    pass


# rsqrt

def rsqrt(x):
    return numpy.reciprocal(numpy.sqrt(x))


class TestRsqrt(unittest.TestCase):

    def test_rsqrt(self):
        x = numpy.random.uniform(0.1, 5, (3, 2)).astype(numpy.float32)
        testing.assert_allclose(F.rsqrt(x).data, rsqrt(x))


testing.run_module(__name__, __file__)
