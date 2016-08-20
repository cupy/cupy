import unittest

import numpy

import chainer.functions as F
from chainer import testing


#
# sqrt

@testing.unary_function_test(F.sqrt)
class TestSqrt(unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy


#
# rsqrt

def rsqrt(x, dtype=numpy.float32):
    return numpy.reciprocal(numpy.sqrt(x, dtype=dtype))


@testing.unary_function_test(F.rsqrt, rsqrt)
class TestRsqrt(unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy


testing.run_module(__name__, __file__)
