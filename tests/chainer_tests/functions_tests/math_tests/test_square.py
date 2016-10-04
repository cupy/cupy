import unittest

import numpy

import chainer.functions as F
from chainer import testing


def make_data(shape, dtype):
    x = numpy.random.uniform(0.1, 5, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    return x, gy


@testing.unary_math_function_test(F.Square(), make_data=make_data)
class TestSquare(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
