import unittest

from chainer import functions as F
from chainer import testing


class TestNoNumpyFunction(unittest.TestCase):

    def test_no_numpy_function(self):
        with self.assertRaises(ValueError):
            testing.unary_function_test(F.rsqrt)  # no numpy.rsqrt


testing.run_module(__name__, __file__)
