import unittest

from chainer import functions as F
from chainer import testing


class TestNoNumpyFunction(unittest.TestCase):

    def test_no_numpy_function(self):
        with self.assertRaises(ValueError):
            testing.math_function_test(F.rsqrt)  # no numpy.rsqrt


class TestInvalidExpectedLabel(unittest.TestCase):

    def test_invalid_expected_label(self):
        with self.assertRaises(ValueError):
            # no numpy.rsqrt
            testing.math_function_test(F.rsqrt, label_expected="FOO")


testing.run_module(__name__, __file__)
