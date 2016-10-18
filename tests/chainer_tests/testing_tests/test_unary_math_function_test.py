import unittest

from chainer import testing


class Dummy(object):
    pass


class TestNoNumpyFunction(unittest.TestCase):

    def test_no_numpy_function(self):
        with self.assertRaises(ValueError):
            testing.unary_math_function_unittest(Dummy())  # no numpy.dummy


testing.run_module(__name__, __file__)
