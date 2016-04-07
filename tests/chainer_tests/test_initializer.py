import unittest

from chainer import initializer
from chainer import testing


@testing.parameterize(
    {'shape': (2, 1), 'expect': (1, 2)},
    {'shape': (2, 3, 4), 'expect': (12, 2)})
class TestGetFans(unittest.TestCase):

    def test_get_fans(self):
        actual = initializer.get_fans(self.shape)
        self.assertTupleEqual(self.expect, actual)


@testing.parameterize(
    {'shape': ()},
    {'shape': (2,)})
class TestGetFansInvalid(unittest.TestCase):

    def test_invalid(self):
        with self.assertRaises(ValueError):
            initializer.get_fans(self.shape)


testing.run_module(__name__, __file__)
