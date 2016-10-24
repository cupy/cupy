import unittest

import mock

from chainer import testing
from chainer.training import extensions


@testing.parameterize(
    {'init': 2.0, 'rate': 0.5, 'target': None, 'expect': [2.0, 1.0, 0.5]},
    {'init': 2.0, 'rate': 0.5, 'target': 1.2, 'expect': [2.0, 1.2, 1.2]},
    {'init': -2.0, 'rate': 0.5, 'target': -1.2, 'expect': [-2.0, -1.2, -1.2]},
    {'init': 2.0, 'rate': 2.0, 'target': None, 'expect': [2.0, 4.0, 8.0]},
    {'init': 2.0, 'rate': 2.0, 'target': 3.0, 'expect': [2.0, 3.0, 3.0]},
    {'init': -2.0, 'rate': 2.0, 'target': -3.0, 'expect': [-2.0, -3.0, -3.0]},
)
class TestExponentialShift(unittest.TestCase):

    def setUp(self):
        self.optimizer = mock.MagicMock()
        self.trainer = mock.MagicMock()
        self.extension = extensions.ExponentialShift(
            'x', self.rate, self.init, self.target, self.optimizer)

    def test_call(self):
        for e in self.expect:
            self.extension(self.trainer)
            self.assertEqual(self.optimizer.x, e)


class TestExponentialShiftInvalidArgument(unittest.TestCase):

    def test_negative_rate(self):
        with self.assertRaises(ValueError):
            extensions.ExponentialShift('x', -1.0)


testing.run_module(__name__, __file__)
