import unittest

import mock

from chainer import testing
from chainer.training import extensions


class TestLinearShift(unittest.TestCase):

    value_range = (2.0, 6.0)
    time_range = (1, 3)
    expect = [2.0, 2.0, 4.0, 6.0, 6.0]

    def setUp(self):
        self.optimizer = mock.MagicMock()
        self.trainer = mock.MagicMock()
        self.extension = extensions.LinearShift(
            'x', self.value_range, self.time_range, self.optimizer)

    def test_call(self):
        for e in self.expect:
            self.extension(self.trainer)
            self.assertEqual(self.optimizer.x, e)


testing.run_module(__name__, __file__)
