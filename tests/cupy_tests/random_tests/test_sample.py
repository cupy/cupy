import mock
import unittest

from cupy import cuda
from cupy import testing
from cupy import random


@testing.gpu
class TestRandint(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        device_id = cuda.Device().id
        self.m = mock.Mock()
        self.m.interval.return_value = 0
        random.generator._random_states = {device_id : self.m}

    def test_value_error(self):
        with self.assertRaises(ValueError):
            random.randint(100, 1)

    def test_high_and_size_are_none(self):
        random.randint(3)
        self.m.interval.assert_called_with(3, None)

    def test_size_is_none(self):
        random.randint(3, 5)
        self.m.interval.assert_called_with(2, None)

    def test_high_is_none(self):
        random.randint(3, None, (1, 2, 3))
        self.m.interval.assert_called_with(3, (1, 2, 3))

    def test_no_none(self):
        random.randint(3, 5, (1, 2, 3))
        self.m.interval.assert_called_with(2, (1, 2, 3))


