import mock
import unittest

import numpy

from cupy import random
from cupy import testing


@testing.gpu
class TestSample(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        random.random_sample = mock.Mock()

    def test_rand(self):
        random.rand(1, 2, 3, dtype=numpy.float32)
        random.random_sample.assert_call_once_with((1, 2, 3), numpy.float32)

    def test_rand_invalid_argument(self):
        with self.assertRaises(TypeError):
            random.rand(1, 2, 3, unnecessary='unnecessary_argument')

    def test_randn(self):
        random.randn(1, 2, 3, dtype=numpy.float32)
        random.random_sample.assert_call_once_with((1, 2, 3), numpy.float32)

    def test_randn_invalid_argument(self):
        with self.assertRaises(TypeError):
            random.randn(1, 2, 3, unnecessary='unnecessary_argument')
