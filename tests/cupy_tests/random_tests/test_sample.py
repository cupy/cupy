import mock
import unittest

import numpy

from cupy import random
from cupy import testing


@testing.gpu
class TestSample(unittest.TestCase):

    _multiprocess_can_split_ = True

    def test_rand(self):
        random.sample_.random_sample = mock.Mock()
        random.rand(1, 2, 3, dtype=numpy.float32)
        random.sample_.random_sample.assert_called_once_with(
            size=(1, 2, 3), dtype=numpy.float32)

    def test_rand_default_dtype(self):
        random.sample_.random_sample = mock.Mock()
        random.rand(1, 2, 3)
        random.sample_.random_sample.assert_called_once_with(
            size=(1, 2, 3), dtype=float)

    def test_rand_invalid_argument(self):
        with self.assertRaises(TypeError):
            random.rand(1, 2, 3, unnecessary='unnecessary_argument')

    def test_randn(self):
        random.distributions.normal = mock.Mock()
        random.randn(1, 2, 3, dtype=numpy.float32)
        random.distributions.normal.assert_called_once_with(
            size=(1, 2, 3), dtype=numpy.float32)

    def test_randn_default_dtype(self):
        random.distributions.normal = mock.Mock()
        random.randn(1, 2, 3)
        random.distributions.normal.assert_called_once_with(
            size=(1, 2, 3), dtype=float)

    def test_randn_invalid_argument(self):
        with self.assertRaises(TypeError):
            random.randn(1, 2, 3, unnecessary='unnecessary_argument')
