import unittest

import numpy

from chainer import testing
from chainer import utils


class TestForceArray(unittest.TestCase):

    def test_scalar(self):
        x = utils.force_array(numpy.float32(1))
        self.assertIsInstance(x, numpy.ndarray)
        self.assertEqual(x.dtype, numpy.float32)

    def test_0dim_array(self):
        x = utils.force_array(numpy.array(1, numpy.float32))
        self.assertIsInstance(x, numpy.ndarray)
        self.assertEqual(x.dtype, numpy.float32)

    def test_array(self):
        x = utils.force_array(numpy.array([1], numpy.float32))
        self.assertIsInstance(x, numpy.ndarray)
        self.assertEqual(x.dtype, numpy.float32)


testing.run_module(__name__, __file__)
