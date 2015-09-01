import unittest

import numpy

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.utils import array


class TestFullLike(unittest.TestCase):

    def test_full_like_cpu(self):
        x = numpy.array([1, 2], numpy.float32)
        y = array.full_like(x, 3)
        self.assertIsInstance(y, numpy.ndarray)
        self.assertEqual(y.shape, (2,))
        self.assertEqual(y[0], 3)
        self.assertEqual(y[1], 3)

    @attr.gpu
    def test_full_like_gpu(self):
        x = cuda.cupy.array([1, 2], numpy.float32)
        y = array.full_like(x, 3)
        self.assertIsInstance(y, cuda.cupy.ndarray)
        y = cuda.to_cpu(y)
        self.assertEqual(y.shape, (2,))
        self.assertEqual(y[0], 3)
        self.assertEqual(y[1], 3)


testing.run_module(__name__, __file__)
