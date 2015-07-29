import unittest

import numpy

from chainer import cuda
from chainer import testing


class TestCuda(unittest.TestCase):

    def test_init_unavailable(self):
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.init()

    def test_to_gpu_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.to_gpu(x)

    def test_to_gpu_async_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.to_gpu_async(x)

    def test_empy_unavailable(self):
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.empty(())

    def test_empy_like_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.empty_like(x)


testing.run_module(__name__, __file__)
