import unittest

import numpy

from chainer import cuda
from chainer import testing


class TestCuda(unittest.TestCase):

    def test_get_dummy_device(self):
        if not cuda.available:
            self.assertIs(cuda.get_device(), cuda.DummyDevice)

    def test_to_gpu_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.to_gpu(x)

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
