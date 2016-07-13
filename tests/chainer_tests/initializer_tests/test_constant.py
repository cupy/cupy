import unittest

from chainer import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr
import numpy


class TestIdentity(unittest.TestCase):

    def setUp(self):
        self.scale = 0.1
        self.shape = (2, 2)
        self.initializer = initializers.Identity(scale=self.scale)
        self.w = numpy.empty((2, 2), dtype=numpy.float32)

    def check_initializer(self, w):
        self.initializer(w)
        xp = cuda.get_array_module(w)
        self.assertIsInstance(w, xp.ndarray)
        testing.assert_allclose(
            w, self.scale * numpy.identity(len(self.shape)))

    def test_initializer_cpu(self):
        self.check_initializer(self.w)

    @attr.gpu
    def test_initializer_gpu(self):
        self.check_initializer(cuda.to_gpu(self.w))


@testing.parameterize(
    {'shape': (2, 3)},
    {'shape': (2, 2, 4)},
    {'shape': ()},
    {'shape': 0})
class TestIdentityInvalid(unittest.TestCase):

    def setUp(self):
        self.initializer = initializers.Identity()

    def test_invalid_shape(self):
        w = numpy.empty(self.shape)
        with self.assertRaises(ValueError):
            self.initializer(w)


class TestConstant(unittest.TestCase):

    def setUp(self):
        self.fill_value = 0.1
        self.initializer = initializers.Constant(fill_value=self.fill_value)
        self.shape = (2, 3)
        self.w = numpy.empty(self.shape, dtype=numpy.float32)

    def check_initializer(self, w):
        self.initializer(w)
        xp = cuda.get_array_module(w)
        self.assertIsInstance(w, xp.ndarray)
        testing.assert_allclose(
            w, numpy.full(self.shape, self.fill_value))

    def test_initializer_cpu(self):
        self.check_initializer(self.w)

    @attr.gpu
    def test_initializer_gpu(self):
        self.check_initializer(cuda.to_gpu(self.w))


testing.run_module(__name__, __file__)
