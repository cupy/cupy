import unittest

from chainer import initializers
from chainer import testing
from chainer import gradient_check
import numpy


class TestIdentity(unittest.TestCase):

    def setUp(self):
        self.scale = 0.1
        self.dim = 2
        self.shape = (self.dim, self.dim)
        self.initializer = initializers.Identity(scale=self.scale)

    def test_shape(self):
        w = self.initializer(self.shape)
        self.assertTupleEqual(w.shape, self.shape)

    def test_value(self):
        w = self.initializer(self.shape)
        gradient_check.assert_allclose(w, self.scale * numpy.identity(self.dim))

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            self.initializer((2, 3))

    def test_invalid_shape2(self):
        with self.assertRaises(ValueError):
            self.initializer((2, 2, 4))


class TestConstant(unittest.TestCase):

    def setUp(self):
        self.fill_value = 0.1
        self.initializer = initializers.Constant(fill_value=self.fill_value)
        self.shape = (2, 3, 4)
    
    def test_shape(self):
        w = self.initializer(self.shape)
        self.assertTupleEqual(w.shape, self.shape)

    def test_value(self):
        w = self.initializer(self.shape)
        gradient_check.assert_allclose(w, numpy.full(self.shape, self.fill_value))


testing.run_module(__name__, __file__)
