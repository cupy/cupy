import unittest

from chainer import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr
import numpy

class UniformBase(object):

    shape = (2, 3, 4)

    def check_initializer(self, w):
        self.initializer(w)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, numpy.float32)
        xp = cuda.get_array_module(w)
        self.assertIsInstance(w, xp.ndarray)

    def test_initializer_cpu(self):
        w = numpy.empty(self.shape, dtype=numpy.float32)
        self.check_initializer(w)

    @attr.gpu
    def test_initializer_gpu(self):
        w = cuda.cupy.empty(self.shape, dtype=numpy.float32)
        self.check_initializer(w)


class TestUniform(unittest.TestCase, UniformBase):

    def setUp(self):
        self.initializer = initializers.Uniform(scale=0.1)


class TestLeCunUniform(unittest.TestCase, UniformBase):

    def setUp(self):
        self.initializer = initializers.LeCunUniform(scale=0.1)


class TestGlorotUniform(unittest.TestCase, UniformBase):

    def setUp(self):
        self.initializer = initializers.GlorotUniform(scale=0.1)


class TestHeUniform(unittest.TestCase, UniformBase):

    def setUp(self):
        self.initializer = initializers.HeUniform(scale=0.1)


testing.run_module(__name__, __file__)
