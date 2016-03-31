import unittest

from chainer import initializers
from chainer import testing


class UniformBase(object):

    shape = (2, 3, 4)

    def test_shape(self):
        w = self.initializer(self.shape)
        self.assertTupleEqual(w.shape, self.shape)


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
