import unittest

from chainer import initializers
from chainer import testing


class NormalBase(object):

    shape = (2, 3, 4)

    def test_shape(self):
        w = self.initializer(self.shape)
        self.assertTupleEqual(w.shape, self.shape)


class TestNormal(unittest.TestCase, NormalBase):

    def setUp(self):
        self.initializer = initializers.Normal(scale=0.1)


class TestGlorotNormal(unittest.TestCase, NormalBase):

    def setUp(self):
        self.initializer = initializers.GlorotNormal(scale=0.1)


class TestHeNormal(unittest.TestCase, NormalBase):

    def setUp(self):
        self.initializer = initializers.HeNormal(scale=1.0)


testing.run_module(__name__, __file__)
