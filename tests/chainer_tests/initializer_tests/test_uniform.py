import unittest

from chainer import initializers
from chainer import testing


class TestUniform(unittest.TestCase):

    def setUp(self):
        self.initializer = initializers.Uniform(scale=0.1)
        self.shape = (2, 3, 4)

    def test_shape(self):
        w = self.initializer(self.shape)
        self.assertTupleEqual(w.shape, self.shape)



testing.run_module(__name__, __file__)
