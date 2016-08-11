import unittest

from chainer import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr
import numpy


@testing.parameterize(*testing.product({
    'target': [
        initializers.Normal,
        initializers.GlorotNormal,
        initializers.HeNormal,
    ],
    'shape': [(2, 3), (2, 3, 4)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class NormalBase(unittest.TestCase):

    def setUp(self):
        pass

    def check_initializer(self, w):
        initializer = self.target(scale=0.1)
        initializer(w)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)

    def test_initializer_cpu(self):
        w = numpy.empty(self.shape, dtype=self.dtype)
        self.check_initializer(w)

    @attr.gpu
    def test_initializer_gpu(self):
        w = cuda.cupy.empty(self.shape, dtype=self.dtype)
        self.check_initializer(w)

    def check_shaped_initializer(self, xp):
        initializer = self.target(scale=0.1, dtype=self.dtype)
        w = initializers.generate_array(initializer, self.shape, xp)
        self.assertIs(cuda.get_array_module(w), xp)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)

    def test_shaped_initializer_cpu(self):
        self.check_shaped_initializer(numpy)

    @attr.gpu
    def test_shaped_initializer_gpu(self):
        self.check_shaped_initializer(cuda.cupy)


testing.run_module(__name__, __file__)
