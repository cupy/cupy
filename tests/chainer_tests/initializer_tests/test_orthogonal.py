import unittest

from chainer import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr
import numpy


@testing.parameterize(*testing.product({
    'shape': [(), (1,), (3, 4), (3, 4, 5)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class OrthogonalBase(unittest.TestCase):

    def setUp(self):
        self.check_options = {}
        if self.dtype == numpy.float16:
            self.check_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_initializer(self, w):
        initializer = initializers.Orthogonal(scale=2.0)
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
        initializer = initializers.Orthogonal(scale=2.0, dtype=self.dtype)
        w = initializers.generate_array(initializer, self.shape, xp)
        self.assertIs(cuda.get_array_module(w), xp)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)

    def test_shaped_initializer_cpu(self):
        self.check_shaped_initializer(numpy)

    @attr.gpu
    def test_shaped_initializer_gpu(self):
        self.check_shaped_initializer(cuda.cupy)

    def check_orthogonality(self, w):
        initializer = initializers.Orthogonal(scale=2.0)
        initializer(w)
        n = 1 if w.ndim == 0 else len(w)
        w = w.astype(numpy.float64).reshape(n, -1)
        dots = w.dot(w.T)
        testing.assert_allclose(
            dots, numpy.identity(n) * 4, **self.check_options)

    def test_orthogonality_cpu(self):
        w = numpy.empty(self.shape, dtype=self.dtype)
        self.check_orthogonality(w)

    @attr.gpu
    def test_orthogonality_gpu(self):
        w = cuda.cupy.empty(self.shape, dtype=self.dtype)
        self.check_orthogonality(w)


class TestEmpty(unittest.TestCase):

    def setUp(self):
        self.w = numpy.empty(0, dtype=numpy.float32)
        self.initializer = initializers.Orthogonal()

    def check_assert(self, w):
        print(w.shape)
        with self.assertRaises(ValueError):
            self.initializer(w)

    def test_cpu(self):
        self.check_assert(self.w)

    @attr.gpu
    def test_gpu(self):
        self.check_assert(cuda.to_gpu(self.w))


@testing.parameterize(
    {'shape': (4, 3)},
    {'shape': (21, 4, 5)})
class TestOverComplete(unittest.TestCase):

    def setUp(self):
        self.w = numpy.empty(self.shape, dtype=numpy.float32)
        self.initializer = initializers.Orthogonal(scale=1.0)

    def check_invalid(self, w):
        with self.assertRaises(ValueError):
            self.initializer(w)

    def test_invalid_cpu(self):
        self.check_invalid(self.w)

    @attr.gpu
    def test_invalid_gpu(self):
        self.check_invalid(cuda.to_gpu(self.w))


testing.run_module(__name__, __file__)
