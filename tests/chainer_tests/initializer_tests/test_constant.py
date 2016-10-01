import unittest

from chainer import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr
import numpy


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestIdentity(unittest.TestCase):

    scale = 0.1
    shape = (2, 2)

    def setUp(self):
        self.check_options = {}
        if self.dtype == numpy.float16:
            self.check_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_initializer(self, w):
        initializer = initializers.Identity(scale=self.scale)
        initializer(w)
        testing.assert_allclose(
            w, self.scale * numpy.identity(len(self.shape)),
            **self.check_options)

    def test_initializer_cpu(self):
        w = numpy.empty(self.shape, dtype=self.dtype)
        self.check_initializer(w)

    @attr.gpu
    def test_initializer_gpu(self):
        w = cuda.cupy.empty(self.shape, dtype=self.dtype)
        self.check_initializer(w)

    def check_shaped_initializer(self, xp):
        initializer = initializers.Identity(
            scale=self.scale, dtype=self.dtype)
        w = initializers.generate_array(initializer, self.shape, xp)
        self.assertIs(cuda.get_array_module(w), xp)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)
        testing.assert_allclose(
            w, self.scale * numpy.identity(len(self.shape)),
            **self.check_options)

    def test_shaped_initializer_cpu(self):
        self.check_shaped_initializer(numpy)

    @attr.gpu
    def test_shaped_initializer_gpu(self):
        self.check_shaped_initializer(cuda.cupy)


@testing.parameterize(
    {'shape': (2, 3)},
    {'shape': (2, 2, 4)},
    {'shape': ()},
    {'shape': 0})
class TestIdentityInvalid(unittest.TestCase):

    def setUp(self):
        self.initializer = initializers.Identity()

    def test_invalid_shape(self):
        w = numpy.empty(self.shape, dtype=numpy.float32)
        with self.assertRaises(ValueError):
            self.initializer(w)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestConstant(unittest.TestCase):

    fill_value = 0.1
    shape = (2, 3)

    def setUp(self):
        self.check_options = {}
        if self.dtype == numpy.float16:
            self.check_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_initializer(self, w):
        initializer = initializers.Constant(fill_value=self.fill_value)
        initializer(w)
        testing.assert_allclose(
            w, numpy.full(self.shape, self.fill_value),
            **self.check_options)

    def test_initializer_cpu(self):
        w = numpy.empty(self.shape, dtype=self.dtype)
        self.check_initializer(w)

    @attr.gpu
    def test_initializer_gpu(self):
        w = cuda.cupy.empty(self.shape, dtype=self.dtype)
        self.check_initializer(w)

    def check_shaped_initializer(self, xp):
        initializer = initializers.Constant(
            fill_value=self.fill_value, dtype=self.dtype)
        w = initializers.generate_array(initializer, self.shape, xp)
        self.assertIs(cuda.get_array_module(w), xp)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)
        testing.assert_allclose(
            w, numpy.full(self.shape, self.fill_value),
            **self.check_options)

    def test_shaped_initializer_cpu(self):
        self.check_shaped_initializer(numpy)

    @attr.gpu
    def test_shaped_initializer_gpu(self):
        self.check_shaped_initializer(cuda.cupy)


testing.run_module(__name__, __file__)
