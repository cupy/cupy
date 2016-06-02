import unittest

from chainer import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr
import numpy


class OrthogonalBase(object):

    def check_initializer(self, w):
        expected_shape = w.shape
        self.initializer(w)
        self.assertTupleEqual(w.shape, expected_shape)
        self.assertEqual(w.dtype, numpy.float32)
        xp = cuda.get_array_module(w)
        self.assertIsInstance(w, xp.ndarray)

    def test_initializer_cpu(self):
        self.check_initializer(self.w)

    @attr.gpu
    def test_initializer_gpu(self):
        self.check_initializer(cuda.to_gpu(self.w))


@testing.parameterize(
    {'shape': (1,)},
    {'shape': (3, 4)},
    {'shape': (3, 4, 5)})
class TestOrthogonal(OrthogonalBase, unittest.TestCase):

    def setUp(self):
        self.w = numpy.empty(self.shape, dtype=numpy.float32)
        self.initializer = initializers.Orthogonal(scale=1.0)

    def check_orthogonality(self, w):
        self.initializer(w)
        xp = cuda.get_array_module(w)
        w = w.reshape(len(w), -1)
        dots = xp.tensordot(w, w, (1, 1))
        testing.assert_allclose(dots, xp.identity(len(w)))

    def test_orthogonality_cpu(self):
        self.check_orthogonality(self.w)

    @attr.gpu
    def test_orthogonality_gpu(self):
        self.check_orthogonality(cuda.to_gpu(self.w))


class TestZeroDim(OrthogonalBase, unittest.TestCase):

    def setUp(self):
        self.w = numpy.empty([], dtype=numpy.float32)
        self.initializer = initializers.Orthogonal(scale=2.0)

    def check_orthogonality(self, w):
        self.initializer(w)
        xp = cuda.get_array_module(w)
        testing.assert_allclose(w, xp.ones((), dtype=numpy.float32) * 2)

    def test_orthogonality_cpu(self):
        self.check_orthogonality(self.w)

    @attr.gpu
    def test_orthogonality_gpu(self):
        self.check_orthogonality(cuda.to_gpu(self.w))


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
