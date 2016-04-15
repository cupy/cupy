import unittest

import numpy

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr


class TestDummyDeviceType(unittest.TestCase):

    def test_int(self):
        self.assertEqual(int(cuda.DummyDeviceType()), -1)

    def test_eq(self):
        self.assertEqual(cuda.DummyDeviceType(), cuda.DummyDeviceType())

    def test_ne(self):
        self.assertNotEqual(cuda.DummyDeviceType(), 1)


class TestCuda(unittest.TestCase):

    def test_get_dummy_device(self):
        if not cuda.available:
            self.assertIs(cuda.get_device(), cuda.DummyDevice)

    @attr.gpu
    def test_get_dummy_device_for_empty_array(self):
        x = cuda.cupy.array([]).reshape((0, 10))
        self.assertIs(cuda.get_device(x), cuda.DummyDevice)

    def test_to_gpu_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.to_gpu(x)

    def test_empy_unavailable(self):
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.empty(())

    def test_empy_like_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.empty_like(x)


@testing.parameterize(
    {'c_contiguous': True},
    {'c_contiguous': False},
)
class TestToCPU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3))

    def test_numpy_array(self):
        y = cuda.to_cpu(self.x)
        self.assertIs(self.x, y)  # Do not copy

    @attr.gpu
    def test_cupy_array(self):
        x = cuda.to_gpu(self.x)
        if not self.c_contiguous:
            x = cuda.cupy.asfortranarray(x)
        y = cuda.to_cpu(x)
        self.assertIsInstance(y, numpy.ndarray)
        numpy.testing.assert_array_equal(self.x, y)

    def test_variable(self):
        x = chainer.Variable(self.x)
        with self.assertRaises(TypeError):
            cuda.to_cpu(x)


class TestWorkspace(unittest.TestCase):

    def setUp(self):
        self.space = cuda.get_max_workspace_size()

    def tearDown(self):
        cuda.set_max_workspace_size(self.space)

    def test_size(self):
        size = 1024
        cuda.set_max_workspace_size(size)
        self.assertEqual(size, cuda.get_max_workspace_size())


testing.run_module(__name__, __file__)
