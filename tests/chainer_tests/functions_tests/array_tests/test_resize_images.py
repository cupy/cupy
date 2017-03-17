import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'in_shape': [(2, 3, 8, 6), (2, 1, 4, 6)],
}))
class TestResizeImagesForwardIdentity(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            size=self.in_shape).astype(numpy.float32)

    def check_forward(self, x, output_shape):
        y = functions.resize_images(x, output_shape)
        testing.assert_allclose(y.data, x)

    def test_forward_cpu(self):
        self.check_forward(self.x, output_shape=self.in_shape[2:])

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), output_shape=self.in_shape[2:])


class TestResizeImagesForwardDownScale(unittest.TestCase):

    in_shape = (2, 2, 4, 4)
    output_shape = (2, 2, 2, 2)

    def setUp(self):
        self.x = numpy.zeros(self.in_shape, dtype=numpy.float32)
        self.x[:, :, :2, :2] = 1
        self.x[:, :, 2:, :2] = 2
        self.x[:, :, :2, 2:] = 3
        self.x[:, :, 2:, 2:] = 4

        self.out = numpy.zeros(self.output_shape, dtype=numpy.float32)
        self.out[:, :, 0, 0] = 1
        self.out[:, :, 1, 0] = 2
        self.out[:, :, 0, 1] = 3
        self.out[:, :, 1, 1] = 4

    def check_forward(self, x, output_shape):
        y = functions.resize_images(x, output_shape)
        testing.assert_allclose(y.data, self.out)

    def test_forward_cpu(self):
        self.check_forward(self.x, output_shape=self.output_shape[2:])

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), output_shape=self.output_shape[2:])


class TestResizeImagesForwardUpScale(unittest.TestCase):

    in_shape = (1, 1, 2, 2)
    output_shape = (1, 1, 3, 3)

    def setUp(self):
        self.x = numpy.zeros(self.in_shape, dtype=numpy.float32)
        self.x[:, :, 0, 0] = 1
        self.x[:, :, 1, 0] = 2
        self.x[:, :, 0, 1] = 3
        self.x[:, :, 1, 1] = 4

        self.out = numpy.zeros(self.output_shape, dtype=numpy.float32)
        self.out[0, 0, :, :] = numpy.array(
            [[1., 2., 3.],
             [1.5, 2.5, 3.5],
             [2., 3., 4.]],
            dtype=numpy.float32)

    def check_forward(self, x, output_shape):
        y = functions.resize_images(x, output_shape)
        testing.assert_allclose(y.data, self.out)

    def test_forward_cpu(self):
        self.check_forward(self.x, output_shape=self.output_shape[2:])

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), output_shape=self.output_shape[2:])


@testing.parameterize(*testing.product({
    'in_shape': [(2, 3, 8, 6), (2, 1, 4, 6)],
    'output_shape': [(10, 5), (3, 4)]
}))
class TestResizeImagesBackward(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            size=self.in_shape).astype(numpy.float32)
        output_shape_4d = self.in_shape[:2] + self.output_shape
        self.grads = numpy.random.uniform(
            size=output_shape_4d).astype(numpy.float32)

    def check_backward(self, x, output_shape, grads):
        gradient_check.check_backward(
            functions.ResizeImages(output_shape),
            (x,), (grads,), atol=1e-2, rtol=1e-3, eps=1e-5)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.output_shape, self.grads)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), self.output_shape,
                            cuda.to_gpu(self.grads))


testing.run_module(__name__, __file__)
