import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSum(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(self.dtype)

    def check_forward(self, x_data, axis=None):
        x = chainer.Variable(x_data)
        y = functions.sum(x, axis=axis)
        self.assertEqual(y.data.dtype, self.dtype)
        y_expect = self.x.sum(axis=axis)

        if self.dtype == numpy.float16:
            options = {'atol': 1e-3, 'rtol': 1e-3}
        else:
            options = {}

        testing.assert_allclose(y_expect, y.data, **options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @condition.retry(3)
    def test_forward_axis_cpu(self):
        for i in range(self.x.ndim):
            self.check_forward(self.x, axis=i)

    @condition.retry(3)
    def test_forward_negative_axis_cpu(self):
        self.check_forward(self.x, axis=-1)

    @condition.retry(3)
    def test_forward_multi_axis_cpu(self):
        self.check_forward(self.x, axis=(0, 1))

    @condition.retry(3)
    def test_forward_multi_axis_invert_cpu(self):
        self.check_forward(self.x, axis=(1, 0))

    @condition.retry(3)
    def test_forward_negative_multi_axis_cpu(self):
        self.check_forward(self.x, axis=(0, -1))

    @condition.retry(3)
    def test_forward_negative_multi_axis_invert_cpu(self):
        self.check_forward(self.x, axis=(-2, 0))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_axis_gpu(self):
        for i in range(self.x.ndim):
            self.check_forward(cuda.to_gpu(self.x), axis=i)

    @attr.gpu
    @condition.retry(3)
    def test_forward_negative_axis_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=-1)

    @attr.gpu
    @condition.retry(3)
    def test_forward_multi_axis_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(0, 1))

    @attr.gpu
    @condition.retry(3)
    def test_forward_multi_axis_invert_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(1, 0))

    @attr.gpu
    @condition.retry(3)
    def test_forward_negative_multi_axis_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(0, -1))

    @attr.gpu
    @condition.retry(3)
    def test_forward_negative_multi_axis_invert_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(-2, 0))

    def check_backward(self, x_data, y_grad, axis=None):
        gradient_check.check_backward(
            functions.Sum(axis), x_data, y_grad, atol=1e-4,
            dtype=numpy.float64)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @condition.retry(3)
    def test_backward_zerodim_cpu(self):
        x = numpy.random.uniform(-1, 1, ()).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, ()).astype(self.dtype)
        self.check_backward(x, gy)

    @condition.retry(3)
    def test_backward_axis_cpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=i)) * self.gy
            self.check_backward(self.x, gy, axis=i)

    @condition.retry(3)
    def test_backward_negative_axis_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=-1)) * self.gy
        self.check_backward(self.x, gy, axis=-1)

    @condition.retry(3)
    def test_backward_multi_axis_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, 1))) * self.gy
        self.check_backward(self.x, gy, axis=(0, 1))

    @condition.retry(3)
    def test_backward_multi_axis_invert_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(1, 0))) * self.gy
        self.check_backward(self.x, gy, axis=(1, 0))

    @condition.retry(3)
    def test_backward_negative_multi_axis_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, -1))) * self.gy
        self.check_backward(self.x, gy, axis=(0, -1))

    @condition.retry(3)
    def test_backward_negative_multi_axis_invert_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(-2, 0))) * self.gy
        self.check_backward(self.x, gy, axis=(-2, 0))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_zerodim_gpu(self):
        x = numpy.random.uniform(-1, 1, ()).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, ()).astype(self.dtype)
        self.check_backward(cuda.to_gpu(x), cuda.to_gpu(gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_axis_gpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=i)) * self.gy
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=i)

    @attr.gpu
    @condition.retry(3)
    def test_backward_negative_axis_gpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=-1)) * self.gy
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=-1)

    @attr.gpu
    @condition.retry(3)
    def test_backward_multi_axis_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, 1))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(0, 1))

    @attr.gpu
    @condition.retry(3)
    def test_backward_multi_axis_invert_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(1, 0))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(1, 0))

    @attr.gpu
    @condition.retry(3)
    def test_backward_negative_multi_axis_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, -1))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(0, -1))

    @attr.gpu
    @condition.retry(3)
    def test_backward_negative_multi_axis_invert_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(-2, 0))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(-2, 0))

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.Sum([0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            functions.Sum((1, 'x'))

    def test_duplicate_axis(self):
        with self.assertRaises(ValueError):
            functions.Sum((0, 0))

    def test_pos_neg_duplicate_axis(self):
        with self.assertRaises(ValueError):
            self.x.sum(axis=(1, -2))

testing.run_module(__name__, __file__)
