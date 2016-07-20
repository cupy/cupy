import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _inv(x):
    if x.ndim == 2:
        return numpy.linalg.inv(x)
    return numpy.array([numpy.linalg.inv(ix) for ix in x])


def _make_eye(shape):
    if len(shape) == 2:
        n = shape[0]
        return numpy.eye(n, dtype=numpy.float32)
    m = shape[0]
    n = shape[1]
    return numpy.array([numpy.eye(n, dtype=numpy.float32)] * m)


@testing.parameterize(*testing.product({
    'shape': [(1, 1), (5, 5)],
}))
class InvFunctionTest(unittest.TestCase):

    def setUp(self):
        self.x = (numpy.eye(self.shape[-1]) +
                  numpy.random.uniform(-0.01, 0.01, self.shape)).astype(
            numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-4}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-4}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.inv(x)
        testing.assert_allclose(
            _inv(self.x), y.data, **self.check_forward_options)

    def check_backward(self, x_data, y_grad, ):
        gradient_check.check_backward(
            functions.Inv(), x_data, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_identity_cpu(self):
        eye = _make_eye(self.x.shape)
        x = chainer.Variable(self.x)
        y = functions.matmul(x, functions.inv(x))
        testing.assert_allclose(
            y.data, eye, **self.check_forward_options)

    @attr.gpu
    @condition.retry(3)
    def test_identity_gpu(self):
        eye = cuda.to_gpu(_make_eye(self.x.shape))
        x = chainer.Variable(cuda.to_gpu(self.x))
        y = functions.matmul(x, functions.inv(x))
        testing.assert_allclose(
            y.data, eye, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'shape': [(5, 1, 1), (3, 5, 5)],
}))
class BatchInvFunctionTest(unittest.TestCase):

    def setUp(self):
        self.x = (numpy.eye(self.shape[-1]) +
                  numpy.random.uniform(-0.01, 0.01, self.shape)).astype(
            numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-4}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-4}

    def check_forward(self, x_data, atol=1e-7, rtol=1e-7):
        x = chainer.Variable(x_data)
        y = functions.batch_inv(x)
        testing.assert_allclose(
            _inv(self.x), y.data, **self.check_forward_options)

    def check_backward(self, x_data, y_grad, **kwargs):
        gradient_check.check_backward(
            functions.BatchInv(), x_data, y_grad,
            **self.check_backward_options)

    @condition.retry(3)
    def test_identity_cpu(self):
        eye = _make_eye(self.x.shape)
        x = chainer.Variable(self.x)
        y = functions.batch_matmul(x, functions.batch_inv(x))
        testing.assert_allclose(
            y.data, eye, **self.check_forward_options)

    @attr.gpu
    @condition.retry(3)
    def test_identity_gpu(self):
        eye = cuda.to_gpu(_make_eye(self.x.shape))
        x = chainer.Variable(cuda.to_gpu(self.x))
        y = functions.batch_matmul(x, functions.batch_inv(x))
        testing.assert_allclose(
            y.data, eye, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class InvFunctionRaiseTest(unittest.TestCase):

    def test_invalid_ndim(self):
        with self.assertRaises(TypeError):
            functions.inv(chainer.Variable(numpy.zeros(1, 2, 2)))

    def test_invalid_shape(self):
        with self.assertRaises(TypeError):
            functions.inv(chainer.Variable(numpy.zeros(1, 2)))


class BatchInvFunctionRaiseTest(unittest.TestCase):

    def test_invalid_ndim(self):
        with self.assertRaises(TypeError):
            functions.batch_inv(chainer.Variable(numpy.zeros(2, 2)))

    def test_invalid_shape(self):
        with self.assertRaises(TypeError):
            functions.batch_inv(chainer.Variable(numpy.zeros(1, 2, 1)))


testing.run_module(__name__, __file__)
