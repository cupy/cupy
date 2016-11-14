import unittest

import mock
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.normalization import batch_normalization
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _batch_normalization(expander, gamma, beta, x, mean, var):
    mean = mean[expander]
    std = numpy.sqrt(var)[expander]
    y_expect = (gamma[expander] * (x - mean) / std + beta[expander])
    return y_expect


@testing.parameterize(*testing.product({
    'ndim': [0, 1, 2, 3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestBatchNormalization(unittest.TestCase):

    def setUp(self):
        self.expander = (None, Ellipsis) + (None,) * self.ndim
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))
        self.eps = 2e-5
        self.decay = 0.9

        self.gamma = numpy.random.uniform(.5, 1, (3,)).astype(self.dtype)
        self.beta = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)

        shape = (5, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)

        self.args = [self.x, self.gamma, self.beta]
        self.mean = self.x.mean(axis=self.aggr_axes)
        self.var = self.x.var(axis=self.aggr_axes) + self.eps
        self.train = True
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, args, use_cudnn=True):
        y = functions.batch_normalization(
            *[chainer.Variable(i) for i in args], running_mean=None,
            running_var=None, decay=self.decay, eps=self.eps,
            use_cudnn=use_cudnn)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean, self.var)

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args])

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args], False)

    def check_backward(self, args, y_grad):
        gradient_check.check_backward(
            batch_normalization.BatchNormalizationFunction(
                mean=None, var=None, train=self.train,
                decay=self.decay, eps=self.eps), args, y_grad,
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'ndim': [0, 1, 2, 3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestFixedBatchNormalization(unittest.TestCase):

    def setUp(self):
        self.gamma = numpy.random.uniform(.5, 1, (3,)).astype(self.dtype)
        self.beta = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)
        self.expander = (None, Ellipsis) + (None,) * self.ndim

        shape = (5, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.eps = 2e-5
        self.decay = 0.0
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))

        self.mean = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)
        self.var = numpy.random.uniform(
            0.5, 1, (3,)).astype(self.dtype)
        self.args = [self.x, self.gamma, self.beta, self.mean, self.var]
        self.train = False
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, args, use_cudnn=True):
        y = functions.fixed_batch_normalization(
            *[chainer.Variable(i) for i in args],
            eps=self.eps, use_cudnn=use_cudnn)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean, self.var)

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args])

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args], False)

    def check_backward(self, args, y_grad):
        gradient_check.check_backward(
            batch_normalization.BatchNormalizationFunction(
                mean=None, var=None, train=self.train,
                decay=self.decay, eps=self.eps),
            args, y_grad,  **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
    # TODO(bkvogel): Check float16 support again in next cuDNN version.
    'dtype': [numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestBatchNormalizationCudnnCall(unittest.TestCase):

    def setUp(self):
        ndim = 0
        self.gamma = cuda.cupy.random.uniform(.5, 1, (3,)).astype(self.dtype)
        self.beta = cuda.cupy.random.uniform(-1, 1, (3,)).astype(self.dtype)
        self.eps = 2e-5
        shape = (7, 3) + (2,) * ndim
        self.x = cuda.cupy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.args = [self.x, self.gamma, self.beta]
        self.aggr_axes = (0,) + tuple(six.moves.range(2, ndim + 2))
        self.mean = self.x.mean(axis=self.aggr_axes)
        self.var = self.x.var(axis=self.aggr_axes) + self.eps
        self.expect = self.use_cudnn and (
            cuda.cudnn.cudnn.getVersion() >= 5000)

    def forward(self):
        return functions.batch_normalization(
            *[chainer.Variable(i) for i in self.args], eps=self.eps,
            running_mean=self.mean, running_var=self.var,
            use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.batchNormalizationForwardTraining') \
                as func:
            self.forward()
            self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        y = self.forward()
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.batchNormalizationBackward') as func:
            y.backward()
            self.assertEqual(func.called, self.expect)

testing.run_module(__name__, __file__)
