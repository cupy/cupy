import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


def _batch_normalization(expander, gamma, beta, x, mean, var, eps, test):
    mean = mean[expander]
    if test:
        std = numpy.sqrt(var[expander])
    else:
        std = numpy.sqrt(var[expander] + eps)
    y_expect = gamma * (x - mean) / std + beta
    return y_expect


@testing.parameterize(*(testing.product({
    'test': [True, False],
    'volatile': ['on'],
    'ndim': [0],
    'dtype': [numpy.float32],
}) + testing.product({
    'test': [True, False],
    'volatile': ['off'],
    'ndim': [0, 1, 2, 3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class BatchNormalizationTest(unittest.TestCase):

    def setUp(self):
        self.expander = (None, Ellipsis) + (None,) * self.ndim
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))

        self.link = links.BatchNormalization(3, dtype=self.dtype)
        gamma = self.link.gamma.data
        gamma[...] = numpy.random.uniform(.5, 1, gamma.shape)
        beta = self.link.beta.data
        beta[...] = numpy.random.uniform(-1, 1, beta.shape)
        self.link.cleargrads()

        self.gamma = gamma.copy()[self.expander]  # fixed on CPU
        self.beta = beta.copy()[self.expander]   # fixed on CPU

        shape = (5, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)

        if self.test:
            self.mean = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)
            self.var = numpy.random.uniform(0.5, 1, (3,)).astype(self.dtype)
            self.link.avg_mean[...] = self.mean
            self.link.avg_var[...] = self.var
        else:
            self.mean = self.x.mean(axis=self.aggr_axes)
            self.var = self.x.var(axis=self.aggr_axes)
        self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_optionss = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_optionss = {'atol': 5e-1, 'rtol': 1e-1}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data, volatile=self.volatile)
        y = self.link(x, test=self.test)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean,
            self.var, self.link.eps, self.test)

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_optionss)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    @attr.cudnn
    def test_forward_gpu_without_cudnn(self):
        self.link.use_cudnn = False
        self.test_forward_gpu()

    @attr.multi_gpu(2)
    @condition.retry(3)
    def test_forward_multi_gpu(self):
        with cuda.get_device(1):
            self.link.to_gpu()
            x = cuda.to_gpu(self.x)
        with cuda.get_device(0):
            self.check_forward(x)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.gamma, self.link.beta),
            eps=1e-2, **self.check_backward_optionss)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.cudnn
    def test_backward_gpu_without_cudnn(self):
        self.link.use_cudnn = False
        self.test_backward_gpu()


@testing.parameterize(
    {'nx': 10, 'ny': 10},
    # TODO(Kenta Oono)
    # Pass the case below (this test does not pass when nx != ny).
    # {'nx': 10, 'ny': 15}
)
class TestPopulationStatistics(unittest.TestCase):

    def setUp(self):
        self.decay = 0.9
        self.size = 3
        self.link = links.BatchNormalization(self.size, self.decay)
        self.x = numpy.random.uniform(
            -1, 1, (self.nx, self.size)).astype(numpy.float32)
        self.y = numpy.random.uniform(
            -1, 1, (self.ny, self.size)).astype(numpy.float32)

    def check_statistics(self, x, y):
        x = chainer.Variable(x)
        self.link(x, finetune=True)
        mean = self.x.mean(axis=0)
        testing.assert_allclose(mean, self.link.avg_mean)
        unbiased_var = self.x.var(axis=0) * self.nx / (self.nx - 1)
        testing.assert_allclose(unbiased_var, self.link.avg_var)

        y = chainer.Variable(y)
        self.link(y, test=True, finetune=True)
        testing.assert_allclose(mean, self.link.avg_mean)
        testing.assert_allclose(unbiased_var, self.link.avg_var)

    @condition.retry(3)
    def test_statistics_cpu(self):
        self.check_statistics(self.x, self.y)

    @attr.gpu
    @condition.retry(3)
    def test_statistics_gpu(self):
        self.link.to_gpu()
        self.check_statistics(cuda.to_gpu(self.x), cuda.to_gpu(self.y))

    @attr.cudnn
    def test_statistics_gpu_without_cudnn(self):
        self.link.use_cudnn = False
        self.test_statistics_gpu()

    def check_statistics2(self, x, y):
        x = chainer.Variable(x)
        y = chainer.Variable(y)
        self.link(x, finetune=True)
        self.link(y, finetune=True)
        mean = (self.x.sum(axis=0) + self.y.sum(axis=0)) / (self.nx + self.ny)
        var = (self.x.var(axis=0) * self.nx +
               self.y.var(axis=0) * self.ny) / (self.nx + self.ny)

        # TODO(Kenta Oono)
        # Fix the estimate of the unbiased variance.
        # Unbiased variance should be (nx + ny) / (nx + ny - 1) times of
        # the variance.
        # But the multiplier is ny / (ny - 1) in current implementation
        # these two values are different when nx is not equal to ny.
        unbiased_var = var * self.ny / (self.ny - 1)
        testing.assert_allclose(mean, self.link.avg_mean)
        testing.assert_allclose(unbiased_var, self.link.avg_var)

    @condition.retry(3)
    def test_statistics2_cpu(self):
        self.check_statistics2(self.x, self.y)

    @attr.gpu
    @condition.retry(3)
    def test_statistics2_gpu(self):
        self.link.to_gpu()
        self.check_statistics2(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.y))

    @attr.cudnn
    def test_statistics2_gpu_without_cudnn(self):
        self.link.use_cudnn = False
        self.test_statistics2_gpu()


@testing.parameterize(*testing.product({
    'test': [True, False],
    'ndim': [0, 1, 2, 3],
}))
class BatchNormalizationTestWithoutGammaAndBeta(unittest.TestCase):

    def setUp(self):
        self.link = links.BatchNormalization(
            3, use_gamma=False, use_beta=False)
        if self.test:
            mean = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
            self.link.avg_mean[...] = mean
            var = numpy.random.uniform(0.5, 1, (3,)).astype(numpy.float32)
            self.link.avg_var[...] = var
        self.link.cleargrads()

        shape = (7, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

        expander = (None, Ellipsis) + (None,) * self.ndim
        gamma = numpy.ones((3,), dtype=numpy.float32)[expander]
        beta = numpy.zeros((3,), dtype=numpy.float32)[expander]
        if self.test:
            mean = self.link.avg_mean
            var = self.link.avg_var
        else:
            aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))
            mean = self.x.mean(axis=aggr_axes)
            var = self.x.var(axis=aggr_axes)
        self.y_expected = _batch_normalization(
            expander, gamma, beta, self.x, mean, var, self.link.eps, self.test)

    def test_no_gamma_and_beta(self):
        self.assertFalse(hasattr(self.link, 'gamma'))
        self.assertFalse(hasattr(self.link, 'beta'))

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x, test=self.test)
        testing.assert_allclose(self.y_expected, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        self.check_forward(x)

    @attr.multi_gpu(2)
    def test_forward_gpu_multi(self):
        with cuda.get_device(0):
            self.link.to_gpu()
            x = cuda.to_gpu(self.x)
        with cuda.get_device(1):
            self.check_forward(x)

    @attr.cudnn
    def test_forward_gpu_without_cudnn(self):
        self.link.use_cudnn = False
        self.test_forward_gpu()

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(self.link, x_data, y_grad,
                                      eps=1e-2, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x, gy)

    @attr.cudnn
    def test_backward_gpu_without_cudnn(self):
        self.link.use_cudnn = False
        self.test_backward_gpu()


@testing.parameterize(*testing.product({
    'size': [3, (2, 3)],
}))
class TestInitialize(unittest.TestCase):

    def setUp(self):
        self.decay = 0.9
        self.initial_gamma = numpy.random.uniform(-1, 1, self.size)
        self.initial_gamma = self.initial_gamma.astype(numpy.float32)
        self.initial_beta = numpy.random.uniform(-1, 1, self.size)
        self.initial_beta = self.initial_beta.astype(numpy.float32)
        self.link = links.BatchNormalization(self.size, self.decay,
                                             initial_gamma=self.initial_gamma,
                                             initial_beta=self.initial_beta)

    @condition.retry(3)
    def test_initialize_cpu(self):
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)

    @attr.gpu
    @condition.retry(3)
    def test_initialize_gpu(self):
        self.link.to_gpu()
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)


class TestDefaultInitializer(unittest.TestCase):

    def setUp(self):
        self.decay = 0.9
        self.size = 3
        self.link = links.BatchNormalization(self.size, self.decay)

    def test_initialize_cpu(self):
        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.data)
        testing.assert_allclose(numpy.zeros(self.size), self.link.beta.data)

    @attr.gpu
    def test_initialize_gpu(self):
        self.link.to_gpu()
        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.data)
        testing.assert_allclose(numpy.zeros(self.size), self.link.beta.data)


@testing.parameterize(*testing.product({
    'shape': [(2, 4), (2, 5, 3, 4)],
}))
class TestInvalidInput(unittest.TestCase):

    def setUp(self):
        self.link = links.BatchNormalization(3)

    def test_invalid_shape_cpu(self):
        with self.assertRaises(type_check.InvalidType):
            self.link(chainer.Variable(numpy.zeros(self.shape, dtype='f')))

    @attr.gpu
    def test_invalid_shape_gpu(self):
        self.link.to_gpu()
        with self.assertRaises(type_check.InvalidType):
            self.link(chainer.Variable(cuda.cupy.zeros(self.shape, dtype='f')))


class TestInvalidInitialize(unittest.TestCase):

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            self.link = links.BatchNormalization({})


testing.run_module(__name__, __file__)
