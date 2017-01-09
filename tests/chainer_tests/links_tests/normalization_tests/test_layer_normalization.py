import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


def _create_ln(*args, **kwargs):
    flag = chainer.disable_experimental_feature_warning
    chainer.disable_experimental_feature_warning = True
    try:
        return links.LayerNormalization(*args, **kwargs)
    finally:
        chainer.disable_experimental_feature_warning = flag


@testing.parameterize(*(testing.product({
    'volatile': ['on', 'off'],
    'batchsize': [1, 5],
    'size': [10, 100],
    'dtype': [numpy.float32],
})))
class LayerNormalizationTest(unittest.TestCase):

    def setUp(self):
        self.link = _create_ln()
        self.link.cleargrads()

        self.shape = (self.batchsize, self.size)
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_optionss = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_optionss = {'atol': 5e-1, 'rtol': 1e-1}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data, volatile=self.volatile)
        y = self.link(x)
        self.assertEqual(y.data.dtype, self.dtype)

        unbatched_concat_y = chainer.functions.concat(
            [self.link(one_x[None, ]) for one_x in x_data], axis=0)

        testing.assert_allclose(
            y.data, unbatched_concat_y.data, **self.check_forward_optionss)

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
            self.link, x_data, y_grad,
            (self.link.gamma, self.link.beta),
            eps=1e-2, **self.check_backward_optionss)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.link(numpy.zeros(self.shape, dtype='f'))
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.link(cuda.cupy.zeros(self.shape, dtype='f'))
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.cudnn
    def test_backward_gpu_without_cudnn(self):
        self.link.use_cudnn = False
        self.link(numpy.zeros(self.shape, dtype='f'))
        self.test_backward_gpu()


@testing.parameterize(*testing.product({
    'size': [3, 50],
}))
class TestInitialize(unittest.TestCase):

    def setUp(self):
        self.initial_gamma = numpy.random.uniform(-1, 1, self.size)
        self.initial_gamma = self.initial_gamma.astype(numpy.float32)
        self.initial_beta = numpy.random.uniform(-1, 1, self.size)
        self.initial_beta = self.initial_beta.astype(numpy.float32)
        self.link = _create_ln(
            initial_gamma=self.initial_gamma,
            initial_beta=self.initial_beta)
        self.shape = (1, self.size)

    @condition.retry(3)
    def test_initialize_cpu(self):
        self.link(numpy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)

    @attr.gpu
    @condition.retry(3)
    def test_initialize_gpu(self):
        self.link.to_gpu()
        self.link(cuda.cupy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)


class TestDefaultInitializer(unittest.TestCase):

    def setUp(self):
        self.size = 3
        self.link = _create_ln()
        self.shape = (1, self.size)

    def test_initialize_cpu(self):
        self.link(numpy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.data)
        testing.assert_allclose(
            numpy.zeros(self.size), self.link.beta.data)

    @attr.gpu
    def test_initialize_gpu(self):
        self.link.to_gpu()
        self.link(cuda.cupy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.data)
        testing.assert_allclose(
            numpy.zeros(self.size), self.link.beta.data)


@testing.parameterize(*testing.product({
    'shape': [(2, 4, 3), (2, 5, 3, 4)],
}))
class TestInvalidInput(unittest.TestCase):

    def setUp(self):
        self.link = _create_ln()

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
        with self.assertRaises(AssertionError):
            self.link = _create_ln(None, 1e-6, {})
            self.link(chainer.Variable(numpy.zeros((1, 5), dtype='f')))


testing.run_module(__name__, __file__)
