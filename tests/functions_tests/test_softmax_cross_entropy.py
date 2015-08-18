import math
import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


class TestSoftmaxCrossEntropy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4,)).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.softmax_cross_entropy(x, t, use_cudnn)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        y = numpy.exp(self.x)
        loss_expect = 0.0
        for i in six.moves.range(y.shape[0]):
            loss_expect -= math.log(y[i, self.t[i]] / y[i].sum())
        loss_expect /= y.shape[0]

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)

    def check_backward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.softmax_cross_entropy(x, t, use_cudnn)
        loss.backward()
        self.assertEqual(None, t.grad)

        func = loss.creator
        f = lambda: func.forward((x.data, t.data))
        gx, = gradient_check.numerical_grad(f, (x.data,), (1,), eps=0.02)

        gradient_check.assert_allclose(gx, x.grad, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)


class TestReplicatedSoftmaxCrossEntropy1(TestSoftmaxCrossEntropy):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4, 2)).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.softmax_cross_entropy(
            x, t, use_cudnn, normalize=True)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        y = numpy.exp(self.x)
        loss_expect = 0.0
        for i in six.moves.range(y.shape[0]):
            for k in six.moves.range(y.shape[2]):
                loss_expect -= math.log(
                    y[i, self.t[i, k], k] / y[i, :, k].sum())
        loss_expect /= y.shape[0] * y.shape[2]

        self.assertAlmostEqual(loss_expect, loss_value, places=4)


class TestReplicatedSoftmaxCrossEntropy2(TestSoftmaxCrossEntropy):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (4, 3, 2, 5)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4, 2, 5)).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.softmax_cross_entropy(
            x, t, use_cudnn, normalize=False)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        y = numpy.exp(self.x)
        loss_expect = 0.0
        for i in six.moves.range(y.shape[0]):
            for k in six.moves.range(y.shape[2]):
                for l in six.moves.range(y.shape[3]):
                    loss_expect -= math.log(
                        y[i, self.t[i, k, l], k, l] / y[i, :, k, l].sum())
        loss_expect /= y.shape[0]

        self.assertAlmostEqual(loss_expect, loss_value, places=4)

testing.run_module(__name__, __file__)
