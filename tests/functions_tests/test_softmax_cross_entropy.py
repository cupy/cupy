import math
from unittest import TestCase

import numpy
from chainer import Variable, cuda
from chainer.cuda import GPUArray, to_cpu, to_gpu
from chainer.functions import softmax_cross_entropy
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.testing import attr

from six.moves import range

if cuda.available:
    cuda.init()


class TestSoftmaxCrossEntropy(TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4,)).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = Variable(x_data)
        t = Variable(t_data)
        loss = softmax_cross_entropy(x, t, use_cudnn)
        loss_value = float(to_cpu(loss.data))

        # Compute expected value
        y = numpy.exp(self.x)
        loss_expect = 0
        for i in range(y.shape[0]):
            loss_expect -= math.log(y[i, self.t[i]] / y[i].sum())
        loss_expect /= y.shape[0]

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.cudnn
    def test_forward_gpu(self):
        self.check_forward(to_gpu(self.x), to_gpu(self.t))

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(to_gpu(self.x), to_gpu(self.t), False)

    def check_backward(self, x_data, t_data, use_cudnn=True):
        x = Variable(x_data)
        t = Variable(t_data)
        loss = softmax_cross_entropy(x, t, use_cudnn)
        loss.backward()
        self.assertEqual(None, t.grad)

        func = loss.creator
        f = lambda: func.forward((x.data, t.data))
        gx, = numerical_grad(f, (x.data,), (1,), eps=0.02)

        assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.cudnn
    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.t))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.t), False)
