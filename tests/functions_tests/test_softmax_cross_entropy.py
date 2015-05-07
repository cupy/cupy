from unittest import TestCase
import math
import numpy
from chainer      import cuda, Variable
from chainer.cuda import to_gpu, GPUArray
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import softmax_cross_entropy

cuda.init()

class TestSoftmaxCrossEntropy(TestCase):
    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4,)).astype(numpy.int32)

    def check_forward(self, x_data, t_data):
        x = Variable(x_data)
        t = Variable(t_data)
        loss = softmax_cross_entropy(x, t)
        if type(loss.data) == GPUArray:
            loss_value = float(loss.data.get())
        else:
            loss_value = float(loss.data)

        # Compute expected value
        y = numpy.exp(self.x)
        loss_expect = 0
        for i in xrange(y.shape[0]):
            loss_expect -= math.log(y[i, self.t[i]] / y[i].sum())
        loss_expect /= y.shape[0]

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    def test_forward_gpu(self):
        self.check_forward(to_gpu(self.x), to_gpu(self.t))

    def check_backward(self, x_data, t_data):
        x = Variable(x_data)
        t = Variable(t_data)
        loss = softmax_cross_entropy(x, t)
        loss.backward()
        self.assertEqual(None, t.grad)

        func = loss.creator
        f = lambda: func.forward((x.data, t.data))
        gx, = numerical_grad(f, (x.data,), (1,), eps=0.02)

        assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.t))
