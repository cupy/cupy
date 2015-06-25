from unittest import TestCase
import math
import numpy
from six.moves import range
from chainer      import cuda, Variable
from chainer.cuda import to_cpu, to_gpu, GPUArray
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import sigmoid_cross_entropy
from chainer.testing import attr

if cuda.available:
    cuda.init()

class TestSigmoidCrossEntropy(TestCase):
    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 2, (4, 3)).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x_val = Variable(x_data)
        t_val = Variable(t_data)
        loss = sigmoid_cross_entropy(x_val, t_val, use_cudnn)
        loss_value = float(to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                xd, td = self.x[i, j], self.t[i, j]
                xdd = 1 if self.x[i, j] >= 0 else 0
                loss_expect -= xd * (td - xdd) - math.log(1 + math.exp(-numpy.abs(xd)))
        loss_expect /= self.t.shape[0]
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
        loss = sigmoid_cross_entropy(x, t, use_cudnn)
        loss.backward()
        self.assertEqual(None, t.grad)

        func = loss.creator
        f = lambda: func.forward((x.data, t.data))
        gx, = numerical_grad(f, (x.data,), (1,), eps=0.01)

        assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.cudnn
    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.t))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.t), False)
