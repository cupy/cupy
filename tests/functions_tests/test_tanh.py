from unittest import TestCase
import numpy
from chainer      import cuda, Variable
from chainer.cuda import to_gpu
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import tanh
from .. import attr
if cuda.available:
    cuda.init()

class TestSigmoid(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-.5, .5, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-.1, .1, (3, 2)).astype(numpy.float32)

    @attr.cudnn
    def test_forward_gpu(self, use_cudnn=True):
        x = Variable(to_gpu(self.x))
        y = tanh(x, use_cudnn=use_cudnn)
        y_expect = tanh(Variable(self.x))
        assert_allclose(y_expect.data, y.data)

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        self.test_forward_gpu(False)

    def check_backward(self, x_data, gy_data, use_cudnn=True):
        x = Variable(x_data)
        y = tanh(x, use_cudnn=use_cudnn)
        y.grad = gy_data
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))
        assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy), False)
