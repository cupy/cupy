from unittest import TestCase
import numpy
from chainer      import cuda, Variable
from chainer.cuda import to_gpu
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import softmax

cuda.init()

class TestSoftmax(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = Variable(x_data)
        y = softmax(x, use_cudnn)

        y_expect = numpy.exp(self.x)
        for i in xrange(y_expect.shape[0]):
            y_expect[i] /= y_expect[i].sum()

        assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_gpu(self):
        self.check_forward(to_gpu(self.x))

    def test_forwrad_gpu_no_cudnn(self):
        self.check_forward(to_gpu(self.x), False)

    def check_backward(self, x_data, gy_data, use_cudnn=True):
        x = Variable(x_data)
        y = softmax(x, use_cudnn)
        y.grad = gy_data
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,), eps=1e-2)

        assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

    def test_backward_gpu_no_cudnn(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy), False)
