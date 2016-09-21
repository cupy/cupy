import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _zoneout(h, x, creator):
    h_next = h * creator.flag_h + x * creator.flag_x
    return h_next


@testing.parameterize(
    {'ratio': 1},
    {'ratio': 0},
    {'ratio': 0.5},
    {'ratio': 0.25},
)
class TestZoneout(unittest.TestCase):

    def setUp(self):
        self.h = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.x = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)

    def check_forward(self, h_data, x_data):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)
        h_next = functions.zoneout(h, x, self.ratio)
        h_next_expect = h * h_next.creator.flag_h + x * h_next.creator.flag_x
        testing.assert_allclose(h_next.data, h_next_expect.data)

    def check_backward(self, h_data, x_data, y_grad):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)
        y = functions.zoneout(h, x, self.ratio)
        creator = y.creator
        y.grad = y_grad
        y.backward()

        def f():
            nonlocal creator
            y = _zoneout(h_data, x_data, creator)
            return y,
        gh, gx, = gradient_check.numerical_grad(f, (h.data, x.data,),
                                                (y.grad,))
        gradient_check.assert_allclose(gh, h.grad, atol=1e-3)
        gradient_check.assert_allclose(gx, x.grad, atol=1e-3)

    def test_forward_cpu(self):
        self.check_forward(self.h, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.h), cuda.to_gpu(self.x))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.h, self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.h),
                            cuda.to_gpu(self.x),
                            cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)
