import math
import numpy
import unittest

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

if cuda.available:
    cuda.init()


class TestCTC(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array([[0.99, 0., 0.01],
                              [0.45, 0.45, 0.1],
                              [0.1, 0.7, 0.2],
                              [0.1, 0.78, 0.12]])
        self.t = numpy.array([0, 1])
        self.l = numpy.array([2, 0, 2, 1, 2])
        self.blank_symbol = 2
        self.g = numpy.array(0.1, dtype=numpy.float32)
        self.gx = self.g

    # recursive forward computation.
    def alpha(self, t, u):
        if u < 0:
            return 0.0
        if t == 0:
            if u == 0:
                return self.x[0][self.blank_symbol]
            elif u == 1:
                return self.x[0][self.l[1]]
            else:
                return 0.0
        elif self.l[u] == self.blank_symbol or self.l[u] == self.l[u-2]:
            return self.x[t][self.l[u]] * \
                (self.alpha(t-1, u-1) + self.alpha(t-1, u))
        else:
            return self.x[t][self.l[u]] * \
                (self.alpha(t-1, u-2)
                 + self.alpha(t-1, u-1)
                 + self.alpha(t-1, u))

    def check_forward(self, t_data, xs_data):
        x = tuple(chainer.Variable(x_data) for x_data in xs_data)
        t = chainer.Variable(t_data)
        loss = functions.connectionist_temporal_classification(2, t, x)
        loss_value = float(loss.data)

        # compute expected value by recursive computation.
        loss_expect = - math.log(self.alpha(self.x.shape[0]-1,
                                            self.l.shape[0]-1)
                                 + self.alpha(self.x.shape[0]-1,
                                              self.l.shape[0]-2))
        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    def test_forward_cpu(self):
        self.check_forward(self.t, tuple(x_data for x_data in self.x))

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.t),
                           tuple(cuda.to_gpu(x_data) for x_data in self.x))

    # expected value(via numerical differentiation) from t_data
    def check_backward(self, t_data, xs_data):
        xs = tuple(chainer.Variable(x_data) for x_data in xs_data)
        t = chainer.Variable(t_data)
        loss = functions.connectionist_temporal_classification(2, t, xs)
        loss.grad = self.g
        loss.backward()

        func = loss.creator
        xs_data = tuple(x.data for x in xs)
        f = lambda: func.forward((t.data,) + xs_data)
        gl_0, gx_0, gx_1, gx_2, gx_3 = gradient_check.numerical_grad(
            f, ((t.data,) + xs_data), (self.gx,))
        gradient_check.assert_allclose(xs[0].grad, gx_0)
        gradient_check.assert_allclose(xs[1].grad, gx_1)
        gradient_check.assert_allclose(xs[2].grad, gx_2)
        gradient_check.assert_allclose(xs[3].grad, gx_3)

    def test_backward_cpu(self):
        self.check_backward(self.t, tuple(x_data for x_data in self.x))

    @attr.gpu
    def test_backward_gpu(self):
        self.gx = cuda.to_gpu(self.g)
        self.check_backward(cuda.to_gpu(self.t),
                            tuple(cuda.to_gpu(x_data) for x_data in self.x))

testing.run_module(__name__, __file__)
