import math
import numpy
import unittest

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestCTC(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 2, 3)).astype(numpy.float32)
        self.t = numpy.array([[0, 1], [1, 0]]).astype(numpy.int32)
        self.l = numpy.array([[2, 0, 2, 1, 2],
                              [2, 1, 2, 0, 2]]).astype(numpy.int32)
        self.blank_symbol = 2
        self.g = numpy.array(0.1, dtype=numpy.float32)
        self.gx = self.g

    # recursive forward computation.
    def alpha(self, x, l, t, u):
        if u < 0:
            return 0.0
        if t == 0:
            if u == 0:
                return x[0][self.blank_symbol]
            elif u == 1:
                return x[0][l[1]]
            else:
                return 0.0
        elif l[u] == self.blank_symbol or l[u] == l[u-2]:
            return x[t][l[u]] * \
                (self.alpha(x, l, t-1, u-1) + self.alpha(x, l, t-1, u))
        else:
            return x[t][l[u]] * \
                (self.alpha(x, l, t-1, u-2)
                 + self.alpha(x, l, t-1, u-1)
                 + self.alpha(x, l, t-1, u))

    def check_forward(self, t_data, xs_data):
        x = tuple(chainer.Variable(x_data) for x_data in xs_data)
        t = chainer.Variable(t_data)
        loss = functions.connectionist_temporal_classification(x, t, 2)
        loss_value = float(loss.data)

        # compute expected value by recursive computation.
        xp = cuda.get_array_module(self.x)
        xt = xp.swapaxes(self.x, 0, 1)
        for b in range(xt.shape[0]):
            for t in range(xt.shape[1]):
                xt[b][t] = numpy.exp(xt[b][t]) / numpy.sum(numpy.exp(xt[b][t]))
        loss_expect = 0
        batch_size = xt.shape[0]
        for b in range(batch_size):
            loss_expect += -math.log(self.alpha(xt[b],
                                                self.l[b],
                                                self.x.shape[0]-1,
                                                self.l[b].shape[0]-1)
                                     + self.alpha(xt[b],
                                                  self.l[b],
                                                  self.x.shape[0]-1,
                                                  self.l[b].shape[0]-2))
        loss_expect /= batch_size
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
        loss = functions.connectionist_temporal_classification(xs, t, 2)
        loss.grad = self.g
        loss.backward()

        func = loss.creator
        xs_data = tuple(x.data for x in xs)
        f = lambda: func.forward((t.data,) + xs_data)
        gl_0, gx_0, gx_1, gx_2, gx_3 = gradient_check.numerical_grad(
            f, ((t.data,) + xs_data), (self.gx,))
        gradient_check.assert_allclose(xs[0].grad, gx_0, atol=1e-05)
        gradient_check.assert_allclose(xs[1].grad, gx_1, atol=1e-05)
        gradient_check.assert_allclose(xs[2].grad, gx_2, atol=1e-05)
        gradient_check.assert_allclose(xs[3].grad, gx_3, atol=1e-05)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.t, tuple(x_data for x_data in self.x))

    @condition.retry(3)
    @attr.gpu
    def test_backward_gpu(self):
        self.g = cuda.to_gpu(self.g)
        self.gx = self.g
        self.check_backward(cuda.to_gpu(self.t),
                            tuple(cuda.to_gpu(x_data) for x_data in self.x))

testing.run_module(__name__, __file__)
