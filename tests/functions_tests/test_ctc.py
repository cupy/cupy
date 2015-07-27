import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr

if cuda.available:
    cuda.init()


class TestCTC(unittest.TestCase):

    def setUp(self):
        """# initialize x as actual value

        self.x = numpy.array([[0.76071646,  0.28901242, 0.89631007],
                              [0.86875586,  0.79419921, 0.98096017],
                              [0.94204353,  0.63607605, 0.90673191],
                              [0.88267389,  0.03590736, 0.71339421]])
        self.t = numpy.array([0,1]) #initialize t as expected label
        self.x = numpy.array([[0.99,0.,0.01], [0.99,0., 0.01],
                              [0.99,0., 0.01], [0.99,0., 0.01]])
        self.t = numpy.array([0])
        """
        self.x = numpy.array([[0.99, 0., 0.01],
                              [0.45, 0.45, 0.1],
                              [0.1, 0.7, 0.2],
                              [0.1, 0.78, 0.12]])
        self.t = numpy.array([0, 1])

    # compute actual value from x_data,
    # expected value(via numerical differentiation) from t_data
    def check_forward(self, t_data, xs_data):
        x = tuple(chainer.Variable(x_data) for x_data in xs_data)
        t = chainer.Variable(t_data)
        loss = functions.connectionist_temporal_classification_cost(t, x)
        loss_value = float(loss.data)

        # Todo: compute expected value
        loss_expect = 1.0

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    def test_forward_cpu(self):
        self.check_forward(self.t, tuple(x_data for x_data in self.x))

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(self.t,
                           tuple(cuda.to_gpu(x_data) for x_data in self.x))

    def check_backward(self, t_data, xs_data):
        xs = tuple(chainer.Variable(x_data) for x_data in xs_data)
        t = chainer.Variable(t_data)
        loss = functions.connectionist_temporal_classification_cost(t, xs)
        loss.backward()

        func = loss.creator
        xs_data = tuple(x.data for x in xs)
        f = lambda: func.forward((t.data,) + xs_data)
        gl_0, gx_0, gx_1, gx_2, gx_3 = gradient_check.numerical_grad(
            f, ((t.data,) + xs_data), (1, ), eps=0.0001)
        gradient_check.assert_allclose(xs[0].grad, gx_0)
        gradient_check.assert_allclose(xs[1].grad, gx_1)
        gradient_check.assert_allclose(xs[2].grad, gx_2)
        gradient_check.assert_allclose(xs[3].grad, gx_3)

    def test_backward_cpu(self):
        self.check_backward(self.t, tuple(x_data for x_data in self.x))

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(self.t,
                            tuple(cuda.to_gpu(x_data) for x_data in self.x))
