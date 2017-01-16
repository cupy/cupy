import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr


def _sigmoid(x):
    xp = cuda.get_array_module(x)
    half = x.dtype.type(0.5)
    return xp.tanh(x * half) * half + half


def _peephole(func, c, h, x):
    xp = cuda.get_array_module(x)
    with cuda.get_device(x):
        lstm_in = x.dot(func.upward.W.data.T)
        lstm_in += h.dot(func.lateral.W.data.T)
        lstm_in = xp.reshape(lstm_in, (len(lstm_in),
                                       lstm_in.shape[1] // 4,
                                       4))
        a, i, f, o = xp.split(lstm_in, 4, 2)
        a = xp.reshape(a, (len(a), a.shape[1]))
        i = xp.reshape(i, (len(i), i.shape[1]))
        f = xp.reshape(f, (len(f), f.shape[1]))
        o = xp.reshape(o, (len(o), o.shape[1]))
        peep_in_i = c.dot(func.peep_i.W.data.T)
        peep_in_f = c.dot(func.peep_f.W.data.T)
        a = xp.tanh(a)
        i = _sigmoid(i + peep_in_i)
        f = _sigmoid(f + peep_in_f)
        c_next = a * i + f * c
        peep_in_o = c_next.dot(func.peep_o.W.data.T)
        o = _sigmoid(o + peep_in_o)
        y = o * xp.tanh(c_next)
    return c_next, y


@testing.parameterize(
    {'in_size': 10, 'out_size': 10},
    {'in_size': 10, 'out_size': 40},
)
class TestPeephole(unittest.TestCase):

    def setUp(self):
        self.link = links.StatefulPeepholeLSTM(self.in_size, self.out_size)
        upward = self.link.upward.W.data
        upward[...] = numpy.random.uniform(-1, 1, upward.shape)
        lateral = self.link.lateral.W.data
        lateral[...] = numpy.random.uniform(-1, 1, lateral.shape)
        peep_i = self.link.peep_i.W.data
        peep_i[...] = numpy.random.uniform(-1, 1, peep_i.shape)
        peep_f = self.link.peep_f.W.data
        peep_f[...] = numpy.random.uniform(-1, 1, peep_f.shape)
        peep_o = self.link.peep_o.W.data
        peep_o[...] = numpy.random.uniform(-1, 1, peep_o.shape)

        c_shape = (1, self.out_size)
        h_shape = (1, self.out_size)
        x_shape = (4, self.in_size)
        gy_shape = (4, self.out_size)
        self.c = numpy.zeros(c_shape).astype(numpy.float32)
        self.h = numpy.zeros(h_shape).astype(numpy.float32)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(numpy.float32)

    def _forward(self, link, x):
        return link(x)

    def check_forward(self, c_data, h_data, x_data):
        x = chainer.Variable(x_data)

        h1 = self.link(x)
        c1_expect, h1_expect = _peephole(self.link, c_data, h_data, x_data)
        testing.assert_allclose(h1.data, h1_expect)
        testing.assert_allclose(self.link.c.data, c1_expect)
        testing.assert_allclose(self.link.h.data, h1_expect)

        h2 = self.link(x)
        c2_expect, h2_expect = _peephole(self.link,
                                         c1_expect, h1_expect, x_data)
        testing.assert_allclose(h2.data, h2_expect)
        testing.assert_allclose(self.link.c.data, c2_expect)
        testing.assert_allclose(self.link.h.data, h2_expect)

    def test_forward_cpu(self):
        self.check_forward(self.c, self.h, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.c),
                           cuda.to_gpu(self.h),
                           cuda.to_gpu(self.x))

    @attr.multi_gpu(2)
    def test_forward_gpu_multi(self):
        with cuda.get_device(0):
            self.link.to_gpu()
            c = cuda.to_gpu(self.c)
            h = cuda.to_gpu(self.h)
            x = cuda.to_gpu(self.x)
        with cuda.get_device(1):
            self.check_forward(c, h, x)

    def check_backward(self, c_data, h_data, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self._forward(self.link, x)
        y.grad = y_grad
        y.backward()

        def f():
            c, y = _peephole(self.link, c_data, h_data, x_data)
            return y,
        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,))
        testing.assert_allclose(gx, x.grad, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.c, self.h, self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.c),
                            cuda.to_gpu(self.h),
                            cuda.to_gpu(self.x),
                            cuda.to_gpu(self.gy))


class TestPeepholeState(unittest.TestCase):

    def setUp(self):
        in_size, out_size = 10, 8
        self.link = links.StatefulPeepholeLSTM(in_size, out_size)

    def check_reset_state(self):
        self.link.reset_state()
        self.assertIsNone(self.link.c)
        self.assertIsNone(self.link.h)

    def test_reset_state_cpu(self):
        self.check_reset_state()

    @attr.gpu
    def test_reset_state_gpu(self):
        self.link.to_gpu()
        self.check_reset_state()


class TestPeepholeToCPUToGPU(unittest.TestCase):

    def setUp(self):
        in_size, out_size = 10, 8
        self.link = links.StatefulPeepholeLSTM(in_size, out_size)
        self.c = chainer.Variable(
            numpy.random.uniform(-1, 1, (1, out_size)).astype(numpy.float32))
        self.h = chainer.Variable(
            numpy.random.uniform(-1, 1, (1, out_size)).astype(numpy.float32))

    def check_to_cpu(self, c, h):
        self.link.c = c
        self.link.h = h
        self.link.to_cpu()
        self.assertIs(self.link.xp, numpy)
        self.assertIsInstance(self.link.c.data, self.link.xp.ndarray)
        self.assertIsInstance(self.link.h.data, self.link.xp.ndarray)
        self.link.to_cpu()
        self.assertIs(self.link.xp, numpy)
        self.assertIsInstance(self.link.c.data, self.link.xp.ndarray)
        self.assertIsInstance(self.link.h.data, self.link.xp.ndarray)

    def test_to_cpu_cpu(self):
        self.check_to_cpu(self.c, self.h)

    @attr.gpu
    def test_to_cpu_gpu(self):
        self.c.to_gpu()
        self.h.to_gpu()
        self.check_to_cpu(self.c, self.h)

    def check_to_cpu_to_gpu(self, c, h):
        self.link.c = c
        self.link.h = h
        self.link.to_gpu()
        self.assertIs(self.link.xp, cuda.cupy)
        self.assertIsInstance(self.link.c.data, self.link.xp.ndarray)
        self.assertIsInstance(self.link.h.data, self.link.xp.ndarray)
        self.link.to_gpu()
        self.assertIs(self.link.xp, cuda.cupy)
        self.assertIsInstance(self.link.c.data, self.link.xp.ndarray)
        self.assertIsInstance(self.link.h.data, self.link.xp.ndarray)
        self.link.to_cpu()
        self.assertIs(self.link.xp, numpy)
        self.assertIsInstance(self.link.c.data, self.link.xp.ndarray)
        self.assertIsInstance(self.link.h.data, self.link.xp.ndarray)
        self.link.to_gpu()
        self.assertIs(self.link.xp, cuda.cupy)
        self.assertIsInstance(self.link.c.data, self.link.xp.ndarray)
        self.assertIsInstance(self.link.h.data, self.link.xp.ndarray)

    @attr.gpu
    def test_to_cpu_to_gpu_cpu(self):
        self.check_to_cpu_to_gpu(self.c, self.h)

    @attr.gpu
    def test_to_cpu_to_gpu_gpu(self):
        self.c.to_gpu()
        self.h.to_gpu()
        self.check_to_cpu_to_gpu(self.c, self.h)


testing.run_module(__name__, __file__)
