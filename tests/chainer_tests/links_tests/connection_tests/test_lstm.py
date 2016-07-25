import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'in_size': 10, 'out_size': 10},
    {'in_size': 10, 'out_size': 40},
)
class TestLSTM(unittest.TestCase):

    def setUp(self):
        self.link = links.LSTM(self.in_size, self.out_size)
        upward = self.link.upward.W.data
        upward[...] = numpy.random.uniform(-1, 1, upward.shape)
        lateral = self.link.lateral.W.data
        lateral[...] = numpy.random.uniform(-1, 1, lateral.shape)
        self.link.zerograds()

        self.upward = upward.copy()  # fixed on CPU
        self.lateral = lateral.copy()  # fixed on CPU

        x_shape = (4, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)

    def check_forward(self, x_data):
        xp = self.link.xp
        x = chainer.Variable(x_data)
        h1 = self.link(x)
        c0 = chainer.Variable(xp.zeros((len(self.x), self.out_size),
                                       dtype=self.x.dtype))
        c1_expect, h1_expect = functions.lstm(c0, self.link.upward(x))
        testing.assert_allclose(h1.data, h1_expect.data)
        testing.assert_allclose(self.link.h.data, h1_expect.data)
        testing.assert_allclose(self.link.c.data, c1_expect.data)

        h2 = self.link(x)
        c2_expect, h2_expect = \
            functions.lstm(c1_expect,
                           self.link.upward(x) + self.link.lateral(h1))
        testing.assert_allclose(h2.data, h2_expect.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))


class TestLSTMState(unittest.TestCase):

    def setUp(self):
        self.link = links.LSTM(5, 7)
        self.x = chainer.Variable(
            numpy.random.uniform(-1, 1, (3, 5)).astype(numpy.float32))
        self.c = chainer.Variable(
            numpy.random.uniform(-1, 1, (3, 5)).astype(numpy.float32))
        self.h = chainer.Variable(
            numpy.random.uniform(-1, 1, (3, 5)).astype(numpy.float32))

    def check_state(self):
        self.assertIsNone(self.link.c)
        self.assertIsNone(self.link.h)
        self.link(self.x)
        self.assertIsNotNone(self.link.c)
        self.assertIsNotNone(self.link.h)

    def test_state_cpu(self):
        self.check_state()

    @attr.gpu
    def test_state_gpu(self):
        self.link.to_gpu()
        self.x.to_gpu()
        self.check_state()

    def check_set_state(self, c, h):
        self.link.set_state(c, h)
        self.assertIsInstance(self.link.c.data, self.link.xp.ndarray)
        testing.assert_allclose(c.data, self.link.c.data)
        self.assertIsInstance(self.link.h.data, self.link.xp.ndarray)
        testing.assert_allclose(h.data, self.link.h.data)

    def test_set_state_cpu(self):
        self.check_set_state(self.c, self.h)

    @attr.gpu
    def test_set_state_gpu(self):
        self.link.to_gpu()
        self.check_set_state(self.c, self.h)

    def check_reset_state(self):
        self.link(self.x)
        self.link.reset_state()
        self.assertIsNone(self.link.c)
        self.assertIsNone(self.link.h)

    def test_reset_state_cpu(self):
        self.check_reset_state()

    @attr.gpu
    def test_reset_state_gpu(self):
        self.link.to_gpu()
        self.x.to_gpu()
        self.check_reset_state()


class TestLSTMToCPUToGPU(unittest.TestCase):

    def setUp(self):
        self.link = links.LSTM(5, 7)
        self.x = chainer.Variable(
            numpy.random.uniform(-1, 1, (3, 5)).astype(numpy.float32))

    def check_to_cpu(self, s):
        self.link.to_cpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)
        self.link.to_cpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)

    def test_to_cpu_cpu(self):
        self.link(self.x)
        self.check_to_cpu(self.link.c)
        self.check_to_cpu(self.link.h)

    @attr.gpu
    def test_to_cpu_gpu(self):
        self.link.to_gpu()
        self.x.to_gpu()
        self.link(self.x)
        self.check_to_cpu(self.link.c)
        self.check_to_cpu(self.link.h)

    def check_to_cpu_to_gpu(self, s):
        self.link.to_gpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)
        self.link.to_gpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)
        self.link.to_cpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)
        self.link.to_gpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)

    @attr.gpu
    def test_to_cpu_to_gpu_cpu(self):
        self.link(self.x)
        self.check_to_cpu_to_gpu(self.link.c)
        self.check_to_cpu_to_gpu(self.link.h)

    @attr.gpu
    def test_to_cpu_to_gpu_gpu(self):
        self.link.to_gpu()
        self.x.to_gpu()
        self.link(self.x)
        self.check_to_cpu_to_gpu(self.link.c)
        self.check_to_cpu_to_gpu(self.link.h)


@testing.parameterize(
    {'in_size': 10, 'out_size': 10},
    {'in_size': 10, 'out_size': 40},
)
class TestStatelessLSTM(unittest.TestCase):

    def setUp(self):
        self.link = links.StatelessLSTM(self.in_size, self.out_size)
        upward = self.link.upward.W.data
        upward[...] = numpy.random.uniform(-1, 1, upward.shape)
        lateral = self.link.lateral.W.data
        lateral[...] = numpy.random.uniform(-1, 1, lateral.shape)
        self.link.zerograds()

        self.upward = upward.copy()  # fixed on CPU
        self.lateral = lateral.copy()  # fixed on CPU

        x_shape = (4, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)

    def check_forward(self, x_data):
        xp = self.link.xp
        x = chainer.Variable(x_data)
        c1, h1 = self.link(None, None, x)
        c0 = chainer.Variable(xp.zeros((len(self.x), self.out_size),
                                       dtype=self.x.dtype))
        c1_expect, h1_expect = functions.lstm(c0, self.link.upward(x))
        testing.assert_allclose(h1.data, h1_expect.data)
        testing.assert_allclose(c1.data, c1_expect.data)

        c2, h2 = self.link(c1, h1, x)
        c2_expect, h2_expect = \
            functions.lstm(c1_expect,
                           self.link.upward(x) + self.link.lateral(h1))
        testing.assert_allclose(h2.data, h2_expect.data)
        testing.assert_allclose(c2.data, c2_expect.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
