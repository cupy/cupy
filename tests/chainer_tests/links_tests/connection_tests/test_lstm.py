import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'in_size': 10, 'out_size': 10},
        {'in_size': 10, 'out_size': 40},
    ],
    [
        {'input_none': False},
        {'input_none': True},
    ],
    [
        {'input_variable': False},
        {'input_variable': True},
    ]
))
class TestLSTM(unittest.TestCase):

    def setUp(self):
        in_size = None if self.input_none else self.in_size
        self.link = links.LSTM(in_size, self.out_size)
        self.link.cleargrads()
        x1_shape = (4, self.in_size)
        self.x1 = numpy.random.uniform(-1, 1, x1_shape).astype(numpy.float32)
        x2_shape = (3, self.in_size)
        self.x2 = numpy.random.uniform(-1, 1, x2_shape).astype(numpy.float32)
        x3_shape = (0, self.in_size)
        self.x3 = numpy.random.uniform(-1, 1, x3_shape).astype(numpy.float32)

    def check_forward(self, x1_data, x2_data, x3_data):
        xp = self.link.xp
        x1 = chainer.Variable(x1_data) if self.input_variable else x1_data
        h1 = self.link(x1)
        with cuda.get_device(x1_data):
            c0 = chainer.Variable(xp.zeros((len(self.x1), self.out_size),
                                           dtype=self.x1.dtype))
            c1_expect, h1_expect = functions.lstm(c0, self.link.upward(x1))
        testing.assert_allclose(h1.data, h1_expect.data)
        testing.assert_allclose(self.link.h.data, h1_expect.data)
        testing.assert_allclose(self.link.c.data, c1_expect.data)

        batch = len(x2_data)
        x2 = chainer.Variable(x2_data) if self.input_variable else x2_data
        h1_in, h1_rest = functions.split_axis(
            self.link.h.data, [batch], axis=0)
        y2 = self.link(x2)
        with cuda.get_device(x1):
            c2_expect, y2_expect = \
                functions.lstm(c1_expect,
                               self.link.upward(x2) + self.link.lateral(h1_in))
        testing.assert_allclose(y2.data, y2_expect.data)
        testing.assert_allclose(self.link.h.data[:batch], y2_expect.data)
        testing.assert_allclose(self.link.h.data[batch:], h1_rest.data)

        x3 = chainer.Variable(x3_data) if self.input_variable else x3_data
        h2_rest = self.link.h
        y3 = self.link(x3)
        c3_expect, y3_expect = \
            functions.lstm(c2_expect, self.link.upward(x3))
        testing.assert_allclose(y3.data, y3_expect.data)
        testing.assert_allclose(self.link.h.data, h2_rest.data)

    def test_forward_cpu(self):
        self.check_forward(self.x1, self.x2, self.x3)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2),
                           cuda.to_gpu(self.x3))

    @attr.multi_gpu(2)
    def test_forward_gpu_multi(self):
        with cuda.get_device(0):
            self.link.to_gpu()
            x1 = cuda.to_gpu(self.x1)
            x2 = cuda.to_gpu(self.x2)
            x3 = cuda.to_gpu(self.x3)
        with cuda.get_device(1):
            self.check_forward(x1, x2, x3)


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


class TestLSTMInvalidSize(unittest.TestCase):

    in_size = 10
    out_size = 20

    def setUp(self):
        self.link = links.LSTM(self.in_size, self.out_size)
        upward = self.link.upward.W.data
        upward[...] = numpy.random.uniform(-1, 1, upward.shape)
        lateral = self.link.lateral.W.data
        lateral[...] = numpy.random.uniform(-1, 1, lateral.shape)

        x1_shape = (4, self.in_size)
        self.x1 = numpy.random.uniform(-1, 1, x1_shape).astype(numpy.float32)
        x2_shape = (5, self.in_size)
        self.x2 = numpy.random.uniform(-1, 1, x2_shape).astype(numpy.float32)

    def check_forward_invalid_size(self, x1_data, x2_data):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        self.link(x1)
        with self.assertRaises(TypeError):
            self.link(x2)

    def test_forward_invalid_size_cpu(self):
        self.check_forward_invalid_size(self.x1, self.x2)

    @attr.gpu
    def test_forward_invalid_size_gpu(self):
        self.link.to_gpu()
        self.check_forward_invalid_size(cuda.to_gpu(self.x1),
                                        cuda.to_gpu(self.x2))


@testing.parameterize(*testing.product_dict(
    [
        {'in_size': 10, 'out_size': 10},
        {'in_size': 10, 'out_size': 40},
    ],
    [
        {'input_none': False},
        {'input_none': True},
    ],
    [
        {'input_variable': False},
        {'input_variable': True},
    ]
))
class TestStatelessLSTM(unittest.TestCase):

    def setUp(self):
        in_size = None if self.input_none else self.in_size
        self.link = links.StatelessLSTM(in_size, self.out_size)
        self.link.cleargrads()

        x_shape = (4, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)

    def check_forward(self, x_data):
        xp = self.link.xp
        x = chainer.Variable(x_data) if self.input_variable else x_data
        c1, h1 = self.link(None, None, x)
        with cuda.get_device(x_data):
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

    @attr.multi_gpu(2)
    def test_forward_gpu_multi(self):
        with cuda.get_device(0):
            self.link.to_gpu()
            x = cuda.to_gpu(self.x)
        with cuda.get_device(1):
            self.check_forward(x)


testing.run_module(__name__, __file__)
