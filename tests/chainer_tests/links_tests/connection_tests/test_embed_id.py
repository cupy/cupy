import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(
    {'x_data': [0, 1, 0], 'ignore_label': None},
    {'x_data': [[0, 1, 0], [1, 0, 1]], 'ignore_label': None},
    {'x_data': [0, 1, -1], 'ignore_label': -1},
    {'x_data': [[0, 1, -1], [-1, 0, 1]], 'ignore_label': -1},
)
class TestEmbedID(unittest.TestCase):

    def setUp(self):
        self.link = links.EmbedID(3, 2, ignore_label=self.ignore_label)
        self.link.ignore_label
        self.link.cleargrads()

        self.W = self.link.W.data.copy()  # fixed on CPU
        self.x = numpy.array(self.x_data, dtype=numpy.int32)
        y_shape = self.x.shape + (2,)
        self.gy = numpy.random.uniform(-1, 1, y_shape).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = numpy.empty_like(self.gy)
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] == -1:
                y_expect[i] = 0
            else:
                y_expect[i] = self.W[int(self.x[i])]

        testing.assert_allclose(y_expect, y.data, atol=0, rtol=0)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, self.link.W)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


@testing.parameterize(
    {'t_value': -1, 'valid': False, 'ignore_label': None},
    {'t_value': 3,  'valid': False, 'ignore_label': None},
    {'t_value': 0,  'valid': True,  'ignore_label': None},
    {'t_value': -1, 'valid': True,  'ignore_label': -1},
    {'t_value': 3,  'valid': False, 'ignore_label': -1},
    {'t_value': 0,  'valid': True,  'ignore_label': -1},
)
class TestEmbedIDValueCheck(unittest.TestCase):

    def setUp(self):
        self.link = links.EmbedID(2, 2, ignore_label=self.ignore_label)
        self.t = numpy.array([self.t_value], dtype=numpy.int32)
        self.original_debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.original_debug)

    def check_value_check(self, t_data):
        t = chainer.Variable(t_data)

        if self.valid:
            # Check if it throws nothing
            self.link(t)
        else:
            with self.assertRaises(ValueError):
                self.link(t)

    def test_value_check_cpu(self):
        self.check_value_check(self.t)

    @attr.gpu
    def test_value_check_gpu(self):
        self.check_value_check(self.t)


class TestEmbedIDUnpickleOldFile(unittest.TestCase):

    def test_old_unpickle(self):
        embed = links.EmbedID(3, 4)
        # To emulate an old pickled file
        delattr(embed, 'ignore_label')
        x = chainer.Variable(numpy.arange(2, dtype=numpy.int32))
        y = embed(x)
        self.assertEqual(y.data.shape, (2, 4))


testing.run_module(__name__, __file__)
