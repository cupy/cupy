import unittest

import numpy

import chainer
from chainer import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr


class TestInceptionBNBase(unittest.TestCase):

    in_channels = 3
    out1, proj3, out3, proj33, out33, proj_pool = 3, 2, 3, 2, 3, 3
    pooltype = 'max'
    stride = 1
    batchsize = 10
    insize = 10

    def setup_data(self):
        self.x = numpy.random.uniform(
            -1, 1, (10, self.in_channels, 5, 5)
        ).astype(numpy.float32)
        self.l = links.InceptionBN(
            self.in_channels, self.out1, self.proj3, self.out3,
            self.proj33, self.out33, self.pooltype, self.proj_pool,
            self.stride)

    def check_backward(self, x_data):
        xp = cuda.get_array_module(x_data)
        x = chainer.Variable(x_data)
        y = self.l(x)
        y.grad = xp.random.randn(*y.data.shape).astype('f')
        y.backward()


class TestInceptionBN(TestInceptionBNBase):

    def setUp(self):
        self.setup_data()

    def test_backward_cpu(self):
        self.check_backward(self.x)

    @attr.gpu
    def test_backward_gpu(self):
        self.l.to_gpu()
        self.check_backward(cuda.to_gpu(self.x))


class TestInceptionBN2(TestInceptionBN):

    pooltype = 'avg'


class TestInceptionBN3(TestInceptionBN):

    out1 = 0


class TestInceptionBN4(TestInceptionBN):

    out1 = 0
    pooltype = 'avg'


class TestInceptionBN5(TestInceptionBN):

    out1 = 0
    proj_pool = None


class TestInceptionBN6(TestInceptionBN):

    out1 = 0
    pooltype = 'avg'
    proj_pool = None


class TestInceptionBN7(TestInceptionBN):

    out1 = 0
    stride = 2


class TestInceptionBN8(TestInceptionBN):

    out1 = 0
    stride = 2
    proj_pool = None


class TestInceptionBN9(TestInceptionBN):

    out1 = 0
    stride = 2
    pooltype = 'avg'


class TestInceptionBN10(TestInceptionBN):

    out1 = 0
    stride = 2
    pooltype = 'avg'
    proj_pool = None


class TestInceptionBNInvalidCall(TestInceptionBNBase):

    proj_pool = None

    def test_invalid_architecture(self):
        with self.assertRaises(AssertionError):
            self.setup_data()


class TestInceptionBNInvalidCall2(TestInceptionBNInvalidCall):

    pooltype = 'avg'
    proj_pool = None


class TestInceptionBNInvalidCall3(TestInceptionBNInvalidCall):

    stride = 2


class TestInceptionBNInvalidCall4(TestInceptionBNInvalidCall):

    stride = 2
    pooltype = 'avg'


class TestInceptionBNInvalidPoolType(TestInceptionBNBase):

    pooltype = 'invalid_pooltype'

    def test_invalid_pooltype(self):
        with self.assertRaises(NotImplementedError):
            self.setup_data()


testing.run_module(__name__, __file__)
