import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


if cuda.available:
    cuda.init()


def _ones(gpu, *shape):
    if gpu:
        return chainer.Variable(cuda.ones(shape).astype(numpy.float32))
    return chainer.Variable(numpy.ones(shape).astype(numpy.float32))


class TestInceptionBNBase(unittest.TestCase):
    in_channels = 3
    out1, proj3, out3, proj33, out33, proj_pool = 3, 2, 3, 2, 3, 3
    pooltype = 'max'
    stride = 1
    batchsize = 10
    insize = 10

    def _setup_inceptionbn(self):
        return functions.InceptionBN(
            self.in_channels, self.out1, self.proj3, self.out3,
            self.proj33, self.out33, self.pooltype, self.proj_pool,
            self.stride)

    def setup_mock(self, gpu):
        self.f.f = mock.MagicMock()
        if self.out1 > 0:
            self.f.f.conv1.return_value = self._ones(gpu, self.out1)
            self.f.f.conv1n.return_value = self._ones(gpu, self.out1)
        self.f.f.proj3.return_value = self._ones(gpu, self.proj3)
        self.f.f.proj3n.return_value = self._ones(gpu, self.proj3)
        self.f.f.conv3.return_value = self._ones(gpu, self.out3)
        self.f.f.conv3n.return_value = self._ones(gpu, self.out3)
        self.f.f.proj33.return_value = self._ones(gpu, self.proj33)
        self.f.f.proj33n.return_value = self._ones(gpu, self.proj33)
        self.f.f.conv33a.return_value = self._ones(gpu, self.out33)
        self.f.f.conv33an.return_value = self._ones(gpu, self.out33)
        self.f.f.conv33b.return_value = self._ones(gpu, self.out33)
        self.f.f.conv33bn.return_value = self._ones(gpu, self.out33)
        self.f.f.pool.return_value = self._ones(gpu, 3)
        if self.proj_pool is not None:
            self.f.f.poolp.return_value = self._ones(gpu, self.proj_pool)
            self.f.f.poolpn.return_value = self._ones(gpu, self.proj_pool)

    def _ones(self, gpu, out_channels):
        return _ones(gpu, self.batchsize, out_channels,
                     self.insize, self.insize)

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (10, self.in_channels, 5, 5)
        ).astype(numpy.float32)


class TestInceptionBN(TestInceptionBNBase):

    def setUp(self):
        super(TestInceptionBN, self).setUp()
        self.f = self._setup_inceptionbn()

    def check_backward(self, x_data, gpu):
        x = chainer.Variable(x_data)
        y = self.f(x)
        y_grad = numpy.random.uniform(
            -1, 1, y.data.shape).astype(numpy.float32)
        if gpu:
            y_grad = cuda.to_gpu(y_grad)
        y.grad = y_grad
        y.backward()

    def test_backward_cpu(self):
        self.check_backward(self.x, False)

    @attr.gpu
    def test_backward_gpu(self):
        self.f.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), True)

    def check_call(self, x, f, gpu):
        self.setup_mock(gpu)
        f(chainer.Variable(x))

        expected = []

        # Variable.__eq__ raises NotImplementedError,
        # so we cannot check arguments
        if self.out1 > 0:
            expected.extend([mock.call.conv1(mock.ANY),
                             mock.call.conv1n(mock.ANY)])

        expected.extend([mock.call.proj3(mock.ANY),
                         mock.call.proj3n(mock.ANY),
                         mock.call.conv3(mock.ANY),
                         mock.call.conv3n(mock.ANY),
                         mock.call.proj33(mock.ANY),
                         mock.call.proj33n(mock.ANY),
                         mock.call.conv33a(mock.ANY),
                         mock.call.conv33an(mock.ANY),
                         mock.call.conv33b(mock.ANY),
                         mock.call.conv33bn(mock.ANY),
                         mock.call.pool(mock.ANY)])

        if self.proj_pool is not None:
            expected.extend([mock.call.poolp(mock.ANY),
                             mock.call.poolpn(mock.ANY)])

        self.assertListEqual(self.f.f.mock_calls, expected)

    def test_call_cpu(self):
        self.check_call(self.x, self.f, False)

    @attr.gpu
    def test_call_gpu(self):
        x = cuda.to_gpu(self.x)
        self.f.to_gpu()
        self.check_call(x, self.f, True)


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
            self.f = self._setup_inceptionbn()


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
            self.f = self._setup_inceptionbn()


testing.run_module(__name__, __file__)
