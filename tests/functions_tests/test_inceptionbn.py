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


class TestInceptionBNBase(unittest.TestCase):
    in_channels = 3
    out1, proj3, out3, proj33, out33, proj_pool = 3, 2, 3, 2, 3, 3

    def _setup_inceptionbn(self, pooltype, proj_pool=None, stride=1):
        return functions.InceptionBN(
            self.in_channels, self.out1, self.proj3, self.out3,
            self.proj33, self.out33, pooltype, proj_pool, stride)


    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (10, self.in_channels, 5, 5)
        ).astype(numpy.float32)
        self.f = self._setup_inceptionbn('max', self.proj_pool)

class TestInceptionBNBackward(TestInceptionBNBase):

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


class TestInceptionBNBackward2(TestInceptionBNBackward):

    def setUp(self):
        super(TestInceptionBNBackward2, self).setUp()
        self.f = self._setup_inceptionbn('avg', self.proj_pool)


class TestInceptionBNBackward3(TestInceptionBNBackward):

    out1 = 0

    def setUp(self):
        super(TestInceptionBNBackward3, self).setUp()
        self.f = self._setup_inceptionbn('max')


class TestInceptionBNBackward4(TestInceptionBNBackward):

    out1 = 0

    def setUp(self):
        super(TestInceptionBNBackward4, self).setUp()
        self.f = self._setup_inceptionbn('avg')


class TestInceptionBNBackward5(TestInceptionBNBackward):

    out1 = 0

    def setUp(self):
        super(TestInceptionBNBackward5, self).setUp()
        self.f = self._setup_inceptionbn('max', self.proj_pool)


class TestInceptionBNBackward6(TestInceptionBNBackward):

    out1 = 0

    def setUp(self):
        super(TestInceptionBNBackward6, self).setUp()
        self.f = self._setup_inceptionbn('avg', self.proj_pool)


class TestInceptionBNBackward7(TestInceptionBNBackward):

    out1 = 0

    def setUp(self):
        super(TestInceptionBNBackward7, self).setUp()
        self.f = self._setup_inceptionbn('max', stride=2)


class TestInceptionBNBackward8(TestInceptionBNBackward):

    out1 = 0

    def setUp(self):
        super(TestInceptionBNBackward8, self).setUp()
        self.f = self._setup_inceptionbn('avg', stride=2)


class TestInceptionBNBackward9(TestInceptionBNBackward):

    out1 = 0

    def setUp(self):
        super(TestInceptionBNBackward9, self).setUp()
        self.f = self._setup_inceptionbn('max', self.proj_pool, stride=2)


class TestInceptionBNBackward10(TestInceptionBNBackward):

    out1 = 0

    def setUp(self):
        super(TestInceptionBNBackward10, self).setUp()
        self.f = self._setup_inceptionbn('avg', self.proj_pool, stride=2)


class TestInceptionBNInvalidForward(TestInceptionBNBase):

    def test_invalid_architecture(self):
        with self.assertRaises(AssertionError):
            self.f = self._setup_inceptionbn('avg')

    def test_invalid_architecture2(self):
        with self.assertRaises(AssertionError):
            self.f = self._setup_inceptionbn('max')

    def test_invalid_architecture3(self):
        with self.assertRaises(AssertionError):
            self.f = self._setup_inceptionbn('avg', self.proj_pool, stride=2)

    def test_invalid_architecture4(self):
        with self.assertRaises(AssertionError):
            self.f = self._setup_inceptionbn('max', self.proj_pool, stride=2)


def _ones(gpu, *shape):
    if gpu:
        return chainer.Variable(cuda.ones(shape).astype(numpy.float32))
    return chainer.Variable(numpy.ones(shape).astype(numpy.float32))


class TestInceptionForward(unittest.TestCase):

    in_channels = 3
    out1, proj3, out3, proj33, out33, proj_pool = 3, 2, 3, 2, 3, 3
    batchsize = 10
    insize = 10

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (10, self.in_channels, 5, 5)
        ).astype(numpy.float32)
        self.f = functions.InceptionBN(
            self.in_channels, self.out1,
            self.proj3, self.out3,
            self.proj33, self.out33, 'max', self.proj_pool)

    def _ones(self, gpu, out_channels):
        return _ones(gpu, self.batchsize, out_channels, self.insize, self.insize)

    def setup_mock(self, gpu):
        self.f.f = mock.MagicMock()
        self.f.f.conv1.return_value = self._ones(gpu, self.out1)
        self.f.f.conv1n.return_value = self._ones(gpu, self.out1)
        self.f.f.proj3.return_value = self._ones(gpu, self.proj3)
        self.f.f.proj3n.return_value = self._ones(gpu,self.proj3)
        self.f.f.conv3.return_value = self._ones(gpu, self.out3)
        self.f.f.conv3n.return_value = self._ones(gpu, self.out3)
        self.f.f.proj33.return_value = self._ones(gpu, self.proj33)
        self.f.f.proj33n.return_value = self._ones(gpu, self.proj33)
        self.f.f.conv33a.return_value = self._ones(gpu, self.out33)
        self.f.f.conv33an.return_value = self._ones(gpu, self.out33)
        self.f.f.conv33b.return_value = self._ones(gpu, self.out33)
        self.f.f.conv33bn.return_value = self._ones(gpu, self.out33)
        self.f.f.pool.return_value = self._ones(gpu, self.proj_pool)
        self.f.f.poolp.return_value = self._ones(gpu, self.proj_pool)
        self.f.f.poolpn.return_value = self._ones(gpu, self.proj_pool)

    def check_call(self, x, f, gpu):
        self.setup_mock(gpu)
        f(chainer.Variable(x))

        # Variable.__eq__ raises NotImplementedError,
        # so we cannot check arguments
        expected = [mock.call.conv1(mock.ANY), mock.call.conv1n(mock.ANY),
                    mock.call.proj3(mock.ANY), mock.call.proj3n(mock.ANY),
                    mock.call.conv3(mock.ANY), mock.call.conv3n(mock.ANY),
                    mock.call.proj33(mock.ANY), mock.call.proj33n(mock.ANY),
                    mock.call.conv33a(mock.ANY), mock.call.conv33an(mock.ANY),
                    mock.call.conv33b(mock.ANY), mock.call.conv33bn(mock.ANY),
                    mock.call.pool(mock.ANY), mock.call.poolp(mock.ANY),
                    mock.call.poolpn(mock.ANY)]

        self.assertListEqual(self.f.f.mock_calls, expected)

    def test_call_cpu(self):
        self.check_call(self.x, self.f, False)

    @attr.gpu
    def test_call_gpu(self):
        x = cuda.to_gpu(self.x)
        self.f.to_gpu()
        self.check_call(x, self.f, True)


testing.run_module(__name__, __file__)
