import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product_dict(
    [
        # we assume insize as (2, 1)
        # standard output size which is estimated with get_deconv_outsize
        # function
        {'cover_all': False, 'outsize': (4, 2)},
        {'cover_all': True, 'outsize': (3, 1)},
        {'cover_all': False, 'outsize': None, 'expected_outsize': (4, 2)},
        {'cover_all': True, 'outsize': None, 'expected_outsize': (3, 1)},
        # another sizes which can be outsize of insize (2, 1)
        {'cover_all': False, 'outsize': (5, 2)},
        {'cover_all': True, 'outsize': (4, 2)},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestUnpooling2D(unittest.TestCase):

    def setUp(self):
        self.N = 2
        self.n_channels = 3
        inh, inw = 2, 1
        self.x = numpy.arange(
            self.N * self.n_channels * inh * inw,
            dtype=self.dtype).reshape(self.N, self.n_channels, inh, inw)
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1

        self.ksize = 2
        outh, outw = self.outsize or self.expected_outsize
        self.gy = numpy.random.uniform(
            -1, 1, (self.N, self.n_channels, outh, outw)).astype(self.dtype)
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.unpooling_2d(x, self.ksize, outsize=self.outsize,
                                   cover_all=self.cover_all)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for i in six.moves.range(self.N):
            for c in six.moves.range(self.n_channels):
                outsize = self.outsize or self.expected_outsize
                assert y_data.shape[2:] == outsize
                if outsize == (5, 2):
                    expect = numpy.zeros(outsize, dtype=self.dtype)
                    expect[:2, :] = self.x[i, c, 0, 0]
                    expect[2:4, :] = self.x[i, c, 1, 0]
                elif outsize == (4, 2):
                    expect = numpy.array([
                        [self.x[i, c, 0, 0], self.x[i, c, 0, 0]],
                        [self.x[i, c, 0, 0], self.x[i, c, 0, 0]],
                        [self.x[i, c, 1, 0], self.x[i, c, 1, 0]],
                        [self.x[i, c, 1, 0], self.x[i, c, 1, 0]],
                    ])
                elif outsize == (3, 1):
                    expect = numpy.array([
                        [self.x[i, c, 0, 0]],
                        [self.x[i, c, 0, 0]],
                        [self.x[i, c, 1, 0]],
                    ])
                else:
                    raise ValueError('Unsupported outsize: {}'.format(outsize))
                testing.assert_allclose(expect, y_data[i, c])

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.Unpooling2D(self.ksize, outsize=self.outsize,
                                  cover_all=self.cover_all),
            x_data, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'h': [5],
    'k': [3],
    's': [3],
    'p': [0],
    'cover_all': [True, False],
}))
class TestMaxPoolingUnpooling(unittest.TestCase):

    def check_left_inverse(self, xp, use_cudnn=False):
        x = xp.arange(self.h * self.h).reshape(
            (1, 1, self.h, self.h)).astype(self.dtype)
        y = chainer.functions.unpooling_2d(
            x, self.k, self.s, self.p, None, self.cover_all)
        x_ = chainer.functions.max_pooling_2d(
            y, self.k, self.s, self.p, self.cover_all, use_cudnn).data

        self.assertEqual(x.shape, x_.shape)
        self.assertEqual(x.dtype, x_.dtype)
        chainer.testing.assert_allclose(x, x_)

    def test_left_inverse_cpu(self):
        self.check_left_inverse(numpy)

    @attr.gpu
    def test_left_inverse_cupy(self):
        self.check_left_inverse(cuda.cupy)

    @attr.gpu
    def test_left_inverse_cudnn(self):
        self.check_left_inverse(cuda.cupy, True)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'h': [5],
    'k': [3],
    's': [3],
    'p': [0],
}))
class TestAveragePoolingUnpooling(unittest.TestCase):

    def check_left_inverse(self, xp, use_cudnn=False):
        x = xp.arange(self.h * self.h).reshape(
            (1, 1, self.h, self.h)).astype(self.dtype)
        # average_pooling_2d does not have cover_all option
        # as max_pooling_2d has.
        y = chainer.functions.unpooling_2d(
            x, self.k, self.s, self.p, None, False)
        x_ = chainer.functions.average_pooling_2d(
            y, self.k, self.s, self.p, use_cudnn).data

        self.assertEqual(x.shape, x_.shape)
        self.assertEqual(x.dtype, x_.dtype)
        chainer.testing.assert_allclose(x, x_)

    def test_left_inverse_cpu(self):
        self.check_left_inverse(numpy)

    @attr.gpu
    def test_left_inverse_cupy(self):
        self.check_left_inverse(cuda.cupy)

    @attr.gpu
    def test_left_inverse_cudnn(self):
        self.check_left_inverse(cuda.cupy, True)


testing.run_module(__name__, __file__)
