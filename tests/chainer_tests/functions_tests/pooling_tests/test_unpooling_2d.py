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


@testing.parameterize(
    # we assume insize as (2, 1)
    # standard output size which is estimated with get_deconv_outsize function
    {'cover_all': False, 'outsize': (4, 2)},
    {'cover_all': True, 'outsize': (3, 1)},
    {'cover_all': False, 'outsize': None, 'expected_outsize': (4, 2)},
    {'cover_all': True, 'outsize': None, 'expected_outsize': (3, 1)},
    # another sizes which can be outsize of insize (2, 1)
    {'cover_all': False, 'outsize': (5, 2)},
    {'cover_all': True, 'outsize': (4, 2)},
)
class TestUnpooling2D(unittest.TestCase):

    def setUp(self):
        self.N = 2
        self.n_channels = 3
        inh, inw = 2, 1
        self.x = numpy.arange(
            self.N * self.n_channels * inh * inw, dtype=numpy.float32)\
            .reshape(self.N, self.n_channels, inh, inw)
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1

        self.ksize = 2
        outh, outw = self.outsize or self.expected_outsize
        self.gy = numpy.random.uniform(
            -1, 1, (self.N, self.n_channels, outh, outw)).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.unpooling_2d(x, self.ksize, outsize=self.outsize,
                                   cover_all=self.cover_all)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for i in six.moves.range(self.N):
            for c in six.moves.range(self.n_channels):
                outsize = self.outsize or self.expected_outsize
                assert y_data.shape[2:] == outsize
                if outsize == (5, 2):
                    expect = numpy.zeros(outsize, dtype=numpy.float32)
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
                gradient_check.assert_allclose(expect, y_data[i, c])

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
            x_data, y_grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
