import unittest

import itertools
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv


def xs_iter(dims):
    return itertools.product(*[range(d) for d in dims])


def kxs_iter(x, outs, ksize, stride, pad):
    return itertools.product(
        *[range(max(0, -p + s * _x), min(-p + s * _x + k, out))
          for (_x, out, k, s, p) in zip(x, outs, ksize, stride, pad)])


@testing.parameterize(*testing.product({
    'dims': [(4, 3)],
    '_ksize': [1, 2, 3],
    '_stride': [1, 2, 3],
    '_pad': [0, 1],
    'dtype': [numpy.float32],
    'cover_all': [True, False],
}))
class TestUnpoolingND(unittest.TestCase):

    def setUp(self):
        N = 2
        c = 3
        ndim = len(self.dims)
        self.ksize = (self._ksize,) * ndim
        self.stride = (self._stride,) * ndim
        self.pad = (self._pad,) * ndim

        x_shape = (N, c) + self.dims
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

        outs = tuple(
            conv.get_deconv_outsize(d, k, s, p, cover_all=self.cover_all)
            for (d, k, s, p)
            in zip(self.dims, self.ksize, self.stride, self.pad))
        gy_shape = (N, c) + outs
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

    def check_forward(self, x_data):
        dims = self.dims
        ksize = self.ksize
        stride = self.stride
        pad = self.pad

        # Compute unpooling.
        x = chainer.Variable(x_data)
        y = functions.unpooling_nd(
            x, self.ksize, self.stride, self.pad, cover_all=self.cover_all)

        # Test output's dtype and shape.
        self.assertEqual(y.data.dtype, self.dtype)
        self.assertEqual(y.data.shape, self.gy.shape)

        # Test output's value.
        N, c = x_data.shape[:2]
        outs = self.gy.shape[2:]
        y_expected_shape = (N, c) + outs
        y_expected = numpy.zeros(y_expected_shape, dtype=x_data.dtype)
        for i in six.moves.range(N):
            for _c in six.moves.range(c):
                for x in xs_iter(dims):
                    x_idx = (i, _c) + x
                    for kx in kxs_iter(x, outs, ksize, stride, pad):
                        y_idx = (i, _c) + kx
                        y_expected[y_idx] += self.x[x_idx]
        testing.assert_allclose(y_expected, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        pass

    @condition.retry(3)
    def test_backward_cpu(self):
        pass

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        pass


# TODO(takgai) test outsize
class TestUnpoolingNDOutsize(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
