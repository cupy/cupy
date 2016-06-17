import unittest

import itertools
import numpy
from six import moves

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv_nd


@testing.parameterize(*testing.product({
    'ds': [(10,), (10, 8), (10, 8, 6)],
}))
class TestIm2ColND(unittest.TestCase):

    def setUp(self):
        shape = (2, 3) + self.ds
        self.img = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

    def check_im2col_nd(self, ks, ss, ps, gpu):
        ds = self.ds
        if gpu:
            im2col = conv_nd.im2col_nd_gpu
            img = cuda.to_gpu(self.img)
        else:
            im2col = conv_nd.im2col_nd_cpu
            img = self.img

        col = im2col(img, ks, ss, ps)
        outs = tuple([conv_nd.get_conv_outsize(d, k, s, p)
                      for (d, k, s, p) in zip(ds, ks, ss, ps)])
        expected_shape = (2, 3) + ks + outs
        self.assertEqual(col.shape, expected_shape)

        col = cuda.to_cpu(col)

        for n in moves.range(2):
            for c in moves.range(3):
                for xs in itertools.product(
                        *[moves.range(out) for out in outs]):
                    for dxs in itertools.product(
                            *[moves.range(k) for k in ks]):
                        oxs = tuple([x * s - p + dx
                                     for (x, s, p, dx)
                                     in zip(xs, ss, ps, dxs)])
                        if all([0 <= ox < d for (ox, d) in zip(oxs, ds)]):
                            col_index = (n, c) + dxs + xs
                            img_index = (n, c) + oxs
                            self.assertEqual(
                                col[col_index], self.img[img_index])
                        else:
                            col_index = (n, c) + dxs + xs
                            self.assertEqual(col[col_index], 0)

    def test_im2col_nd_1_cpu(self):
        N = len(self.ds)
        ks = (1,) * N
        ss = (1,) * N
        ps = (1,) * N
        self.check_im2col_nd(ks, ss, ps, gpu=False)

    def test_im2col_nd_2_cpu(self):
        N = len(self.ds)
        ks = (2,) * N
        ss = (2,) * N
        ps = (2,) * N
        self.check_im2col_nd(ks, ss, ps, gpu=False)

    def test_im2col_nd_3_cpu(self):
        N = len(self.ds)
        ks = (1, 2, 1)[:N]
        ss = (2, 1, 2)[:N]
        ps = (1, 2, 1)[:N]
        self.check_im2col_nd(ks, ss, ps, gpu=False)

    @attr.gpu
    def test_im2col_nd_1_gpu(self):
        N = len(self.ds)
        ks = (1,) * N
        ss = (1,) * N
        ps = (1,) * N
        self.check_im2col_nd(ks, ss, ps, gpu=True)

    @attr.gpu
    def test_im2col_nd_2_gpu(self):
        N = len(self.ds)
        ks = (2,) * N
        ss = (2,) * N
        ps = (2,) * N
        self.check_im2col_nd(ks, ss, ps, gpu=True)

    @attr.gpu
    def test_im2col_nd_3_gpu(self):
        N = len(self.ds)
        ks = (1, 2, 1)[:N]
        ss = (2, 1, 2)[:N]
        ps = (1, 2, 1)[:N]
        self.check_im2col_nd(ks, ss, ps, gpu=True)


@testing.parameterize(*testing.product({
    'ds': [(10,), (10, 8), (10, 8, 6)],
}))
class TestCol2ImND(unittest.TestCase):

    def setUp(self):
        pass

    def check_col2im_nd(self, ks, ss, ps, gpu):
        ds = self.ds
        outs = tuple([conv_nd.get_conv_outsize(d, k, s, p)
                      for (d, k, s, p) in zip(ds, ks, ss, ps)])
        col_shape = (2, 3) + ks + outs
        col = numpy.random.uniform(-1, 1, col_shape).astype(numpy.float32)

        if gpu:
            col2im = conv_nd.col2im_nd_gpu
            col_data = cuda.to_gpu(col)
        else:
            col2im = conv_nd.col2im_nd_cpu
            col_data = col

        img = col2im(col_data, ss, ps, ds)
        img = cuda.to_cpu(img)
        img_shape = (2, 3) + ds
        self.assertEqual(img.shape, img_shape)
        for n in moves.range(2):
            for c in moves.range(3):
                for xs in itertools.product(
                        *[moves.range(d) for d in ds]):
                    v = numpy.float32(0.0)
                    for dxs in itertools.product(
                            *[moves.range(k) for k in ks]):
                        oxs = tuple([(x + p - dx) // s
                                     for (x, p, dx, s)
                                     in zip(xs, ps, dxs, ss)])
                        if all([(x + p - dx) % s == 0
                                for (x, p, dx, s)
                                in zip(xs, ps, dxs, ss)]) and \
                           all([0 <= ox < out
                                for (ox, out) in zip(oxs, outs)]):
                            col_index = (n, c) + dxs + oxs
                            v += col[col_index]
                    img_index = (n, c) + xs
                    self.assertAlmostEqual(img[img_index], v)

    def test_col2im_1_cpu(self):
        N = len(self.ds)
        ks = (1,) * N
        ss = (1,) * N
        ps = (1,) * N
        self.check_col2im_nd(ks, ss, ps, gpu=False)

    def test_col2im_2_cpu(self):
        N = len(self.ds)
        ks = (2,) * N
        ss = (2,) * N
        ps = (2,) * N
        self.check_col2im_nd(ks, ss, ps, gpu=False)

    def test_col2im_3_cpu(self):
        N = len(self.ds)
        ks = (1, 2, 1)[:N]
        ss = (2, 1, 2)[:N]
        ps = (1, 2, 1)[:N]
        self.check_col2im_nd(ks, ss, ps, gpu=False)

    @attr.gpu
    def test_col2im_1_gpu(self):
        N = len(self.ds)
        ks = (1,) * N
        ss = (1,) * N
        ps = (1,) * N
        self.check_col2im_nd(ks, ss, ps, gpu=True)

    @attr.gpu
    def test_col2im_2_gpu(self):
        N = len(self.ds)
        ks = (2,) * N
        ss = (2,) * N
        ps = (2,) * N
        self.check_col2im_nd(ks, ss, ps, gpu=True)

    @attr.gpu
    def test_col2im_3_gpu(self):
        N = len(self.ds)
        ks = (1, 2, 1)[:N]
        ss = (2, 1, 2)[:N]
        ps = (1, 2, 1)[:N]
        self.check_col2im_nd(ks, ss, ps, gpu=True)

testing.run_module(__name__, __file__)
