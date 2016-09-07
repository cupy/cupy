import unittest

import numpy
from six import moves

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv


class TestConv(unittest.TestCase):

    def check_conv_outsize(self, size, k, s, p):
        # When cover_all == False, `outsize` is the maximum integer that
        # satisfies "(outsize - 1) * s + k <= w"
        w = size + p * 2
        outsize = conv.get_conv_outsize(size, k, s, p, cover_all=False)
        self.assertTrue((outsize - 1) * s + k <= w < outsize * s + k)

    def check_conv_outsize_cover_all(self, size, k, s, p):
        # When cover_all == True, `outsize` is the minimum integer that
        # satisfies "w <= (outsize - 1) * s + k"
        w = size + p * 2
        outsize = conv.get_conv_outsize(size, k, s, p, cover_all=True)
        self.assertTrue((outsize - 2) * s + k < w <= (outsize - 1) * s + k)

    def test_conv_outsize1(self):
        self.check_conv_outsize(10, 4, 3, 2)

    def test_conv_outsize2(self):
        self.check_conv_outsize(10, 4, 4, 2)

    def test_conv_outsize_cover_all1(self):
        self.check_conv_outsize_cover_all(10, 4, 3, 2)

    def test_conv_outsize_cover_all2(self):
        self.check_conv_outsize_cover_all(10, 4, 4, 2)


class TestIm2Col(unittest.TestCase):

    def setUp(self):
        self.w = 10
        self.h = 8
        shape = (2, 3, self.h, self.w)
        self.img = numpy.random.uniform(-1, 1, shape).astype(
            numpy.float32)

    def check_im2col(self, kh, kw, sy, sx, ph, pw, gpu):
        if gpu:
            im2col = conv.im2col_gpu
            img = cuda.to_gpu(self.img)
        else:
            im2col = conv.im2col_cpu
            img = self.img

        col = im2col(img, kh, kw, sy, sx, ph, pw)
        col_h = conv.get_conv_outsize(self.h, kh, sy, ph)
        col_w = conv.get_conv_outsize(self.w, kw, sx, pw)
        self.assertEqual(col.shape, (2, 3, kh, kw, col_h, col_w))

        col = cuda.to_cpu(col)

        for n in moves.range(2):
            for c in moves.range(3):
                for y in moves.range(col_h):
                    for x in moves.range(col_w):
                        for dy in moves.range(kh):
                            for dx in moves.range(kw):
                                oy = y * sy - ph + dy
                                ox = x * sx - pw + dx
                                if 0 <= oy < self.h and 0 <= ox < self.w:
                                    self.assertEqual(
                                        col[n, c, dy, dx, y, x],
                                        self.img[n, c, oy, ox])
                                else:
                                    self.assertEqual(col[n, c, dy, dx, y, x],
                                                     0)

    def test_im2col_1_cpu(self):
        self.check_im2col(1, 1, 1, 1, 1, 1, gpu=False)

    def test_im2col_2_cpu(self):
        self.check_im2col(2, 2, 2, 2, 2, 2, gpu=False)

    def test_im2col_3_cpu(self):
        self.check_im2col(1, 2, 2, 1, 1, 2, gpu=False)

    @attr.gpu
    def test_im2col_1_gpu(self):
        self.check_im2col(1, 1, 1, 1, 1, 1, gpu=True)

    @attr.gpu
    def test_im2col_2_gpu(self):
        self.check_im2col(2, 2, 2, 2, 2, 2, gpu=True)

    @attr.gpu
    def test_im2col_3_gpu(self):
        self.check_im2col(1, 2, 2, 1, 1, 2, gpu=True)


class TestCol2Im(unittest.TestCase):

    def setUp(self):
        self.w = 10
        self.h = 8

    def check_col2im(self, kh, kw, sy, sx, ph, pw, gpu):
        col_h = conv.get_conv_outsize(self.h, kh, sy, ph)
        col_w = conv.get_conv_outsize(self.w, kw, sx, pw)
        shape = (2, 3, kh, kw, col_h, col_w)
        col = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

        if gpu:
            col2im = conv.col2im_gpu
            col_data = cuda.to_gpu(col)
        else:
            col2im = conv.col2im_cpu
            col_data = col

        img = col2im(col_data, sy, sx, ph, pw, self.h, self.w)
        img = cuda.to_cpu(img)
        self.assertEqual(img.shape, (2, 3, self.h, self.w))
        for n in moves.range(2):
            for c in moves.range(3):
                for y in moves.range(self.h):
                    for x in moves.range(self.w):
                        v = numpy.float32(0.0)
                        for dy in moves.range(kh):
                            for dx in moves.range(kw):
                                oy = (y + ph - dy) // sy
                                ox = (x + pw - dx) // sx
                                if (y + ph - dy) % sy == 0 and \
                                   (x + pw - dx) % sx == 0 and \
                                   0 <= oy < col_h and \
                                   0 <= ox < col_w:
                                    v += col[n, c, dy, dx, oy, ox]
                        self.assertAlmostEqual(img[n, c, y, x], v)

    def test_col2im_1_cpu(self):
        self.check_col2im(1, 1, 1, 1, 1, 1, gpu=False)

    def test_col2im_2_cpu(self):
        self.check_col2im(2, 2, 2, 2, 2, 2, gpu=False)

    def test_col2im_3_cpu(self):
        self.check_col2im(1, 2, 2, 1, 1, 2, gpu=False)

    @attr.gpu
    def test_col2im_1_gpu(self):
        self.check_col2im(1, 1, 1, 1, 1, 1, gpu=True)

    @attr.gpu
    def test_col2im_2_gpu(self):
        self.check_col2im(2, 2, 2, 2, 2, 2, gpu=True)

    @attr.gpu
    def test_col2im_3_gpu(self):
        self.check_col2im(1, 2, 2, 1, 1, 2, gpu=True)


testing.run_module(__name__, __file__)
