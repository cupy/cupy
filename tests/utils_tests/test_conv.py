import unittest

import numpy

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv


if cuda.available:
    cuda.init()


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
        self.img = numpy.random.uniform(-1, 1, (2, 3, 4, 3)).astype(
            numpy.float32)

    def check_im2col(self, ph, pw, gpu):
        if gpu:
            im2col = conv.im2col_gpu
            img = cuda.to_gpu(self.img)
        else:
            im2col = conv.im2col_cpu
            img = self.img

        col = im2col(img, 1, 1, 1, 1, ph, pw)
        col_h = 4 + ph * 2
        col_w = 3 + pw * 2
        self.assertEqual(col.shape, (2, 3, 1, 1, col_h, col_w))

        col = cuda.to_cpu(col)

        for n in range(2):
            for c in range(3):
                for y in range(col_h):
                    for x in range(col_w):
                        if 0 <= y - ph < 4 and 0 <= x - pw < 3:
                            self.assertEqual(
                                col[n, c, 0, 0, y, x],
                                self.img[n, c, y - ph, x - pw])
                        else:
                            self.assertEqual(col[n, c, 0, 0, y, x], 0)

    def test_im2col_cpu(self):
        self.check_im2col(1, 1, gpu=False)

    @attr.gpu
    def test_im2col_gpu(self):
        self.check_im2col(1, 1, gpu=True)


class TestCol2Im(unittest.TestCase):

    def setUp(self):
        self.col = numpy.random.uniform(-1, 1, (2, 3, 1, 1, 4, 3)).astype(
            numpy.float32)

    def check_col2im(self, ph, pw, gpu):
        if gpu:
            col2im = conv.col2im_gpu
            col = cuda.to_gpu(self.col)
        else:
            col2im = conv.col2im_cpu
            col = self.col

        img_h = 4 - ph * 2
        img_w = 3 - pw * 2
        img = col2im(col, 1, 1, ph, pw, img_h, img_w)
        img = cuda.to_cpu(img)
        self.assertEqual(img.shape, (2, 3, img_h, img_w))
        for n in range(2):
            for c in range(3):
                for y in range(img_h):
                    for x in range(img_w):
                        if 0 <= y + ph < 4 and 0 <= x + pw < 3:
                            self.assertEqual(
                                img[n, c, y, x],
                                self.col[n, c, 0, 0, y + ph, x + pw])
                        else:
                            self.assertEqual(img[n, c, y, x], 0)

    def test_col2im_cpu(self):
        self.check_col2im(1, 1, gpu=False)

    @attr.gpu
    def test_col2im_gpu(self):
        self.check_col2im(1, 1, gpu=True)


testing.run_module(__name__, __file__)
