#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from chainer import cuda
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv


def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


class Unpooling2D(pooling_2d.Pooling2D):

    """Unpooling over a set of 2d planes."""

    def forward_cpu(self, x):
        h, w = x[0].shape[2:]
        out_h = get_deconv_outsize(h, self.kh, self.sy, self.ph)
        out_w = get_deconv_outsize(w, self.kw, self.sx, self.pw)
        col = numpy.tile(x[0][:, :, numpy.newaxis, numpy.newaxis],
                         (1, 1, self.kh, self.kw, 1, 1))
        y = conv.col2im_cpu(col, self.sy, self.sx, self.ph, self.pw,
                            out_h, out_w)
        return y,

    def forward_gpu(self, x):
        h, w = x[0].shape[2:]
        out_h = get_deconv_outsize(h, self.kh, self.sy, self.ph)
        out_w = get_deconv_outsize(w, self.kw, self.sx, self.pw)
        col = cuda.cupy.tile(x[0][:, :, cuda.cupy.newaxis, cuda.cupy.newaxis],
                             (1, 1, self.kh, self.kw, 1, 1))
        y = conv.col2im_gpu(col, self.sy, self.sx, self.ph, self.pw,
                            out_h, out_w)
        return y,

    def backward_cpu(self, x, gy):
        gcol = conv.im2col_cpu(
            gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)
        gx = gcol.sum(axis=(2, 3))
        return gx,

    def backward_gpu(self, x, gy):
        gcol = conv.im2col_gpu(
            gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)
        gx = gcol.sum(axis=(2, 3))
        return gx,


def unpooling_2d(x, ksize, stride=None, pad=0, cover_all=True, use_cudnn=True):
    return Unpooling2D(ksize, stride, pad, cover_all, use_cudnn)(x)
