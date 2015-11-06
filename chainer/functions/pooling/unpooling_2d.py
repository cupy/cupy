#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six

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
        n, c, h, w = x[0].shape
        gx = numpy.zeros((n, c, h, w), dtype=numpy.float32)

        # TODO(wkentaro): Make it fast
        for i in numpy.ndindex(n, c, h, w):
            j = i[0], i[1], i[2] * self.kh, i[3] * self.kw
            for k in six.moves.xrange(self.kh):
                for l in six.moves.xrange(self.kw):
                    gx[i] += gy[0][(j[0], j[1], j[2] + k, j[3] + l)]
        return gx,

    def backward_gpu(self, x, gy):
        n, c, h, w = x[0].shape
        gx = cuda.cupy.zeros((n, c, h, w), dtype=cuda.cupy.float32)

        # TODO(wkentaro): Make it fast
        for i0 in six.moves.xrange(n):
            for i1 in six.moves.xrange(c):
                for i2 in six.moves.xrange(h):
                    for i3 in six.moves.xrange(w):
                        i = (i0, i1, i2, i3)
                        j = i0, i1, i2 * self.kh, i3 * self.kw
                        for k in six.moves.xrange(self.kh):
                            for l in six.moves.xrange(self.kw):
                                m = (j[0], j[1], j[2] + k, j[3] + l)
                                gx[i] += gy[0][m]
        return gx,


def unpooling_2d(x, ksize, stride=None, pad=0, cover_all=True, use_cudnn=True):
    return Unpooling2D(ksize, stride, pad, cover_all, use_cudnn)(x)
