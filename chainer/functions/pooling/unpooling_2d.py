#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer.functions.pooling import pooling_2d
import numpy
import six


class Unpooling2D(pooling_2d.Pooling2D):

    """Unpooling over a set of 2d planes."""

    def forward_cpu(self, x):
        y = x[0]
        y = y.repeat(self.kh, axis=2)
        y = y.repeat(self.kw, axis=3)
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


def unpooling_2d(x, ksize, stride=None, pad=0, cover_all=True, use_cudnn=True):
    return Unpooling2D(ksize, stride, pad, cover_all, use_cudnn)(x)
