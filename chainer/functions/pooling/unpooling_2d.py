import numpy

from chainer import cuda
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv


class Unpooling2D(pooling_2d.Pooling2D):

    """Unpooling over a set of 2d planes."""

    def __init__(self, ksize, stride=None, pad=0,
                 outsize=None, cover_all=True, use_cudnn=True):
        super(Unpooling2D, self).__init__(
            ksize, stride, pad, cover_all, use_cudnn)
        self.outh, self.outw = (None, None) if outsize is None else outsize

    def forward_cpu(self, x):
        h, w = x[0].shape[2:]
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(h, self.kh, self.sy, self.ph)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(w, self.kw, self.sx, self.pw)
        col = numpy.tile(x[0][:, :, numpy.newaxis, numpy.newaxis],
                         (1, 1, self.kh, self.kw, 1, 1))
        y = conv.col2im_cpu(col, self.sy, self.sx, self.ph, self.pw,
                            self.outh, self.outw)
        return y,

    def forward_gpu(self, x):
        h, w = x[0].shape[2:]
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(h, self.kh, self.sy, self.ph)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(w, self.kw, self.sx, self.pw)
        col = cuda.cupy.tile(x[0][:, :, cuda.cupy.newaxis, cuda.cupy.newaxis],
                             (1, 1, self.kh, self.kw, 1, 1))
        y = conv.col2im_gpu(col, self.sy, self.sx, self.ph, self.pw,
                            self.outh, self.outw)
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


def unpooling_2d(x, ksize, stride=None, pad=0,
                 outsize=None, cover_all=True, use_cudnn=True):
    return Unpooling2D(ksize, stride, pad, outsize, cover_all, use_cudnn)(x)
