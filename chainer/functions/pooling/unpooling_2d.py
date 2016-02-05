import numpy

from chainer import cuda
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
from chainer.utils import type_check


class Unpooling2D(pooling_2d.Pooling2D):

    """Unpooling over a set of 2d planes."""

    def __init__(self, ksize, stride=None, pad=0,
                 outsize=None, cover_all=True, use_cudnn=True):
        super(Unpooling2D, self).__init__(
            ksize, stride, pad, cover_all, use_cudnn)
        self.outh, self.outw = (None, None) if outsize is None else outsize

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
        )

        if self.outh is not None:
            type_check.expect(
                x_type.shape[2] ==
                conv.get_conv_outsize(self.outh, self.kh, self.sy, self.ph)
            )
        if self.outw is not None:
            type_check.expect(
                x_type.shape[3] ==
                conv.get_conv_outsize(self.outw, self.kw, self.sx, self.pw),
            )

    def forward(self, x):
        h, w = x[0].shape[2:]
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(h, self.kh, self.sy, self.ph)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(w, self.kw, self.sx, self.pw)
        xp = cuda.get_array_module(x)
        col = xp.tile(x[0][:, :, xp.newaxis, xp.newaxis],
                      (1, 1, self.kh, self.kw, 1, 1))
        if any(isinstance(xi, cuda.ndarray) for xi in x):
            y = conv.col2im_gpu(col, self.sy, self.sx, self.ph, self.pw,
                                self.outh, self.outw)
        else:
            y = conv.col2im_cpu(col, self.sy, self.sx, self.ph, self.pw,
                                self.outh, self.outw)
        return y,

    def backward(self, x, gy):
        if any(isinstance(xi, cuda.ndarray) for xi in x):
            gcol = conv.im2col_gpu(
                gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)
        else:
            gcol = conv.im2col_cpu(
                gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)
        gx = gcol.sum(axis=(2, 3))
        return gx,


def unpooling_2d(x, ksize, stride=None, pad=0,
                 outsize=None, cover_all=True, use_cudnn=True):
    return Unpooling2D(ksize, stride, pad, outsize, cover_all, use_cudnn)(x)
