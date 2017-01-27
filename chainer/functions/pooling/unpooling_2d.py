import numpy

from chainer import cuda
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
from chainer.utils import type_check


class Unpooling2D(pooling_2d.Pooling2D):

    """Unpooling over a set of 2d planes."""

    def __init__(self, ksize, stride=None, pad=0,
                 outsize=None, cover_all=True):
        super(Unpooling2D, self).__init__(ksize, stride, pad, cover_all)
        self.outh, self.outw = (None, None) if outsize is None else outsize

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4,
        )

        if self.outh is not None:
            expected_h = conv.get_conv_outsize(
                self.outh, self.kh, self.sy, self.ph, cover_all=self.cover_all)
            type_check.expect(x_type.shape[2] == expected_h)
        if self.outw is not None:
            expected_w = conv.get_conv_outsize(
                self.outw, self.kw, self.sx, self.pw, cover_all=self.cover_all)
            type_check.expect(x_type.shape[3] == expected_w)

    def forward(self, x):
        h, w = x[0].shape[2:]
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(
                h, self.kh, self.sy, self.ph, cover_all=self.cover_all)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(
                w, self.kw, self.sx, self.pw, cover_all=self.cover_all)
        xp = cuda.get_array_module(*x)
        col = xp.tile(x[0][:, :, None, None],
                      (1, 1, self.kh, self.kw, 1, 1))
        if xp is numpy:
            y = conv.col2im_cpu(col, self.sy, self.sx, self.ph, self.pw,
                                self.outh, self.outw)
        else:
            y = conv.col2im_gpu(col, self.sy, self.sx, self.ph, self.pw,
                                self.outh, self.outw)
        return y,

    def backward(self, x, gy):
        if isinstance(gy[0], cuda.ndarray):
            gcol = conv.im2col_gpu(
                gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)
        else:
            gcol = conv.im2col_cpu(
                gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)
        gx = gcol.sum(axis=(2, 3))
        return gx,


def unpooling_2d(x, ksize, stride=None, pad=0, outsize=None, cover_all=True):
    """Inverse operation of pooling for 2d array.

    This function acts similarly to :class:`~functions.Deconvolution2D`, but
    it spreads input 2d array's value without any parameter instead of
    computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or pair of ints): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k)`` are equivalent.
        stride (int, pair of ints or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent. If ``None`` is
            specified, then it uses same stride as the pooling window size.
        pad (int or pair of ints): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        outsize (None or pair of ints): Expected output size (height, width)
            of array after the operation.  If ``None``, the size
            (height or width) is estimated from the size of input array
            in first batch with
            :func:`~chainer.utils.conv.get_deconv_outsize`.
            If outsize is not ``None``, the result of outsize applied to
            :func:`~chainer.utils.conv.get_conv_outsize` must be equal to
            the shape of the 2d array in the input batch ``x``.
        cover_all (bool): If ``True``, the output size may be smaller than
            the size if ``cover_all`` is ``False``. This flag serves to
            align behavior to the pooling functions which can cover all
            input locations, see :func:`~chainer.functions.max_pooling_2d`
            and :func:`~chainer.functions.convolution_2d`.


    Returns:
        ~chainer.Variable: Output variable.

    """
    return Unpooling2D(ksize, stride, pad, outsize, cover_all)(x)
