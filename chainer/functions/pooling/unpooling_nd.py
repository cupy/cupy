import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import conv_nd
from chainer.utils import type_check


class UnpoolingND(function.Function):
    """Unpooling over a set of N-dimensional planes."""

    def __init__(self, ndim, ksize, stride=None, pad=0, outsize=None,
                 cover_all=True):
        if stride is None:
            stride = ksize

        self.ndim = ndim
        self.ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = conv_nd.as_tuple(stride, ndim)
        self.pad = conv_nd.as_tuple(pad, ndim)
        self.outs = outsize
        self.cover_all = cover_all

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 2 + self.ndim,
        )
        if self.outs is not None:
            expected_dims = tuple(
                conv.get_conv_outsize(out, k, s, p, cover_all=self.cover_all)
                for (out, k, s, p)
                in zip(self.outs, self.ksize, self.stride, self.pad))
            type_check.expect(x_type.shape[2:] == expected_dims)

    def forward(self, x):
        dims = x[0].shape[2:]
        ndim = self.ndim
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        if self.outs is None:
            self.outs = tuple(
                conv.get_deconv_outsize(d, k, s, p, cover_all=self.cover_all)
                for (d, k, s, p) in zip(dims, ksize, stride, pad))

        xp = cuda.get_array_module(*x)

        colon = slice(None)
        # (:, :, None, None, ..., None)
        tile_index = (colon, colon) + (None,) * ndim
        # (1, 1, k_1, k_2, ..., k_n, 1, 1, ..., 1)
        tile_reps = (1, 1) + ksize + (1,) * ndim
        col = xp.tile(x[0][tile_index], tile_reps)

        if xp is numpy:
            col2im_nd = conv_nd.col2im_nd_cpu
        else:
            col2im_nd = conv_nd.col2im_nd_gpu
        y = col2im_nd(col, stride, pad, self.outs)

        return y,

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        if xp is numpy:
            im2col_nd = conv_nd.im2col_nd_cpu
        else:
            im2col_nd = conv_nd.im2col_nd_gpu
        gcol = im2col_nd(gy[0], self.ksize, self.stride, self.pad,
                         cover_all=self.cover_all)
        gcol_axis = tuple(six.moves.range(2, 2 + self.ndim))
        gx = gcol.sum(axis=gcol_axis)
        return gx,


def unpooling_nd(x, ksize, stride=None, pad=0, outsize=None, cover_all=True):
    """To be described."""
    ndim = len(x.data.shape[2:])
    return UnpoolingND(ndim, ksize, stride, pad, outsize, cover_all)(x)
