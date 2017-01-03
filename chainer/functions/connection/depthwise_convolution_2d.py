import numpy
from six import moves

from chainer import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class DepthwiseConvolution2D(function.Function):

    def __init__(self, stride=1, pad=0):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[1],
        )

        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0] * w_type.shape[1],
            )

    def forward(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        kh, kw = W.shape[2:]

        xp = cuda.get_array_module(*x)
        if xp is numpy:
            self.col = conv.im2col_cpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw)
        else:
            self.col = conv.im2col_gpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw)

        arys = [xp.tensordot(self.col[:, i, :, :, :, :], W[:, i, :, :],
                             ((1, 2), (1, 2))).astype(x.dtype, copy=False)
                for i in moves.range(W.shape[1])]

        # along input channel axis
        y = xp.concatenate(arys, axis=3)

        if b is not None:
            y += b
        return xp.rollaxis(y, 3, 1),

    def backward(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        h, w = x.shape[2:]

        xp = cuda.get_array_module(*x)
        gy = xp.rollaxis(gy, 1, 4)
        garys = xp.split(gy, W.shape[1], axis=3)
        gW = xp.empty_like(W)
        gcol = xp.empty_like(self.col)
        gcol = xp.rollaxis(self.col, 0, 4)
        for i in moves.range(W.shape[1]):
            gW[:, i, :, :] = xp.tensordot(
                garys[i], self.col[:, i, :, :, :, :], ((0, 1, 2), (0, 3, 4)))
            gcol[i, :, :, :, :, :] = xp.tensordot(
                W[:, i, :, :], garys[i], (0, 3))
        gW = gW.astype(W.dtype, copy=False)
        gcol = gcol.astype(x.dtype, copy=False)
        gcol = xp.rollaxis(gcol, 3)

        if xp is numpy:
            gx = conv.col2im_cpu(gcol, self.sy, self.sx,
                                 self.ph, self.pw, h, w)
        else:
            gx = conv.col2im_gpu(gcol, self.sy, self.sx,
                                 self.ph, self.pw, h, w)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 1, 2))
            return gx, gW, gb


def depthwise_convolution_2d(x, W, b=None, stride=1, pad=0):
    """Two-dimensional depthwise convolution function.

    This is an implementation of two-dimensional depthwise convolution.
    It takes three variables: the input image ``x``, the filter weight ``W``,
    and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input.
    - :math:`c_M` is the number of the channel multiplier.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`h_O` and :math:`w_O` are the height and width of the output image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.

    Args:
        x (~chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input variable of shape :math:`(n, c_I, h, w)`.
        W (~chainer.Variable): Weight variable of shape
            :math:`(c_M, c_I, k_H, k_W)`.
        b (~chainer.Variable):
            Bias variable of length :math:`c_M * C_I` (optional).
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.


    Returns:
        ~chainer.Variable:
            Output variable. The shape is :math:`(n, c_I * c_M, h_O, w_O)`.

    Like ``Convolution2D``, ``DepthwiseConvolution2D`` function computes
    correlations between filters and patches of size :math:`(k_H, k_W)` in
    ``x``.
    Unlike ``Convolution2D``, with not adds up between output channels of
    filters but concatenates.
    For that reason, the shape of outputs of depthwise convolution are
    :math:`(n, c_I * c_M, h_O, w_O)`, ``c_M`` called channel_multiplier.

    :math:`(h_O, w_O)` is determined by the equivalent equation with
    ``Convolution2D``.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.


    See: `L. Sifre. Rigid-motion scattering for image classification\
          <http://www.di.ens.fr/data/publications/papers/phd_sifre.pdf>`_

    .. seealso:: :class:`~chainer.links.DepthwiseConvolution2D`

    """
    func = DepthwiseConvolution2D(stride, pad)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
