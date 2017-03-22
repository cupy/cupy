import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class Dropconnect(function.Function):

    """Linear unit regularized by dropconnect."""

    def __init__(self, ratio, mask=None):
        self.ratio = ratio
        self.mask = mask

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        scale = inputs[1].dtype.type(1. / (1 - self.ratio))
        xp = cuda.get_array_module(*inputs)
        mask_shape = (inputs[0].shape[0], inputs[1].shape[0],
                      inputs[1].shape[1])
        if self.mask is None:
            if xp == numpy:
                self.mask = xp.random.rand(*mask_shape) >= self.ratio
            else:
                self.mask = xp.random.rand(*mask_shape,
                                           dtype=numpy.float32) >= self.ratio
        elif isinstance(self.mask, chainer.Variable):
            self.mask = self.mask.data

        x = _as_mat(inputs[0])
        W = inputs[1] * scale * self.mask

        # ijk,ik->ij
        y = xp.matmul(W, x[:, :, None])
        y = y.reshape(y.shape[0], y.shape[1]).astype(x.dtype, copy=False)

        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        scale = inputs[1].dtype.type(1. / (1 - self.ratio))
        x = _as_mat(inputs[0])
        W = inputs[1] * scale * self.mask
        gy = grad_outputs[0]
        xp = cuda.get_array_module(*inputs)

        # ij,ijk->ik
        gx = xp.matmul(gy[:, None, :], W).reshape(inputs[0].shape)
        gx = gx.astype(x.dtype, copy=False)

        # ij,ik,ijk->jk
        gW = (gy[:, :, None] * x[:, None, :] * self.mask).sum(0) * scale
        gW = gW.astype(W.dtype, copy=False)

        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW


def dropconnect(x, W, b=None, ratio=.5, train=True, mask=None):
    """Linear unit regularized by dropconnect.

    Dropconnect drops weight matrix elements randomly with probability
    ``ratio`` and scales the remaining elements by factor ``1 / (1 - ratio)``.
    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    :math:`Y = xW^\\top + b`.

    In testing mode, zero will be used as dropconnect ratio instead of
    ``ratio``.

    Notice:
    This implementation cannot be used for reproduction of the paper.
    There is a difference between the current implementation and the
    original one.
    The original version uses sampling with gaussian distribution before
    passing activation function, the current implementation averages
    before activation.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input variable. Its first dimension ``n`` is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``.
        ratio (float):
            Dropconnect ratio.
        train (bool):
            If ``True``, executes dropconnect.
            Otherwise, dropconnect function works as a linear function.
        mask (None or chainer.Variable or :class:`numpy.ndarray` or
            cupy.ndarray):
            If ``None``, randomized dropconnect mask is generated.
            Otherwise, The mask must be ``(n, M, N)`` shaped array.
            Main purpose of this option is debugging.
            `mask` array will be used as a dropconnect mask.



    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Dropconnect`

    """
    if not train:
        ratio = 0
    if b is None:
        return Dropconnect(ratio, mask)(x, W)
    else:
        return Dropconnect(ratio, mask)(x, W, b)
