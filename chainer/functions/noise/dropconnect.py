import numpy

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
        if xp == numpy:
            flag = xp.random.rand(*inputs[1].shape) >= self.ratio
        else:
            flag = (xp.random.rand(*inputs[1].shape, dtype=numpy.float32) >=
                    self.ratio)
        if self.mask is None:
            self.mask = scale * flag

        x = _as_mat(inputs[0])
        W = inputs[1] * self.mask
        y = x.dot(W.T).astype(x.dtype, copy=False)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1] * self.mask
        gy = grad_outputs[0]

        gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        gW = gy.T.dot(x).astype(W.dtype, copy=False) * self.mask
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

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``.
        ratio (float):
            Dropconnect ratio.
            If mask is not ``None``, this value is ignored.
        train (bool):
            If ``True``, executes dropconnect.
            Otherwise, does nothing.
        mask (:class:`numpy.ndarray` or cupy.ndarray):
            If ``None``, randomized dropconnect mask is generated.
            If not ``None``, this value is used as dropconnect mask.
            And scaling will not be executed.
            Then mask shape must be ``(M, N)``.
            Main purpose of latter option is debugging.

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
