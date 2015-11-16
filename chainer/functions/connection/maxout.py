import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _as_mat(x):
    if x.dim == 2:
        return x
    return x.reshape(len(x), -1)


class MaxoutFunction(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() >= 2,
            in_types.size() <= 3
        )
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 3,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1]
        )

        if n_int.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype.kind == 'f',
                b_type.ndim == 2,
                b_type.shape[0] == w_type.shape[0]
            )

    def forward(self, inputs):
        xp = cuda.get_array_module(inputs)

        x = _as_mat(inputs[0])
        W = inputs[1]
        ys = xp.tensordot(x, W, axis=1)
        if len(inputs == 3):
            b = inputs[2]
            ys += b
        self.argmax = xp.argmax(ys, axis=1)
        return xp.max(ys, axis=1),

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        x, W = inputs[:2]

        xp = cuda.get_array_module(inputs)
        gxW = xp.zeros((gy.shape[0], W.shape[1], gy.shape[1]))
        gxW_r = xp.rollaxis(gx, 1)
        for i in numpy.ndindex(gy.shape):
            gxW_r[self.argmax[i]][i] = gy[i]
        gx = xp.tensordot(gxW, W, ((1, 2), (1, 2))).reshape(x.shape)
        gW = xp.tensordot(x, gW, (0, 0))

        if len(inputs) == 3:
            gb = gy.sum(axis=0)
            return gx, gW, gb
        else:
            return gx, gW


def maxout(x, W, b=None):
    """Maxout function

    It accepts two or three arguments: an input minibatch ``x``,
    a weight tensor ``W``, and optionally a bias matrix ``b``.
    It computes :math:`Y_{i} = \mathrm{max}_{j} (xW_{\cdot ij} + b_{ij})`

    Args:
       x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
       W (~chainer.Variable): Weight variable of shape ``(N, C, M)``
       b (~chainer.Variable): Bias variable (optional) of shape ``(C, M)```
    Returns:
        ~chainer.Variable: Outputvariable

    .. seealso:: :class:`~chainer.links.Maxout`
    """

    if b is None:
        return MaxoutFunction()(x, W)
    else:
        return MaxoutFunction()(x, W, b)
