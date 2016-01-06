import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _as_mat(x):
    if x.ndim == 2:
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
            type_check.prod(x_type.shape[1:]) == w_type.shape[0]
        )

        if in_types.size().eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype.kind == 'f',
                b_type.ndim == 2,
                b_type.shape == w_type.shape[1:]
            )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x = _as_mat(inputs[0])
        W = inputs[1]
        ys = xp.tensordot(x, W, axes=1)
        if len(inputs) == 3:
            ys += inputs[2]
        self.argmax = xp.argmax(ys, axis=1)
        return xp.max(ys, axis=1),

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        x = _as_mat(inputs[0])
        W = inputs[1]

        xp = cuda.get_array_module(*inputs)
        # gradient of z = xW + b
        gz = xp.zeros((gy.shape[0], W.shape[1], gy.shape[1]), x.dtype)
        if xp == numpy:
            idx0 = xp.arange(len(gy))[:, None]
            idx1 = xp.arange(gy.shape[1])
            gz[idx0, self.argmax, idx1] = gy
        else:
            gz_r = xp.rollaxis(gz, 1)
            cuda.elementwise(
                'T gy, S argmax, int32 n', 'raw T gz',
                'gz[argmax * n + i] = gy', 'maxout_bwd'
            )(gy, self.argmax, gz_r.size // len(gz_r), gz_r)
        gx = xp.tensordot(gz, W, ((1, 2), (1, 2))).reshape(inputs[0].shape)
        gW = xp.tensordot(x, gz, (0, 0))

        if len(inputs) == 3:
            gb = gz.sum(axis=0)
            return gx, gW, gb
        else:
            return gx, gW


def maxout(x, W, b=None):
    """non parametrized Maxout activation function

    It accepts two or three arguments: an input minibatch ``x``,
    a weight tensor ``W``, and optionally a bias matrix ``b``.
    It computes

    .. math::

      Y_{i} = \\mathrm{max}_{j} (x^{T}W_{\\cdot ij} + b_{ij})

    where :math:`x` is a input vector and :math:`W_{\\cdot ij}`
    is a sub-vector extracted from :math:`W` by fixing second
    and third dimensions to :math:`i` and :math:`j`, respectively.
    Minibatch dimension is omitted in the above equation.

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
