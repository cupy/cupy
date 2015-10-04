import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class GetItem(function.Function):

    """Get elements stored in given indicies."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'i',
            x_type.ndim == 2,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
        )

    def forward_cpu(self, inputs):
        x, t = inputs
        return x[six.moves.range(t.size), t],

    def forward_gpu(self, inputs):
        x, t = inputs
        y = cuda.elementwise(
            'S t, raw T x, int32 n_channel',
            'T y',
            'y = x[i * n_channel + t]',
            'getitem_fwd'
        )(t, x, x.shape[1])
        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        gx = numpy.zeros_like(x)
        gx[six.moves.range(t.size), t] = gloss
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        gx = cuda.cupy.zeros_like(x)
        gx = cuda.elementwise(
            'S t, T gloss, raw T gx, int32 n_channel',
            'T gx',
            'gx[i * n_channel + t] = gloss',
            'getitem_bwd'
        )(t, gloss, gx, x.shape[1])
        return gx, None


def getitem(x, t):
    """Get elements stored in given indicies.

    Args:
        x (Variable): Variable storing arrays.
        t (Variable): Variable storing index numbers.

    Returns:
        ~chainer.Variable: Variable that holds ```t```-th element of ```x```.

    """
    return GetItem()(x, t)
