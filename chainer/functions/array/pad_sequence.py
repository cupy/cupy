import six

from chainer import function
from chainer import cuda
from chainer.utils import type_check


class PadSequence(function.Function):

    def __init__(self, length, padding):
        self.length = length
        self.padding = padding

    def check_type_forward(self, in_types):
        for in_type in in_types:
            type_check.expect(
                in_type.ndim > 0,
                in_type.shape[1:] == in_types[0].shape[1:],
                in_type.dtype == in_types[0].dtype)

        if self.length is not None:
            for in_type in in_types:
                type_check.expect(in_type.shape[0] <= self.length)

    def forward(self, xs):
        xp = cuda.get_array_module(*xs)

        if self.length is None:
            length = max(len(x) for x in xs)
        else:
            length = self.length

        shape = (len(xs), length) + xs[0].shape[1:]
        y = xp.empty(shape, xs[0].dtype)

        for i, x in enumerate(xs):
            l = len(x)
            if l == length:
                y[i] = x
            else:
                y[i, 0:l] = x
                y[i, l:] = self.padding

        return y,

    def backward(self, xs, grad):
        xp = cuda.get_array_module(*xs)
        g = grad[0]
        gs = xp.split(g, len(xs), axis=0)
        return tuple([g[0, 0:len(x)] for g, x in six.moves.zip(gs, xs)])


def pad_sequence(xs, length=None, padding=0):
    """Pad given arrays to make their lengths same.

    Args:
        xs (list of ~chainer.Variable): Variables you want to reshpae.
        length (None or int): Size of the first dimension of resphed arrays.
            If it is ``None``, longest size of the first dimension of ``xs``
            are used.
        padding (int or float): Value to fill.

    Returns:
        tuple: Returns a tuple of ~chainer.Variable. Each variable has the same
            shape.

    """
    return PadSequence(length, padding)(*xs)
