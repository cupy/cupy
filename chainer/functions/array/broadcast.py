import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _backward_one(x, g):
    if g is None:
        xp = cuda.get_array_module(x)
        return xp.zeros_like(x)

    if g.ndim != x.ndim:
        g = g.sum(axis=tuple(range(g.ndim - x.ndim)))
        # An input variable is always an array, not a scalar.
        # We need to convert a scalar value to a zero-dim array.
        xp = cuda.get_array_module(x)
        if xp.isscalar(g):
            g = xp.array(g)

    axis = tuple(i for i, sx in enumerate(x.shape) if sx == 1)
    if len(axis) > 0:
        return g.sum(keepdims=True, axis=axis)
    else:
        return g


class Broadcast(function.Function):

    """Function that broadcast given arrays."""

    def check_type_forward(self, in_types):
        shapes = [t.eval().shape for t in in_types]
        r_shapes = [s[::-1] for s in shapes]
        r_filled = six.moves.zip_longest(*r_shapes, fillvalue=1)
        for ss in r_filled:
            d = max(ss)
            if not all(s == d or s == 1 for s in ss):
                expect = 'each dimension has the same size or is 1'
                actual = 'shapes: ' + ', '.join(map(str, shapes))
                raise type_check.InvalidType(expect, actual)

    def forward(self, xs):
        xp = cuda.get_array_module(*xs)
        return tuple(xp.broadcast_arrays(*xs))

    def backward(self, xs, grads):
        return tuple(_backward_one(x, g) for x, g in six.moves.zip(xs, grads))


def broadcast(*args):
    """Broadcast given variables.

    Args:
      args (Variables): Variables to be broadcasted.

    Returns:
      ``tuple``: Tuple of :class:`~chainer.Variable` objects which are
          broadcasted from given arguments.
    """
    return Broadcast()(*args)
