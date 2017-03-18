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

    """Function that broadcasts given arrays."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

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
        args (:class:`~chainer.Variable` or :class:`numpy.ndarray` \
        or :class:`cupy.ndarray`):
            Input variables to be broadcasted. Each dimension of the shapes \
            of the input variables must have the same size.

    Returns:
        ~chainer.Variable: :class:`~chainer.Variable` or tuple of \
            :class:`~chainer.Variable` objects which are broadcasted \
            from given arguments.

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (3, 2)).astype('f')
        >>> y = F.broadcast(x)
        >>> np.all(x == y.data)
        True
        >>> z = np.random.uniform(0, 1, (3, 2)).astype('f')
        >>> y, w = F.broadcast(x, z)
        >>> np.all(x == y.data) & np.all(z == w.data)
        True

    """
    return Broadcast()(*args)


class BroadcastTo(function.Function):

    """Function that broadcasts an array to a new shape."""

    def __init__(self, shape):
        self._shape = tuple(shape)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

        ndim = type_check.Variable(len(self._shape), 'len(shape)')
        type_check.expect(in_types[0].ndim <= ndim)

        shape = in_types[0].shape.eval()
        # check the shape in inverse order
        for i in six.moves.range(-1, -len(shape) - 1, -1):
            if shape[i] == self._shape[i] or shape[i] == 1:
                continue
            expect = 'in_type[0].shape[%d] == %d' % (i, self._shape[i])
            if self._shape[i] != 1:
                expect += ' or in_type[0].shape[%d] == 1' % i
            actual = 'in_type[0].shape: %s' % str(shape)
            raise type_check.InvalidType(expect, actual)

    def forward(self, xs):
        xp = cuda.get_array_module(*xs)
        x = xs[0]
        if hasattr(xp, 'broadcast_to'):
            return xp.broadcast_to(x, self._shape),
        else:
            # numpy 1.9 doesn't support broadcast_to method
            dummy = xp.empty(self._shape)
            bx, _ = xp.broadcast_arrays(x, dummy)
            return bx,

    def backward(self, xs, grads):
        return _backward_one(xs[0], grads[0]),


def broadcast_to(x, shape):
    """Broadcast a given variable to a given shape.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable be broadcasted. A \
            :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        shape (tuple): Tuple of :class:`int` of the shape of the \
            output variable.

    Returns:
        ~chainer.Variable: Output variable broadcasted to the given shape.

    .. admonition:: Example

        >>> x = np.arange(0, 3)
        >>> x
        array([0, 1, 2])
        >>> y = F.broadcast_to(x, (3, 3))
        >>> y.data
        array([[0, 1, 2],
               [0, 1, 2],
               [0, 1, 2]])

    """
    return BroadcastTo(shape)(x)
