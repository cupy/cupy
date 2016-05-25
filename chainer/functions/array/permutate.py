import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _check_indices(indices):
    if len(indices) == 0:
        return
    for i in indices:
        if 0 <= i < len(indices):
            continue
        raise ValueError('Out of bounds index: {}'.format(i))
    sort = numpy.sort(indices)
    for s, t in six.moves.zip(sort, sort[1:]):
        if s == t:
            raise ValueError('indices contains duplicate value: {}'.format(s))


def _inverse_indices(indices):
    r = numpy.empty(len(indices), 'i')
    for i, ind in enumerate(indices):
        r[ind] = i
    return r


class Permutate(function.Function):

    """Permutate function."""

    def __init__(self, indices, axis=0, inv=False):
        _check_indices(indices)
        self.indices = indices
        self.axis = axis
        self.inv = inv

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]
        if self.axis < 0:
            type_check.expect(x_type.ndim >= -self.axis)
        else:
            type_check.expect(x_type.ndim > self.axis)

        type_check.expect(x_type.shape[self.axis] == len(self.indices))

    def _permutate(self, x, inv):
        xp = cuda.get_array_module(x)
        if inv:
            indices = _inverse_indices(self.indices)
        else:
            indices = self.indices

        if xp is not numpy:
            indices = xp.array(indices, 'i')
        return xp.take(x, indices, axis=self.axis)

    def forward(self, inputs):
        x = inputs[0]
        return self._permutate(x, self.inv),

    def backward(self, inputs, grads):
        g = grads[0]
        return self._permutate(g, not self.inv),


def permutate(x, indices, axis=0, inv=False):
    """Permutates a given variable along an axis.

    This function permutate ``x`` with given ``indices``.
    That means ``y[i] = x[indices[i]]`` for all ``i``.
    Note that this result is same as ``y = x.take(indices)``.
    ``indices`` must be a permutation of ``[0, 1, ..., len(x) - 1]``.

    When ``inv`` is ``True``, ``indices`` is treated as its inverse.
    That means ``y[indices[i]] = x[i]``.

    Args:
        x (~chainer.Variable): Variable to permutate.
        indices (numpy.ndarray): Indices to extract from the variable.
        axis (int): Axis that the input array is permutate along.
        inv (bool): If ``True``, ``indices`` is treated as its inverse.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Permutate(indices, axis=axis, inv=inv)(x)
