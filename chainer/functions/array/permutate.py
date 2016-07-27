import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _check_indices(indices):
    if len(indices) == 0:
        return
    # TODO(unno): Check indices without cpu
    indices = cuda.to_cpu(indices)
    for i in indices:
        if 0 <= i < len(indices):
            continue
        raise ValueError('Out of bounds index: {}'.format(i))
    sort = numpy.sort(indices)
    for s, t in six.moves.zip(sort, sort[1:]):
        if s == t:
            raise ValueError('indices contains duplicate value: {}'.format(s))


def _inverse_indices(indices):
    xp = cuda.get_array_module(indices)
    r = xp.empty_like(indices)
    if xp is numpy:
        for i, ind in enumerate(indices):
            r[ind] = i
    else:
        cuda.elementwise(
            'int32 ind', 'raw int32 r',
            'r[ind] = i',
            'inverse_indices'
        )(indices, r)
    return r


class Permutate(function.Function):

    """Permutate function."""

    def __init__(self, axis=0, inv=False):
        self.axis = axis
        self.inv = inv

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, ind_type = in_types
        if self.axis < 0:
            type_check.expect(x_type.ndim >= -self.axis)
        else:
            type_check.expect(x_type.ndim > self.axis)

        type_check.expect(
            ind_type.dtype == numpy.int32,
            ind_type.ndim == 1,
            x_type.shape[self.axis] == ind_type.shape[0],
        )

    def _permutate(self, x, indices, inv):
        xp = cuda.get_array_module(x)
        if inv:
            indices = _inverse_indices(indices)

        return xp.take(x, indices, axis=self.axis)

    def forward(self, inputs):
        x, inds = inputs

        if chainer.is_debug():
            _check_indices(inds)

        return self._permutate(x, inds, self.inv),

    def backward(self, inputs, grads):
        inds = inputs[1]
        g = grads[0]
        return self._permutate(g, inds, not self.inv), None


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
        indices (~chainer.Variable): Indices to extract from the variable.
        axis (int): Axis that the input array is permutate along.
        inv (bool): If ``True``, ``indices`` is treated as its inverse.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Permutate(axis=axis, inv=inv)(x, indices)
