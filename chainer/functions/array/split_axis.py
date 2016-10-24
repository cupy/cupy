import collections

import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


class SplitAxis(function.Function):

    """Function that splits multiple arrays along the specified axis."""

    def __init__(self, indices_or_sections, axis):
        if not isinstance(
                indices_or_sections,
                six.integer_types + (collections.Iterable,)):
            raise TypeError('indices_or_sections must be integer or 1-D array')
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].ndim > self.axis)

        if isinstance(self.indices_or_sections, collections.Iterable):
            if len(self.indices_or_sections) > 0:
                max_index = type_check.Variable(
                    self.indices_or_sections[-1], 'max_index')
                type_check.expect(in_types[0].shape[self.axis] > max_index)
        else:
            sections = type_check.Variable(
                self.indices_or_sections, 'sections')
            type_check.expect(in_types[0].shape[self.axis] % sections == 0)

    def forward(self, x):
        if isinstance(self.indices_or_sections, collections.Iterable):
            cdimx = x[0].shape[self.axis]
            ind = list(self.indices_or_sections)
            ind.append(cdimx)
            prev_i = 0
            for i in ind:
                cdimy = max(0, min(i, cdimx) - prev_i)
                if cdimy == 0:
                    raise ValueError('Not support if shape contains 0')
                prev_i = i
        xp = cuda.get_array_module(*x)
        return tuple(xp.split(x[0], self.indices_or_sections, self.axis))

    def backward(self, x, gys):
        xp = cuda.get_array_module(*x)
        if any(gy is None for gy in gys):
            gx = xp.zeros_like(x[0])
            gxs = xp.split(gx, self.indices_or_sections, self.axis)
            for gxi, gy in six.moves.zip(gxs, gys):
                if gy is None:
                    continue
                gxi[:] = gy
            return gx,
        else:
            return xp.concatenate(gys, axis=self.axis),


def split_axis(x, indices_or_sections, axis, force_tuple=False):
    """Splits given variables along an axis.

    Args:
        x (tuple of Variables): Variables to be split.
        indices_or_sections (int or 1-D array): If this argument is an integer,
            N, the array will be divided into N equal arrays along axis.
            If it is a 1-D array of sorted integers, it
            indicates the positions where the array is split.
        axis (int): Axis that the input array is split along.
        force_tuple (bool): If ``True``, this method returns a tuple even when
            the number of outputs is one.

    Returns:
        tuple or Variable: Tuple of :class:`~chainer.Variable` objects
             if the number of outputs is more than 1 or
             :class:`~chainer.Variable` otherwise.
             When ``force_tuple`` is ``True``, returned value is always a tuple
             regardless of the number of outputs.

    .. note::
        This function raises :class:`ValueError` if at least
        one of the outputs is split to zero-size
        (i.e. ``axis``-th value of its shape is zero).

    """
    res = SplitAxis(indices_or_sections, axis)(x)
    if force_tuple and isinstance(res, chainer.Variable):
        res = (res,)
    return res
