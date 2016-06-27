import collections

from chainer import cuda
from chainer import function
from chainer.utils import type_check
from chainer import variable


class GetItem(function.Function):

    """Function that slices array and extract elements."""

    def __init__(self, slices):
        if not isinstance(slices, collections.Iterable):
            slices = tuple([slices])
        self.slices = slices

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        valid_slice = len(self.slices) - self.slices.count(None)
        type_check.expect(in_types[0].ndim == valid_slice)

    def forward(self, xs):
        ary = xs[0]
        return ary[tuple(self.slices)],

    def backward(self, xs, gys):
        xp = cuda.get_array_module(*xs)
        gy = gys[0]
        gx = xp.zeros_like(xs[0])
        gx[tuple(self.slices)] = gy
        return gx,


def get_item(x, slices):
    """Extract elements from array with specified shape, axes and offsets.

    Args:
        x (tuple of Variables): Variable to be sliced.
        slices (slice or tuple of slices): Slice objects to slice variable.

    Returns:
        Variable: :class:`~chainer.Variable` object
            which contains sliced array of ``x``.

    """
    return GetItem(slices)(x)


def install_variable_get_item():
    variable.Variable.__getitem__ = get_item
