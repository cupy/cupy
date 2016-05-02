import collections

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Crop(function.Function):

    """Function that crops one array with the other with axes and offset."""

    def __init__(self, axes, offset):
        if isinstance(axes, int):
            axes = tuple(axes)
        elif not isinstance(axes, collections.Iterable):
            raise TypeError('axes must be integer or 1-D array')
        if not isinstance(offset, int):
            raise TypeError('offset must be integer')
        self.axes = axes
        self.offset = offset

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        for axis in self.axes:
            type_check.expect(in_types[0].ndim > axis)
            type_check.expect(in_types[1].ndim > axis)

    def forward(self, xs):
        ary, ref_ary = xs
        for axis in self.axes:
            ary = ary.swapaxes(axis, 0)
            size = ref_ary.shape[axis]
            start = self.offset
            end = start + size
            ary = ary[start:end].swapaxes(axis, 0)
        return ary,

    def backward(self, xs, gys):
        xp = cuda.get_array_module(*xs)
        ref_ary = xs[1]
        gy = gys[0]
        gx = xp.zeros_like(xs[0])
        gx_orig = gx
        for axis in self.axes:
            gx = gx.swapaxes(axis, 0)
            size = ref_ary.shape[axis]
            start = self.offset
            end = start + size
            gx = gx[start:end].swapaxes(axis, 0)
        gx[...] = gy
        return gx_orig, None


def crop(x0, x1, axes, offset=0):
    """Crop one array with the other with axes and offset.

    Args:
        x0 (tuple of Variables): Variable to be cropped.
        x1 (tuple of Variables): Variable to crop the ``x0`` with.
        axes (int or tuple of ints):
            Axes that the input array is cropped along with.
        offset (int): Offset for the starting point of cropping.

    Returns:
        Variable: :class:``~chainer.Variable`` object
            which is cropped array of ``x0``.

    """
    return Crop(axes, offset)(x0, x1)
