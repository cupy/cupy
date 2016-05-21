import collections

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Crop(function.Function):

    """Function that crops one array with the other with axes and offset."""

    def __init__(self, shape, axes, offset):
        if not isinstance(shape, collections.Iterable):
            raise TypeError('shape must be 1-D array')
        if isinstance(axes, int):
            axes = tuple(axes)
        elif not isinstance(axes, collections.Iterable):
            raise TypeError('axes must be integer or 1-D array')
        if not isinstance(offset, int):
            raise TypeError('offset must be integer')
        self.shape = shape
        self.axes = axes
        self.offset = offset

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        for axis in self.axes:
            type_check.expect(in_types[0].ndim > axis)

    def forward(self, xs):
        ary = xs[0]
        slices = [slice(None)] * ary.ndim
        for axis in self.axes:
            slices[axis] = slice(self.offset, self.offset + self.shape[axis])
        return ary[tuple(slices)],

    def backward(self, xs, gys):
        xp = cuda.get_array_module(*xs)
        gy = gys[0]
        gx = xp.zeros_like(xs[0])
        gx_orig = gx
        for axis in self.axes:
            gx = gx.swapaxes(axis, 0)
            size = self.shape[axis]
            start = self.offset
            end = start + size
            gx = gx[start:end].swapaxes(axis, 0)
        gx[...] = gy
        return gx_orig,


def crop(x, shape, axes, offset=0):
    """Crop one array with the other with axes and offset.

    Args:
        x (tuple of Variables): Variable to be cropped.
        shape (tuple or list of ints): Reference shape for cropping.
            The size in each axis specified by 'axes' is used.
        axes (int or tuple of ints):
            Axes that the input array is cropped along with.
        offset (int): Offset for the starting point of cropping.

    Returns:
        Variable: :class:``~chainer.Variable`` object
            which is cropped array of ``x0``.

    """
    return Crop(shape, axes, offset)(x)
