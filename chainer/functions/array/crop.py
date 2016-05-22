import collections

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Crop(function.Function):

    """Function that crops array with specified shape, axes and offsets."""

    def __init__(self, shape, axes, offsets):
        if not isinstance(shape, collections.Iterable):
            raise TypeError('shape must be 1-D array')
        if isinstance(axes, int):
            axes = tuple([axes])
        elif not isinstance(axes, collections.Iterable):
            raise TypeError('axes must be integer or 1-D array')
        if isinstance(offsets, int):
            offsets = tuple([offsets] * len(shape))
        elif not isinstance(offsets, collections.Iterable):
            raise TypeError('offsets must be integer or 1-D array')
        elif len(offsets) != len(shape):
            raise ValueError('offsets must have same length as shape')
        self.shape = shape
        self.axes = axes
        self.offsets = offsets

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        for axis in self.axes:
            type_check.expect(in_types[0].ndim > axis)

    def forward(self, xs):
        ary = xs[0]
        slices = [slice(None)] * ary.ndim
        for axis in self.axes:
            slices[axis] = slice(self.offsets[axis],
                                 self.offsets[axis] + self.shape[axis])
        return ary[tuple(slices)],

    def backward(self, xs, gys):
        xp = cuda.get_array_module(*xs)
        gy = gys[0]
        slices = [slice(None)] * gy.ndim
        for axis in self.axes:
            slices[axis] = slice(self.offsets[axis],
                                 self.offsets[axis] + self.shape[axis])
        gx = xp.zeros_like(xs[0])
        gx[tuple(slices)] = gy
        return gx,


def crop(x, shape, axes, offsets=0):
    """Crop array with specified shape and offsets at specified axes.

    Args:
        x (tuple of Variables): Variable to be cropped.
        shape (tuple or list of ints): Reference shape for cropping.
            The size in each axis specified by 'axes' is used.
        axes (int or sequence of ints):
            Axes that the input array is cropped along with.
        offsets (int or sequence of ints): Offsets for the starting
            point of cropping for each axis. If the type is int,
            it is copied to be same length as ``shape``.
            If type is sequence, the length must be same as ``shape``.
            Default is 0.

    Returns:
        Variable: :class:`~chainer.Variable` object
            which is cropped array of ``x``.

    """
    return Crop(shape, axes, offsets)(x)
