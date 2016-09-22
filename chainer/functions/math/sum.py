from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Sum(function.Function):
    """Sum of array elements over a given axis."""

    def __init__(self, axis=None):
        if axis is None:
            self.axis = None
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all(isinstance(a, int) for a in axis):
            if len(set(axis)) != len(axis):
                raise ValueError('duplicate value in axis: ({})'.format(
                    ', '.join(map(str, axis))))
            self.axis = axis
        else:
            raise TypeError('None, int or tuple of int are required')

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

        if self.axis is not None:
            for axis in self.axis:
                if axis >= 0:
                    type_check.expect(
                        axis < in_types[0].ndim,
                    )
                else:
                    type_check.expect(
                        -axis - 1 < in_types[0].ndim,
                    )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return xp.asarray(x[0].sum(axis=self.axis)),

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)

        gx = xp.empty_like(x[0])
        if gx.ndim == 0:
            gx = gy[0]
        elif self.axis is None:
            gx[:] = gy[0]
        else:
            gy = gy[0]
            actual_axis = []
            for axis in self.axis:
                if axis < 0:
                    axis += len(gx.shape)
                actual_axis.append(axis)
            for axis in sorted(actual_axis):
                gy = xp.expand_dims(gy, axis=axis)
            gx[:] = gy

        return gx,


def sum(x, axis=None):
    """Sum of array elements over a given axis.

    Args:
        x (~chainer.Variable): Elements to sum.
        axis (None, int, or tuple of int): Axis which a sum is performed.
            The default (axis = None) is perform a sum over all the dimensions
            of the input array.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Sum(axis)(x)
