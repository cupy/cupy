from chainer import cuda
from chainer import function
from chainer.utils import type_check


class SelectorBase(function.Function):
    """Select an array element from a given axis or set of axes."""

    def __init__(self, axis=None, keepdims=False):
        self.keepdims = keepdims
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

    def _fwd(self, x, xp):
        raise NotImplementedError('_fwd should be implemented in sub-class.')

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f'
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
        self.y = xp.asarray(self._fwd(x[0], xp))
        return self.y,

    def backward(self, x, gy):
        x = x[0]

        if self.axis is None:
            axis = range(x.ndim)
        else:
            axis = [ax % x.ndim for ax in self.axis]

        # Add broadcastable dimensions to y and gy
        # for each one that was reduced in the forward operation
        shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
        gy = gy[0].reshape(shape)
        y = self.y.reshape(shape)

        # Compute the gradient
        return gy * (x == y),


class Max(SelectorBase):

    def _fwd(self, x, xp):
        return xp.amax(x, axis=self.axis, keepdims=self.keepdims)


class Min(SelectorBase):

    def _fwd(self, x, xp):
        return xp.amin(x, axis=self.axis, keepdims=self.keepdims)


def max(x, axis=None, keepdims=False):
    """Maximum of array elements over a given axis.

    Args:
        x (~chainer.Variable): Array to be maximized.
        axis (None, int, or tuple of int): Axis over which a max is performed.
            The default (axis = None) is perform a max over all the dimensions
            of the input array.
    Returns:
        ~chainer.Variable: Output variable.

    """
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    """Minimum of array elements over a given axis.

    Args:
        x (~chainer.Variable): Array to be minimized.
        axis (None, int, or tuple of int): Axis over which a min is performed.
            The default (axis = None) is perform a min over all the dimensions
            of the input array.
    Returns:
        ~chainer.Variable: Output variable.

    """
    return Min(axis, keepdims)(x)
