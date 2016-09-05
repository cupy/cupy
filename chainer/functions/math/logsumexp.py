from chainer import cuda
from chainer import function
from chainer.utils import type_check


class LogSumExp(function.Function):

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

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)

        x, = inputs
        m = x.max(axis=self.axis, keepdims=True)
        y = x - m
        xp.exp(y, out=y)
        y_sum = y.sum(axis=self.axis)
        self.y = xp.asarray(xp.log(y_sum) + m.reshape(y_sum.shape))
        return self.y,

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        x, = inputs
        gy, = grads

        y = self.y
        if self.axis is not None:
            actual_axis = []
            for axis in self.axis:
                if axis < 0:
                    axis = len(x.shape) + axis
                actual_axis.append(axis)
            for axis in sorted(actual_axis):
                gy = xp.expand_dims(gy, axis=axis)
                y = xp.expand_dims(y, axis=axis)
        gx = gy * xp.exp(x - y)
        return gx,


def logsumexp(x, axis=None):
    """Log-sum-exp of array elements over a given axis.

    This function calculates logarithm of sum of exponential of array elements.

    .. math::

       y_i = \\log\\left(\\sum_j \\exp(x_{ij})\\right)

    Args:
        x (~chainer.Variable): Elements to log-sum-exp.
        axis (None, int, or tuple of int): Axis which a sum is performed.
            The default (axis = None) is perform a sum over all the dimensions
            of the input array.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return LogSumExp(axis)(x)
