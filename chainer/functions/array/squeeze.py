import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def argone(iterable):
    result = []
    for i, x in enumerate(iterable):
        if not isinstance(x, six.integer_types):
            raise ValueError('elements in iterable must be int')
        if x == 1:
            result.append(i)
    return result


class Squeeze(function.Function):

    """Remove demensions of size one from the shape of a ndarray."""

    def __init__(self, axis=None):
        if axis is None:
            self.axis = None
        elif isinstance(axis, six.integer_types):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all(
                isinstance(x, six.integer_types) for x in axis):
            self.axis = axis
        else:
            raise TypeError('axis must be None, int or tuple of ints')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        if self.axis is not None:
            for x in self.axis:
                if x >= 0:
                    type_check.expect(x < x_type.ndim)
                else:
                    type_check.expect(-x_type.ndim <= x)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        if self.axis is None:
            self._axis = tuple(argone(inputs[0].shape))
        return xp.squeeze(inputs[0], self.axis),

    def backward(self, inputs, grads):
        if self.axis is None:
            axis = self._axis
        else:
            axis = self.axis
            axis = [x + inputs[0].ndim if x < 0 else x for x in axis]
            axis.sort()

        shape = list(grads[0].shape)
        for x in axis:          # axis needs to be sorted
            shape.insert(x, 1)
        return grads[0].reshape(shape),


def squeeze(x, axis=None):
    """Remove demensions of size one from the shape of a ndarray.

    Args:
        x (chainer.Variable or :class:``numpy.ndarray`` or cupy.ndarray): Input
            data.
        axis (None or int or tuple of ints): A subset of the single-dimensional
            entries in the shape to remove. If ``None`` is supplied, all of
            them are removed. The dimension index starts at zero. If an axis
            with dimension greater than one is selected, an error is raised.

    Returns:
        ~chainer.Variable: Variable whose dimensions of size 1 are removed.

    """
    return Squeeze(axis)(x)
