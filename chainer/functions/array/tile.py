import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Tile(function.Function):
    """Tiling of an array."""

    def __init__(self, reps):
        if isinstance(reps, six.integer_types):
            self.reps = (reps,)
        elif isinstance(reps, tuple) and all(
                isinstance(x, six.integer_types) for x in reps):
            self.reps = reps
        else:
            raise TypeError('reps must be int or tuple of ints')

        if not all(x >= 0 for x in self.reps):
            raise ValueError('all elements in reps must be zero or larger')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.tile(inputs[0], self.reps),

    def backward(self, inputs, grads):
        x = inputs[0]
        reps = self.reps

        # Ensure input and reps have the same length.
        if x.ndim > len(reps):
            reps = (1,) * (x.ndim - len(reps)) + reps
        elif x.ndim < len(reps):
            x = x.reshape((1,) * (len(reps) - x.ndim) + x.shape)

        if grads[0].shape == ():
            # This case should be treated differently because numpy.num would
            # return a scalar (even if keepdims=True).
            return grads[0],

        # Reshape so that base axis and reps axis can be distinguished.
        new_shape = []
        for i in range(grads[0].ndim):
            new_shape.append(reps[i])
            new_shape.append(x.shape[i])
        new_shape = tuple(new_shape)

        # Sum along reps axis
        reps_axis = tuple(range(0, 2 * grads[0].ndim, 2))
        gy = grads[0].reshape(new_shape).sum(axis=reps_axis)

        if inputs[0].ndim < len(reps):
            return gy.reshape(inputs[0].shape),
        else:
            return gy,


def tile(x, reps):
    """Construct an array by tiling a given array.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input data.
        reps (int or tuple of ints): The number of times for each axis with
            which x is replicated.

    Returns:
        ~chainer.Variable: Variable tiled the given array.
    """
    return Tile(reps)(x)
