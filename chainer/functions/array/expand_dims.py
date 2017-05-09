from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ExpandDims(function.Function):

    """Expands dimenstions of an input array without copy."""

    def __init__(self, axis):
        self.axis = int(axis)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        if self.axis >= 0:
            type_check.expect(x_type.ndim >= self.axis)
        else:
            type_check.expect(x_type.ndim >= -self.axis - 1)

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return xp.expand_dims(x[0], self.axis),

    def backward(self, x, gy):
        return gy[0].reshape(x[0].shape),


def expand_dims(x, axis):
    """Expands dimensions of an input variable without copy.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        axis (int):
            Position where new axis is to be inserted. The ``axis`` parameter
            is acceptable when :math:`-ndim - 1 \\leq axis \\leq ndim`.
            (``ndim`` is the dimension of input variables). When
            :math:`axis < 0`, the result is the same with
            :math:`ndim + 1 - |axis|`.

    Returns:
        ~chainer.Variable: Variable that holds a expanded input. The ``ndim``
        of output is one grater than that of ``x``.

    .. admonition:: Example

        >>> x = np.array([1, 2, 3])
        >>> x.shape
        (3,)
        >>> y = F.expand_dims(x, axis=0)
        >>> y.shape
        (1, 3)
        >>> y.data
        array([[1, 2, 3]])
        >>> y = F.expand_dims(x, axis=1)
        >>> y.shape
        (3, 1)
        >>> y.data
        array([[1],
               [2],
               [3]])
        >>> y = F.expand_dims(x, axis=-2)
        >>> y.shape
        (1, 3)
        >>> y.data
        array([[1, 2, 3]])

    """
    return ExpandDims(axis)(x)
