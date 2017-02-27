from chainer import cuda
from chainer import function
from chainer.utils import type_check


class CReLU(function.Function):

    """Concatenated Rectified Linear Unit."""

    def __init__(self, axis=1):
        if not isinstance(axis, int):
            raise TypeError('axis must be an integer value')
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim > self.axis,
            in_types[0].ndim >= -self.axis
        )

    def get_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] *= 2
        return tuple(output_shape)

    def forward(self, x):
        x, = x
        xp = cuda.get_array_module(x)
        y = xp.empty(self.get_output_shape(x.shape), dtype=x.dtype)
        y_former, y_latter = xp.split(y, 2, axis=self.axis)
        zero = x.dtype.type(0)
        xp.maximum(zero, x, out=y_former)
        xp.maximum(zero, -x, out=y_latter)
        return y,

    def backward(self, x, gy):
        x, = x
        xp = cuda.get_array_module(x)
        gy, = gy
        gy_former, gy_latter = xp.split(gy, 2, axis=self.axis)
        return gy_former * (x > 0) - gy_latter * (-x > 0),


def crelu(x, axis=1):
    """Concatenated Rectified Linear Unit function.

    This function is expressed as follows

     .. math:: f(x) = (\\max(0, x), \\max(0, -x)).

    Here, two output values are concatenated along an axis.

    See: https://arxiv.org/abs/1603.05201

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        axis (int): Axis that the output values are concatenated along.
            Default is 1.

    Returns:
        ~chainer.Variable: Output variable of concatenated array.
        If the axis is 1, A :math:`(s_1, s_2 \\times 2, ..., s_N)`-shaped float
        array.

    .. admonition:: Example

        >>> x = np.random.uniform(-10, 10, (3, 2)).astype('f')
        >>> x
        array([[ 0.97627008,  4.30378723],
               [ 2.05526757,  0.89766365],
               [-1.52690399,  2.9178822 ]], dtype=float32)
        >>> y = F.crelu(x, axis=1)
        >>> y.data
        array([[ 0.97627008,  4.30378723,  0.        ,  0.        ],
               [ 2.05526757,  0.89766365,  0.        ,  0.        ],
               [ 0.        ,  2.9178822 ,  1.52690399,  0.        ]], \
dtype=float32)

    """
    return CReLU(axis=axis)(x)
