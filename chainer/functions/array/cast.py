from chainer import function
from chainer.utils import type_check


class Cast(function.Function):

    """Cast function."""

    def __init__(self, typ):
        self.type = typ

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        type_check.expect(x_type.dtype.kind == 'f')

    def forward(self, x):
        return x[0].astype(self.type, copy=False),

    def backward(self, x, g):
        return g[0].astype(x[0].dtype.type, copy=False),


def cast(x, typ):
    """Cast an input variable to a given type.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable to be casted. A \
            :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        typ (:class:`str` of dtype or :class:`numpy.dtype`):
            Typecode or data type to cast.

    Returns:
        ~chainer.Variable: Variable holding a casted array.

    .. admonition:: Example

        >>> x = np.arange(0, 3, dtype=np.float64)
        >>> x.dtype
        dtype('float64')
        >>> y = F.cast(x, np.float32)
        >>> y.dtype
        dtype('float32')
        >>> y = F.cast(x, 'float16')
        >>> y.dtype
        dtype('float16')

    """
    return Cast(typ)(x)
