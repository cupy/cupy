from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
import numpy


class ClippedReLU(function.Function):

    """Clipped Rectifier Unit function.

    Clipped ReLU is written as
    :math:`ClippedReLU(x, z) = \\min(\\max(0, x), z)`,
    where :math:`z(>0)` is a parameter to cap return value of ReLU.

    """

    def __init__(self, z):
        if not isinstance(z, float):
            raise TypeError('z must be float value')
        # z must be positive.
        assert z > 0
        self.cap = z

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        x = x[0]
        return utils.force_array(numpy.minimum(numpy.maximum(0, x), self.cap),
                                 x.dtype),

    def backward_cpu(self, x, gy):
        x = x[0]
        return utils.force_array(gy[0] * (0 < x) * (x < self.cap), x.dtype),

    def forward_gpu(self, x):
        return cuda.elementwise(
            'T x, T cap', 'T y', 'y = min(max(x, (T)0), cap)',
            'clipped_relu_fwd')(x[0], self.cap),

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T z', 'T gx',
            'gx = ((x > 0) & (x < z))? gy : (T)0',
            'clipped_relu_bwd')(x[0], gy[0], self.cap)
        return gx,


def clipped_relu(x, z=20.0):
    """Clipped Rectifier Unit function.

    For a clipping value :math:`z(>0)`, it computes

     .. math::`ClippedReLU(x, z) = \\min(\\max(0, x), z)`.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_n)`-shaped float array.
        z (float): Clipping value. (default = 20.0)

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_n)`-shaped float array.

    .. admonition:: Example

        >>> x = np.random.uniform(-100, 100, (10, 20)).astype('f')
        >>> z = 10.0
        >>> np.any(x < 0)
        True
        >>> np.any(x > z)
        True
        >>> y = F.clipped_relu(x, z=z)
        >>> np.any(y.data < 0)
        False
        >>> np.any(y.data > z)
        False

    """
    return ClippedReLU(z)(x)
