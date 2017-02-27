import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ELU(function.Function):

    """Exponential Linear Unit."""

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (numpy.exp(y[neg_indices]) - 1)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : (T)(alpha * (exp(x) - 1))',
            'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        neg_indices = x[0] < 0
        gx[neg_indices] *= self.alpha * numpy.exp(x[0][neg_indices])
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T alpha', 'T gx',
            'gx = x >= 0 ? gy : (T)(gy * alpha * exp(x))',
            'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,


def elu(x, alpha=1.0):
    """Exponential Linear Unit function.

    For a parameter :math:`\\alpha`, it is expressed as

    .. math::
        f(x) = \\left \\{ \\begin{array}{ll}
        x & {\\rm if}~ x \\ge 0 \\\\
        \\alpha (\\exp(x) - 1) & {\\rm if}~ x < 0,
        \\end{array} \\right.

    See: https://arxiv.org/abs/1511.07289

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        alpha (float): Parameter :math:`\\alpha`. Default is 1.0.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.random.randint(-10, 10, (3, 2)).astype('f')
        >>> x
        array([[  2.,   5.],
               [-10.,  -7.],
               [ -7.,  -3.]], dtype=float32)
        >>> y = F.elu(x, alpha=1.)
        >>> y.data
        array([[ 2.        ,  5.        ],
               [-0.99995458, -0.99908811],
               [-0.99908811, -0.95021296]], dtype=float32)

    """
    return ELU(alpha=alpha)(x)
