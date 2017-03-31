import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Softplus(function.Function):

    """Softplus function."""

    def __init__(self, beta=1.0):
        self.beta = float(beta)
        self.beta_inv = float(1.0 / beta)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, inputs):
        x, = inputs
        # y = log(1 + exp(beta * x)) / beta
        bx = self.beta * x
        y = (numpy.fmax(bx, 0) +
             numpy.log1p(numpy.exp(-numpy.fabs(bx)))) * self.beta_inv
        return utils.force_array(y, x.dtype),

    def forward_gpu(self, inputs):
        x, = inputs
        y = cuda.elementwise(
            'T x, T beta, T beta_inv', 'T y',
            '''
              T bx = beta * x;
              y = (max(bx, (T)0) + log1p(exp(-fabs(bx)))) * beta_inv;
            ''',
            'softplus_fwd'
        )(x, self.beta, self.beta_inv)
        return y,

    def backward_cpu(self, inputs, grads):
        x, = inputs
        g, = grads
        gx = (1 - 1 / (1 + numpy.exp(self.beta * x))) * g
        return utils.force_array(gx, x.dtype),

    def backward_gpu(self, inputs, grads):
        x, = inputs
        g, = grads
        gx = cuda.elementwise(
            'T x, T g, T beta', 'T gx',
            'gx = (1 - 1 / (1 + exp(beta * x))) * g',
            'softplus_bwd'
        )(x, g, self.beta)
        return gx,


def softplus(x, beta=1.0):
    """Element-wise softplus function.

    The softplus function is the smooth approximation of ReLU.

    .. math:: f(x)=\\frac{1}{\\beta}\\log(1 + \\exp(\\beta x)),

    where :math:`\\beta` is a parameter. The function becomes curved
    and akin to ReLU as the :math:`\\beta` is increasing.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        beta (float): Parameter :math:`\\beta`.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.arange(-2, 3, 2).astype('f')
        >>> x
        array([-2.,  0.,  2.], dtype=float32)
        >>> F.softplus(x, beta=1.0).data
        array([ 0.126928  ,  0.69314718,  2.12692809], dtype=float32)

    """
    return Softplus(beta=beta)(x)
