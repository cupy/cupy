import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Softplus(function.Function):

    """Softplus function."""

    def __init__(self, beta=1.0):
        self.beta = numpy.float32(beta)
        self.beta_inv = numpy.float32(1.0 / beta)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, inputs):
        x, = inputs
        # y = log(1 + exp(beta * x)) / beta
        bx = self.beta * x
        y = (numpy.fmax(bx, numpy.float32(0.0)) +
             numpy.log1p(numpy.exp(-numpy.fabs(bx)))) * self.beta_inv
        return y,

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
        return (1 - 1 / (1 + numpy.exp(self.beta * x))) * g,

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
    """Elementwise softplus function.

    This function is expressed as
    :math:`f(x) = \\frac{1}{\\beta}\\log(1 + \\exp(\\beta x))`,
    where :math:`\\beta` is a parameter.

    Args:
        x (~chainer.Variable): Input variable.
        beta (float): Parameter :math:`\\beta`.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Softplus(beta=beta)(x)
