import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Gaussian(function.Function):

    """Gaussian sampling function.

    In forward calculation, this funciton takes mean and logarithm of variance
    as inputs, and draw a sample from a gaussian distribution.
    """

    def __init__(self):
        self.eps = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        m_type, v_type = in_types
        type_check.expect(
            m_type.dtype == numpy.float32,
            v_type.dtype == numpy.float32,
            m_type.shape == v_type.shape,
        )

    def forward_cpu(self, inputs):
        mean, ln_var = inputs
        if self.eps is None:
            self.eps = numpy.random.standard_normal(ln_var.shape) \
                                   .astype(numpy.float32)

        self.noise = numpy.exp(ln_var * 0.5) * self.eps
        return mean + self.noise,

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        mean, ln_var = inputs
        if self.eps is None:
            self.eps = cupy.random.standard_normal(
                ln_var.shape, dtype=mean.dtype)

        self.noise = cupy.empty_like(mean)
        self.noise = cuda.elementwise(
            'T v, T e', 'T noise',
            'noise = exp(v / 2) * e',
            'gaussian_forward'
        )(ln_var, self.eps)
        return mean + self.noise,

    def backward(self, inputs, grad_output):
        g, = grad_output
        return g, g * self.noise * g.dtype.type(0.5),


def gaussian(mean, ln_var):
    """Gaussian sampling function.

    It takes mean :math:`\\mu` and logarithm of variance
    :math:`\\log(\\sigma^2)` as input and output a sample drawn from gaussian
    :math:`N(\\mu, \\sigma)`.

    Args:
        mean (~chainer.Variable): Input variable representing mean
            :math:`\\mu`.
        ln_var (~chainer.Variable): Input variable representing logarithm of
            variance :math:`\\log(\\sigma^2)`.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Gaussian()(mean, ln_var)
