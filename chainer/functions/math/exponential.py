import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Exp(function.Function):

    @property
    def label(self):
        return 'exp'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        self.y = utils.force_array(numpy.exp(x[0]))
        return self.y,

    def forward_gpu(self, x):
        self.y = cuda.cupy.exp(x[0])
        return self.y,

    def backward(self, x, gy):
        return utils.force_array(self.y * gy[0]),


def exp(x):
    """Elementwise exponential function."""
    return Exp()(x)


class Log(function.Function):

    @property
    def label(self):
        return 'log'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        return utils.force_array(numpy.log(x[0])),

    def forward_gpu(self, x):
        return cuda.cupy.log(x[0]),

    def backward(self, x, gy):
        return utils.force_array(gy[0] / x[0]),


def log(x):
    """Elementwise natural logarithm function."""
    return Log()(x)


class Log2(function.Function):

    @property
    def label(self):
        return 'log2'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.log2(x[0])),

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        gx = utils.force_array(xp.reciprocal(x[0]))
        gx /= xp.log(2)
        gx *= gy[0]
        return gx,


def log2(x):
    """Elementwise logarithm function to the base 2.

    .. math::
       y_i = \\log_2 x_i.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Log2()(x)


class Log10(function.Function):

    @property
    def label(self):
        return 'log10'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.log10(x[0])),

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        gx = utils.force_array(xp.reciprocal(x[0]))
        gx /= xp.log(10)
        gx *= gy[0]
        return gx,


def log10(x):
    """Elementwise logarithm function to the base 10.

    .. math::
       y_i = \\log_{10} x_i.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Log10()(x)
