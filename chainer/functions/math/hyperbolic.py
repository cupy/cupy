from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Cosh(function.Function):

    @property
    def label(self):
        return 'cosh'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.cosh(x[0])),

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        gx = utils.force_array(xp.sinh(x[0]))
        gx *= gy[0]
        return gx,


def cosh(x):
    """Elementwise hyperbolic cosine function.

    .. math::
       y_i = \\cosh x_i.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Cosh()(x)


class Sinh(function.Function):

    @property
    def label(self):
        return 'sinh'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.sinh(x[0])),

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        gx = utils.force_array(xp.cosh(x[0]))
        gx *= gy[0]
        return gx,


def sinh(x):
    """Elementwise hyperbolic sine function.

    .. math::
       y_i = \\sinh x_i.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Sinh()(x)
