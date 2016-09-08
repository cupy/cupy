from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Fmod(function.Function):
    @property
    def label(self):
        return 'fmod'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types.size() == 2,
            in_types[0].dtype.kind == in_types[1].dtype.kind,
        )

    def forward(self, inputs):
        x, divisor = inputs
        xp = cuda.get_array_module(*x)
        m = xp.fmod(x, divisor)
        return utils.force_array(m, x[0].dtype),

    def backward(self, inputs, grad_outputs):
        gw, = grad_outputs
        x, divisor = inputs
        xp = cuda.get_array_module(*x)
        return gw, -1 * xp.fix(x / divisor) * gw


def fmod(x, divisor):
    """Elementwise mod function.

    .. math::
       y_i = x_i mod div.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Fmod()(x, divisor)
