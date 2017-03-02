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
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].dtype.kind == 'f',
            in_types[1].dtype.kind == 'f',
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, divisor = inputs
        m = xp.fmod(x, divisor)
        return utils.force_array(m, x.dtype),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gw, = grad_outputs
        x, divisor = inputs
        return gw, utils.force_array(xp.fix(x / divisor) * -1 * gw, x.dtype)


def fmod(x, divisor):
    """Elementwise mod function.

    .. math::
       y_i = x_i \\bmod \\mathrm{divisor}.

    Args:
        x (~chainer.Variable): Input variable.
        divisor (~chainer.Variable): Input divisor.
    Returns:
        ~chainer.Variable: Output variable.
    """
    return Fmod()(x, divisor)
