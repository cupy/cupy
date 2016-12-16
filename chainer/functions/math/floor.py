from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Floor(function.Function):

    @property
    def label(self):
        return 'floor'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.floor(x[0]), x[0].dtype),

    def backward(self, x, grad_outputs):
        xp = cuda.get_array_module(*x)
        return xp.zeros_like(x[0]),


def floor(x):
    """Elementwise floor function.

    .. math::
       y_i = \\lfloor x_i \\rfloor

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Floor()(x)
