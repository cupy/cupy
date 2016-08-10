from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Sqrt(function.Function):

    @property
    def label(self):
        return 'sqrt'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.sqrt(x[0], dtype=x[0].dtype)),

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        gx = utils.force_array(xp.sqrt(x[0], dtype=x[0].dtype))
        gx *= 2.0
        xp.reciprocal(gx, out=gx)
        gx *= gy[0]
        return gx,


def sqrt(x):
    """Elementwise square root function."""
    return Sqrt()(x)
