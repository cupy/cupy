import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Sin(function.Function):

    @property
    def label(self):
        return 'sin'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.sin(x[0])),

    def backward_cpu(self, x, gy):
        gx = utils.force_array(numpy.cos(x[0]))
        gx *= gy[0]
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx', 'gx = cos(x) * gy', 'sin_bwd'
        )(x[0], gy[0])
        return gx,


def sin(x):
    """Elementwise sin function."""
    return Sin()(x)


class Cos(function.Function):

    @property
    def label(self):
        return 'cos'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.cos(x[0])),

    def backward_cpu(self, x, gy):
        gx = utils.force_array(numpy.sin(x[0]))
        numpy.negative(gx, out=gx)
        gx *= gy[0]
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx', 'gx = -sin(x) * gy', 'cos_bwd'
        )(x[0], gy[0])
        return gx,


def cos(x):
    """Elementwise cos function."""
    return Cos()(x)
