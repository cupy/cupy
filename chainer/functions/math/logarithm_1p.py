import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Log1p(function.Function):

    @property
    def label(self):
        return 'log1p'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        return utils.force_array(numpy.log1p(x[0])),

    def forward_gpu(self, x):
        return cuda.cupy.log1p(x[0]),

    def backward(self, x, gy):
        return utils.force_array(gy[0] / (x[0] + x[0].dtype.type(1.0))),


def log1p(x):
    """Elementwise natural logarithm plus one function."""
    return Log1p()(x)
