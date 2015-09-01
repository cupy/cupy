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
