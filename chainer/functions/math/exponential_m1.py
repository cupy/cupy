import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check



class Expm1(function.Function):

    @property
    def label(self):
        return 'expm1'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        self.y = utils.force_array(numpy.expm1(x[0]))
        return self.y,

    def forward_gpu(self, x):
        self.y = cuda.cupy.expm1(x[0])
        return self.y,

    def backward(self, x, gy):
        return utils.force_array((self.y+1.0).astype('float32') * gy[0]),


def expm1(x):
    """Elementwise exponential function."""
    return Expm1()(x)


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
        return utils.force_array((gy[0] / (x[0]+1.0)).astype('float32')),


def log1p(x):
    """Elementwise natural logarithm function."""
    return Log1p()(x)
