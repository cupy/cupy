import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Sum(function.Function):

    """Summation over all elements."""

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype == numpy.float32
        )

    def forward_cpu(self, x):
        return numpy.array(x[0].sum()),

    def forward_gpu(self, x):
        return cuda.gpuarray.sum(x[0]),

    def backward_cpu(self, x, gy):
        return numpy.full_like(x[0], gy[0]),

    def backward_gpu(self, x, gy):
        # TODO(beam2d): Make it async
        return cuda.full_like(x[0], gy[0].get()),


def sum(x):
    """Computes sum of all elements."""
    return Sum()(x)
