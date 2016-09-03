import numpy

from chainer import cuda
from chainer import function
from chainer.utils import array
from chainer.utils import type_check


class NormalizeL2(function.Function):

    """L2 normalization"""

    def __init__(self, eps=1e-5):
        self.eps = eps

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 2,
        )

    def forward_cpu(self, inputs):
        x = array.as_mat(inputs[0])
        norm = numpy.linalg.norm(x, axis=1) + self.eps
        return x / norm[:, numpy.newaxis],

    def forward_gpu(self, inputs):
        x = array.as_mat(inputs[0])
        l2norm_kernel = cuda.cupy.ReductionKernel(
            'T x, float32 eps',
            'T y',
            'x * x',
            'a + b',
            'y = sqrt(a) + eps',
            '0',
            'l2norm'
        )
        norm = cuda.cupy.broadcast_to(
            l2norm_kernel(x, self.eps, axis=1).reshape(-1, 1),
            x.shape
        )

        return x / norm,

    def backward_cpu(self, inputs, gy):
        x = inputs[0]
        gy = gy[0]

        norm = numpy.linalg.norm(x, axis=1) + self.eps
        norm = norm[:, numpy.newaxis]

        gx = gy * norm - (x * gy).sum(axis=1)[:, numpy.newaxis] * x / norm
        gx = gx / norm ** 2

        return gx,

    def backward_gpu(self, inputs, gy):
        x = inputs[0]
        gy = gy[0]

        l2norm_kernel = cuda.cupy.ReductionKernel(
            'T x, float32 eps',
            'T y',
            'x * x',
            'a + b',
            'y = sqrt(a) + eps',
            '0',
            'l2norm'
        )
        norm = cuda.cupy.broadcast_to(
            l2norm_kernel(x, self.eps, axis=1).reshape(-1, 1),
            x.shape
        )
        x_gy = cuda.cupy.broadcast_to(
            (x * gy).sum(axis=1, keepdims=True),
            x.shape
        )
        gx = cuda.elementwise(
            'T gy, T x, T x_gy, T norm',
            'T gx',
            'gx = (gy * norm - x_gy * x / norm) / (norm * norm)',
            'l2_bwd')(gy, x, x_gy, norm)

        return gx,


def normalize(x, eps=1e-5):
    """L2 norm squared (a.k.a. Euclidean norm).

    This function implements L2 normalization on a 1D vector. No reduction
    is done along batch axis.  Let :math:`x` be an input vector of dimension
    :math:`(N, K)`, where :math:`N` and :math:`K` denote mini-batch size and
    the dimension of the input variable. Then, this function computes an output
    vector :math:`y` by the following equation:

    .. math::
       y_i = {x_i \\over \\| x_i \\|_2}

    :math:`eps` is used to avoid division by zero when :math:`x_i=0`

    Args:
        x (~chainer.Variable): Two dimensional output variable. The first
            dimension is assumed to be the mini-batch dimension.
        eps (float): Epsilon value for numerical stability.

    Returns:
        ~chainer.Variable: Two dimensional output variable, the same shape
            as :math:`x`.

    """
    return NormalizeL2(eps)(x)
