import numpy as np

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class L2NormSquared(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == np.float32,
            x_type.ndim >= 2,
        )

    def forward_cpu(self, inputs):
        x, = inputs
        N = x.shape[0]
        self.norm = np.zeros(N, dtype=np.float32)
        for i in range(N):
            self.norm[i] = x[i].dot(x[i])
        return self.norm,

    def forward_gpu(self, inputs):
        x, = inputs
        l2normsquared_kernel = cuda.cupy.ReductionKernel(
            'T x', 'T y', 'x * x', 'a + b', 'y = a', '0', 'l2normsquared'
        )
        return l2normsquared_kernel(x, axis=1),

    def backward(self, inputs, gy):
        return 2 * inputs * gy[0],


def l2_norm_squared(x):
    """L2 normalization (a.k.a. Euclidean norm) squared.

    This function implements the square of L2 normalization on a vector.

    """
    return L2NormSquared()(x)
