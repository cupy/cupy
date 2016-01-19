import numpy as np
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _as_two_dim(x):
    if x.ndim == 2:
        return x
    return x.reshape((len(x), -1))


class BatchL2NormSquared(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == np.float32,
            x_type.ndim >= 2,
        )

    def forward_cpu(self, inputs):
        x = _as_two_dim(inputs[0])
        return np.array([x[i].dot(x[i]) for i in six.moves.range(len(x))],
                        dtype=np.float32),

    def forward_gpu(self, inputs):
        x = _as_two_dim(inputs[0])
        l2normsquared_kernel = cuda.cupy.ReductionKernel(
            'T x', 'T y', 'x * x', 'a + b', 'y = a', '0', 'l2normsquared'
        )
        return l2normsquared_kernel(x, axis=1),

    def backward_cpu(self, inputs, gy):
        x = _as_two_dim(inputs[0])
        xp = cuda.get_array_module(x, gy)
        return xp.array([2 * x[i] * gy[0][i] for i in six.moves.range(len(x))],
                        dtype=np.float32).reshape(inputs[0].shape),

    def backward_gpu(self, inputs, gy):
        x = _as_two_dim(inputs[0])
        gy_repeat = gy[0].reshape(len(gy[0]), 1).repeat(x.shape[1], axis=1)
        kernel = cuda.elementwise(
            'T x, T gy', 'T gx', 'gx = 2 * x * gy', 'l2normsquared_bwd')
        gx = kernel(x, gy_repeat).reshape(inputs[0].shape)
        return gx,


def batch_l2_norm_squared(x):
    """L2 norm (a.k.a. Euclidean norm) squared.

    This function implements the square of L2 norm on a vector. No reduction
    along batch axis is done.

    Args:
        x (~chainer.Variable): Input variable. The first dimension is assumed
            to be the *minibatch dimension*. If x has more than two dimensions
            all but the first dimension are flattened to one dimension.

    Returns:
        ~chainer.Variable: Two dimensional output variable.

    """
    return BatchL2NormSquared()(x)
