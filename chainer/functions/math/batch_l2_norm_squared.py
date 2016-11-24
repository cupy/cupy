import numpy

from chainer import cuda
from chainer import function
from chainer.utils import array
from chainer.utils import type_check


class BatchL2NormSquared(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= 2,
        )

    def forward_cpu(self, inputs):
        x = array.as_mat(inputs[0])
        return (x * x).sum(axis=1),

    def forward_gpu(self, inputs):
        x = array.as_mat(inputs[0])
        l2normsquared_kernel = cuda.cupy.ReductionKernel(
            'T x', 'T y', 'x * x', 'a + b', 'y = a', '0', 'l2normsquared'
        )
        return l2normsquared_kernel(x, axis=1),

    def backward(self, inputs, gy):
        x = inputs[0]
        xp = cuda.get_array_module(x)
        gy0 = gy[0].reshape(-1, *((1,) * (x.ndim - 1)))
        if xp is numpy:
            gx = 2 * x * gy0
        else:
            kernel = cuda.elementwise(
                'T x, T gy', 'T gx', 'gx = 2 * x * gy',
                'l2normsquared_bwd')
            gx = kernel(x, gy0)
        return gx,


def batch_l2_norm_squared(x):
    """L2 norm (a.k.a. Euclidean norm) squared.

    This function implements the square of L2 norm on a vector. No reduction
    along batch axis is done.

    Args:
        x (~chainer.Variable): Input variable. The first dimension is assumed
            to be the *minibatch dimension*. If ``x`` has more than two
            dimensions all but the first dimension are flattened to one
            dimension.

    Returns:
        ~chainer.Variable: Two dimensional output variable.

    """
    return BatchL2NormSquared()(x)
