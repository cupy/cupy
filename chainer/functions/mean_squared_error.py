import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class MeanSquaredError(function.Function):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return numpy.array(diff.dot(diff) / diff.size, numpy.float32),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        ret = cuda.reduce(
            'const float* x0, const float* x1',
            '(x0[i] - x1[i]) * (x0[i] - x1[i])',
            'a+b', '0', 'mse_fwd', numpy.float32)(x0, x1)
        ret /= x0.size
        return ret,

    def backward_cpu(self, inputs, gy):
        coeff = 2. * gy[0] / self.diff.size
        gx0 = coeff * self.diff
        return gx0, -gx0

    def backward_gpu(self, inputs, gy):
        x0, x1 = inputs
        gx0 = cuda.empty_like(x0)
        gx1 = cuda.empty_like(x1)
        coeff = gy[0] * (2. / x0.size)
        cuda.elementwise(
            '''float* gx0, float* gx1, const float* x0, const float* x1,
               const float* coeff''',
            '''gx0[i] = *coeff * (x0[i] - x1[i]);
               gx1[i] = -gx0[i];''',
            'mse_bwd')(gx0, gx1, x0, x1, coeff)
        return gx0, gx1


def mean_squared_error(x0, x1):
    """Mean squared error function.

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return MeanSquaredError()(x0, x1)
