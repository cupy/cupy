import numpy
from chainer import cuda, Function

class MeanSquaredError(Function):
    """Mean squared error (a.k.a. Euclidean loss) function."""

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return numpy.array([diff.dot(diff) / diff.size]),

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

    This function computes mean squared error between two variables. The mean is
    taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return MeanSquaredError()(x0, x1)
