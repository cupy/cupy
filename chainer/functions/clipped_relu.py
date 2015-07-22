import numpy
from chainer import cuda
from chainer import function


class ClippedReLU(function.Function):

    """clipped ReLU function. clipped ReLU is written as below,

    :math:`ClippedReLU(x, z) = min{max{0,x},z}`
    z is a parameter to cap return value of ReLU.

    """

    def __init__(self, z):
        self.cap = z

    def forward_cpu(self, x):
        return numpy.minimum(numpy.maximum(0, x[0]), self.cap),

    def backward_cpu(self, x, gy):
        return gy[0] * (0 < x[0]) * (x[0] < self.cap),

    def forward_gpu(self, x):
        return cuda.gpuarray.minimum(cuda.gpuarray.maximum(0, x[0]), self.cap),

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(x[0])
        cuda.elementwise(
            'float* gx, const float* x, const float* gy, const float z',
            'gx[i] = ((x[i] > 0) and (x[i] < z))? gy[i] : 0',
            'relu_bwd')(gx, x[0], gy[0], self.cap)
        return gx,


def clipped_relu(x, z=20):
    """Clipped Rectifier Unit function :math:`CReLU(x, z) = min{max{0,x},z}`
    Args:
        x (~chainer.Variable): Input variable.
        z (integer): clipping value. (default = 20)

    Returns:
        ~chainer.Variable: Output variable.

    """

    return ClippedReLU(z)(x)
