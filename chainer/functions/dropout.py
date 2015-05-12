import numpy
from chainer import cuda, Function

class Dropout(Function):
    """Dropout regularization."""

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward_cpu(self, x):
        scale = numpy.float32(1. / (1 - self.dropout_ratio))
        self.mask = scale * (numpy.random.rand(*x[0].shape) >= self.dropout_ratio)
        return x[0] * self.mask,

    def forward_gpu(self, x):
        self.rand = cuda.empty_like(x[0])
        y = cuda.empty_like(x[0])

        cuda.get_generator().fill_uniform(self.rand)
        self.scale = 1. / (1 - self.dropout_ratio)

        self.kernel = cuda.elementwise(
            '''float* y, const float* x, const float* rand, float dropout_ratio,
               float scale''',
            'y[i] = rand[i] < dropout_ratio ? 0 : scale * x[i]',
            'dropout')
        self.kernel(y, x[0], self.rand, self.dropout_ratio, self.scale)
        return y,

    def backward_cpu(self, x, gy):
        return gy[0] * self.mask,

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(gy[0])
        self.kernel(gx, gy[0], self.rand, self.dropout_ratio, self.scale)
        return gx,


def dropout(x, ratio=.5, train=True):
    if train:
        return Dropout(ratio)(x)
    return x
