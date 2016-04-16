import numpy

from chainer import cuda
from chainer import function


class HardSigmoid(function.Function):

    def forward_cpu(self, inputs):
        x = inputs[0]
        return numpy.minimum(1.0, numpy.maximum(0.0, x * 0.2 + 0.5)),

    def forward_gpu(self, inputs):
        x = inputs[0]
        return cuda.elementwise(
            'T x', 'T y',
            'y = min(1.0, max(0.0, x * 0.2 + 0.5))',
            'hard_sigmoid_fwd'
        )(x),

    def backward_cpu(self, inputs, grads):
        x = inputs[0]
        g = grads[0]
        return ((-2.5 < x) & (x < 2.5)) * g * 0.2,

    def backward_gpu(self, inputs, grads):
        x = inputs[0]
        g = grads[0]
        return cuda.elementwise(
            'T x, T g', 'T gx',
            'gx = fabs(x) < 2.5 ? 0.2 * g : 0',
            'hard_sigmoid_bwd'
        )(x, g),


def hard_sigmoid(x):
    return HardSigmoid()(x)
