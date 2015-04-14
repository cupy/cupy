import numpy
import pycuda.gpuarray as gpuarray
from chain import Function
from softmax import Softmax

class SoftmaxCrossEntropy(Function):
    """Softmax activation followed by a cross entropy loss."""

    def forward_cpu(self, inputs):
        x, t = inputs
        self.y = x - numpy.amax(x, axis=1, keepdims=True)
        numpy.exp(self.y, out=self.y)
        self.y /= self.y.sum(axis=1, keepdims=True)
        return -numpy.log(self.y[xrange(len(t)), t]).sum(keepdims=True),

    def backward_cpu(self, inputs, grad_outputs):
        t = inputs[1]
        gloss = grad_outputs[0]
        gy = self.y
        gy[xrange(len(t)), t] -= 1
        if gloss is not None:
            gy *= gloss[0]
        return gy, None

    # TODO(beam2d): Implement GPU forward/backward

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
