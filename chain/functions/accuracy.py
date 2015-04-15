import numpy
import pycuda.gpuarray as gpuarray
from chain import Function

class Accuracy(Function):
    """Compute accuracy within minibatch."""

    def forward_cpu(self, inputs):
        y, t = inputs
        pred = y.argmax(axis=1)
        return (pred == t).mean(keepdims=True),

    def forward_gpu(self, inputs):
        # Fallback to CPU
        # TODO(beam2d): Pure GPU version
        accuracy = self.forward_cpu((a.get() for a in inputs))
        return gpuarray.to_gpu(accuracy),


def accuracy(y, t):
    return Accuracy()(y, t)
