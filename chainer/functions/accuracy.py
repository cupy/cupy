import numpy
from chainer import cuda, Function

class Accuracy(Function):
    """Compute accuracy within minibatch."""

    def forward_cpu(self, inputs):
        y, t = inputs
        y = y.reshape(y.shape[0], y.size / y.shape[0])  # flatten
        pred = y.argmax(axis=1)
        return (pred == t).mean(dtype=numpy.float32),

    def forward_gpu(self, inputs):
        # Fallback to CPU
        # TODO(beam2d): Pure GPU version
        accuracy, = self.forward_cpu((a.get() for a in inputs))
        return cuda.to_gpu_async(numpy.array(accuracy)),


def accuracy(y, t):
    return Accuracy()(y, t)
