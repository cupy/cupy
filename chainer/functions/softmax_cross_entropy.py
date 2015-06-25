import numpy
import six

from chainer import cuda
from chainer import function
from chainer.functions import softmax


class SoftmaxCrossEntropy(function.Function):

    """Softmax activation followed by a cross entropy loss."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def forward_cpu(self, inputs):
        x, t = inputs
        self.y, = softmax.Softmax().forward_cpu((x,))
        p = self.y[six.moves.range(len(t)), t]
        return -numpy.log(p).sum(keepdims=True) / t.size,

    def forward_gpu(self, inputs):
        x, t = inputs
        self.y, = softmax.Softmax(self.use_cudnn).forward_gpu((x,))
        ret = cuda.reduce(
            'int* t, float* y, int n_channel', '-log(y[i * n_channel + t[i]])',
            'a+b', '0', 'crossent_fwd', numpy.float32
        )(t, self.y, self.y.shape[1])
        ret /= t.size
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = self.y.copy()
        gx[six.moves.range(len(t)), t] -= 1
        gx *= gloss[0] / t.size
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = cuda.empty_like(self.y)
        coeff = gloss / t.size
        cuda.elementwise(
            '''
               float* gx, const float* y, const int* t, const float* coeff,
               int n_channel
            ''',
            'gx[i] = *coeff * (y[i] - ((i % n_channel) == t[i / n_channel]))',
            'softmax_crossent_bwd')(gx, self.y, t, coeff, self.y.shape[1])
        return gx, None


def softmax_cross_entropy(x, t, use_cudnn=True):
    """Computes cross entropy loss on softmax of the prediction using the
    groundtruth label vector.

    Args:
        x (Variable): Variable holding a matrix whose (i, j)-th element
            indicates unnormalized log probability of the class j at the i-th
            example.
        t (Variable): Variable holding an int32 vector of groundtruth labels.

    Returns:
        Variable: A variable holding a scalar array of the cross entropy loss.

    .. note::

       This function is differentiable only by ``x``.

    """
    return SoftmaxCrossEntropy(use_cudnn)(x, t)
