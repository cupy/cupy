import numpy
from chainer import cuda, cudnn, Function
from chainer.functions.softmax import Softmax

class SoftmaxCrossEntropy(Function):
    """Softmax activation followed by a cross entropy loss."""

    def forward_cpu(self, inputs):
        x, t = inputs
        self.y, = Softmax().forward_cpu((x,))
        return -numpy.log(self.y[xrange(len(t)), t]).sum(keepdims=True) / t.size,

    def forward_gpu(self, inputs):
        x, t = inputs
        self.y, = Softmax().forward_gpu((x,))
        ret = cuda.reduce(
            'int* t, float* y, int n_channel', '-log(y[i * n_channel + t[i]])',
            'a+b', '0', 'crossent_fwd', numpy.float32)(t, self.y, self.y.shape[1])
        ret /= t.size
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = self.y.copy()
        gx[xrange(len(t)), t] -= 1
        gx *= gloss[0] / t.size
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = cuda.empty_like(self.y)
        coeff = gloss / t.size
        cuda.elementwise(
            'float* gx, const float* y, const int* t, const float* coeff, int n_channel',
            'gx[i] = *coeff * (y[i] - ((i % n_channel) == t[i / n_channel]))',
            'softmax_crossent_bwd')(gx, self.y, t, coeff, self.y.shape[1])
        return gx, None


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
