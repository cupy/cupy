import numpy
import six

from chainer import cuda
from chainer import function
from chainer.functions import softmax
from chainer.utils import type_check


class SoftmaxCrossEntropy(function.Function):

    """Softmax activation followed by a cross entropy loss."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )

    def check_type_backward(self, in_types, out_types):
        type_check.expect(
            in_types.size() == 2,
            out_types.size() == 1,
        )
        y_type, = out_types
        type_check.expect(y_type.ndim == 0)  # means scalar

    def forward_cpu(self, inputs):
        x, t = inputs
        self.y, = softmax.Softmax().forward_cpu((x,))
        yd = self.y.transpose(
            [0] + list(six.moves.range(2, self.y.ndim)) + [1])
        yd = yd.reshape(numpy.prod(yd.shape[:-1]), -1)
        p = yd[six.moves.range(t.size), t.flat]
        y = -numpy.log(p).sum(keepdims=True) / t.shape[0]
        return y.reshape(()),

    def forward_gpu(self, inputs):
        x, t = inputs
        self.y, = softmax.Softmax(self.use_cudnn).forward_gpu((x,))
        n_unit = int(numpy.prod(self.y.shape[2:]))
        # the map_expr is equivalent to the pseudo code -log(y[n, c, m]),
        # where n = i / n_unit, c = t[i], and m = i % n_unit
        ret = cuda.reduce(
            'int* t, float* y, int n_channel, int n_unit',
            '-log(y[n_unit * ((i / n_unit) * n_channel + t[i])'
            '       + (i % n_unit)])',
            'a+b', '0', 'crossent_fwd', numpy.float32
        )(t, self.y, self.y.shape[1], n_unit)
        ret /= t.shape[0]
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        if self.y.ndim == 2:
            gx = self.y.copy()
            gx[six.moves.xrange(len(t)), t] -= 1
            gx *= gloss / t.size
        else:
            # in the case where y.ndim is higher than 2,
            # we think that a current implementation is inefficient
            # because it yields two provisional arrays for indexing.
            n_unit = int(numpy.prod(self.y.shape[2:]))
            gx = self.y.copy().reshape(self.y.shape[0], self.y.shape[1], -1)
            fst_index = numpy.arange(t.size) // n_unit
            trd_index = numpy.arange(t.size) % n_unit
            gx[fst_index, t.flat, trd_index] -= 1
            gx = (gloss / t.shape[0]) * gx.reshape(self.y.shape)
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        n_unit = int(numpy.prod(self.y.shape[2:]))
        gx = cuda.empty_like(self.y)
        coeff = gloss / t.shape[0]
        cuda.elementwise(
            '''
               float* gx, const float* y, const int* t, const float* coeff,
               int n_channel, int n_unit
            ''',
            '''
               const int n = i / (n_channel * n_unit);
               const int c = (i % (n_channel * n_unit)) / n_unit;
               const int m = (i % (n_channel * n_unit)) % n_unit;
               gx[i] = *coeff * (y[i] - (c == t[n * n_unit + m]));
            ''',
            'softmax_crossent_bwd')(
                gx, self.y, t, coeff, self.y.shape[1], n_unit)
        return gx, None


def softmax_cross_entropy(x, t, use_cudnn=True):
    """Computes cross entropy loss for pre-softmax activations.

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
