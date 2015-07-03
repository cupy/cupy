import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Accuracy(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= 2,
            t_type.dtype == numpy.int32,
            t_type.ndim == 1,
            t_type.shape[0] == x_type.shape[0],
        )
        for i in range(2, x_type.ndim.eval()):
            type_check.expect(x_type.shape[i] == 1)

    def forward_cpu(self, inputs):
        y, t = inputs
        y = y.reshape(y.shape[0], y.size / y.shape[0])  # flatten
        pred = y.argmax(axis=1)
        return numpy.array((pred == t).mean(dtype=numpy.float32)),

    def forward_gpu(self, inputs):
        x, t = inputs
        fragments = cuda.empty((x.shape[0],), dtype=numpy.int8)
        cuda.elementwise(
            'char* fragments, const float* x, const int* t, int c',
            '''
               x += i * c;
               float maxval = x[0];
               int   argmax = 0;
               for (int j = 1; j < c; ++j) {
                 if (maxval < x[j]) {
                   maxval = x[j];
                   argmax = j;
                 }
               }
               fragments[i] = argmax == t[i];
            ''', 'accuracy_fwd_map')(fragments, x, t, x.shape[1])
        y = cuda.gpuarray.sum(fragments, dtype=numpy.float32)
        y /= x.shape[0]
        return y,


def accuracy(y, t):
    """Computes muticlass classification accuracy of the minibatch.

    Args:
        y (Variable): Variable holding a matrix whose (i, j)-th element
            indicates the score of the class j at the i-th example.
        t (Variable): Variable holding an int32 vector of groundtruth labels.

    Returns:
        Variable: A variable holding a scalar array of the accuracy.

    .. note:: This function is non-differentiable.

    """
    return Accuracy()(y, t)
