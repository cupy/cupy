import numpy

from chainer import cuda
from chainer import cudnn
from chainer import function
from chainer.utils import type_check

if cudnn.available:
    from chainer.cudnn import libcudnn
    _algorithm = libcudnn.cudnnSoftmaxAlgorithm['CUDNN_SOFTMAX_ACCURATE']
    _mode = libcudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_CHANNEL']


class Softmax(function.Function):

    """Softmax activation function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim > 1,
        )

    def check_type_backward(self, in_types, out_types):
        type_check.expect(
            in_types.size() == 1,
            out_types.size() == 1,
        )
        x_type, = in_types
        y_type, = out_types

        type_check.expect(
            y_type.ndim > 1,
            y_type.shape == x_type.shape,
        )

    def forward_cpu(self, x):
        self.y = x[0] - numpy.amax(x[0], axis=1, keepdims=True)
        numpy.exp(self.y, out=self.y)
        self.y /= self.y.sum(axis=1, keepdims=True)
        return self.y,

    def forward_gpu(self, x):
        y = cuda.empty_like(x[0])
        n_units = int(numpy.prod(x[0].shape[2:]))
        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(x[0], n_units, 1)
            libcudnn.cudnnSoftmaxForward(
                handle, _algorithm, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(y))
            self.y = y
        else:
            maxes_shape = (x[0].shape[0],) + x[0].shape[2:]
            maxes = cuda.empty(maxes_shape, dtype=numpy.float32)
            c = x[0].shape[1]
            cuda.elementwise(
                'float* maxes, const float* x, int n_channel, int n_units',
                '''
                   const int n = i / n_units;
                   const int m = i % n_units;
                   const float* row = x + n * n_channel * n_units + m;
                   float maxval = row[0];
                   for (int c = 1; c < n_channel; ++c) {
                     const int v = c * n_units;
                     if (maxval < row[v]) {
                       maxval = row[v];
                     }
                   }
                   maxes[i] = maxval;
                ''', 'softmax_rowmax')(maxes, x[0], c, n_units)
            cuda.elementwise(
                '''
                   float* y, const float* x, const float* maxes,
                   int n_channel, int n_units
                ''',
                '''
                   const int n = i / (n_channel * n_units);
                   const int m = (i % (n_channel * n_units)) % n_units;
                   y[i] = __expf(x[i] - maxes[n * n_units + m]);
                ''',
                'softmax_exp')(y, x[0], maxes, c, n_units)
            coeff = maxes  # reuse memory
            cuda.elementwise(
                'float* coeff, const float* y, int n_channel, int n_units',
                '''
                   const int n = i / n_units;
                   const int m = i % n_units;
                   const float* row = y + n * n_channel * n_units + m;
                   float sum = 0;
                   for (int c = 0; c < n_channel; ++c) {
                     sum += row[c * n_units];
                   }
                   coeff[i] = 1 / sum;
                ''', 'softmax_invrowsum')(coeff, y, c, n_units)
            cuda.elementwise(
                'float* y, const float* coeff, int n_channel, int n_units',
                '''
                   const int n = i / (n_channel * n_units);
                   const int m = (i % (n_channel * n_units)) % n_units;
                   y[i] *= coeff[n * n_units + m];
                ''',
                'softmax_rowmul')(y, coeff, c, n_units)
            self.y = y

        return y,

    def backward_cpu(self, x, gy):
        gx = self.y * gy[0]
        sumdx = gx.sum(axis=1, keepdims=True)
        gx -= self.y * sumdx
        return gx,

    def backward_gpu(self, x, gy):
        n_units = int(numpy.prod(x[0].shape[2:]))
        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            gx = cuda.empty_like(x[0])
            desc = cudnn.get_tensor_desc(x[0], n_units, 1)
            libcudnn.cudnnSoftmaxBackward(
                handle, _algorithm, _mode, 1, desc.value, cudnn.get_ptr(
                    self.y),
                desc.value, cudnn.get_ptr(gy[0]), 0, desc.value,
                cudnn.get_ptr(gx))
        else:
            gx = self.y * gy[0]
            c = gx.shape[1]
            sum_ydy_shape = (gx.shape[0],) + gx.shape[2:]
            sum_ydy = cuda.empty(sum_ydy_shape, dtype=numpy.float32)
            cuda.elementwise(
                'float* sum_ydy, const float* ydy, int n_channel, int n_units',
                '''
                   int n = i / n_units;
                   int m = i % n_units;
                   const float* row = ydy + n * n_channel * n_units + m;
                   float sum = 0;
                   for (int c = 0; c < n_channel; ++c) {
                     sum += row[c * n_units];
                   }
                   sum_ydy[i] = sum;
                ''', 'softmax_bwd_sum_ydy')(sum_ydy, gx, c, n_units)
            cuda.elementwise(
                '''
                   float* gx, const float* y, const float* sum_ydy,
                   int n_channel, int n_units
                ''',
                '''
                   const int n = i / (n_channel * n_units);
                   const int m = (i % (n_channel * n_units)) % n_units;
                   gx[i] -= y[i] * sum_ydy[n * n_units + m];
                ''',
                'softmax_bwd_diff')(gx, self.y, sum_ydy, c, n_units)

        return gx,


def softmax(x, use_cudnn=True):
    """Channelwise softmax function.

    This function only accepts a two dimensional input array, and computes its
    softmax along the second axis. For each index :math:`i, j` of the input
    matrix :math:`x`, it computes
    :math:`f_{ij}(x)={\\exp(x_{ij}) \\over \\sum_j \\exp(x_{ij})}`.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If True and CuDNN is enabled, then this function uses
            CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Softmax(use_cudnn)(x)
