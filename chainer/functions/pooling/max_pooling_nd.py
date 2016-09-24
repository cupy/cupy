import numpy

import functools
from operator import mul

from chainer import cuda
from chainer.functions.pooling import max_pooling_nd_kernel
from chainer.functions.pooling import pooling_nd
from chainer.utils import conv_nd


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn


class MaxPoolingND(pooling_nd.PoolingND):

    """Max pooling over a set of N-dimensional planes."""

    def forward_cpu(self, x):
        col = conv_nd.im2col_nd_cpu(
            x[0], self.ksize, self.stride, self.pad, pval=-float('inf'),
            cover_all=self.cover_all)
        n, c = col.shape[:2]
        mid = (len(col.shape) - 2) // 2 + 2
        ksize = col.shape[2:mid]
        outs = col.shape[mid:]
        # (n, c, k_1 * k_2 * ... * k_N, out_1, out_2, ..., out_N)
        col_shape = (n, c) + (functools.reduce(mul, ksize),) + outs
        col = col.reshape(col_shape)

        # We select maximum twice, since the implementation using numpy.choose
        # hits its bug when kh * kw >= 32.
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y,

    def forward_gpu(self, x):
        if (cuda.cudnn_enabled and self.use_cudnn and self.ndim > 1 and
                pooling_nd._check_cudnn_acceptable_type(x[0].dtype)):
            return super(MaxPoolingND, self).forward_gpu(x)

        n, c = x[0].shape[:2]
        dims = x[0].shape[2:]
        ys = tuple(conv_nd.get_conv_outsize(d, k, s, p, self.cover_all)
                   for (d, k, s, p) in zip(
                       dims, self.ksize, self.stride, self.pad))
        # (n, c, y_1, y_2, ..., y_N)
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x[0].dtype)
        self.indexes = cuda.cupy.empty(y_shape, dtype=numpy.int32)

        in_params, out_params, operation, name = \
            max_pooling_nd_kernel.MaxPoolingNDKernelForward.generate(self.ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            x[0].reduced_view(),
            *(dims + ys + self.ksize + self.stride + self.pad +
              (y, self.indexes)))

        return y,

    def backward_cpu(self, x, gy):
        n, c = gy[0].shape[:2]
        outs = gy[0].shape[2:]
        dims = x[0].shape[2:]
        # (n, c, k_1, k_2, ..., k_N, out_1, out_2, ..., out_N)
        gcol_shape = (n, c) + self.ksize + outs
        gcol = numpy.zeros(gcol_shape, dtype=x[0].dtype)

        # TODO(takagi) Make it fast
        gcol_shape = (n, c, -1) + outs
        gcol_r = numpy.rollaxis(gcol.reshape(gcol_shape), 2)
        indeces = (n, c) + outs
        for i in numpy.ndindex(indeces):
            gcol_r[self.indexes[i]][i] = gy[0][i]

        gx = conv_nd.col2im_nd_cpu(gcol, self.stride, self.pad, dims)
        return gx,

    def backward_gpu(self, x, gy):
        if (cuda.cudnn_enabled and self.use_cudnn and self.ndim > 1 and
                pooling_nd._check_cudnn_acceptable_type(x[0].dtype)):
            return super(MaxPoolingND, self).backward_gpu(x, gy)

        n, c = x[0].shape[:2]
        dims = x[0].shape[2:]
        ys = gy[0].shape[2:]
        gx = cuda.cupy.empty_like(x[0])

        ndim = self.ndim
        in_params, out_params, operation, name = \
            max_pooling_nd_kernel.MaxPoolingNDKernelBackward.generate(ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            gy[0].reduced_view(), self.indexes.reduced_view(),
            *(dims + ys + self.ksize + self.stride + self.pad + (gx,)))
        return gx,

    def create_pool_desc(self):
        return cudnn.create_pooling_descriptor(
            self.ksize, self.stride, self.pad, libcudnn.CUDNN_POOLING_MAX)


def max_pooling_nd(x, ksize, stride=None, pad=0, cover_all=True,
                   use_cudnn=True):
    """N-dimensionally spatial max pooling function.

    This function provides a N-dimensionally generalized version of
    :func:`~functions.max_pooling_2d`. This acts similarly to
    :class:`~functions.ConvolutionND`, but it computes the maximum of input
    spatial patch for each channel without any parameter instead of computing
    the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or tuple of ints): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k, ..., k)`` are equivalent.
        stride (int or tuple of ints or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s,s, ..., s)`` are equivalent. If
            ``None`` is specified, then it uses same stride as the pooling
            window size.
        pad (int or tuple of ints): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
        cover_all (bool): If ``True``, all spatial locations are pooled into
            some output pixels. It may make the output size larger.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation. cuDNN supports more than
            one-dimensional pooling.

    Returns:
        ~chainer.Variable: Output variable.

    """
    ndim = len(x.data.shape[2:])
    return MaxPoolingND(ndim, ksize, stride, pad, cover_all, use_cudnn)(x)
