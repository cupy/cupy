import numpy

import six

import functools
import operator

from chainer import cuda
from chainer.functions.pooling import average_pooling_nd_kernel
from chainer.functions.pooling import pooling_nd
from chainer.utils import conv_nd


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn


class AveragePoolingND(pooling_nd.PoolingND):

    """Average pooling over a set of N-dimensional planes."""
    # TODO(takagi) Support cover_all mode.

    def forward_cpu(self, x):
        col = conv_nd.im2col_nd_cpu(
            x[0], self.ksize, self.stride, self.pad, cover_all=False)

        # mean along (_, _, k_1, k_2, ..., k_N, _, ..., _)
        y_axis = tuple(six.moves.range(2, 2+len(self.ksize)))
        y = col.mean(axis=y_axis)
        return y,

    def forward_gpu(self, x):
        if (cuda.cudnn_enabled and self.use_cudnn and self.ndim > 1 and
                pooling_nd._check_cudnn_acceptable_type(x[0].dtype)):
            return super(AveragePoolingND, self).forward_gpu(x)

        n, c = x[0].shape[:2]
        dims = x[0].shape[2:]
        ys = tuple(conv_nd.get_conv_outsize(d, k, s, p, cover_all=False)
                   for (d, k, s, p) in zip(
                       dims, self.ksize, self.stride, self.pad))
        # (n, c, y_1, y_2, ..., y_N)
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x[0].dtype)
        coeff = 1. / functools.reduce(operator.mul, self.ksize)

        ndim = self.ndim
        in_params, out_params, operation, name = \
            average_pooling_nd_kernel.AveragePoolingNDForward.generate(ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            x[0].reduced_view(),
            *(dims + ys + self.ksize + self.stride + self.pad + (coeff, y)))

        return y,

    def backward_cpu(self, x, gy):
        dims = x[0].shape[2:]
        outs = gy[0].shape[2:]
        colon = slice(None, None, None)
        gy_index = (colon, colon) + (None,) * len(dims)
        gcol_reps = (1, 1) + self.ksize + (1,) * len(outs)
        gcol = numpy.tile(gy[0][gy_index], gcol_reps)
        gx = conv_nd.col2im_nd_cpu(gcol, self.stride, self.pad, dims)
        gx /= functools.reduce(operator.mul, self.ksize)
        return gx,

    def backward_gpu(self, x, gy):
        if (cuda.cudnn_enabled and self.use_cudnn and self.ndim > 1 and
                pooling_nd._check_cudnn_acceptable_type(x[0].dtype)):
            return super(AveragePoolingND, self).backward_gpu(x, gy)

        n, c = x[0].shape[:2]
        dims = x[0].shape[2:]
        ys = gy[0].shape[2:]
        gx = cuda.cupy.empty_like(x[0])
        coeff = 1. / functools.reduce(operator.mul, self.ksize)

        ndim = self.ndim
        in_params, out_params, operation, name = \
            average_pooling_nd_kernel.AveragePoolingNDKernelBackward.generate(
                ndim)
        cuda.elementwise(in_params, out_params, operation, name)(
            gy[0].reduced_view(),
            *(dims + ys + self.ksize + self.stride + self.pad + (coeff, gx)))

        return gx,

    def create_pool_desc(self):
        return cudnn.create_pooling_descriptor(
            self.ksize, self.stride, self.pad,
            libcudnn.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)


def average_pooling_nd(x, ksize, stride=None, pad=0, use_cudnn=True):
    """N-dimensionally spatial average pooling function.

    This function provides a N-dimensionally generalized version of
    :func:`~functions.average_pooling_2d`. This acts similarly to
    :class:`~functions.ConvolutionND`, but it computes the average of input
    spatial patch for each channel without any parameter instead of computing
    the inner products.

    Args:
        x(~chainer.Variable): Input variable.
        ksize (int or tuple of ints): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k, ..., k)`` are equivalent.
        stride (int or tuple of ints or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s,s, ..., s)`` are equivalent. If
            ``None`` is specified, then it uses same stride as the pooling
            window size.
        pad (int or tuple of ints): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation. cuDNN supports more than
            one-dimensional pooling.

    Returns:
        ~chainer.Variable: Output variable.

    .. note::

       This function currently does not support ``cover_all`` mode as
       :func:`max_pooling_nd`. Average pooling runs in non-cover-all mode.

    """
    ndim = len(x.data.shape[2:])
    return AveragePoolingND(ndim, ksize, stride, pad, False, use_cudnn)(x)
