import numpy
import six

from chainer import cuda
from chainer import function
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
from chainer.utils import conv_nd
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()


_check_cudnn_acceptable_type = pooling_2d._check_cudnn_acceptable_type


class _PoolingND(function.Function):

    """Base class of pooling function over a set of N-dimensional planes."""

    def __init__(self, ndim, ksize, stride=None, pad=0, cover_all=True,
                 use_cudnn=True):

        if stride is None:
            stride = ksize

        self.ndim = ndim
        self.ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = conv_nd.as_tuple(stride, ndim)
        self.pad = conv_nd.as_tuple(pad, ndim)

        self.cover_all = cover_all
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 2 + self.ndim
        )

    def forward_gpu(self, x):
        # Implementation using cuDNN.
        x = x[0]
        n, c = x.shape[:2]
        dims = x.shape[2:]
        ys = tuple(conv.get_conv_outsize(d, k, s, p, self.cover_all)
                   for d, k, s, p in six.moves.zip(
                       dims, self.ksize, self.stride, self.pad))
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)

        handle = cudnn.get_handle()
        pool_desc = self.create_pool_desc()
        x_desc = cudnn.create_tensor_descriptor(x)
        y_desc = cudnn.create_tensor_descriptor(y)

        oz_dtype = 'd' if x.dtype == 'd' else 'f'
        one = numpy.array(1, dtype=oz_dtype).ctypes
        zero = numpy.array(0, dtype=oz_dtype).ctypes
        libcudnn.poolingForward(
            handle, pool_desc.value, one.data, x_desc.value,
            x.data.ptr, zero.data, y_desc.value, y.data.ptr)
        self.y = y

        return y,

    def backward_gpu(self, x, gy):
        # Implementation using cudnn
        x = x[0]
        handle = cudnn.get_handle()
        pool_desc = self.create_pool_desc()

        # Pooling of cuDNNv2 does not seem to support non-contiguous gradients
        gy = cuda.cupy.ascontiguousarray(gy[0])

        x_desc = cudnn.create_tensor_descriptor(x)
        y_desc = cudnn.create_tensor_descriptor(gy)

        oz_dtype = 'd' if x.dtype == 'd' else 'f'
        one = numpy.array(1, dtype=oz_dtype).ctypes
        zero = numpy.array(0, dtype=oz_dtype).ctypes
        gx = cuda.cupy.empty_like(x)
        libcudnn.poolingBackward(
            handle, pool_desc.value, one.data, y_desc.value,
            self.y.data.ptr, y_desc.value, gy.data.ptr, x_desc.value,
            x.data.ptr, zero.data, x_desc.value, gx.data.ptr)
        return gx,

    def create_pool_desc(self):
        raise NotImplementedError()
