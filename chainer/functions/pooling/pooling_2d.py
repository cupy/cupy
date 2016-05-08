import collections

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()


def _check_cudnn_acceptable_type(x_dtype):
    return _cudnn_version >= 3000 or x_dtype != numpy.float16


def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return x, x


class Pooling2D(function.Function):

    """Base class of pooling function over a set of 2d planes."""

    def __init__(self, ksize, stride=None, pad=0, cover_all=True,
                 use_cudnn=True):
        if stride is None:
            stride = ksize

        self.kh, self.kw = _pair(ksize)
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)

        self.cover_all = cover_all
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 4
        )

    def forward_gpu(self, x):
        # Implementation using cudnn
        x = x[0]
        n, c, h, w = x.shape
        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        y = cuda.cupy.empty((n, c, y_h, y_w), dtype=x.dtype)

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
