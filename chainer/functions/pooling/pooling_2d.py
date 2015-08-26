import collections
import ctypes

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn


def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return (x, x)


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
            in_types[0].dtype == numpy.float32,
            in_types[0].ndim == 4
        )

    def forward_gpu(self, x):
        # Implementation using cudnn
        n, c, h, w = x[0].shape
        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        y = cuda.empty((n, c, y_h, y_w), dtype=numpy.float32)

        handle = cudnn.get_handle()
        pool_desc = self.create_pool_desc()
        x_desc = cudnn.create_tensor_descriptor(x[0])
        y_desc = cudnn.create_tensor_descriptor(y)

        libcudnn.poolingForward(
            handle, pool_desc.value, ctypes.c_float(1), x_desc.value,
            x[0].data.ptr, ctypes.c_float(0), y_desc.value, y.data.ptr)
        self.y = y

        return y,

    def backward_gpu(self, x, gy):
        # Implementation using cudnn
        handle = cudnn.get_handle()
        pool_desc = self.create_pool_desc()

        # Pooling of cuDNNv2 does not seem to support non-contiguous gradients
        gy = cuda.cupy.ascontiguousarray(gy[0])

        x_desc = cudnn.create_tensor_descriptor(x[0])
        y_desc = cudnn.create_tensor_descriptor(gy)

        gx = cuda.empty_like(x[0])
        libcudnn.poolingBackward(
            handle, pool_desc.value, ctypes.c_float(1), y_desc.value,
            self.y.data.ptr, y_desc.value, gy.data.ptr, x_desc.value,
            x[0].data.ptr, ctypes.c_float(0), x_desc.value, gx.data.ptr)
        return gx,

    def create_pool_desc(self):
        raise NotImplementedError()
