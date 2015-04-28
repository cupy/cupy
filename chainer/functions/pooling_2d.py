from collections import Iterable
import libcudnn
import numpy
from chainer import cuda, cudnn, Function

def _pair(x):
    if isinstance(x, Iterable):
        return x
    return (x, x)

class Pooling2D(Function):
    """Base class of pooling function over a set of 2d planes."""

    def __init__(self, ksize, stride=None, pad=0):
        if stride is None:
            stride = ksize

        self.kh, self.kw = _pair(ksize)
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)

    def forward_gpu(self, x):
        handle = cudnn.get_default_handle()
        pool_desc = self.create_pool_desc()
        x_desc = cudnn.get_tensor_desc(x[0], x[0].shape[2], x[0].shape[3])

        y_h = 1 + (x[0].shape[2] + self.ph * 2 - self.kh) / self.sy
        y_w = 1 + (x[0].shape[3] + self.pw * 2 - self.kw) / self.sx
        self.y = cuda.empty((x[0].shape[0], x[0].shape[1], y_h, y_w),
                            dtype=numpy.float32)
        y_desc = cudnn.get_tensor_desc(self.y, y_h, y_w)

        libcudnn.cudnnPoolingForward(
            handle, pool_desc.value, 1, x_desc.value, cudnn.get_ptr(x[0]),
            0, y_desc.value, cudnn.get_ptr(self.y))
        return self.y,

    def backward_gpu(self, x, gy):
        handle = cudnn.get_default_handle()
        pool_desc = self.create_pool_desc()

        x_desc = cudnn.get_tensor_desc( x[0],  x[0].shape[2],  x[0].shape[3])
        y_desc = cudnn.get_tensor_desc(gy[0], gy[0].shape[2], gy[0].shape[3])

        gx = cuda.empty_like(x[0])
        libcudnn.cudnnPoolingBackward(
            handle, pool_desc.value, 1, y_desc.value, cudnn.get_ptr(self.y),
            y_desc.value, cudnn.get_ptr(gy[0]), x_desc.value, cudnn.get_ptr(x[0]),
            0, x_desc.value, cudnn.get_ptr(gx))
        return gx,

    def create_pool_desc(self):
        raise NotImplementedError()


class MaxPooling2D(Pooling2D):
    """Max pooling over a set of 2d planes."""
    # TODO(beam2d): Implement CPU version.

    def create_pool_desc(self):
        return cudnn.get_pool2d_desc(
            (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw),
            'CUDNN_POOLING_MAX')

def max_pooling_2d(x, ksize, stride=None, pad=0):
    return MaxPooling2D(ksize, stride, pad)(x)


class AveragePooling2D(Pooling2D):
    """Average pooling over a set of 2d planes."""
    # TODO(beam2d): Implement CPU version.

    def create_pool_desc(self):
        return cudnn.get_pool2d_desc(
            (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw),
            'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING')

def average_pooling_2d(x, ksize, stride=None, pad=0):
    return AveragePooling2D(ksize, stride, pad)(x)
