import numpy
from chainer import cuda, Function

class Copy(Function):
    """Copy an input GPUArray onto another device."""

    def __init__(self, src_device, dst_device):
        self.src_device = src_device
        self.dst_device = dst_device

    def forward_cpu(self, x):
        return x[0].copy(),

    def forward_gpu(self, x):
        with cuda.using_device(self.src_device):
            if self.src_device == self.dst_device:
                return cuda.copy_async(x[0]),
            # Use sync function, since async version is not provided by PyCUDA
            return cuda.copy_peer(x[0], self.dst_device, self.src_device),

    def backward_cpu(self, x, gy):
        return gy[0].copy(),

    def backward_gpu(self, x, gy):
        with cuda.using_device(self.dst_device):
            if self.src_device == self.dst_device:
                return cuda.copy_async(gy[0]),
            # Use sync function, since async version is not provided by PyCUDA
            return cuda.copy_peer(gy[0], self.src_device, self.dst_device),

def copy(x, src, dst):
    return Copy(src, dst)(x)
