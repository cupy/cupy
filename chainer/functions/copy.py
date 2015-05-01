import numpy
from chainer import cuda, Function

class Copy(Function):
    """Copy an input GPUArray onto another device."""

    def __init__(self, from_device, to_device):
        self.from_device = from_device
        self.to_device   = to_device

    def forward_cpu(self, x):
        return x[0].copy(),

    def forward_gpu(self, x):
        if self.from_device == self.to_device:
            return cuda.copy_async(x[0]),
        # Use sync function, since async version is not provided by PyCUDA
        return cuda.copy_peer(x[0], self.to_device, self.from_device),

    def backward_cpu(self, x, gy):
        return gy[0].copy(),

    def backward_gpu(self, x, gy):
        if self.from_device == self.to_device:
            return cuda.copy_async(gy[0]),
        # Use sync function, since async version is not provided by PyCUDA
        return cuda.copy_peer(gy[0], self.from_device, self.to_device),

def copy(x, from_device, to_device):
    return Copy(from_device, to_device)(x)
