import ctypes
import numpy
from chainer import cuda, Function

try:
    import libcudnn
    from chainer import cudnn
    _mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_TANH']
    use_cudnn = cudnn.enabled
except:
    use_cudnn = False

class Tanh(Function):
    """Hyperbolic tangent function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def forward_cpu(self, x):
        self.y = numpy.tanh(x[0])
        return self.y,

    def forward_gpu(self, x):
        self.y = cuda.empty_like(x[0])
        if use_cudnn and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(x[0], 1, 1)
            libcudnn.cudnnActivationForward(
                handle, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(self.y))
        else:
            cuda.elementwise('float* y, const float* x', 'y[i] = tanhf(x[i])',
                             'tanh_fwd')(self.y, x[0])
        return self.y,

    def backward_cpu(self, x, gy):
        return gy[0] * (1 - self.y * self.y),

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(self.y)
        if use_cudnn and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(self.y, 1, 1)
            libcudnn.cudnnActivationBackward(
                handle, _mode, 1, desc.value, cudnn.get_ptr(self.y),
                desc.value, cudnn.get_ptr(gy[0]), desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(gx))
        else:
            cuda.elementwise(
                'float* gx, const float* y, const float* gy',
                'gx[i] = gy[i] * (1 - y[i] * y[i])',
                'tanh_bwd')(gx, self.y, gy[0])
        return gx,

def tanh(x, use_cudnn=True):
    return Tanh(use_cudnn)(x)
