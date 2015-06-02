import numpy
from chainer import cuda, cudnn, Function

if cudnn.available:
    from chainer.cudnn import libcudnn
    _mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_RELU']

class ReLU(Function):
    """Rectified Linear Unit."""
    # TODO(beam2d): Implement in-place version.

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def forward_cpu(self, x):
        return numpy.maximum(0, x[0]),

    def forward_gpu(self, x):
        y = cuda.empty_like(x[0])
        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(x[0], 1, 1)
            libcudnn.cudnnActivationForward(
                handle, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(y))
            self.y = y
        else:
            cuda.elementwise('float* y, const float* x', 'y[i] = max(0.f, x[i])',
                             'relu_fwd')(y, x[0])
        return y,

    def backward_cpu(self, x, gy):
        return gy[0] * (x[0] > 0),

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(x[0])
        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(self.y, 1, 1)
            libcudnn.cudnnActivationBackward(
                handle, _mode, 1, desc.value, cudnn.get_ptr(self.y),
                desc.value, cudnn.get_ptr(gy[0]), desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(gx))
        else:
            cuda.elementwise(
                'float* gx, const float* x, const float* gy',
                'gx[i] = x[i] > 0 ? gy[i] : 0',
                'relu_bwd')(gx, x[0], gy[0])
        return gx,

def relu(x, use_cudnn=True):
    """Rectified Linear Unit function :math:`f(x)=\\max(0, x)`.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If True and CuDNN is enabled, then this function uses
            CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return ReLU(use_cudnn)(x)
