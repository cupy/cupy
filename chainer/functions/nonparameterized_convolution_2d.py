import numpy

from chainer import cuda
from chainer import function
from chainer.functions import convolution_2d as conv2d_module


class NonparameterizedConvolution2D(function.Function):

    """Two-dimensional nonparameterized convolution function.

    Args:
        stride (int or (int, int)): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or (int, int)): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If True, then this function uses CuDNN if available.

    .. seealso:: :meth:`Convolution2D`

    """
    def __init__(self, stride=1, pad=0, use_cudnn=True):
        self.stride = stride
        self.pad = pad

        self.use_cudnn = use_cudnn

    def forward(self, x):
        W = x[1]
        b = None
        if len(x) == 3:
            b = x[2]
        func = conv2d_module.Convolution2D(
            W.shape[0], W.shape[1], W.shape[2:],
            stride=self.stride, pad=self.pad, use_cudnn=self.use_cudnn,
            init_params=False)
        func.W = W
        func.b = b
        self.func = func
        return func.forward(x[:1])

    def backward_cpu(self, x, gy):
        func = self.func
        func.gW = numpy.zeros_like(x[1])
        if func.b is not None:
            func.gb = numpy.zeros_like(func.b)
        gx = func.backward_cpu(x[:1], gy)
        if func.gb is None:
            return (gx[0], func.gW)
        return (gx[0], func.gW, func.gb)

    def backward_gpu(self, x, gy):
        func = self.func
        func.gW = cuda.zeros_like(func.W)
        if func.b is not None:
            func.gb = cuda.zeros_like(func.b)
        gx = func.backward_gpu(x[:1], gy)
        if func.gb is None:
            return (gx[0], func.gW)
        return (gx[0], func.gW, func.gb)


def convolution_2d(x, W, b=None, stride=1, pad=0, use_cudnn=True):
    """Two-dimensional convolution function.

    Args:
        x (~chainer.Variable): Input variable.
        W (~chainer.Variable): Weight variable.
        b (~chainer.Variable): Bias  variable.
        stride (int or (int, int)): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or (int, int)): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If True, then this function uses CuDNN if available.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :meth:`Convolution2D`

    """
    return NonparameterizedConvolution2D(
        stride=stride, pad=pad, use_cudnn=use_cudnn)(x, W, b)
