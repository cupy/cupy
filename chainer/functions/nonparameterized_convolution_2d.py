import numpy

from chainer import cuda
from chainer import function
from chainer.functions import convolution_2d as conv2d_module
from chainer.utils import type_check


class NonparameterizedConvolution2D(function.Function):

    """Two-dimensional nonparameterized convolution class.

    Args:
        stride (int or (int, int)): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or (int, int)): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If True, then this function uses CuDNN if available.

    .. seealso:: :class:`Convolution2D`

    """
    def __init__(self, stride=1, pad=0, use_cudnn=True):
        self.stride = stride
        self.pad = pad

        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            2 <= in_types.size(),
            in_types.size() <= 3,
        )

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[1],
        )

        if in_types.size().eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, x):
        W = x[1]
        b = None
        if len(x) == 3:
            b = x[2]
        func = conv2d_module.Convolution2D(
            W.shape[1], W.shape[0], W.shape[2:],
            stride=self.stride, pad=self.pad, use_cudnn=self.use_cudnn,
            initialW=W, initial_bias=b)
        self.func = func
        if any(isinstance(i, cuda.GPUArray) for i in x):
            func.to_gpu()
        return func.forward(x[:1])

    def backward(self, x, gy):
        func = self.func
        func.zero_grads()
        gx = func.backward(x[:1], gy)
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

    .. seealso:: :class:`Convolution2D`

    """
    return NonparameterizedConvolution2D(
        stride=stride, pad=pad, use_cudnn=use_cudnn)(x, W, b)
