import numpy

from chainer import cuda
from chainer import function
from chainer.functions import linear as linear_module
from chainer.utils import type_check


class NonparameterizedLinear(function.Function):

    """Nonparameterized linear class.

    .. seealso:: :class:`Linear`

    """

    def check_type_forward(self, in_types):
        type_check.expect(
            2 <= in_types.size(),
            in_types.size() <= 3,
        )
        x_type = in_types[0]
        w_type = in_types[1]

        prod = type_check.Variable(numpy.prod, 'prod')
        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim >= 2,
            w_type.ndim == 2,
            prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if in_types.size().eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, x):
        W = x[1]
        b = None
        if len(x) == 3:
            b = x[2]
        out_size, in_size = W.shape
        func = linear_module.Linear(
            in_size, out_size, initialW=W, initial_bias=b)
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


def linear(x, W, b=None, stride=1, pad=0, use_cudnn=True):
    """Nonparameterized linear function.

    Args:
        x (~chainer.Variable): Input variable.
        W (~chainer.Variable): Weight variable.
        b (~chainer.Variable): Bias variable.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`Linear`

    """

    return NonparameterizedLinear()(x, W, b)
